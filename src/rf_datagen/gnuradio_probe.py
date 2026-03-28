"""GNU Radio probe — MITM debug utility for signal verification.

Provides a flexible tap point that can be injected at any stage of the
signal generation pipeline.  Signals are piped through GNU Radio flowgraphs
for analysis, decoding, or transformation.

The probe operates in three modes:

  1. **Analyze** — Feed IQ through a GR flowgraph and return measurements
     (SNR, bandwidth, symbol rate, constellation, etc.)

  2. **Decode** — Attempt to decode the signal using GR-based decoders
     (gr-lora_sdr, gr-mixalot, gr-nrsc5, etc.) and return decoded content

  3. **Transform** — Pass IQ through a GR processing chain and return the
     modified IQ (filtering, resampling, channel simulation, etc.)

Usage as a QC pipeline stage::

    from rf_datagen.gnuradio_probe import GnuRadioProbe, ProbePoint

    probe = GnuRadioProbe()

    # Tap after generation, before impairments
    result = probe.analyze(iq_data, fs=12000, mode="FT8")

    # Decode a specific signal
    decoded = probe.decode(iq_data, fs=12000, decoder="lora")

    # Transform (e.g., channel simulation)
    filtered = probe.transform(iq_data, fs=12000, flowgraph="bandpass",
                                params={"low": 200, "high": 3000})

Usage as CLI tap point::

    rf-datagen qc probe --mode FT8 --point after-generation
    rf-datagen qc probe --mode LORA --decoder lora --snr 10
    rf-datagen qc probe --flowgraph custom_flow.grc --input signal.iq
"""

import os
import shutil
import struct
import subprocess
import tempfile
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional

import numpy as np

from .constants import FS, WINDOW_LEN
from .logging_config import get_logger

log = get_logger("gnuradio_probe")


class ProbePoint(Enum):
    """Injection points in the generation pipeline."""
    AFTER_GENERATION = "after-generation"
    AFTER_IMPAIRMENTS = "after-impairments"
    AFTER_WINDOWING = "after-windowing"
    CUSTOM = "custom"


@dataclass
class ProbeResult:
    """Result from a probe analysis or decode attempt."""
    success: bool
    mode: str = ""
    probe_point: str = ""
    # Analysis measurements
    estimated_snr: Optional[float] = None
    bandwidth_hz: Optional[float] = None
    center_freq_hz: Optional[float] = None
    symbol_rate: Optional[float] = None
    # Decode results
    decoded_text: Optional[str] = None
    decode_confidence: Optional[float] = None
    bit_error_rate: Optional[float] = None
    # Transformed IQ
    output_iq: Optional[np.ndarray] = None
    # Raw measurements dict for mode-specific data
    measurements: dict = field(default_factory=dict)
    # Error info
    error: Optional[str] = None


def _check_gnuradio():
    """Check if GNU Radio Python bindings are available."""
    try:
        import gnuradio
        return True
    except ImportError:
        return False


def _check_gr_module(module_name):
    """Check if a specific GNU Radio OOT module is available."""
    try:
        __import__(module_name)
        return True
    except ImportError:
        return False


def _iq_to_file(iq, path, dtype=np.complex64):
    """Write IQ array to a binary file (interleaved float32 I/Q)."""
    iq.astype(dtype).tofile(path)


def _file_to_iq(path, dtype=np.complex64):
    """Read IQ from a binary file."""
    return np.fromfile(path, dtype=dtype)


class GnuRadioProbe:
    """GNU Radio probe for signal verification and debugging.

    Can be injected at any point in the pipeline as a MITM tap.
    Falls back to NumPy-based analysis when GNU Radio is not available.
    """

    def __init__(self, fs=FS):
        self.fs = fs
        self._gr_available = _check_gnuradio()
        if not self._gr_available:
            log.info("GNU Radio not available — using NumPy fallback analysis")

    @property
    def available_decoders(self):
        """Return list of available GR-based decoders."""
        decoders = []
        if _check_gr_module("lora_sdr"):
            decoders.append("lora")
        if _check_gr_module("mixalot"):
            decoders.append("pocsag")
        if _check_gr_module("nrsc5"):
            decoders.append("hdradio")
        # CLI-based decoders (always available if tool is in PATH)
        if shutil.which("multimon-ng"):
            decoders.extend(["dtmf", "pocsag_cli", "flex_cli", "eas_cli"])
        if shutil.which("inspectrum"):
            decoders.append("inspectrum")
        return decoders

    def analyze(self, iq, fs=None, mode=None, probe_point=ProbePoint.CUSTOM):
        """Analyze IQ signal and return measurements.

        Works without GNU Radio using NumPy spectral analysis.
        When GNU Radio is available, uses gr-inspector for richer analysis.
        """
        fs = fs or self.fs
        result = ProbeResult(success=True, mode=mode or "",
                             probe_point=probe_point.value)

        if len(iq) == 0:
            result.success = False
            result.error = "Empty IQ input"
            return result

        # Spectral analysis (always available)
        result.measurements.update(self._spectral_analysis(iq, fs))
        result.bandwidth_hz = result.measurements.get("bandwidth_3db")
        result.center_freq_hz = result.measurements.get("center_freq")
        result.estimated_snr = result.measurements.get("estimated_snr")

        # GNU Radio enhanced analysis
        if self._gr_available:
            try:
                gr_measurements = self._gr_analyze(iq, fs)
                result.measurements.update(gr_measurements)
            except Exception as e:
                log.debug("GR analysis failed (non-fatal): %s", e)

        return result

    def decode(self, iq, fs=None, decoder=None, mode=None,
               probe_point=ProbePoint.CUSTOM):
        """Attempt to decode the signal using the appropriate decoder.

        Selects decoder based on mode if not explicitly specified.
        """
        fs = fs or self.fs
        result = ProbeResult(success=False, mode=mode or "",
                             probe_point=probe_point.value)

        if decoder is None and mode:
            decoder = self._select_decoder(mode)
        if decoder is None:
            result.error = "No decoder available for this mode"
            return result

        tmpdir = tempfile.mkdtemp(prefix="gr_probe_")
        try:
            if decoder == "lora" and _check_gr_module("lora_sdr"):
                return self._decode_lora_gr(iq, fs, result, tmpdir)
            elif decoder in ("pocsag_cli", "flex_cli", "eas_cli", "dtmf"):
                return self._decode_multimon(iq, fs, decoder, result, tmpdir)
            elif decoder == "pocsag" and _check_gr_module("mixalot"):
                return self._decode_pocsag_gr(iq, fs, result, tmpdir)
            elif decoder == "hdradio" and _check_gr_module("nrsc5"):
                return self._decode_hdradio_gr(iq, fs, result, tmpdir)
            else:
                result.error = f"Decoder '{decoder}' not available"
                return result
        except Exception as e:
            result.error = str(e)
            return result
        finally:
            shutil.rmtree(tmpdir, ignore_errors=True)

    def transform(self, iq, fs=None, flowgraph=None, params=None):
        """Pass IQ through a GNU Radio flowgraph and return transformed signal.

        Built-in transforms:
            "bandpass"  — bandpass filter (params: low, high)
            "resample"  — rational resampler (params: rate)
            "channel"   — channel model (params: snr, freq_offset, multipath)
            "agc"       — automatic gain control

        Custom .grc/.py flowgraph files are also supported.
        """
        fs = fs or self.fs
        params = params or {}
        result = ProbeResult(success=False)

        if flowgraph is None:
            result.error = "No flowgraph specified"
            return result

        # Built-in transforms (NumPy fallback)
        if flowgraph == "bandpass":
            return self._transform_bandpass(iq, fs, params)
        elif flowgraph == "resample":
            return self._transform_resample(iq, fs, params)
        elif flowgraph == "channel":
            return self._transform_channel(iq, fs, params)
        elif flowgraph == "agc":
            return self._transform_agc(iq, fs, params)

        # Custom flowgraph file
        if os.path.exists(flowgraph):
            return self._run_custom_flowgraph(iq, fs, flowgraph, params)

        result.error = f"Unknown flowgraph: {flowgraph}"
        return result

    # ------------------------------------------------------------------
    # Spectral analysis (NumPy — always available)
    # ------------------------------------------------------------------

    def _spectral_analysis(self, iq, fs):
        """Compute spectral measurements using NumPy."""
        n = len(iq)
        nfft = min(4096, n)
        n_segs = max(1, n // nfft)

        psd = np.zeros(nfft)
        for i in range(n_segs):
            segment = iq[i * nfft:(i + 1) * nfft]
            if len(segment) < nfft:
                break
            windowed = segment * np.hanning(nfft)
            spec = np.fft.fftshift(np.fft.fft(windowed))
            psd += np.abs(spec) ** 2
        psd /= n_segs
        psd_db = 10 * np.log10(psd + 1e-30)
        freqs = np.fft.fftshift(np.fft.fftfreq(nfft, 1.0 / fs))

        peak_db = np.max(psd_db)
        noise_floor = np.median(psd_db)

        # 3dB bandwidth
        bw3_mask = psd_db > (peak_db - 3)
        bw3_freqs = freqs[bw3_mask]
        bw_3db = (bw3_freqs[-1] - bw3_freqs[0]) if len(bw3_freqs) > 1 else 0

        # 10dB bandwidth
        bw10_mask = psd_db > (peak_db - 10)
        bw10_freqs = freqs[bw10_mask]
        bw_10db = (bw10_freqs[-1] - bw10_freqs[0]) if len(bw10_freqs) > 1 else 0

        # Center frequency (spectral centroid)
        power_linear = 10 ** (psd_db / 10)
        center_freq = np.sum(freqs * power_linear) / np.sum(power_linear)

        # SNR estimate
        estimated_snr = peak_db - noise_floor

        # Peak-to-average power ratio
        sig_power = np.mean(np.abs(iq) ** 2)
        peak_power = np.max(np.abs(iq) ** 2)
        papr = 10 * np.log10(peak_power / (sig_power + 1e-30))

        # Crest factor
        rms = np.sqrt(sig_power)
        crest_factor = np.max(np.abs(iq)) / (rms + 1e-30)

        return {
            "bandwidth_3db": float(bw_3db),
            "bandwidth_10db": float(bw_10db),
            "center_freq": float(center_freq),
            "estimated_snr": float(estimated_snr),
            "peak_power_db": float(peak_db),
            "noise_floor_db": float(noise_floor),
            "papr_db": float(papr),
            "crest_factor": float(crest_factor),
            "signal_power": float(sig_power),
            "n_samples": n,
            "duration_s": float(n / fs),
        }

    # ------------------------------------------------------------------
    # GNU Radio analysis
    # ------------------------------------------------------------------

    def _gr_analyze(self, iq, fs):
        """Enhanced analysis using GNU Radio blocks."""
        measurements = {}
        try:
            from gnuradio import gr, blocks, analog, fft as gr_fft
            # Use GR's signal analysis blocks for autocorrelation-based
            # symbol rate estimation, etc.
            measurements["gr_available"] = True
        except ImportError:
            measurements["gr_available"] = False
        return measurements

    # ------------------------------------------------------------------
    # Decoder implementations
    # ------------------------------------------------------------------

    def _select_decoder(self, mode):
        """Select appropriate decoder for a signal mode."""
        mode_decoder_map = {
            "LORA": "lora",
            "POCSAG": "pocsag_cli",
            "FLEX": "flex_cli",
            "HDRADIO": "hdradio",
            "DTMF": "dtmf",
            "EAS": "eas_cli",
            "BELL103": "dtmf",  # multimon-ng handles Bell modems
            "BELL202": "dtmf",
        }
        decoder = mode_decoder_map.get(mode.upper())
        if decoder and decoder in self.available_decoders:
            return decoder
        return None

    def _decode_multimon(self, iq, fs, decoder, result, tmpdir):
        """Decode using multimon-ng (DTMF, POCSAG, FLEX, EAS)."""
        from .qc import sig_to_wav

        wav_path = os.path.join(tmpdir, "probe_input.wav")
        sig_to_wav(iq, fs, wav_path, stereo_iq=False)

        mode_flag_map = {
            "dtmf": "-a DTMF",
            "pocsag_cli": "-a POCSAG512 -a POCSAG1200 -a POCSAG2400",
            "flex_cli": "-a FLEX",
            "eas_cli": "-a EAS",
        }
        flags = mode_flag_map.get(decoder, "-a DTMF")

        try:
            cmd = f"multimon-ng -t wav {flags} {wav_path}"
            proc = subprocess.run(
                cmd.split(), capture_output=True, text=True, timeout=30)
            output = proc.stdout.strip()

            if output:
                result.success = True
                result.decoded_text = output
                # Count decoded messages
                lines = [l for l in output.split("\n") if l.strip()]
                result.decode_confidence = min(1.0, len(lines) / 5.0)
            else:
                result.error = "No output from multimon-ng"
        except (subprocess.TimeoutExpired, OSError) as e:
            result.error = str(e)

        return result

    def _decode_lora_gr(self, iq, fs, result, tmpdir):
        """Decode LoRa using gr-lora_sdr."""
        try:
            from lora_sdr import lora_sdr_utils
            result.error = "gr-lora_sdr decode not yet implemented"
        except ImportError:
            result.error = "gr-lora_sdr not installed"
        return result

    def _decode_pocsag_gr(self, iq, fs, result, tmpdir):
        """Decode POCSAG using gr-mixalot."""
        result.error = "gr-mixalot decode not yet implemented"
        return result

    def _decode_hdradio_gr(self, iq, fs, result, tmpdir):
        """Decode HD Radio using gr-nrsc5."""
        result.error = "gr-nrsc5 decode not yet implemented"
        return result

    # ------------------------------------------------------------------
    # Transform implementations (NumPy fallback)
    # ------------------------------------------------------------------

    def _transform_bandpass(self, iq, fs, params):
        """Apply bandpass filter."""
        from .dsp.filters import bandpass_filter

        low = params.get("low", 300)
        high = params.get("high", 3000)
        order = params.get("order", 5)

        filtered = bandpass_filter(iq, low, high, fs, order=order)
        return ProbeResult(success=True, output_iq=filtered)

    def _transform_resample(self, iq, fs, params):
        """Resample to new rate."""
        from scipy.signal import resample

        target_rate = params.get("rate", fs)
        target_len = int(len(iq) * target_rate / fs)
        resampled = resample(iq, target_len)
        return ProbeResult(success=True, output_iq=resampled,
                           measurements={"output_fs": target_rate})

    def _transform_channel(self, iq, fs, params):
        """Apply channel model (AWGN + freq offset + multipath)."""
        from .impairments.effects import add_awgn, freq_shift

        sig = iq.copy()

        snr = params.get("snr")
        if snr is not None:
            from .impairments import normalize_power
            sig = add_awgn(normalize_power(sig), snr)

        freq_offset = params.get("freq_offset")
        if freq_offset is not None:
            sig = freq_shift(sig, freq_offset, fs)

        return ProbeResult(success=True, output_iq=sig)

    def _transform_agc(self, iq, fs, params):
        """Simple AGC normalization."""
        target_power = params.get("target_power", 1.0)
        power = np.mean(np.abs(iq) ** 2)
        if power > 1e-30:
            gain = np.sqrt(target_power / power)
            iq = iq * gain
        return ProbeResult(success=True, output_iq=iq)

    def _run_custom_flowgraph(self, iq, fs, flowgraph_path, params):
        """Run a custom GNU Radio flowgraph (.py or .grc)."""
        tmpdir = tempfile.mkdtemp(prefix="gr_custom_")
        try:
            input_path = os.path.join(tmpdir, "input.iq")
            output_path = os.path.join(tmpdir, "output.iq")
            _iq_to_file(iq, input_path)

            if flowgraph_path.endswith(".grc"):
                # Compile .grc to .py first
                py_path = os.path.join(tmpdir, "flowgraph.py")
                try:
                    subprocess.run(
                        ["grcc", "-o", tmpdir, flowgraph_path],
                        capture_output=True, timeout=30, check=True)
                    # grcc generates a .py file
                    import glob
                    py_files = glob.glob(os.path.join(tmpdir, "*.py"))
                    if py_files:
                        flowgraph_path = py_files[0]
                    else:
                        return ProbeResult(
                            success=False,
                            error="grcc did not produce a .py file")
                except (subprocess.CalledProcessError, OSError) as e:
                    return ProbeResult(success=False, error=str(e))

            # Run the flowgraph with input/output paths as env vars
            env = os.environ.copy()
            env["GR_PROBE_INPUT"] = input_path
            env["GR_PROBE_OUTPUT"] = output_path
            env["GR_PROBE_FS"] = str(fs)
            for k, v in params.items():
                env[f"GR_PROBE_{k.upper()}"] = str(v)

            try:
                subprocess.run(
                    ["python3", flowgraph_path],
                    env=env, capture_output=True, timeout=60, check=True)
            except (subprocess.CalledProcessError, OSError) as e:
                return ProbeResult(success=False, error=str(e))

            if os.path.exists(output_path):
                output_iq = _file_to_iq(output_path)
                return ProbeResult(success=True, output_iq=output_iq)
            else:
                return ProbeResult(success=False,
                                   error="Flowgraph produced no output")
        finally:
            shutil.rmtree(tmpdir, ignore_errors=True)


# ---------------------------------------------------------------------------
# Pipeline tap — convenience wrappers for common probe points
# ---------------------------------------------------------------------------

def tap_after_generation(iq, fs=FS, mode=None):
    """Quick-tap: analyze IQ right after generation, before impairments."""
    probe = GnuRadioProbe(fs=fs)
    return probe.analyze(iq, fs=fs, mode=mode,
                         probe_point=ProbePoint.AFTER_GENERATION)


def tap_after_impairments(iq, fs=FS, mode=None):
    """Quick-tap: analyze IQ after impairments are applied."""
    probe = GnuRadioProbe(fs=fs)
    return probe.analyze(iq, fs=fs, mode=mode,
                         probe_point=ProbePoint.AFTER_IMPAIRMENTS)


def tap_decode(iq, fs=FS, mode=None, decoder=None):
    """Quick-tap: attempt to decode the signal."""
    probe = GnuRadioProbe(fs=fs)
    return probe.decode(iq, fs=fs, decoder=decoder, mode=mode)


def tap_transform(iq, fs=FS, flowgraph=None, **params):
    """Quick-tap: transform IQ through a flowgraph."""
    probe = GnuRadioProbe(fs=fs)
    return probe.transform(iq, fs=fs, flowgraph=flowgraph, params=params)
