"""Digital voice generator — FreeDV, M17, DMR, D-STAR, YSF, P25, NXDN."""

import os
import shutil
import subprocess
import tempfile
import wave

import numpy as np
from scipy.signal import resample

from ..constants import FS, WINDOW_LEN
from ..dsp import audio_to_iq, _4fsk_mod, _gmsk_mod
from ..dsp.filters import rrc_filter
from ..content.ham_text import gen_speech_text
from ..content.tts import (TTSEngine, apply_ptt_transients, apply_mic_effects,
                            apply_tx_audio_clipping)
from ..protocols import frame_dmr, frame_dstar, frame_ysf, frame_p25, frame_nxdn
from ..impairments import extract_windows, apply_impairments, configure_impairments
from ..logging_config import get_logger
from ..output import atomic_save_npy, atomic_write_csv
from .base import BaseGenerator

log = get_logger("digivoice")

DIGIVOICE_MODES = ["FREEDV", "M17", "DMR", "DSTAR", "YSF", "P25", "NXDN"]
TIER1_MODES = {"FREEDV", "M17"}
TIER2_MODES = {"DMR", "DSTAR", "YSF", "P25", "NXDN"}


def modulate_4fsk(dibit_stream, sym_rate, dev_outer, dev_inner, rolloff=0.2, fs=FS):
    """4-level FSK modulator with RRC pulse shaping."""
    dibit_to_dev = {
        0b01: dev_outer, 0b00: dev_inner,
        0b10: -dev_inner, 0b11: -dev_outer,
    }
    sps = max(1, int(fs / sym_rate))
    n = len(dibit_stream) * sps
    freq = np.zeros(n)
    for i, d in enumerate(dibit_stream):
        freq[i * sps:(i + 1) * sps] = dibit_to_dev.get(int(d) & 0x03, 0.0)
    # RRC pulse shaping
    n_taps = min(8 * sps, n)
    if n_taps > 2:
        h = rrc_filter(n_taps, rolloff, sps)
        from scipy.signal import fftconvolve
        freq = fftconvolve(freq, h, mode="same")
    phase = 2 * np.pi * np.cumsum(freq) / fs
    phase += np.random.uniform(0, 2 * np.pi)
    return np.exp(1j * phase)


def modulate_gmsk(bit_stream, bit_rate, bt=0.5, fs=FS):
    """GMSK modulator for D-STAR."""
    return _gmsk_mod(bit_stream, bit_rate, bt, fs)


def codec2_encode(raw_8k_path, tmpdir, mode="3200"):
    """Encode raw 8kHz s16le audio with Codec2, return encoded bits."""
    c2_path = os.path.join(tmpdir, "codec2.bin")
    try:
        result = subprocess.run(
            ["c2enc", mode, raw_8k_path, c2_path],
            capture_output=True, timeout=30,
        )
    except (subprocess.TimeoutExpired, OSError):
        return None
    if result.returncode != 0 or not os.path.exists(c2_path):
        return None
    data = np.fromfile(c2_path, dtype=np.uint8)
    bits = np.unpackbits(data)
    try:
        os.remove(c2_path)
    except OSError:
        pass
    return bits


def generate_freedv(raw_8k_path, tmpdir, submode=None, *, target_fs=FS,
                    freedv_modes=None):
    """Generate FreeDV IQ using freedv_tx CLI."""
    if submode is None:
        modes = freedv_modes or ["1600", "700C", "700D", "700E"]
        submode = np.random.choice(modes)
    out_path = os.path.join(tmpdir, "freedv_out.raw")
    try:
        result = subprocess.run(
            ["freedv_tx", submode, raw_8k_path, out_path],
            capture_output=True, timeout=30,
        )
    except (subprocess.TimeoutExpired, OSError):
        return np.array([], dtype=np.complex128)
    if result.returncode != 0 or not os.path.exists(out_path):
        return np.array([], dtype=np.complex128)
    raw = np.fromfile(out_path, dtype=np.int16)
    try:
        os.remove(out_path)
    except OSError:
        pass
    if len(raw) == 0:
        return np.array([], dtype=np.complex128)
    return audio_to_iq(raw, 8000, target_fs=target_fs)


def generate_m17(raw_8k_path, tmpdir, *, target_fs=FS):
    """Generate M17 IQ using m17-mod CLI."""
    out_path = os.path.join(tmpdir, "m17_out.raw")
    try:
        with open(raw_8k_path, "rb") as fin:
            result = subprocess.run(
                ["m17-mod", "-S", "W1AW", "-D", "ALL"],
                stdin=fin, capture_output=True, timeout=30,
            )
    except (subprocess.TimeoutExpired, OSError):
        return np.array([], dtype=np.complex128)
    if result.returncode != 0 or len(result.stdout) == 0:
        return np.array([], dtype=np.complex128)
    raw = np.frombuffer(result.stdout, dtype=np.int16)
    if len(raw) == 0:
        return np.array([], dtype=np.complex128)
    return audio_to_iq(raw, 48000, target_fs=target_fs)


def generate_tier2(raw_8k_path, tmpdir, mode, *, fs=FS, codec2_mode="3200"):
    """Generate tier2 digital voice (DMR, D-STAR, YSF, P25, NXDN)."""
    codec_bits = codec2_encode(raw_8k_path, tmpdir, mode=codec2_mode)
    if codec_bits is None or len(codec_bits) < 100:
        codec_bits = np.random.randint(0, 2, 5000).astype(np.uint8)

    if mode == "DMR":
        dibits = frame_dmr(codec_bits)
        sig = modulate_4fsk(dibits, 4800, 1944, 648, fs=fs)
        # TDMA envelope
        frame_samples = int(0.060 * fs)
        slot_samples = int(0.0275 * fs)
        gap_samples = frame_samples - 2 * slot_samples
        n = len(sig)
        env = np.zeros(n)
        pos = 0
        while pos < n:
            end1 = min(pos + slot_samples, n)
            env[pos:end1] = 1.0
            pos += slot_samples + gap_samples // 2
            end2 = min(pos + slot_samples, n)
            env[pos:end2] = 1.0
            pos += slot_samples + gap_samples // 2
        return sig * env
    elif mode == "DSTAR":
        bits = frame_dstar(codec_bits)
        return modulate_gmsk(bits, 4800, bt=0.5, fs=fs)
    elif mode == "YSF":
        dibits = frame_ysf(codec_bits)
        return modulate_4fsk(dibits, 4800, 1800, 600, fs=fs)
    elif mode == "P25":
        dibits = frame_p25(codec_bits)
        return modulate_4fsk(dibits, 4800, 1800, 600, fs=fs)
    elif mode == "NXDN":
        if np.random.random() < 0.5:
            dibits = frame_nxdn(codec_bits)
            return modulate_4fsk(dibits, 4800, 2400, 800, fs=fs)
        else:
            dibits = frame_nxdn(codec_bits)
            return modulate_4fsk(dibits, 2400, 1050, 350, fs=fs)
    return np.array([], dtype=np.complex128)


class DigivoiceGenerator(BaseGenerator):
    name = "digivoice"
    required_tools = ["freedv_tx", "m17-mod", "c2enc", "piper"]
    signal_classes = DIGIVOICE_MODES

    def generate_class(self, class_name, rng=None):
        raise NotImplementedError("Use run() for digivoice generation")

    def run(self, output_dir, seed=42):
        # Pre-flight: skip entirely if all classes cached
        cached = self._check_all_cached(output_dir)
        if cached is not None:
            log.info("digivoice: all %d classes cached — skipping",
                     len(cached))
            return cached

        configure_impairments(self.impairment_config)

        parts_dir = os.path.join(output_dir, "parts", self.name)
        os.makedirs(parts_dir, exist_ok=True)

        classes = self._resolve_classes()
        results = {}
        if not classes:
            return results

        voice_cache = self.config.voice_cache
        utterances = self.config.utterances_per_class
        tts = TTSEngine(voice_cache)
        stride = self.impairment_config.effective_stride(self.window_len)
        power_threshold = self.impairment_config.window_power_threshold

        # Check tool availability
        tier1_available = {}
        for tool, mode in [("freedv_tx", "FREEDV"), ("m17-mod", "M17")]:
            tier1_available[mode] = shutil.which(tool) is not None

        tmpdir = tempfile.mkdtemp(prefix="digivoice_gen_")

        try:
            for mode_name in classes:
                n_samples = self._boosted_count(mode_name)
                npy_path = os.path.join(parts_dir, f"{mode_name}.npy")
                meta_path = os.path.join(parts_dir, f"{mode_name}_meta.csv")
                hash_path = os.path.join(parts_dir, f"{mode_name}.hash")
                cfg_hash = self._config_hash(mode_name, n_samples)

                if self._check_checkpoint(npy_path, meta_path, hash_path,
                                          n_samples, cfg_hash):
                    log.info("%15s: cached", mode_name)
                    results[mode_name] = {"status": "cached",
                                          "samples": n_samples}
                    continue

                # Two-tier cache: try raw stream before expensive codecs
                cached_raw = self._load_raw_stream(parts_dir, mode_name)
                if cached_raw is not None:
                    log.info("%15s: raw stream cached, re-windowing...",
                             mode_name)
                    combined = cached_raw
                else:
                    mode_iq_segments = []
                    np.random.seed(seed + hash(mode_name) % 10000)

                    for i in range(utterances):
                        text, _ = gen_speech_text()
                        audio, wav_fs = tts.synthesize(text, tmpdir)
                        if len(audio) < 1000:
                            continue

                        audio = apply_mic_effects(audio, wav_fs)
                        audio = apply_tx_audio_clipping(audio, wav_fs)
                        audio = apply_ptt_transients(audio, wav_fs)

                        # Write raw 8kHz s16le for codec input
                        from scipy.signal import resample as sig_resample
                        audio_8k = sig_resample(audio, int(len(audio) * 8000 / wav_fs))
                        raw_path = os.path.join(tmpdir, "speech_8k.raw")
                        (audio_8k * 32767).astype(np.int16).tofile(raw_path)

                        if mode_name in TIER1_MODES:
                            if not tier1_available.get(mode_name, False):
                                continue
                            if mode_name == "FREEDV":
                                iq = generate_freedv(raw_path, tmpdir, target_fs=self.fs,
                                                     freedv_modes=self.config.freedv_modes)
                            else:
                                iq = generate_m17(raw_path, tmpdir, target_fs=self.fs)
                        else:
                            iq = generate_tier2(raw_path, tmpdir, mode_name, fs=self.fs,
                                                codec2_mode=self.config.codec2_mode)

                        if len(iq) >= self.window_len:
                            mode_iq_segments.append(iq)

                    if not mode_iq_segments:
                        log.warning("%15s: FAILED (no audio segments)", mode_name)
                        results[mode_name] = {"status": "failed",
                                              "reason": "no audio segments"}
                        continue

                    combined = np.concatenate(mode_iq_segments)
                    del mode_iq_segments
                    self._save_raw_stream(parts_dir, mode_name, combined)

                raw_windows = extract_windows(
                    combined, window_len=self.window_len,
                    stride=stride, power_threshold=power_threshold,
                    max_windows=n_samples)
                del combined
                if len(raw_windows) == 0:
                    log.warning("%15s: FAILED (no valid windows)", mode_name)
                    results[mode_name] = {"status": "failed",
                                          "reason": "no valid windows"}
                    continue

                samples, meta = apply_impairments(
                    raw_windows, n_samples, fs=self.fs,
                    window_len=self.window_len, return_metadata=True)

                atomic_save_npy(npy_path, samples)
                snrs = meta.get("snrs", [])
                meta_rows = []
                for i, s in enumerate(meta["scenarios"]):
                    snr = snrs[i] if i < len(snrs) else ""
                    meta_rows.append([s, snr])
                atomic_write_csv(meta_path, ["scenario", "snr"], meta_rows)
                self._write_hash(hash_path, cfg_hash)

                log.info("%15s: %d raw -> %d samples",
                         mode_name, len(raw_windows), len(samples))
                results[mode_name] = {"status": "ok",
                                      "samples": len(samples),
                                      "raw_windows": len(raw_windows)}

        finally:
            shutil.rmtree(tmpdir, ignore_errors=True)

        return results
