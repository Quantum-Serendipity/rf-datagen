"""OP25 digital voice generator — authentic DMR, D-STAR, YSF, P25 via dv_tx.py."""

import os
import shutil
import subprocess
import tempfile

import numpy as np
from scipy.signal import resample as sig_resample

from ..constants import FS
from ..dsp import audio_to_iq
from ..content.ham_text import gen_speech_text
from ..content.tts import (TTSEngine, apply_ptt_transients, apply_mic_effects,
                            apply_tx_audio_clipping)
from ..logging_config import get_logger
from .base import BaseGenerator

log = get_logger("op25")

OP25_PROTOCOLS = {
    "DMR":   {"flag": "dmr",   "rate": 48000},
    "DSTAR": {"flag": "dstar", "rate": 48000},
    "YSF":   {"flag": "ysf",   "rate": 48000},
    "P25":   {"flag": "p25",   "rate": 48000},
}


def generate_dv_audio(raw_8k_path, protocol, tmpdir):
    """Generate digital voice baseband audio using OP25 dv_tx.py.

    Args:
        raw_8k_path: Path to raw 8kHz s16le speech input.
        protocol: One of "dmr", "dstar", "ysf", "p25".
        tmpdir: Temp directory for output.

    Returns:
        (audio_array, sample_rate) or (empty_array, 0).
    """
    out_path = os.path.join(tmpdir, f"op25_{protocol}.raw")
    params = OP25_PROTOCOLS.get(protocol.upper(), {})
    flag = params.get("flag", protocol.lower())
    rate = params.get("rate", 48000)

    try:
        result = subprocess.run(
            ["dv_tx.py", "-p", flag, "-f", raw_8k_path, "-o", out_path],
            capture_output=True, text=True, timeout=60,
        )
    except (subprocess.TimeoutExpired, OSError):
        return np.array([]), 0

    if result.returncode != 0:
        return np.array([]), 0

    if not os.path.exists(out_path) or os.path.getsize(out_path) == 0:
        return np.array([]), 0

    try:
        raw = np.fromfile(out_path, dtype=np.int16)
        return raw.astype(np.float64) / 32768.0, rate
    except Exception as e:
        log.debug("op25 tx_imbe failed: %s", e)
        return np.array([]), 0
    finally:
        try:
            os.remove(out_path)
        except OSError:
            pass


class Op25Generator(BaseGenerator):
    name = "op25"
    required_tools = ["dv_tx.py", "piper"]
    signal_classes = list(OP25_PROTOCOLS.keys())

    def generate_class(self, class_name, rng=None):
        tmpdir = tempfile.mkdtemp(prefix="op25_gen_")
        voice_cache = self.config.voice_cache
        tts = TTSEngine(voice_cache)

        try:
            segments = []
            n_utterances = self.config.utterances_per_class

            for _ in range(n_utterances):
                text, _ = gen_speech_text()
                audio, wav_fs = tts.synthesize(text, tmpdir)
                if len(audio) < 1000:
                    continue

                # Apply mic/TX effects like digivoice generator
                audio = apply_mic_effects(audio, wav_fs)
                audio = apply_tx_audio_clipping(audio, wav_fs)
                audio = apply_ptt_transients(audio, wav_fs)

                # Resample to 8kHz for vocoder input
                audio_8k = sig_resample(audio, int(len(audio) * 8000 / wav_fs))
                raw_path = os.path.join(tmpdir, "speech_8k.raw")
                (audio_8k * 32767).astype(np.int16).tofile(raw_path)

                # Generate via OP25
                dv_audio, sr = generate_dv_audio(
                    raw_path, class_name, tmpdir)
                if len(dv_audio) < 1000 or sr == 0:
                    continue

                iq = audio_to_iq(dv_audio, sr, target_fs=self.fs)
                if len(iq) >= self.window_len:
                    segments.append(iq)

            if not segments:
                return np.array([], dtype=np.complex128)
            return np.concatenate(segments)

        finally:
            shutil.rmtree(tmpdir, ignore_errors=True)
