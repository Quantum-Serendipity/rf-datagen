"""SAME/EAS generator — uses sameeas Python package to generate SAME AFSK WAV."""

import os
import shutil
import subprocess
import tempfile
import wave

import numpy as np

from ..constants import FS
from ..dsp import audio_to_iq
from ..content.ham_text import gen_same_message
from ..logging_config import get_logger
from .base import BaseGenerator

log = get_logger("sameeas")

SAMEEAS_FS = 22050  # typical sameeas output rate


def _try_import_sameeas():
    """Try to import sameeas package, return module or None."""
    try:
        import sameeas
        return sameeas
    except ImportError:
        return None


def generate_same_signal(message):
    """Generate EAS/SAME audio from a SAME header string.

    Tries Python import first, then falls back to CLI script.
    Returns float64 audio array and sample rate, or (empty, 0).
    """
    mod = _try_import_sameeas()
    if mod is not None:
        try:
            # sameeas.generate() returns (audio_data, sample_rate)
            audio, sr = mod.generate(message)
            if isinstance(audio, bytes):
                audio = np.frombuffer(audio, dtype=np.int16)
            return audio.astype(np.float64) / 32768.0, sr
        except Exception as e:
            log.warning("SAME/EAS generation failed: %s", e)

    # Fallback: try CLI script
    tmpdir = tempfile.mkdtemp(prefix="sameeas_")
    wav_path = os.path.join(tmpdir, "eas.wav")
    try:
        result = subprocess.run(
            ["sameeas", "--message", message, "--output", wav_path],
            capture_output=True, text=True, timeout=30)
        if result.returncode != 0:
            return np.array([]), 0
        if not os.path.exists(wav_path) or os.path.getsize(wav_path) == 0:
            return np.array([]), 0

        with wave.open(wav_path, "rb") as wf:
            sr = wf.getframerate()
            nframes = wf.getnframes()
            raw = np.frombuffer(wf.readframes(nframes), dtype=np.int16)
        return raw.astype(np.float64) / 32768.0, sr
    except Exception as e:
        log.warning("SAME/EAS CLI fallback failed: %s", e)
        return np.array([]), 0
    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)


class SameeasGenerator(BaseGenerator):
    name = "sameeas"
    required_tools = []  # checked dynamically via import or CLI

    signal_classes = ["EAS"]

    def check_prerequisites(self):
        if _try_import_sameeas() is not None:
            return []
        if shutil.which("sameeas") is not None:
            return []
        return ["sameeas"]

    def generate_class(self, class_name, rng=None):
        segments = []
        n_messages = self.config.messages_per_mode

        for _ in range(n_messages):
            msg = gen_same_message()
            audio, sr = generate_same_signal(msg)
            if len(audio) < 1000 or sr == 0:
                continue
            iq = audio_to_iq(audio, sr, target_fs=self.fs)
            if len(iq) >= self.window_len:
                segments.append(iq)

        if not segments:
            return np.array([], dtype=np.complex128)
        return np.concatenate(segments)
