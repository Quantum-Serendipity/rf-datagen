"""CW CLI generator — uses ebook2cw (primary) or cwwav (fallback)."""

import os
import shutil
import subprocess
import tempfile
import wave

import numpy as np

from ..constants import FS
from ..dsp import audio_to_iq
from ..content.ham_text import CW_PHRASES
from ..logging_config import get_logger
from .base import BaseGenerator

log = get_logger("cw")


def _get_cw_text():
    parts = []
    n_parts = np.random.randint(2, 6)
    for _ in range(n_parts):
        parts.append(np.random.choice(CW_PHRASES))
    return " ".join(parts)


def _generate_ebook2cw(text, wpm, tone_freq, tmpdir):
    """Generate CW WAV using ebook2cw."""
    out_prefix = os.path.join(tmpdir, "cw")
    cmd = [
        "ebook2cw", "-w", str(wpm), "-f", str(int(tone_freq)),
        "-o", out_prefix, "-s", "48000", "-Q", "h",
        "-T", "1",  # single output file
    ]
    try:
        result = subprocess.run(
            cmd, input=text, capture_output=True, text=True, timeout=30)
    except (subprocess.TimeoutExpired, OSError):
        return np.array([])
    if result.returncode != 0:
        return np.array([])

    # ebook2cw produces files like cw0000.wav
    wav_path = out_prefix + "0000.wav"
    if not os.path.exists(wav_path) or os.path.getsize(wav_path) == 0:
        return np.array([])

    try:
        with wave.open(wav_path, "rb") as wf:
            nframes = wf.getnframes()
            raw = np.frombuffer(wf.readframes(nframes), dtype=np.int16)
        return raw.astype(np.float64) / 32768.0
    except Exception as e:
        log.warning("CW tool failed: %s", e)
        return np.array([])
    finally:
        try:
            os.remove(wav_path)
        except OSError:
            pass


def _generate_cwwav(text, wpm, tone_freq, tmpdir):
    """Generate CW WAV using cwwav (fallback)."""
    wav_path = os.path.join(tmpdir, "cw_out.wav")
    cmd = [
        "cwwav", "-w", str(wpm), "-f", str(int(tone_freq)),
        "-o", wav_path, "--rate", "48000",
    ]
    try:
        result = subprocess.run(
            cmd, input=text, capture_output=True, text=True, timeout=30)
    except (subprocess.TimeoutExpired, OSError):
        return np.array([])
    if result.returncode != 0:
        return np.array([])
    if not os.path.exists(wav_path) or os.path.getsize(wav_path) == 0:
        return np.array([])

    try:
        with wave.open(wav_path, "rb") as wf:
            nframes = wf.getnframes()
            raw = np.frombuffer(wf.readframes(nframes), dtype=np.int16)
        return raw.astype(np.float64) / 32768.0
    except Exception as e:
        log.warning("CW tool failed: %s", e)
        return np.array([])
    finally:
        try:
            os.remove(wav_path)
        except OSError:
            pass


class CwCliGenerator(BaseGenerator):
    name = "cw"
    required_tools = []  # checked dynamically — ebook2cw or cwwav
    signal_classes = ["CW"]

    def check_prerequisites(self):
        if shutil.which("ebook2cw") or shutil.which("cwwav"):
            return []
        return ["ebook2cw"]

    def generate_class(self, class_name, rng=None):
        use_ebook2cw = shutil.which("ebook2cw") is not None
        tmpdir = tempfile.mkdtemp(prefix="cw_gen_")
        try:
            segments = []
            wpm_lo, wpm_hi = self.config.cw_wpm_range
            tone_lo, tone_hi = self.config.cw_tone_range
            n_messages = self.config.messages_per_mode

            for _ in range(n_messages):
                text = _get_cw_text()
                wpm = int(np.random.uniform(wpm_lo, wpm_hi))
                tone = int(np.random.uniform(tone_lo, tone_hi))

                if use_ebook2cw:
                    audio = _generate_ebook2cw(text, wpm, tone, tmpdir)
                else:
                    audio = _generate_cwwav(text, wpm, tone, tmpdir)

                if len(audio) < 1000:
                    continue
                iq = audio_to_iq(audio, 48000, target_fs=self.fs)
                if len(iq) >= self.window_len:
                    segments.append(iq)

            if not segments:
                return np.array([], dtype=np.complex128)
            return np.concatenate(segments)
        finally:
            shutil.rmtree(tmpdir, ignore_errors=True)
