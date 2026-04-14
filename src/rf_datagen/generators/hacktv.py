"""HackTV generator — analog TV signal via hacktv CLI (IQ file output)."""

import os
import shutil
import struct
import subprocess
import tempfile

import numpy as np

from ..constants import FS
from ..dsp import audio_to_iq
from ..logging_config import get_logger
from .base import BaseGenerator

log = get_logger("hacktv")

HACKTV_FS = 48000  # output sample rate for hacktv

# TV standards to cycle through
TV_STANDARDS = ["pal", "ntsc", "secam"]

# Test patterns available in hacktv
TEST_PATTERNS = [
    "test:colourbars",
    "test:pm5544",
    "test:ueitm",
]


def generate_hacktv_iq(standard, pattern, tmpdir, duration_s=2.0):
    """Generate analog TV IQ using hacktv.

    hacktv can output complex IQ to a file using -o file:path.
    At 48 kHz sample rate, we capture the baseband structure of the
    analog TV signal — sync pulses, color burst, and luminance patterns.

    Returns complex IQ array or empty array on failure.
    """
    out_path = os.path.join(tmpdir, "hacktv_out.iq")
    n_samples = int(duration_s * HACKTV_FS)

    cmd = [
        "hacktv",
        "-m", standard,
        "-s", str(HACKTV_FS),
        "-o", f"file:{out_path}",
        pattern,
    ]

    try:
        result = subprocess.run(
            cmd, capture_output=True, text=True, timeout=30)
    except (subprocess.TimeoutExpired, OSError):
        return np.array([], dtype=np.complex128)

    if result.returncode != 0:
        return np.array([], dtype=np.complex128)

    if not os.path.exists(out_path) or os.path.getsize(out_path) == 0:
        return np.array([], dtype=np.complex128)

    try:
        # hacktv outputs interleaved int16 I/Q samples
        raw = np.fromfile(out_path, dtype=np.int16)
        if len(raw) < 4:
            return np.array([], dtype=np.complex128)

        # Deinterleave I/Q
        raw = raw[:len(raw) - len(raw) % 2]
        i_samples = raw[0::2].astype(np.float64) / 32768.0
        q_samples = raw[1::2].astype(np.float64) / 32768.0
        iq = i_samples + 1j * q_samples

        # Resample to target fs
        if HACKTV_FS != FS:
            from scipy.signal import resample
            target_len = int(len(iq) * FS / HACKTV_FS)
            if target_len < 1:
                return np.array([], dtype=np.complex128)
            iq = resample(iq, target_len)

        return iq

    except Exception as e:
        log.warning("HackTV IQ processing failed: %s", e)
        return np.array([], dtype=np.complex128)
    finally:
        try:
            os.remove(out_path)
        except OSError:
            pass


class HacktvGenerator(BaseGenerator):
    name = "hacktv"
    required_tools = ["hacktv"]
    signal_classes = ["ATV"]

    def generate_class(self, class_name, rng=None):
        tmpdir = tempfile.mkdtemp(prefix="hacktv_gen_")

        try:
            segments = []
            n_messages = self.config.messages_per_mode

            for i in range(n_messages):
                standard = TV_STANDARDS[i % len(TV_STANDARDS)]
                pattern = TEST_PATTERNS[i % len(TEST_PATTERNS)]
                duration = np.random.uniform(1.0, 3.0)

                iq = generate_hacktv_iq(standard, pattern, tmpdir,
                                        duration_s=duration)
                if len(iq) >= self.window_len:
                    segments.append(iq)

            if not segments:
                return np.array([], dtype=np.complex128)
            return np.concatenate(segments)

        finally:
            shutil.rmtree(tmpdir, ignore_errors=True)
