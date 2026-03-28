"""MSK144 generator — uses msk144gensim CLI encoder."""

import shutil
import subprocess
import tempfile

import numpy as np

from ..constants import FS
from ..dsp import audio_to_iq
from ..content.ham_text import gen_ft8_message
from .base import BaseGenerator

SYNTH_FS = 12000


def encode_msk144(message, tmpdir):
    """Encode an MSK144 message using msk144gensim, returns audio array."""
    try:
        result = subprocess.run(
            ["msk144gensim", message],
            capture_output=True, text=True, timeout=15, cwd=tmpdir,
        )
    except (subprocess.TimeoutExpired, OSError):
        return None
    if result.returncode != 0:
        return None

    # msk144gensim outputs space-separated float samples on stdout
    lines = result.stdout.strip().split("\n")
    samples = []
    for line in lines:
        parts = line.strip().split()
        for p in parts:
            try:
                samples.append(float(p))
            except ValueError:
                continue
    if len(samples) < 100:
        return None
    return np.array(samples, dtype=np.float64)


class Msk144Generator(BaseGenerator):
    name = "msk144"
    required_tools = ["msk144gensim"]
    signal_classes = ["MSK144"]

    def generate_class(self, class_name, rng=None):
        tmpdir = tempfile.mkdtemp(prefix="msk144_gen_")
        try:
            segments = []
            n_messages = self.config.messages_per_mode

            for _ in range(n_messages):
                msg = gen_ft8_message()
                audio = encode_msk144(msg, tmpdir)
                if audio is not None and len(audio) > 0:
                    iq = audio_to_iq(audio, SYNTH_FS, target_fs=self.fs)
                    if len(iq) >= self.window_len:
                        segments.append(iq)

            if not segments:
                return np.array([], dtype=np.complex128)
            return np.concatenate(segments)
        finally:
            shutil.rmtree(tmpdir, ignore_errors=True)
