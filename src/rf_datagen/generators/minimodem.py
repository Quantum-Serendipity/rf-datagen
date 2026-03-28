"""Minimodem generator — RTTY, Bell 103, Bell 202 via minimodem CLI."""

import subprocess

import numpy as np

from ..constants import FS
from ..dsp import audio_to_iq
from ..content.ham_text import gen_minimodem_text
from .base import BaseGenerator

MINIMODEM_FS = 48000

# Mapping from our mode names to minimodem CLI arguments
MODE_PARAMS = {
    "rtty":    {"args": ["--tx", "45.45"], "class": "RTTY"},
    "bell103": {"args": ["--tx", "300"],   "class": "BELL103"},
    "bell202": {"args": ["--tx", "1200"],  "class": "BELL202"},
}


def generate_minimodem(mode, text):
    """Generate audio via minimodem stdin->stdout pipe.

    minimodem writes raw float32 audio to stdout at 48 kHz.
    """
    params = MODE_PARAMS[mode]
    cmd = ["minimodem"] + params["args"] + [
        "-R", str(MINIMODEM_FS),   # sample rate
        "--float-samples",          # float32 output on stdout
    ]
    try:
        result = subprocess.run(
            cmd, input=text.encode("ascii", errors="replace"),
            capture_output=True, timeout=60)
    except (subprocess.TimeoutExpired, OSError):
        return np.array([])
    if result.returncode != 0:
        return np.array([])
    if len(result.stdout) < 400:  # need at least some samples
        return np.array([])

    audio = np.frombuffer(result.stdout, dtype=np.float32).astype(np.float64)
    return audio


class MinimodemGenerator(BaseGenerator):
    name = "minimodem"
    required_tools = ["minimodem"]
    signal_classes = ["RTTY", "BELL103", "BELL202"]

    def _active_modes(self):
        """Return minimodem modes to generate based on config."""
        configured = self.config.minimodem_modes
        return [m for m in configured if m in MODE_PARAMS]

    def generate_class(self, class_name, rng=None):
        # Find which minimodem mode produces this class
        target_modes = [
            m for m, p in MODE_PARAMS.items() if p["class"] == class_name
        ]
        if not target_modes:
            return np.array([], dtype=np.complex128)

        mode = target_modes[0]
        segments = []
        n_messages = self.config.messages_per_mode

        for _ in range(n_messages):
            text = gen_minimodem_text(mode.upper() if mode == "rtty" else mode)
            audio = generate_minimodem(mode, text)
            if len(audio) < 1000:
                continue
            iq = audio_to_iq(audio, MINIMODEM_FS, target_fs=self.fs)
            if len(iq) >= self.window_len:
                segments.append(iq)

        if not segments:
            return np.array([], dtype=np.complex128)
        return np.concatenate(segments)
