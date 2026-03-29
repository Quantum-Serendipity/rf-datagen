"""Minimodem validators — BELL103, BELL202."""

import os
import subprocess

import numpy as np

from rf_datagen.generators.minimodem import generate_minimodem, MINIMODEM_FS

from ._base import BaseRoundtripValidator, register
from ._helpers import write_mono_wav, add_awgn_audio


# Minimodem mode params: mode_key -> (baud_rate, class_name)
_MODE_MAP = {
    "BELL103": ("300",  "bell103"),
    "BELL202": ("1200", "bell202"),
}


@register("BELL103", "BELL202")
class MinimodemValidator(BaseRoundtripValidator):
    """Validate BELL103/BELL202 via minimodem TX→RX pipe."""

    required_tools = ["minimodem"]
    tier = 1

    def make_trial(self, mode):
        baud, mm_mode = _MODE_MAP[mode]

        def run(snr_db, trial_idx, tmpdir):
            # Generate known text
            text = f"CQ CQ DE W1AW TEST {trial_idx} K\n"

            # TX: generate audio via minimodem
            audio = generate_minimodem(mm_mode, text)
            if len(audio) < 1000:
                return False

            # Add noise
            if snr_db is not None:
                audio = add_awgn_audio(audio, snr_db)

            # Write WAV for RX
            wav_path = os.path.join(tmpdir, f"minimodem_{mode}.wav")
            write_mono_wav(audio, MINIMODEM_FS, wav_path)

            # RX: decode via minimodem --rx
            cmd = ["minimodem", "--rx", baud,
                   "-R", str(MINIMODEM_FS),
                   "--float-samples",
                   "-f", wav_path]
            try:
                result = subprocess.run(
                    cmd, capture_output=True, text=True, timeout=30)
            except (subprocess.TimeoutExpired, FileNotFoundError):
                return False

            decoded = result.stdout.strip()
            if not decoded:
                return False

            # Compare: check if key words from TX appear in decoded
            expected_words = text.upper().split()
            decoded_upper = decoded.upper()
            matches = sum(1 for w in expected_words if w in decoded_upper)
            return matches >= len(expected_words) * 0.5

        return run
