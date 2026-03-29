"""CW validator via multimon-ng MORSE_CW decoder."""

import os
import subprocess

import numpy as np

from ._base import BaseRoundtripValidator, register
from ._helpers import write_mono_wav, add_awgn_audio


@register("CW")
class CWValidator(BaseRoundtripValidator):
    """Validate CW round-trip via multimon-ng MORSE_CW decoder."""

    required_tools = ["multimon-ng"]
    tier = 1
    CW_FREQ = 700.0

    def generate(self, text, tmpdir):
        fs = 8000
        wpm = np.random.choice([15, 18, 20, 25])
        dit_dur = 1.2 / wpm
        dah_dur = dit_dur * 3
        gap_dur = dit_dur

        MORSE = {
            'A': '.-', 'B': '-...', 'C': '-.-.', 'D': '-..', 'E': '.',
            'F': '..-.', 'G': '--.', 'H': '....', 'I': '..', 'J': '.---',
            'K': '-.-', 'L': '.-..', 'M': '--', 'N': '-.', 'O': '---',
            'P': '.--.', 'Q': '--.-', 'R': '.-.', 'S': '...', 'T': '-',
            'U': '..-', 'V': '...-', 'W': '.--', 'X': '-..-', 'Y': '-.--',
            'Z': '--..', '0': '-----', '1': '.----', '2': '..---',
            '3': '...--', '4': '....-', '5': '.....', '6': '-....',
            '7': '--...', '8': '---..', '9': '----.',
        }

        audio_parts = []
        for ch in text.upper():
            if ch == ' ':
                audio_parts.append(np.zeros(int(fs * dit_dur * 4)))
                continue
            code = MORSE.get(ch, '')
            for i, elem in enumerate(code):
                dur = dit_dur if elem == '.' else dah_dur
                n = int(fs * dur)
                t = np.arange(n) / fs
                tone = 0.8 * np.sin(2 * np.pi * self.CW_FREQ * t)
                ramp = min(int(fs * 0.005), n // 4)
                if ramp > 0:
                    tone[:ramp] *= np.linspace(0, 1, ramp)
                    tone[-ramp:] *= np.linspace(1, 0, ramp)
                audio_parts.append(tone)
                if i < len(code) - 1:
                    audio_parts.append(np.zeros(int(fs * gap_dur)))
            audio_parts.append(np.zeros(int(fs * dit_dur * 3)))

        if not audio_parts:
            return None
        audio = np.concatenate(audio_parts)
        return audio, fs

    def decode(self, wav_path):
        try:
            result = subprocess.run(
                ["multimon-ng", "-t", "wav", "-a", "MORSE_CW", wav_path],
                capture_output=True, text=True, timeout=30)
        except (subprocess.TimeoutExpired, FileNotFoundError):
            return ""
        lines = result.stdout.strip().split("\n")
        decoded = []
        for line in lines:
            line = line.strip()
            if line.startswith("Enabled demodulators:") or not line:
                continue
            decoded.append(line)
        return " ".join(decoded)

    def compare(self, decoded_text, expected_text):
        expected = "".join(c for c in expected_text.upper() if c.isalnum())
        decoded = "".join(c for c in decoded_text.upper() if c.isalnum())
        if not expected:
            return False
        matches = sum(1 for c in expected if c in decoded)
        return matches >= len(expected) * 0.5

    def make_trial(self, mode):
        words = ["CQ", "TEST", "DE", "W1AW", "K", "73", "QSL", "RST", "599"]
        def run(snr_db, trial_idx, tmpdir):
            text = " ".join(np.random.choice(words, size=3))
            result = self.generate(text, tmpdir)
            if result is None:
                return False
            audio, fs = result
            if snr_db is not None:
                audio = add_awgn_audio(audio, snr_db)
            wav_path = os.path.join(tmpdir, "cw_signal.wav")
            write_mono_wav(audio, fs, wav_path)
            decoded = self.decode(wav_path)
            return self.compare(decoded, text)
        return run
