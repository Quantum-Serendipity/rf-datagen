"""WSJT-X validators — FT8, WSPR."""

import os
import subprocess

import numpy as np

from rf_datagen.generators.wsjtx import encode_ft8, SYNTH_FS
from rf_datagen.content.ham_text import gen_ft8_message, gen_wspr_message

from ._base import BaseRoundtripValidator, register
from ._helpers import write_mono_wav, add_awgn_audio


@register("FT8")
class FT8Validator(BaseRoundtripValidator):
    """Validate FT8 via jt9 --ft8 decoder."""

    required_tools = ["jt9"]
    tier = 1

    def generate(self, message):
        return encode_ft8(message)

    def prepare_wav(self, audio, tmpdir, snr_db=None):
        total_samples = int(15 * SYNTH_FS)
        offset_samples = int(0.5 * SYNTH_FS)
        padded = np.zeros(total_samples, dtype=np.float64)
        end = min(offset_samples + len(audio), total_samples)
        padded[offset_samples:end] = audio[:end - offset_samples]
        if snr_db is not None:
            padded = add_awgn_audio(padded, snr_db)
        wav_path = os.path.join(tmpdir, "ft8_signal.wav")
        write_mono_wav(padded, SYNTH_FS, wav_path)
        return wav_path

    def decode(self, wav_path, tmpdir):
        decode_dir = os.path.join(tmpdir, "decode")
        os.makedirs(decode_dir, exist_ok=True)
        cmd = ["jt9", "--ft8", "-p", "15", "-d", "3",
               "-a", decode_dir, "-t", decode_dir, wav_path]
        try:
            result = subprocess.run(
                cmd, capture_output=True, text=True, timeout=60)
        except subprocess.TimeoutExpired:
            return []
        messages = []
        for line in result.stdout.strip().split("\n"):
            line = line.strip()
            if not line:
                continue
            parts = line.split()
            if len(parts) >= 5:
                try:
                    float(parts[0])
                    float(parts[1])
                    float(parts[2])
                    float(parts[3])
                    msg_start = 4
                    if parts[msg_start] == "~":
                        msg_start = 5
                    msg = " ".join(parts[msg_start:])
                    messages.append(msg)
                except (ValueError, IndexError):
                    continue
        return messages

    def compare(self, decoded_messages, expected_message):
        expected_upper = expected_message.upper().strip()
        for msg in decoded_messages:
            msg_norm = " ".join(msg.upper().split())
            if expected_upper in msg_norm or msg_norm in expected_upper:
                return True
            exp_parts = expected_upper.split()
            if len(exp_parts) >= 2:
                matches = sum(1 for p in exp_parts if p in msg_norm)
                if matches >= len(exp_parts) - 1:
                    return True
        return False

    def make_trial(self, mode):
        def run(snr_db, trial_idx, tmpdir):
            msg = gen_ft8_message()
            audio = self.generate(msg)
            if audio is None or len(audio) == 0:
                return False
            wav_path = self.prepare_wav(audio, tmpdir, snr_db)
            decoded = self.decode(wav_path, tmpdir)
            return self.compare(decoded, msg)
        return run


@register("WSPR")
class WSPRValidator(BaseRoundtripValidator):
    """Validate WSPR via fst4sim encoder + jt9 --fst4w decoder."""

    required_tools = ["jt9", "fst4sim"]
    tier = 1

    WSPR_AUDIO_FREQ = 1500.0

    def generate(self, message, tmpdir, snr_db=None):
        snr_arg = "99" if snr_db is None else str(snr_db)
        try:
            result = subprocess.run(
                ["fst4sim", message, "120", str(self.WSPR_AUDIO_FREQ),
                 "0.0", "0.0", "1.0", "1", snr_arg, "T"],
                capture_output=True, text=True, timeout=30,
                cwd=tmpdir)
            if result.returncode != 0:
                return None
        except subprocess.TimeoutExpired:
            return None
        wav_path = os.path.join(tmpdir, "000000_0001.wav")
        if not os.path.exists(wav_path):
            return None
        return wav_path

    def decode(self, wav_path, tmpdir):
        decode_dir = os.path.join(tmpdir, "decode")
        os.makedirs(decode_dir, exist_ok=True)
        cmd = ["jt9", "--fst4w", "-p", "120", "-d", "3",
               "-f", str(int(self.WSPR_AUDIO_FREQ)), "-F", "200",
               "-a", decode_dir, "-t", decode_dir, wav_path]
        try:
            result = subprocess.run(
                cmd, capture_output=True, text=True, timeout=120)
        except subprocess.TimeoutExpired:
            return []
        messages = []
        for line in result.stdout.strip().split("\n"):
            line = line.strip()
            if not line or "DecodeFinished" in line:
                continue
            parts = line.split()
            if len(parts) >= 5:
                try:
                    float(parts[1])
                    float(parts[2])
                    float(parts[3])
                    msg_start = 4
                    if parts[msg_start] == "`":
                        msg_start = 5
                    msg = " ".join(parts[msg_start:])
                    messages.append(msg)
                except (ValueError, IndexError):
                    continue
        return messages

    def compare(self, decoded_messages, expected_message):
        parts = expected_message.split()
        if len(parts) < 1:
            return False
        expected_call = parts[0].upper()
        expected_grid = parts[1].upper() if len(parts) > 1 else ""
        expected_power = parts[2] if len(parts) > 2 else ""
        for msg in decoded_messages:
            msg_upper = msg.upper()
            if expected_call in msg_upper:
                return True
            if "<" in msg and expected_grid and expected_power:
                if expected_grid in msg_upper and expected_power in msg:
                    return True
        return False

    def make_trial(self, mode):
        def run(snr_db, trial_idx, tmpdir):
            msg = gen_wspr_message()
            wav_path = self.generate(msg, tmpdir, snr_db)
            if wav_path is None:
                return False
            decoded = self.decode(wav_path, tmpdir)
            return self.compare(decoded, msg)
        return run
