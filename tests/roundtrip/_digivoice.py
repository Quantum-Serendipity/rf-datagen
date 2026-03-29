"""FreeDV and M17 validators."""

import os
import shutil
import subprocess
import tempfile

import numpy as np

from ._base import BaseRoundtripValidator, register
from ._helpers import add_awgn_audio, generate_test_speech


@register("FREEDV")
class FreeDVValidator(BaseRoundtripValidator):
    """Validate FreeDV round-trip via freedv_tx/freedv_rx."""

    required_tools = ["freedv_tx", "freedv_rx"]
    tier = 2

    def __init__(self):
        self._speech_tmpdir = None
        self._speech_path = None

    def setup(self):
        self._speech_tmpdir = tempfile.mkdtemp(prefix="val_speech_")
        self._speech_path = generate_test_speech(self._speech_tmpdir)

    def teardown(self):
        if self._speech_tmpdir:
            shutil.rmtree(self._speech_tmpdir, ignore_errors=True)

    def generate(self, speech_raw_path, tmpdir, submode="1600"):
        out_path = os.path.join(tmpdir, f"freedv_{submode}_modem.raw")
        try:
            result = subprocess.run(
                ["freedv_tx", submode, speech_raw_path, out_path],
                capture_output=True, timeout=30)
            if result.returncode != 0:
                return None
        except (subprocess.TimeoutExpired, FileNotFoundError):
            return None
        if not os.path.exists(out_path) or os.path.getsize(out_path) < 100:
            return None
        return out_path

    def add_noise(self, modem_path, snr_db):
        audio = np.fromfile(modem_path, dtype=np.int16).astype(np.float64) / 32768.0
        noisy = add_awgn_audio(audio, snr_db)
        noisy_s16 = (np.clip(noisy, -1, 1) * 32767).astype(np.int16)
        out_path = modem_path.replace(".raw", "_noisy.raw")
        noisy_s16.tofile(out_path)
        return out_path

    def decode(self, modem_path, tmpdir, submode="1600"):
        decoded_path = os.path.join(tmpdir, f"freedv_{submode}_decoded.raw")
        try:
            result = subprocess.run(
                ["freedv_rx", submode, modem_path, decoded_path],
                capture_output=True, timeout=30)
        except subprocess.TimeoutExpired:
            return False
        if not os.path.exists(decoded_path):
            return False
        return os.path.getsize(decoded_path) > 0

    def make_trial(self, mode):
        def run(snr_db, trial_idx, tmpdir):
            modem_path = self.generate(self._speech_path, tmpdir, "1600")
            if modem_path is None:
                return False
            if snr_db is not None:
                modem_path = self.add_noise(modem_path, snr_db)
            return self.decode(modem_path, tmpdir, "1600")
        return run


@register("M17")
class M17Validator(BaseRoundtripValidator):
    """Validate M17 round-trip via m17-mod/m17-demod."""

    required_tools = ["m17-mod", "m17-demod"]
    tier = 2

    def __init__(self):
        self._speech_tmpdir = None
        self._speech_path = None

    def setup(self):
        self._speech_tmpdir = tempfile.mkdtemp(prefix="val_speech_")
        self._speech_path = generate_test_speech(self._speech_tmpdir)

    def teardown(self):
        if self._speech_tmpdir:
            shutil.rmtree(self._speech_tmpdir, ignore_errors=True)

    def generate(self, speech_raw_path, tmpdir):
        out_path = os.path.join(tmpdir, "m17_baseband.raw")
        try:
            with open(speech_raw_path, "rb") as infile:
                result = subprocess.run(
                    ["m17-mod", "-S", "N0CALL"],
                    stdin=infile, capture_output=True, timeout=30)
            if result.returncode != 0 or len(result.stdout) < 100:
                return None
        except (subprocess.TimeoutExpired, FileNotFoundError):
            return None
        with open(out_path, "wb") as f:
            f.write(result.stdout)
        return out_path

    def add_noise(self, baseband_path, snr_db):
        audio = np.fromfile(baseband_path, dtype=np.int16).astype(np.float64) / 32768.0
        noisy = add_awgn_audio(audio, snr_db)
        noisy_s16 = (np.clip(noisy, -1, 1) * 32767).astype(np.int16)
        out_path = baseband_path.replace(".raw", "_noisy.raw")
        noisy_s16.tofile(out_path)
        return out_path

    def decode(self, baseband_path):
        try:
            with open(baseband_path, "rb") as infile:
                result = subprocess.run(
                    ["m17-demod", "-l"],
                    stdin=infile, capture_output=True, timeout=30)
        except subprocess.TimeoutExpired:
            return False
        return len(result.stdout) > 0

    def make_trial(self, mode):
        def run(snr_db, trial_idx, tmpdir):
            baseband_path = self.generate(self._speech_path, tmpdir)
            if baseband_path is None:
                return False
            if snr_db is not None:
                baseband_path = self.add_noise(baseband_path, snr_db)
            return self.decode(baseband_path)
        return run
