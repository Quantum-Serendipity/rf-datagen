"""DSD-CC validators — DMR, D-STAR, YSF, NXDN, P25."""

import subprocess

import numpy as np
from scipy.signal import resample

from rf_datagen.constants import FS
from rf_datagen.generators.digivoice import generate_tier2

from ._base import BaseRoundtripValidator, register
from ._helpers import add_awgn_iq, generate_test_speech


@register("DMR", "DSTAR", "YSF", "NXDN", "P25")
class DSDCCValidator(BaseRoundtripValidator):
    """Validate digital voice framing via dsdccx sync detection."""

    required_tools = ["dsdccx", "c2enc"]
    tier = 2

    MODE_FLAGS = {
        "DMR":   "-fr",
        "DSTAR": "-fd",
        "YSF":   "-fy",
        "NXDN":  "-fn",
        "P25":   "-fp",
    }

    def generate_iq(self, mode, tmpdir):
        speech_path = generate_test_speech(tmpdir, duration_s=2.0)
        iq = generate_tier2(speech_path, tmpdir, mode)
        if len(iq) < 100:
            return None
        return iq

    def iq_to_discriminator(self, iq):
        disc = np.angle(iq[1:] * np.conj(iq[:-1]))
        target_len = int(len(disc) * 8000 / FS)
        if target_len < 100:
            return None
        disc_8k = resample(disc, target_len)
        peak = np.max(np.abs(disc_8k))
        if peak > 0:
            disc_8k /= peak
        return (disc_8k * 16000).astype(np.int16)

    def decode(self, disc_s16, mode):
        flag = self.MODE_FLAGS.get(mode)
        if not flag:
            return False
        try:
            result = subprocess.run(
                ["dsdccx", flag, "-i", "-", "-o", "/dev/null", "-n", "-q"],
                input=disc_s16.tobytes(),
                capture_output=True, timeout=30)
        except (subprocess.TimeoutExpired, FileNotFoundError):
            return False
        output = (result.stdout + result.stderr).decode("utf-8", errors="replace")
        sync_indicators = ["Sync", "sync", "Frame", "frame",
                           "voice", "Voice", "data", "Data"]
        return any(ind in output for ind in sync_indicators)

    def make_trial(self, mode):
        def run(snr_db, trial_idx, tmpdir):
            iq = self.generate_iq(mode, tmpdir)
            if iq is None:
                return False
            if snr_db is not None:
                iq = add_awgn_iq(iq, snr_db)
            disc = self.iq_to_discriminator(iq)
            if disc is None:
                return False
            return self.decode(disc, mode)
        return run
