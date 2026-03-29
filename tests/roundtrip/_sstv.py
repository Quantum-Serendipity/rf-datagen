"""SSTV spectral validator."""

import numpy as np

from rf_datagen.generators.sstv import encode_sstv, SSTV_AUDIO_FS

from ._base import BaseRoundtripValidator, register
from ._helpers import add_awgn_audio


@register("SSTV")
class SSTVSpectralValidator(BaseRoundtripValidator):
    """Validate SSTV signals by checking spectral structure."""

    required_tools = []
    tier = 3

    def generate(self, tmpdir):
        mode = np.random.choice(["Robot36", "MartinM1", "ScottieS1"])
        try:
            audio = encode_sstv(mode)
            if len(audio) < 1000:
                return None, None
            return audio, mode
        except Exception:
            return None, None

    def validate_spectral(self, audio):
        from scipy.fft import rfft, rfftfreq

        n = len(audio)
        if n < 4096:
            return False
        header_len = min(int(SSTV_AUDIO_FS * 2), n)
        header = audio[:header_len]
        freqs = rfftfreq(header_len, 1 / SSTV_AUDIO_FS)
        spectrum = np.abs(rfft(header))

        sync_band = (freqs >= 1100) & (freqs <= 1300)
        cal_band = (freqs >= 1800) & (freqs <= 2000)
        video_band = (freqs >= 1500) & (freqs <= 2300)
        total_band = (freqs >= 500) & (freqs <= 3000)

        sync_energy = np.sum(spectrum[sync_band] ** 2)
        cal_energy = np.sum(spectrum[cal_band] ** 2)
        video_energy = np.sum(spectrum[video_band] ** 2)
        total_energy = np.sum(spectrum[total_band] ** 2)

        if total_energy < 1e-10:
            return False

        sync_ratio = (sync_energy + cal_energy) / total_energy
        video_ratio = video_energy / total_energy

        return sync_ratio > 0.05 and video_ratio > 0.3

    def make_trial(self, mode):
        def run(snr_db, trial_idx, tmpdir):
            audio, sstv_mode = self.generate(tmpdir)
            if audio is None:
                return False
            if snr_db is not None:
                audio = add_awgn_audio(audio, snr_db)
            return self.validate_spectral(audio)
        return run
