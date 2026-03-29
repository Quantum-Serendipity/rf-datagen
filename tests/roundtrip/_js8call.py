"""JS8Call validator via js8call TCP API.

JS8 uses a GFSK modulation similar to FT8 but with a continuous QSO
protocol. Validation requires a headless js8call instance.
"""

import os
import shutil
import subprocess
import tempfile

import numpy as np

from rf_datagen.generators.wsjtx import SYNTH_FS

from ._base import BaseRoundtripValidator, register
from ._helpers import write_mono_wav, add_awgn_audio


@register("JS8")
class JS8Validator(BaseRoundtripValidator):
    """Validate JS8 round-trip.

    Since js8call headless mode is complex to orchestrate, this uses
    spectral validation as a fallback: verify the 8-GFSK structure
    matches expected bandwidth and symbol timing.
    """

    required_tools = ["jt9"]
    tier = 2
    expected_fail = True  # Spectral-only validation for now

    def make_trial(self, mode):
        def run(snr_db, trial_idx, tmpdir):
            from rf_datagen.generators.synthetic import synth_js8
            from rf_datagen.constants import FS
            from scipy.fft import rfft, rfftfreq

            iq = synth_js8()
            if len(iq) < 100:
                return False

            # Spectral validation: JS8 should have energy in a narrow
            # band (~50 Hz) centered around the carrier
            if snr_db is not None:
                from ._helpers import add_awgn_iq
                iq = add_awgn_iq(iq, snr_db)

            spectrum = np.abs(np.fft.fft(iq)) ** 2
            freqs = np.fft.fftfreq(len(iq), 1.0 / FS)

            # Signal bandwidth should be < 100 Hz (8-GFSK at ~6.25 baud)
            total_power = np.sum(spectrum)
            if total_power < 1e-20:
                return False

            # Find peak frequency
            peak_idx = np.argmax(spectrum)
            peak_freq = abs(freqs[peak_idx])

            # Energy within ±100 Hz of peak should contain >30% of power
            mask = np.abs(np.abs(freqs) - peak_freq) < 100
            band_power = np.sum(spectrum[mask])
            ratio = band_power / total_power

            return ratio > 0.3

        return run
