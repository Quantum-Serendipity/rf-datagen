"""Tests for wideband (20 MHz) signal synthesizers.

Uses a reduced window length for test speed — full WIDEBAND_WINDOW_LEN
(2M samples) is only needed for actual dataset generation.
"""

import numpy as np
import pytest

from rf_datagen.generators.synthetic_wideband import (
    WIDEBAND_SYNTHESIZERS, WIDEBAND_CLASSES,
)
from rf_datagen.domains import WIDEBAND

_WIDEBAND_FS = WIDEBAND.sample_rate

# Reduced window for test speed (still exercises all signal generation logic)
_TEST_WL = 65536


@pytest.mark.parametrize("mode", WIDEBAND_CLASSES)
def test_wideband_synth_output_valid(mode):
    """Each wideband synthesizer must produce complex, finite, nonzero IQ."""
    fn = WIDEBAND_SYNTHESIZERS[mode]
    sig = fn(fs=_WIDEBAND_FS, window_len=_TEST_WL)
    assert np.iscomplexobj(sig), f"{mode}: output not complex"
    assert len(sig) >= _TEST_WL, \
        f"{mode}: output too short ({len(sig)} < {_TEST_WL})"
    assert np.all(np.isfinite(sig)), f"{mode}: output contains NaN/Inf"
    power = np.mean(np.abs(sig) ** 2)
    assert power > 0, f"{mode}: output has zero power"


def test_wideband_class_count():
    assert len(WIDEBAND_CLASSES) == 8


def test_all_wideband_in_domain():
    from rf_datagen.domains import labels_for_domain
    domain_labels = labels_for_domain("wideband")
    for cls in WIDEBAND_CLASSES:
        assert cls in domain_labels, f"{cls} not in wideband domain"


# ---------------------------------------------------------------------------
# Per-class spectral property tests
# ---------------------------------------------------------------------------

_WIDEBAND_NYQUIST = 10_000_000  # fs / 2

# Expected occupied bandwidth per mode (Hz).  Conservative upper bounds —
# the test verifies >50 % of spectral energy falls within this band.
_WIDEBAND_EXPECTED_BW = {
    "WIFI_PREAMBLE":  18_000_000,
    "LTE_FRAME":      18_000_000,
    "FIVEG_NR":       18_000_000,
    "GPS_L1":          3_000_000,
    "ZIGBEE":          5_000_000,
    "DAB":            18_000_000,   # 2048-pt OFDM at 20 MHz → near full BW
    "DVB_T":          18_000_000,
    "LORAN_C_WIDE":    1_000_000,
}


@pytest.mark.parametrize("mode", WIDEBAND_CLASSES)
def test_wideband_synth_energy_concentrated(mode):
    """More than 50 % of spectral energy must sit within the expected BW."""
    fn = WIDEBAND_SYNTHESIZERS[mode]
    sig = fn(fs=_WIDEBAND_FS, window_len=_TEST_WL)

    spectrum = np.abs(np.fft.fftshift(np.fft.fft(sig[:_TEST_WL]))) ** 2
    total_energy = np.sum(spectrum)
    assert total_energy > 0, f"{mode}: no spectral energy"

    n = _TEST_WL
    freqs = np.fft.fftshift(np.fft.fftfreq(n, d=1.0 / _WIDEBAND_FS))

    occ_bw = _WIDEBAND_EXPECTED_BW[mode]
    mask = np.abs(freqs) <= occ_bw / 2.0
    energy_in_band = np.sum(spectrum[mask])
    ratio = energy_in_band / total_energy

    assert ratio > 0.50, (
        f"{mode}: only {ratio:.1%} of energy within ±{occ_bw/2e6:.1f} MHz "
        f"(expected >50 %)"
    )


def test_wifi_has_preamble_structure():
    """WiFi STF is periodic with period 16 samples — verify repetition."""
    from rf_datagen.generators.synthetic_wideband import WIDEBAND_SYNTHESIZERS

    fn = WIDEBAND_SYNTHESIZERS["WIFI_PREAMBLE"]
    sig = fn(fs=_WIDEBAND_FS, window_len=_TEST_WL)

    # The first 160 samples are the STF (10 repeats of a 16-sample pattern).
    # Check that shifting by 16 samples gives high correlation.
    stf_region = sig[:160]
    period = 16

    # Correlate stf_region with itself shifted by 16 samples
    a = stf_region[:160 - period]
    b = stf_region[period:160]
    cross = np.abs(np.vdot(a, b))
    auto = np.sqrt(np.vdot(a, a).real * np.vdot(b, b).real)
    correlation = cross / (auto + 1e-30)

    assert correlation > 0.85, (
        f"WiFi STF period-16 correlation = {correlation:.3f}, expected > 0.85"
    )
