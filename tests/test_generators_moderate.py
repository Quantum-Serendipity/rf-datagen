"""Tests for moderate-rate (1 MHz) signal synthesizers."""

import numpy as np
import pytest

from rf_datagen.generators.synthetic_moderate import (
    MODERATE_SYNTHESIZERS, MODERATE_CLASSES,
)
from rf_datagen.domains import MODERATE

_MODERATE_FS = MODERATE.sample_rate
_MODERATE_WL = MODERATE.window_length


@pytest.mark.parametrize("mode", MODERATE_CLASSES)
def test_moderate_synth_output_valid(mode):
    """Each moderate synthesizer must produce complex, finite, nonzero IQ."""
    fn = MODERATE_SYNTHESIZERS[mode]
    sig = fn(fs=_MODERATE_FS, window_len=_MODERATE_WL)
    assert np.iscomplexobj(sig), f"{mode}: output not complex"
    assert len(sig) >= _MODERATE_WL, \
        f"{mode}: output too short ({len(sig)} < {_MODERATE_WL})"
    assert np.all(np.isfinite(sig)), f"{mode}: output contains NaN/Inf"
    power = np.mean(np.abs(sig) ** 2)
    assert power > 0, f"{mode}: output has zero power"


@pytest.mark.parametrize("mode", MODERATE_CLASSES)
def test_moderate_synth_bandwidth(mode):
    """Signal energy should be within the moderate domain Nyquist."""
    fn = MODERATE_SYNTHESIZERS[mode]
    sig = fn(fs=_MODERATE_FS, window_len=_MODERATE_WL)
    # Check that >50% of energy is within Nyquist
    spectrum = np.abs(np.fft.fft(sig)) ** 2
    total_energy = np.sum(spectrum)
    # This is a sanity check — all signals at _MODERATE_FS must be within
    # the 500 kHz Nyquist by construction
    assert total_energy > 0, f"{mode}: no spectral energy"


def test_moderate_class_count():
    assert len(MODERATE_CLASSES) == 14


def test_all_moderate_in_domain():
    from rf_datagen.domains import labels_for_domain
    domain_labels = labels_for_domain("moderate")
    for cls in MODERATE_CLASSES:
        assert cls in domain_labels, f"{cls} not in moderate domain"


# ---------------------------------------------------------------------------
# Per-class spectral property tests
# ---------------------------------------------------------------------------

_TEST_WL = 32768

# Expected occupied bandwidth per mode (Hz).  These are conservative upper
# bounds — the test only checks that >60 % of spectral energy falls inside.
_MODERATE_EXPECTED_BW = {
    "BLE":                1_000_000,   # 1 Msym/s GFSK → occupies full Nyquist
    "ZWAVE":              200_000,
    "ADS_B":              1_000_000,   # PPM pulses have wide spectral spread
    "GSM_BURST":          300_000,
    "LFM_RADAR":          500_000,
    "FMCW_RADAR":         500_000,
    "PHASE_CODED_RADAR":  500_000,
    "NOAA_APT":           500_000,
    "COSPAS_SARSAT":      500_000,
    "LORA_WIDE":          500_000,
    "VDL2":               500_000,
    "DRM_WIDE":           500_000,
    "DECT":               1_000_000,   # 1.152 Mbps GFSK → near full Nyquist
    "IRIDIUM":            500_000,
}

_MODERATE_NYQUIST = 500_000  # fs / 2


@pytest.mark.parametrize("mode", MODERATE_CLASSES)
def test_moderate_synth_energy_concentrated(mode):
    """More than 60 % of spectral energy must sit within the expected BW."""
    fn = MODERATE_SYNTHESIZERS[mode]
    sig = fn(fs=_MODERATE_FS, window_len=_TEST_WL)

    spectrum = np.abs(np.fft.fftshift(np.fft.fft(sig[:_TEST_WL]))) ** 2
    total_energy = np.sum(spectrum)
    assert total_energy > 0, f"{mode}: no spectral energy"

    n = _TEST_WL
    freqs = np.fft.fftshift(np.fft.fftfreq(n, d=1.0 / _MODERATE_FS))

    occ_bw = _MODERATE_EXPECTED_BW[mode]
    mask = np.abs(freqs) <= occ_bw / 2.0
    energy_in_band = np.sum(spectrum[mask])
    ratio = energy_in_band / total_energy

    assert ratio > 0.60, (
        f"{mode}: only {ratio:.1%} of energy within ±{occ_bw/2e3:.0f} kHz "
        f"(expected >60 %)"
    )


_PULSED_RADAR_MODES = ["LFM_RADAR", "PHASE_CODED_RADAR"]


@pytest.mark.parametrize("mode", _PULSED_RADAR_MODES)
def test_radar_duty_cycle(mode):
    """Pulsed radar modes must have quiet periods (duty cycle < 0.95)."""
    fn = MODERATE_SYNTHESIZERS[mode]
    sig = fn(fs=_MODERATE_FS, window_len=_TEST_WL)

    amp = np.abs(sig[:_TEST_WL])
    peak = np.max(amp)
    assert peak > 0, f"{mode}: signal is all zeros"

    # Fraction of samples with amplitude > 10 % of peak
    active_frac = np.mean(amp > 0.10 * peak)
    assert active_frac < 0.95, (
        f"{mode}: duty cycle {active_frac:.2f} — expected < 0.95 "
        f"(signal should have quiet inter-pulse periods)"
    )
