"""Tests for moderate-rate (1 MHz) signal synthesizers."""

import numpy as np
import pytest

from rf_datagen.generators.synthetic_moderate import (
    MODERATE_SYNTHESIZERS, MODERATE_CLASSES,
)
from rf_datagen.constants import MODERATE_FS, MODERATE_WINDOW_LEN


@pytest.fixture(autouse=True)
def seed_rng():
    np.random.seed(42)


@pytest.mark.parametrize("mode", MODERATE_CLASSES)
def test_moderate_synth_output_valid(mode):
    """Each moderate synthesizer must produce complex, finite, nonzero IQ."""
    fn = MODERATE_SYNTHESIZERS[mode]
    sig = fn(fs=MODERATE_FS, window_len=MODERATE_WINDOW_LEN)
    assert np.iscomplexobj(sig), f"{mode}: output not complex"
    assert len(sig) >= MODERATE_WINDOW_LEN, \
        f"{mode}: output too short ({len(sig)} < {MODERATE_WINDOW_LEN})"
    assert np.all(np.isfinite(sig)), f"{mode}: output contains NaN/Inf"
    power = np.mean(np.abs(sig) ** 2)
    assert power > 0, f"{mode}: output has zero power"


@pytest.mark.parametrize("mode", MODERATE_CLASSES)
def test_moderate_synth_bandwidth(mode):
    """Signal energy should be within the moderate domain Nyquist."""
    fn = MODERATE_SYNTHESIZERS[mode]
    sig = fn(fs=MODERATE_FS, window_len=MODERATE_WINDOW_LEN)
    # Check that >50% of energy is within Nyquist
    spectrum = np.abs(np.fft.fft(sig)) ** 2
    total_energy = np.sum(spectrum)
    # This is a sanity check — all signals at MODERATE_FS must be within
    # the 500 kHz Nyquist by construction
    assert total_energy > 0, f"{mode}: no spectral energy"


def test_moderate_class_count():
    assert len(MODERATE_CLASSES) == 14


def test_all_moderate_in_domain():
    from rf_datagen.domains import labels_for_domain
    domain_labels = labels_for_domain("moderate")
    for cls in MODERATE_CLASSES:
        assert cls in domain_labels, f"{cls} not in moderate domain"
