"""Tests for wideband (20 MHz) signal synthesizers.

Uses a reduced window length for test speed — full WIDEBAND_WINDOW_LEN
(2M samples) is only needed for actual dataset generation.
"""

import numpy as np
import pytest

from rf_datagen.generators.synthetic_wideband import (
    WIDEBAND_SYNTHESIZERS, WIDEBAND_CLASSES,
)
from rf_datagen.constants import WIDEBAND_FS

# Reduced window for test speed (still exercises all signal generation logic)
_TEST_WL = 65536


@pytest.fixture(autouse=True)
def seed_rng():
    np.random.seed(42)


@pytest.mark.parametrize("mode", WIDEBAND_CLASSES)
def test_wideband_synth_output_valid(mode):
    """Each wideband synthesizer must produce complex, finite, nonzero IQ."""
    fn = WIDEBAND_SYNTHESIZERS[mode]
    sig = fn(fs=WIDEBAND_FS, window_len=_TEST_WL)
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
