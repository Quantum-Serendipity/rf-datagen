"""Tests for rf_datagen.spectral_fingerprint — spectral validation."""

import csv
import os

import numpy as np
import pytest

from rf_datagen.spectral_fingerprint import (
    SPECTRAL_SPECS,
    _measure_spectral,
    check_class_spectral,
    validate_spectral,
)
from rf_datagen.output import atomic_save_npy, write_csv


FS = 12_000


# ---------------------------------------------------------------------------
# SPECTRAL_SPECS coverage
# ---------------------------------------------------------------------------

def test_specs_cover_all_classes():
    """Every signal class should have a spectral spec entry."""
    from rf_datagen.domains import ALL_SIGNAL_LABELS
    missing = [l for l in ALL_SIGNAL_LABELS if l not in SPECTRAL_SPECS]
    assert not missing, f"Missing spectral specs: {missing}"


def test_specs_bw_ranges_valid():
    """All bw_3db ranges should have min < max and be non-negative."""
    for cls, spec in SPECTRAL_SPECS.items():
        if spec.get("exempt"):
            continue
        lo, hi = spec["bw_3db"]
        assert lo >= 0, f"{cls}: negative bw_3db lower bound"
        assert hi > lo, f"{cls}: bw_3db upper <= lower ({lo}, {hi})"


def test_specs_papr_ranges_valid():
    """All PAPR ranges should have min < max and be non-negative."""
    for cls, spec in SPECTRAL_SPECS.items():
        if spec.get("exempt"):
            continue
        lo, hi = spec["papr"]
        assert lo >= 0, f"{cls}: negative PAPR lower bound"
        assert hi > lo, f"{cls}: PAPR upper <= lower ({lo}, {hi})"


# ---------------------------------------------------------------------------
# _measure_spectral
# ---------------------------------------------------------------------------

def test_measure_tone():
    """A pure tone should have narrow bandwidth and low PAPR."""
    n = 2048
    t = np.arange(n) / FS
    iq = np.exp(2j * np.pi * 1000 * t)  # 1 kHz tone
    m = _measure_spectral(iq, FS)
    assert m["bandwidth_3db"] < 200  # Narrow
    assert m["papr_db"] < 5  # Low for constant-envelope


def test_measure_wideband_noise():
    """White noise should have wide bandwidth."""
    n = 4096
    iq = (np.random.randn(n) + 1j * np.random.randn(n)) / np.sqrt(2)
    m = _measure_spectral(iq, FS)
    assert m["bandwidth_3db"] > 3000  # Wide


def test_measure_ofdm_like():
    """Multi-tone signal should have higher PAPR than single tone."""
    n = 2048
    t = np.arange(n) / FS
    # Sum of 10 tones at different frequencies
    iq = sum(np.exp(2j * np.pi * f * t) for f in range(100, 1100, 100))
    m_multi = _measure_spectral(iq, FS)

    # Single tone for comparison
    iq_single = np.exp(2j * np.pi * 500 * t)
    m_single = _measure_spectral(iq_single, FS)

    assert m_multi["papr_db"] > m_single["papr_db"]


# ---------------------------------------------------------------------------
# check_class_spectral
# ---------------------------------------------------------------------------

def test_check_class_passes_on_matching_signal():
    """Signal within spec should pass."""
    n = 2048
    t = np.arange(n) / FS
    # FSK-like signal that should match FT8 spec (BW ~50Hz, low PAPR)
    iq = np.exp(2j * np.pi * 500 * t)  # Narrow tone
    spec = {"bw_3db": (5, 200), "papr": (0, 8)}

    result = check_class_spectral([iq] * 10, FS, spec)
    assert result["status"] == "PASS"
    assert result["bw_ok"]
    assert result["papr_ok"]


def test_check_class_fails_on_bandwidth_mismatch():
    """Multi-tone signal against narrow BW spec should fail."""
    n = 2048
    t = np.arange(n) / FS
    # Wide multi-tone has high spectral SNR but wide BW
    iq = sum(np.exp(2j * np.pi * f * t) for f in range(200, 4000, 200))
    spec = {"bw_3db": (10, 100), "papr": (0, 20)}  # Very narrow expected BW

    result = check_class_spectral([iq] * 10, FS, spec)
    assert result["status"] == "FAIL"
    assert not result["bw_ok"]  # Median BW should exceed narrow spec


def test_check_class_exempt_always_passes():
    """Exempt classes should always pass."""
    n = 2048
    iq = np.zeros(n, dtype=np.complex128)
    spec = {"bw_3db": (10, 100), "papr": (0, 5), "exempt": True}

    result = check_class_spectral([iq], FS, spec)
    assert result["status"] == "PASS"


# ---------------------------------------------------------------------------
# validate_spectral (integration with on-disk dataset)
# ---------------------------------------------------------------------------

def _make_spectral_dataset(tmp_path, classes, n_per_class=25,
                           window_len=2048):
    """Create an on-disk dataset with synthetic IQ for spectral testing."""
    tags = []
    scen = []
    windows = []
    for cls in classes:
        for _ in range(n_per_class):
            tags.append(cls)
            scen.append("hf_clean")
            # Generate a tone-like signal (narrow BW, low PAPR)
            t = np.arange(window_len) / FS
            freq = np.random.uniform(200, 2000)
            iq = np.exp(2j * np.pi * freq * t)
            # Add small noise
            iq += 0.01 * (np.random.randn(window_len)
                          + 1j * np.random.randn(window_len))
            windows.append(iq)

    iq_data = np.array(windows, dtype=np.complex128)
    iq_path = os.path.join(str(tmp_path), "rf_datagen_iq.npy")
    csv_path = os.path.join(str(tmp_path), "rf_datagen_tags.csv")
    atomic_save_npy(iq_path, iq_data)
    write_csv(tags, scen, csv_path)


def test_validate_spectral_runs(tmp_path):
    """validate_spectral should complete without error on a small dataset."""
    # Use 3 narrowband classes that have narrow-BW specs (tones will match)
    classes = ["CW", "WSPR", "JT9"]
    _make_spectral_dataset(tmp_path, classes, n_per_class=10)

    result = validate_spectral(str(tmp_path), ["narrowband"],
                               n_samples=5, seed=42)
    # Should have results for the 3 classes we created
    assert "per_class" in result
    for cls in classes:
        assert cls in result["per_class"]


def test_validate_spectral_skips_missing_data(tmp_path):
    """Classes not in dataset should be marked SKIP, not error."""
    classes = ["CW"]
    _make_spectral_dataset(tmp_path, classes, n_per_class=5)

    result = validate_spectral(str(tmp_path), ["narrowband"],
                               n_samples=3, seed=42)
    # FT8 not in our tiny dataset → should be SKIP
    assert result["per_class"]["FT8"]["status"] == "SKIP"


def test_validate_spectral_catches_wrong_bandwidth(tmp_path):
    """Wideband multi-tone labeled as SIGFOX should fail narrow BW spec."""
    n_per_class = 25
    window_len = 2048

    # SIGFOX is non-exempt with narrow spec: bw_3db (0, 320)
    tags = ["SIGFOX"] * n_per_class
    scen = ["hf_clean"] * n_per_class
    windows = []
    t = np.arange(window_len) / FS
    for _ in range(n_per_class):
        # Multi-tone spanning wide bandwidth → high spectral SNR but wide BW
        iq = sum(np.exp(2j * np.pi * f * t) for f in range(200, 4000, 200))
        windows.append(iq)

    iq_data = np.array(windows, dtype=np.complex128)
    iq_path = os.path.join(str(tmp_path), "rf_datagen_iq.npy")
    csv_path = os.path.join(str(tmp_path), "rf_datagen_tags.csv")
    atomic_save_npy(iq_path, iq_data)
    write_csv(tags, scen, csv_path)

    result = validate_spectral(str(tmp_path), ["narrowband"],
                               n_samples=20, seed=42)
    sigfox_result = result["per_class"]["SIGFOX"]
    # Wide multi-tone should fail SIGFOX's narrow BW spec
    assert sigfox_result["status"] == "FAIL"
    assert not sigfox_result["bw_ok"]
