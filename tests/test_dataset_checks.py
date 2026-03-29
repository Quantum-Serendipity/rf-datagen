"""Tests for rf_datagen.dataset_checks — structural validation."""

import csv
import os

import numpy as np
import pytest

from rf_datagen.dataset_checks import run_structural_checks
from rf_datagen.output import atomic_save_npy, write_csv


def _make_dataset(tmp_path, classes, n_per_class=20, window_len=2048,
                  dtype=np.complex128, scenarios=None, prefix="rf_datagen"):
    """Create a minimal on-disk dataset for testing structural checks."""
    tags = []
    scen = []
    for cls in classes:
        tags.extend([cls] * n_per_class)
        scen.extend([scenarios or "hf_clean"] * n_per_class)

    n = len(tags)
    iq = (np.random.randn(n, window_len)
          + 1j * np.random.randn(n, window_len)).astype(dtype)

    iq_path = os.path.join(str(tmp_path), f"{prefix}_iq.npy")
    csv_path = os.path.join(str(tmp_path), f"{prefix}_tags.csv")
    atomic_save_npy(iq_path, iq)
    write_csv(tags, scen, csv_path)
    return iq, tags


# ---------------------------------------------------------------------------
# Happy path
# ---------------------------------------------------------------------------

def test_structural_all_pass(tmp_path):
    """Full narrowband dataset with all classes passes all checks."""
    from rf_datagen.domains import labels_for_domain
    classes = labels_for_domain("narrowband")
    _make_dataset(tmp_path, classes, n_per_class=10)

    result = run_structural_checks(str(tmp_path), ["narrowband"])
    assert result["status"] == "PASS"
    for name, check in result["checks"].items():
        assert check["status"] == "PASS", f"{name}: {check}"


# ---------------------------------------------------------------------------
# Failure injection
# ---------------------------------------------------------------------------

def test_missing_iq_file(tmp_path):
    """Missing .npy fails the file_exists check."""
    result = run_structural_checks(str(tmp_path), ["narrowband"])
    assert result["status"] == "FAIL"
    assert "narrowband_file_exists" in result["checks"]
    assert result["checks"]["narrowband_file_exists"]["status"] == "FAIL"


def test_missing_class_detected(tmp_path):
    """Dataset missing a class fails class_completeness."""
    from rf_datagen.domains import labels_for_domain
    classes = labels_for_domain("narrowband")
    # Drop last 3 classes
    subset = classes[:-3]
    _make_dataset(tmp_path, subset, n_per_class=10)

    result = run_structural_checks(str(tmp_path), ["narrowband"])
    completeness = result["checks"]["narrowband_class_completeness"]
    assert completeness["status"] == "FAIL"
    assert "Missing" in completeness["detail"]


def test_nan_detected(tmp_path):
    """NaN values in IQ data are caught."""
    from rf_datagen.domains import labels_for_domain
    classes = labels_for_domain("narrowband")
    iq, tags = _make_dataset(tmp_path, classes, n_per_class=10)

    # Inject NaN into the saved file
    iq_path = os.path.join(str(tmp_path), "rf_datagen_iq.npy")
    iq_rw = np.load(iq_path)
    iq_rw[0, 0] = np.nan
    np.save(iq_path, iq_rw)

    result = run_structural_checks(str(tmp_path), ["narrowband"])
    nan_check = result["checks"]["narrowband_no_nan_inf"]
    assert nan_check["status"] == "FAIL"


def test_inf_detected(tmp_path):
    """Inf values in IQ data are caught."""
    from rf_datagen.domains import labels_for_domain
    classes = labels_for_domain("narrowband")
    _make_dataset(tmp_path, classes, n_per_class=10)

    iq_path = os.path.join(str(tmp_path), "rf_datagen_iq.npy")
    iq_rw = np.load(iq_path)
    iq_rw[5, 100] = np.inf
    np.save(iq_path, iq_rw)

    result = run_structural_checks(str(tmp_path), ["narrowband"])
    inf_check = result["checks"]["narrowband_no_nan_inf"]
    assert inf_check["status"] == "FAIL"


def test_zero_power_detected(tmp_path):
    """All-zero windows are caught."""
    from rf_datagen.domains import labels_for_domain
    classes = labels_for_domain("narrowband")
    _make_dataset(tmp_path, classes, n_per_class=10)

    iq_path = os.path.join(str(tmp_path), "rf_datagen_iq.npy")
    iq_rw = np.load(iq_path)
    iq_rw[0, :] = 0.0  # Zero out first window
    np.save(iq_path, iq_rw)

    result = run_structural_checks(str(tmp_path), ["narrowband"])
    zero_check = result["checks"]["narrowband_no_zero_power"]
    assert zero_check["status"] == "FAIL"


def test_metadata_row_mismatch(tmp_path):
    """CSV row count != IQ row count fails metadata_alignment."""
    from rf_datagen.domains import labels_for_domain
    classes = labels_for_domain("narrowband")
    _make_dataset(tmp_path, classes, n_per_class=10)

    # Truncate the IQ file to fewer rows
    iq_path = os.path.join(str(tmp_path), "rf_datagen_iq.npy")
    iq = np.load(iq_path)
    np.save(iq_path, iq[:100])  # Only 100 rows, CSV has more

    result = run_structural_checks(str(tmp_path), ["narrowband"])
    align_check = result["checks"]["narrowband_metadata_alignment"]
    assert align_check["status"] == "FAIL"


def test_wrong_shape_detected(tmp_path):
    """Wrong window length fails shape check."""
    from rf_datagen.domains import labels_for_domain
    classes = labels_for_domain("narrowband")
    # Create with wrong window_len (512 instead of 2048)
    _make_dataset(tmp_path, classes, n_per_class=10, window_len=512)

    result = run_structural_checks(str(tmp_path), ["narrowband"])
    shape_check = result["checks"]["narrowband_shape"]
    assert shape_check["status"] == "FAIL"


# ---------------------------------------------------------------------------
# Multi-domain
# ---------------------------------------------------------------------------

def test_multi_domain(tmp_path):
    """Multi-domain checks use per-domain subdirectories."""
    from rf_datagen.domains import labels_for_domain

    for domain_name in ["narrowband", "moderate"]:
        domain_dir = os.path.join(str(tmp_path), domain_name)
        os.makedirs(domain_dir, exist_ok=True)
        classes = labels_for_domain(domain_name)
        from rf_datagen.domains import DOMAINS
        domain = DOMAINS[domain_name]
        _make_dataset(
            domain_dir, classes, n_per_class=5,
            window_len=domain.window_length,
            dtype=domain.dtype,
            prefix=f"rf_datagen_{domain_name}",
        )

    result = run_structural_checks(str(tmp_path), ["narrowband", "moderate"])
    assert result["status"] == "PASS"
