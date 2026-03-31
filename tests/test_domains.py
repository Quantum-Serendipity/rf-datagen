"""Tests for the multi-rate domain registry."""

import os

import numpy as np

from rf_datagen.domains import (
    DOMAINS, NARROWBAND, MODERATE, WIDEBAND,
    SIGNAL_DOMAIN_MAP, labels_for_domain, all_signal_labels,
    ALL_SIGNAL_LABELS,
)
from rf_datagen.constants import SIGNAL_LABELS, FS, WINDOW_LEN
from rf_datagen.output import assemble_parts


def test_domains_registry():
    assert "narrowband" in DOMAINS
    assert "moderate" in DOMAINS
    assert "wideband" in DOMAINS


def test_narrowband_defaults_match_legacy():
    assert NARROWBAND.sample_rate == FS
    assert NARROWBAND.window_length == WINDOW_LEN
    assert NARROWBAND.dtype == np.dtype(np.complex128)


def test_moderate_domain():
    assert MODERATE.sample_rate == 1_000_000
    assert MODERATE.window_length == 131_072
    assert MODERATE.dtype == np.dtype(np.complex64)


def test_wideband_domain():
    assert WIDEBAND.sample_rate == 20_000_000
    assert WIDEBAND.window_length == 2_097_152
    assert WIDEBAND.dtype == np.dtype(np.complex64)


def test_signal_domain_map_covers_all():
    """Every label in ALL_SIGNAL_LABELS must map to a domain."""
    for label in ALL_SIGNAL_LABELS:
        assert label in SIGNAL_DOMAIN_MAP, f"{label} missing from SIGNAL_DOMAIN_MAP"


def test_signal_labels_match_all():
    """SIGNAL_LABELS must match ALL_SIGNAL_LABELS (all 90 classes)."""
    assert SIGNAL_LABELS == list(ALL_SIGNAL_LABELS)


def test_all_labels_starts_with_narrowband():
    """ALL_SIGNAL_LABELS starts with narrowband for backward compat."""
    all_labels = all_signal_labels()
    nb_labels = labels_for_domain("narrowband")
    assert all_labels[:len(nb_labels)] == nb_labels


def test_no_duplicate_labels():
    """No signal label appears in more than one domain."""
    all_labels = all_signal_labels()
    assert len(all_labels) == len(set(all_labels)), \
        f"Duplicate labels found: {[l for l in all_labels if all_labels.count(l) > 1]}"


def test_labels_for_domain_moderate():
    labels = labels_for_domain("moderate")
    assert len(labels) == 14
    assert "BLE" in labels
    assert "ADS_B" in labels


def test_labels_for_domain_wideband():
    labels = labels_for_domain("wideband")
    assert len(labels) == 8
    assert "WIFI_PREAMBLE" in labels
    assert "GPS_L1" in labels


def test_total_label_count():
    """Target: ~90 signal classes across all domains."""
    assert len(ALL_SIGNAL_LABELS) == 68 + 14 + 8  # 90


# ---------------------------------------------------------------------------
# Integration tests for assemble_parts with multi-domain checkpoints
# ---------------------------------------------------------------------------


def test_assemble_parts_multi_domain(tmp_path):
    """Assemble checkpoints from two domains with different window_len/dtype."""
    parts_dir = tmp_path / "parts"
    parts_dir.mkdir()

    # Narrowband checkpoint: FT8 — complex128, window_len=2048
    ft8_data = np.ones((5, 2048), dtype=np.complex128) * (1 + 2j)
    np.save(str(parts_dir / "FT8.npy"), ft8_data)

    # Moderate checkpoint: BLE — complex64, window_len=131072
    ble_data = np.ones((3, 131072), dtype=np.complex64) * (0.5 + 0.5j)
    np.save(str(parts_dir / "BLE.npy"), ble_data)

    # Assemble narrowband domain
    iq_nb, tags_nb, _ = assemble_parts(str(tmp_path), window_len=2048, labels=["FT8"])
    assert iq_nb.shape == (5, 2048)
    assert np.iscomplexobj(iq_nb)
    assert iq_nb.dtype == np.complex128
    assert tags_nb == ["FT8"] * 5

    # Assemble moderate domain
    iq_mod, tags_mod, _ = assemble_parts(str(tmp_path), window_len=131072, labels=["BLE"])
    assert iq_mod.shape == (3, 131072)
    assert np.iscomplexobj(iq_mod)
    assert iq_mod.dtype == np.complex64
    assert tags_mod == ["BLE"] * 3


def test_assemble_parts_validates_shape(tmp_path):
    """Checkpoint with wrong window_len is skipped with a warning."""
    parts_dir = tmp_path / "parts"
    parts_dir.mkdir()

    # Save a checkpoint with window_len=512, but assemble expects 2048
    wrong_shape = np.ones((4, 512), dtype=np.complex128)
    np.save(str(parts_dir / "FT8.npy"), wrong_shape)

    iq_data, tags, _ = assemble_parts(str(tmp_path), window_len=2048, labels=["FT8"])
    assert len(tags) == 0
    assert iq_data.size == 0


def test_assemble_parts_skips_nan(tmp_path):
    """Checkpoint containing NaN values is skipped."""
    parts_dir = tmp_path / "parts"
    parts_dir.mkdir()

    nan_data = np.ones((3, 2048), dtype=np.complex128)
    nan_data[1, 100] = np.nan + 0j  # inject a NaN
    np.save(str(parts_dir / "FT8.npy"), nan_data)

    iq_data, tags, _ = assemble_parts(str(tmp_path), window_len=2048, labels=["FT8"])
    assert len(tags) == 0
    assert iq_data.size == 0
