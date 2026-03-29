"""Tests for the multi-rate domain registry."""

import numpy as np

from rf_datagen.domains import (
    DOMAINS, NARROWBAND, MODERATE, WIDEBAND,
    SIGNAL_DOMAIN_MAP, labels_for_domain, all_signal_labels,
    ALL_SIGNAL_LABELS,
)
from rf_datagen.constants import SIGNAL_LABELS, FS, WINDOW_LEN


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


def test_narrowband_labels_match_constants():
    """SIGNAL_LABELS (legacy) must exactly match narrowband domain labels."""
    nb_labels = labels_for_domain("narrowband")
    assert nb_labels == SIGNAL_LABELS


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
