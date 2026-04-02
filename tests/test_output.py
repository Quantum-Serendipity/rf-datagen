"""Tests for rf_datagen.output — save_dataset with train/val/test split."""

import csv
import os

import numpy as np
import pytest

from rf_datagen.output import (save_dataset, atomic_save_npy,
                                assemble_parts, atomic_write_csv)


def _make_dataset(n_per_class=20, n_classes=3, window_len=64):
    """Create a small synthetic dataset for testing."""
    labels = [f"CLASS_{i}" for i in range(n_classes)]
    tags = []
    scenarios = []
    for label in labels:
        tags.extend([label] * n_per_class)
        scenarios.extend(["hf_clean"] * n_per_class)
    n = n_per_class * n_classes
    iq = (np.random.randn(n, window_len)
          + 1j * np.random.randn(n, window_len)).astype(np.complex128)
    return iq, tags, scenarios


def test_save_dataset_no_split(tmp_path):
    """Without split_ratios, produces single file pair."""
    iq, tags, scenarios = _make_dataset()
    result = save_dataset(iq, tags, scenarios, str(tmp_path))
    iq_path, csv_path = result
    assert os.path.exists(iq_path)
    assert os.path.exists(csv_path)
    loaded = np.load(iq_path)
    assert loaded.shape == iq.shape


def test_save_dataset_with_split(tmp_path):
    """With split_ratios, produces separate files per split."""
    iq, tags, scenarios = _make_dataset(n_per_class=50)
    result = save_dataset(iq, tags, scenarios, str(tmp_path),
                          split_ratios=(0.8, 0.1, 0.1))
    assert isinstance(result, dict)
    assert "train" in result
    assert "val" in result
    assert "test" in result

    total = 0
    for split_name in ["train", "val", "test"]:
        iq_path, csv_path = result[split_name]
        assert os.path.exists(iq_path)
        assert os.path.exists(csv_path)
        loaded = np.load(iq_path)
        total += len(loaded)

    # All samples accounted for
    assert total == len(iq)


def test_save_dataset_split_stratified(tmp_path):
    """Each split should contain samples from all classes."""
    iq, tags, scenarios = _make_dataset(n_per_class=30, n_classes=4)
    result = save_dataset(iq, tags, scenarios, str(tmp_path),
                          split_ratios=(0.7, 0.15, 0.15))

    for split_name in ["train", "val", "test"]:
        iq_path, csv_path = result[split_name]
        # Read CSV to check class distribution
        import csv
        with open(csv_path) as f:
            reader = csv.DictReader(f)
            split_tags = [row["mode"] for row in reader]
        classes_present = set(split_tags)
        assert len(classes_present) == 4, \
            f"{split_name} missing classes: {classes_present}"


def test_save_dataset_split_deterministic(tmp_path):
    """Same seed produces same split."""
    iq, tags, scenarios = _make_dataset(n_per_class=20)
    dir1 = str(tmp_path / "run1")
    dir2 = str(tmp_path / "run2")
    r1 = save_dataset(iq, tags, scenarios, dir1,
                      split_ratios=(0.8, 0.1, 0.1), seed=99)
    r2 = save_dataset(iq, tags, scenarios, dir2,
                      split_ratios=(0.8, 0.1, 0.1), seed=99)
    for split in ["train", "val", "test"]:
        d1 = np.load(r1[split][0])
        d2 = np.load(r2[split][0])
        np.testing.assert_array_equal(d1, d2)


# ---------------------------------------------------------------------------
# Multi-generator assembly tests
# ---------------------------------------------------------------------------

def test_assemble_multi_generator(tmp_path):
    """Data from two generator subdirs is concatenated per class."""
    from rf_datagen.constants import SIGNAL_LABELS, WINDOW_LEN
    label = SIGNAL_LABELS[0]

    gen_a = tmp_path / "parts" / "gen_a"
    gen_b = tmp_path / "parts" / "gen_b"
    gen_a.mkdir(parents=True)
    gen_b.mkdir(parents=True)

    arr_a = np.ones((100, WINDOW_LEN), dtype=np.complex128)
    arr_b = np.ones((50, WINDOW_LEN), dtype=np.complex128) * 2
    np.save(str(gen_a / f"{label}.npy"), arr_a)
    np.save(str(gen_b / f"{label}.npy"), arr_b)

    iq, tags, scenarios, snrs = assemble_parts(str(tmp_path), labels=[label])
    assert iq.shape == (150, WINDOW_LEN)
    assert tags == [label] * 150
    assert len(scenarios) == 150


def test_assemble_multi_generator_with_meta(tmp_path):
    """Scenario metadata from multiple generators is concatenated."""
    from rf_datagen.constants import SIGNAL_LABELS, WINDOW_LEN
    label = SIGNAL_LABELS[0]

    gen_a = tmp_path / "parts" / "gen_a"
    gen_b = tmp_path / "parts" / "gen_b"
    gen_a.mkdir(parents=True)
    gen_b.mkdir(parents=True)

    arr_a = np.ones((3, WINDOW_LEN), dtype=np.complex128)
    arr_b = np.ones((2, WINDOW_LEN), dtype=np.complex128)
    np.save(str(gen_a / f"{label}.npy"), arr_a)
    np.save(str(gen_b / f"{label}.npy"), arr_b)

    # Write meta CSVs
    atomic_write_csv(str(gen_a / f"{label}_meta.csv"),
                     ["scenario", "snr"], [["hf_clean", "10"]] * 3)
    atomic_write_csv(str(gen_b / f"{label}_meta.csv"),
                     ["scenario", "snr"], [["vhf_mobile", "20"]] * 2)

    iq, tags, scenarios, snrs = assemble_parts(str(tmp_path), labels=[label])
    assert len(scenarios) == 5
    assert scenarios[:3] == ["hf_clean"] * 3
    assert scenarios[3:] == ["vhf_mobile"] * 2
    assert snrs[:3] == ["10"] * 3
    assert snrs[3:] == ["20"] * 2


def test_assemble_legacy_flat_files(tmp_path):
    """Legacy flat parts/*.npy files are loaded with deprecation warning."""
    from rf_datagen.constants import SIGNAL_LABELS, WINDOW_LEN
    label = SIGNAL_LABELS[0]

    parts = tmp_path / "parts"
    parts.mkdir()
    arr = np.ones((10, WINDOW_LEN), dtype=np.complex128)
    np.save(str(parts / f"{label}.npy"), arr)

    iq, tags, scenarios, snrs = assemble_parts(str(tmp_path), labels=[label])
    assert iq.shape == (10, WINDOW_LEN)
    assert tags == [label] * 10


def test_assemble_empty_returns_four(tmp_path):
    """Empty assembly returns 4-tuple with empty scenarios and snrs."""
    iq, tags, scenarios, snrs = assemble_parts(str(tmp_path))
    assert len(iq) == 0
    assert tags == []
    assert scenarios == []
    assert snrs == []
