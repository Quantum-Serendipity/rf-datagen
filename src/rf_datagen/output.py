"""Dataset output — save, assemble, and write CSV metadata."""

import csv
import json
import os
import tempfile
from pathlib import Path

import numpy as np

from .constants import SIGNAL_LABELS
from .logging_config import get_logger

log = get_logger("output")


def atomic_save_npy(path, array):
    """Save a numpy array atomically via temp file + rename.

    Writes to a temp file in the same directory, then os.replace() to the
    final path.  This guarantees the file is either fully written or absent
    — never a corrupt partial write.
    """
    dirn = os.path.dirname(path) or "."
    fd, tmp = tempfile.mkstemp(dir=dirn, suffix=".tmp.npy")
    try:
        os.close(fd)
        np.save(tmp, array, allow_pickle=False)
        # np.save won't append .npy since tmp already ends with .npy
        os.replace(tmp, path)
    except BaseException:
        try:
            os.unlink(tmp)
        except OSError:
            pass
        raise


def atomic_write_csv(path, header, rows):
    """Write a CSV file atomically via temp file + rename."""
    dirn = os.path.dirname(path) or "."
    fd, tmp = tempfile.mkstemp(dir=dirn, suffix=".csv.tmp")
    try:
        with os.fdopen(fd, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(header)
            writer.writerows(rows)
        os.replace(tmp, path)
    except BaseException:
        try:
            os.unlink(tmp)
        except OSError:
            pass
        raise


def atomic_write_json(path, data, indent=2):
    """Write a JSON file atomically via temp file + rename."""
    dirn = os.path.dirname(path) or "."
    fd, tmp = tempfile.mkstemp(dir=dirn, suffix=".json.tmp")
    try:
        with os.fdopen(fd, "w") as f:
            json.dump(data, f, indent=indent)
        os.replace(tmp, path)
    except BaseException:
        try:
            os.unlink(tmp)
        except OSError:
            pass
        raise


def save_checkpoint(windows, class_name, output_dir):
    """Save per-class checkpoint to parts/ directory."""
    parts_dir = os.path.join(output_dir, "parts")
    os.makedirs(parts_dir, exist_ok=True)
    atomic_save_npy(os.path.join(parts_dir, f"{class_name}.npy"), windows)


def load_checkpoint(class_name, output_dir):
    """Load per-class checkpoint if it exists. Returns array or None."""
    path = os.path.join(output_dir, "parts", f"{class_name}.npy")
    if os.path.exists(path):
        return np.load(path)
    return None


def write_csv(tags, scenarios, output_path):
    """Write metadata CSV with idx, mode, scenario, domain, category, subcategory."""
    from .taxonomy import get_category
    from .domains import SIGNAL_DOMAIN_MAP

    rows = []
    for i, (tag, scenario) in enumerate(zip(tags, scenarios)):
        category, subcategory = get_category(tag)
        domain_obj = SIGNAL_DOMAIN_MAP.get(tag)
        domain_name = domain_obj.name if domain_obj else ""
        sample_rate = domain_obj.sample_rate if domain_obj else 0
        window_length = domain_obj.window_length if domain_obj else 0
        rows.append([i, tag, scenario, domain_name, sample_rate,
                     window_length, category, subcategory])

    header = ["idx", "mode", "scenario", "domain", "sample_rate",
              "window_length", "category", "subcategory"]
    atomic_write_csv(output_path, header, rows)


def assemble_parts(output_dir, generator_name=None, window_len=None,
                    labels=None):
    """Assemble per-class .npy checkpoints into a single dataset.

    Validates each checkpoint on load: correct shape, complex dtype, no
    NaN/Inf.  Skips invalid checkpoints with a warning.

    Args:
        output_dir: Directory containing parts/ subdirectory.
        generator_name: Unused (kept for backward compat).
        window_len: Expected window length. If None, uses WINDOW_LEN.
        labels: Ordered list of labels to include. If None, uses SIGNAL_LABELS.

    Returns (iq_data, tags) where:
        iq_data: complex array of shape [N, window_len]
        tags: list of class name strings, length N
    """
    if window_len is None:
        from .constants import WINDOW_LEN
        window_len = WINDOW_LEN
    if labels is None:
        labels = SIGNAL_LABELS

    parts_dir = os.path.join(output_dir, "parts")
    if not os.path.exists(parts_dir):
        return np.array([]), []

    all_windows = []
    all_tags = []

    for label in labels:
        path = os.path.join(parts_dir, f"{label}.npy")
        if not os.path.exists(path):
            continue
        try:
            windows = np.load(path)
        except Exception as e:
            log.warning("Skipping %s — failed to load: %s", label, e)
            continue
        if len(windows) == 0:
            continue
        # Shape validation
        if windows.ndim != 2 or windows.shape[1] != window_len:
            log.warning("Skipping %s — wrong shape %s (expected (N, %d))",
                        label, windows.shape, window_len)
            continue
        # Dtype validation
        if not np.iscomplexobj(windows):
            log.warning("Skipping %s — expected complex dtype, got %s",
                        label, windows.dtype)
            continue
        # NaN/Inf check (sample first 100 rows for speed)
        check_slice = windows[:min(100, len(windows))]
        if np.any(np.isnan(check_slice)) or np.any(np.isinf(check_slice)):
            log.warning("Skipping %s — contains NaN or Inf values", label)
            continue
        all_windows.append(windows)
        all_tags.extend([label] * len(windows))

    if not all_windows:
        return np.array([]), []

    iq_data = np.concatenate(all_windows, axis=0)
    return iq_data, all_tags


def save_dataset(iq_data, tags, scenarios, output_dir, prefix="rf_datagen"):
    """Save assembled dataset to output directory."""
    os.makedirs(output_dir, exist_ok=True)

    iq_path = os.path.join(output_dir, f"{prefix}_iq.npy")
    csv_path = os.path.join(output_dir, f"{prefix}_tags.csv")

    atomic_save_npy(iq_path, iq_data)
    write_csv(tags, scenarios, csv_path)

    log.info("Saved %d windows to %s", len(iq_data), iq_path)
    log.info("Saved metadata to %s", csv_path)

    return iq_path, csv_path
