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
    """Write metadata CSV with idx, mode, snr, scenario columns."""
    rows = [[i, tag, scenario] for i, (tag, scenario) in
            enumerate(zip(tags, scenarios))]
    atomic_write_csv(output_path, ["idx", "mode", "scenario"], rows)


def assemble_parts(output_dir, generator_name=None):
    """Assemble per-class .npy checkpoints into a single dataset.

    Validates each checkpoint on load: correct shape, complex dtype, no
    NaN/Inf.  Skips invalid checkpoints with a warning.

    Returns (iq_data, tags) where:
        iq_data: complex128 array of shape [N, window_len]
        tags: list of class name strings, length N
    """
    from .constants import WINDOW_LEN
    parts_dir = os.path.join(output_dir, "parts")
    if not os.path.exists(parts_dir):
        return np.array([]), []

    all_windows = []
    all_tags = []

    for label in SIGNAL_LABELS:
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
        if windows.ndim != 2 or windows.shape[1] != WINDOW_LEN:
            log.warning("Skipping %s — wrong shape %s (expected (N, %d))",
                        label, windows.shape, WINDOW_LEN)
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
