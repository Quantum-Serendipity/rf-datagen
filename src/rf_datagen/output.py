"""Dataset output — save, assemble, and write CSV metadata."""

import csv
import os
from pathlib import Path

import numpy as np

from .constants import SIGNAL_LABELS


def save_checkpoint(windows, class_name, output_dir):
    """Save per-class checkpoint to parts/ directory."""
    parts_dir = os.path.join(output_dir, "parts")
    os.makedirs(parts_dir, exist_ok=True)
    np.save(os.path.join(parts_dir, f"{class_name}.npy"), windows)


def load_checkpoint(class_name, output_dir):
    """Load per-class checkpoint if it exists. Returns array or None."""
    path = os.path.join(output_dir, "parts", f"{class_name}.npy")
    if os.path.exists(path):
        return np.load(path)
    return None


def write_csv(tags, scenarios, output_path):
    """Write metadata CSV with idx, mode, snr, scenario columns."""
    with open(output_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["idx", "mode", "scenario"])
        for i, (tag, scenario) in enumerate(zip(tags, scenarios)):
            writer.writerow([i, tag, scenario])


def assemble_parts(output_dir, generator_name=None):
    """Assemble per-class .npy checkpoints into a single dataset.

    Returns (iq_data, tags) where:
        iq_data: complex128 array of shape [N, 2048]
        tags: list of class name strings, length N
    """
    parts_dir = os.path.join(output_dir, "parts")
    if not os.path.exists(parts_dir):
        return np.array([]), []

    all_windows = []
    all_tags = []

    for label in SIGNAL_LABELS:
        path = os.path.join(parts_dir, f"{label}.npy")
        if not os.path.exists(path):
            continue
        windows = np.load(path)
        if len(windows) == 0:
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

    np.save(iq_path, iq_data)
    write_csv(tags, scenarios, csv_path)

    print(f"  Saved {len(iq_data)} windows to {iq_path}")
    print(f"  Saved metadata to {csv_path}")

    return iq_path, csv_path
