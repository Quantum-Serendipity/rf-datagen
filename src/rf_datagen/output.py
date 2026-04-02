"""Dataset output — save, assemble, and write CSV metadata."""

import csv
import json
import os
import tempfile
from pathlib import Path

import numpy as np

from .domains import ALL_SIGNAL_LABELS
from .logging_config import get_logger

log = get_logger("output")


def _atomic_replace(path, suffix, write_fn):
    """Generic atomic file write: write to temp, then os.replace().

    Args:
        path: Final destination path.
        suffix: Temp file suffix (e.g. ".tmp.npy").
        write_fn: Callable(tmp_path) that writes the file content.
            If write_fn needs an open fd, it should accept the path
            and open it itself.  For npy we pass the path directly;
            for text formats we open via os.fdopen in the caller.
    """
    dirn = os.path.dirname(path) or "."
    fd, tmp = tempfile.mkstemp(dir=dirn, suffix=suffix)
    try:
        os.close(fd)
        write_fn(tmp)
        os.replace(tmp, path)
    except BaseException:
        try:
            os.unlink(tmp)
        except OSError:
            pass
        raise


def atomic_save_npy(path, array):
    """Save a numpy array atomically via temp file + rename."""
    def _write(tmp):
        np.save(tmp, array, allow_pickle=False)
    _atomic_replace(path, ".tmp.npy", _write)


def atomic_write_csv(path, header, rows):
    """Write a CSV file atomically via temp file + rename."""
    def _write(tmp):
        with open(tmp, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(header)
            writer.writerows(rows)
    _atomic_replace(path, ".tmp.csv", _write)


def atomic_write_json(path, data, indent=2):
    """Write a JSON file atomically via temp file + rename."""
    def _write(tmp):
        with open(tmp, "w") as f:
            json.dump(data, f, indent=indent)
    _atomic_replace(path, ".tmp.json", _write)


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


def write_csv(tags, scenarios, output_path, snrs=None):
    """Write metadata CSV with idx, mode, scenario, snr, domain, category, subcategory."""
    from .taxonomy import get_category
    from .domains import SIGNAL_DOMAIN_MAP

    rows = []
    for i, (tag, scenario) in enumerate(zip(tags, scenarios)):
        category, subcategory = get_category(tag)
        domain_obj = SIGNAL_DOMAIN_MAP.get(tag)
        domain_name = domain_obj.name if domain_obj else ""
        sample_rate = domain_obj.sample_rate if domain_obj else 0
        window_length = domain_obj.window_length if domain_obj else 0
        snr = snrs[i] if snrs is not None and i < len(snrs) else ""
        rows.append([i, tag, scenario, snr, domain_name, sample_rate,
                     window_length, category, subcategory])

    header = ["idx", "mode", "scenario", "snr", "domain", "sample_rate",
              "window_length", "category", "subcategory"]
    atomic_write_csv(output_path, header, rows)


def _validate_checkpoint(windows, label, source, window_len):
    """Validate a loaded checkpoint array. Returns True if valid."""
    if len(windows) == 0:
        return False
    if windows.ndim != 2 or windows.shape[1] != window_len:
        log.warning("Skipping %s/%s — wrong shape %s (expected (N, %d))",
                    source, label, windows.shape, window_len)
        return False
    if not np.iscomplexobj(windows):
        log.warning("Skipping %s/%s — expected complex dtype, got %s",
                    source, label, windows.dtype)
        return False
    check_slice = windows[:min(100, len(windows))]
    if np.any(np.isnan(check_slice)) or np.any(np.isinf(check_slice)):
        log.warning("Skipping %s/%s — contains NaN or Inf values",
                    source, label)
        return False
    return True


def _load_meta(meta_path):
    """Load scenario and snr data from a _meta.csv sidecar file.

    Returns (scenarios, snrs) where each is a list of strings.
    """
    import csv as _csv
    if not os.path.exists(meta_path):
        return [], []
    try:
        with open(meta_path) as f:
            reader = _csv.reader(f)
            header = next(reader, None)
            if header is None:
                return [], []
            # Find column indices (backward compat: old files may lack snr)
            scenario_idx = 0
            snr_idx = header.index("snr") if "snr" in header else -1
            scenarios = []
            snrs = []
            for row in reader:
                if not row:
                    continue
                scenarios.append(row[scenario_idx] if scenario_idx < len(row) else "")
                if snr_idx >= 0 and snr_idx < len(row):
                    snrs.append(row[snr_idx])
                else:
                    snrs.append("")
            return scenarios, snrs
    except Exception:
        return [], []


def assemble_parts(output_dir, generator_name=None, window_len=None,
                    labels=None):
    """Assemble per-class .npy checkpoints into a single dataset.

    Discovers generator subdirectories under parts/ and concatenates
    data from all generators per class. Also loads scenario metadata
    from _meta.csv sidecar files.

    Args:
        output_dir: Directory containing parts/ subdirectory.
        generator_name: Unused (kept for backward compat).
        window_len: Expected window length. If None, uses WINDOW_LEN.
        labels: Ordered list of labels to include. If None, uses SIGNAL_LABELS.

    Returns (iq_data, tags, scenarios, snrs) where:
        iq_data: complex array of shape [N, window_len]
        tags: list of class name strings, length N
        scenarios: list of scenario name strings, length N
        snrs: list of snr strings, length N
    """
    if window_len is None:
        from .constants import WINDOW_LEN
        window_len = WINDOW_LEN
    if labels is None:
        labels = ALL_SIGNAL_LABELS

    parts_dir = os.path.join(output_dir, "parts")
    if not os.path.exists(parts_dir):
        return np.array([]), [], [], []

    # Discover generator subdirectories
    gen_dirs = sorted(
        d for d in os.listdir(parts_dir)
        if os.path.isdir(os.path.join(parts_dir, d))
    )

    # Check for legacy flat files (backward compat)
    has_flat = any(
        os.path.exists(os.path.join(parts_dir, f"{label}.npy"))
        for label in labels
    )
    if has_flat:
        log.warning("Found flat parts/*.npy files (legacy layout). "
                     "Re-run generation to use parts/{generator}/ layout.")

    all_windows = []
    all_tags = []
    all_scenarios = []
    all_snrs = []

    for label in labels:
        label_windows = []
        label_scenarios = []
        label_snrs = []
        contributions = {}

        # Load from generator subdirectories
        for gen_name in gen_dirs:
            npy_path = os.path.join(parts_dir, gen_name, f"{label}.npy")
            if not os.path.exists(npy_path):
                continue
            try:
                windows = np.load(npy_path)
            except Exception as e:
                log.warning("Skipping %s/%s — failed to load: %s",
                            gen_name, label, e)
                continue
            if not _validate_checkpoint(windows, label, gen_name, window_len):
                continue
            label_windows.append(windows)
            contributions[gen_name] = len(windows)

            meta_path = os.path.join(parts_dir, gen_name, f"{label}_meta.csv")
            scenarios, snrs = _load_meta(meta_path)
            if len(scenarios) == len(windows):
                label_scenarios.extend(scenarios)
                label_snrs.extend(snrs)
            else:
                label_scenarios.extend([""] * len(windows))
                label_snrs.extend([""] * len(windows))

        # Legacy flat files fallback
        if has_flat:
            flat_path = os.path.join(parts_dir, f"{label}.npy")
            if os.path.exists(flat_path):
                try:
                    windows = np.load(flat_path)
                except Exception:
                    windows = None
                if windows is not None and _validate_checkpoint(
                        windows, label, "flat", window_len):
                    label_windows.append(windows)
                    contributions["flat"] = len(windows)
                    flat_meta = os.path.join(parts_dir, f"{label}_meta.csv")
                    scenarios, snrs = _load_meta(flat_meta)
                    if len(scenarios) == len(windows):
                        label_scenarios.extend(scenarios)
                        label_snrs.extend(snrs)
                    else:
                        label_scenarios.extend([""] * len(windows))
                        label_snrs.extend([""] * len(windows))

        if not label_windows:
            continue

        total = sum(contributions.values())
        if len(contributions) > 1:
            detail = ", ".join(f"{g}={n}" for g, n in
                               sorted(contributions.items()))
            log.info("%15s: %s -> %d total", label, detail, total)

        combined = np.concatenate(label_windows, axis=0)
        all_windows.append(combined)
        all_tags.extend([label] * len(combined))
        all_scenarios.extend(label_scenarios)
        all_snrs.extend(label_snrs)

    if not all_windows:
        return np.array([]), [], [], []

    iq_data = np.concatenate(all_windows, axis=0)
    return iq_data, all_tags, all_scenarios, all_snrs


def save_dataset(iq_data, tags, scenarios, output_dir, prefix="rf_datagen",
                 split_ratios=None, seed=42, snrs=None):
    """Save assembled dataset to output directory.

    Args:
        iq_data: Complex IQ array of shape [N, window_len].
        tags: List of class name strings, length N.
        scenarios: List of scenario name strings, length N.
        output_dir: Output directory path.
        prefix: Filename prefix.
        split_ratios: Optional (train, val, test) ratios that sum to 1.0.
            When provided, produces separate files per split with
            stratified class balance.  Default None = no split.
        seed: RNG seed for reproducible splitting.
        snrs: Optional list of SNR values, length N.

    Returns:
        (iq_path, csv_path) for unsplit, or dict of paths per split.
    """
    os.makedirs(output_dir, exist_ok=True)

    if split_ratios is None:
        iq_path = os.path.join(output_dir, f"{prefix}_iq.npy")
        csv_path = os.path.join(output_dir, f"{prefix}_tags.csv")
        atomic_save_npy(iq_path, iq_data)
        write_csv(tags, scenarios, csv_path, snrs=snrs)
        log.info("Saved %d windows to %s", len(iq_data), iq_path)
        log.info("Saved metadata to %s", csv_path)
        return iq_path, csv_path

    # Stratified train/val/test split
    train_r, val_r, test_r = split_ratios
    rng = np.random.RandomState(seed)

    # Group indices by class for stratification
    from collections import defaultdict
    class_indices = defaultdict(list)
    for i, tag in enumerate(tags):
        class_indices[tag].append(i)

    train_idx, val_idx, test_idx = [], [], []
    for cls in sorted(class_indices.keys()):
        indices = np.array(class_indices[cls])
        rng.shuffle(indices)
        n = len(indices)
        n_train = max(1, int(n * train_r))
        n_val = max(0, int(n * val_r))
        # test gets the remainder
        train_idx.extend(indices[:n_train])
        val_idx.extend(indices[n_train:n_train + n_val])
        test_idx.extend(indices[n_train + n_val:])

    paths = {}
    for split_name, idxs in [("train", train_idx), ("val", val_idx),
                              ("test", test_idx)]:
        if not idxs:
            continue
        idxs = sorted(idxs)
        split_iq = iq_data[idxs]
        split_tags = [tags[i] for i in idxs]
        split_scenarios = [scenarios[i] for i in idxs]
        split_snrs = [snrs[i] for i in idxs] if snrs else None

        iq_path = os.path.join(output_dir, f"{prefix}_{split_name}_iq.npy")
        csv_path = os.path.join(output_dir, f"{prefix}_{split_name}_tags.csv")
        atomic_save_npy(iq_path, split_iq)
        write_csv(split_tags, split_scenarios, csv_path, snrs=split_snrs)
        log.info("Saved %d %s windows to %s", len(split_iq), split_name,
                 iq_path)
        paths[split_name] = (iq_path, csv_path)

    return paths
