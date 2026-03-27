"""BaseGenerator ABC — shared generation pipeline."""

import os
import time
from abc import ABC, abstractmethod

import numpy as np

from .._state import shutdown_requested
from ..config import GeneratorConfig, ImpairmentConfig, checkpoint_config_hash
from ..constants import FS, WINDOW_LEN
from ..impairments import extract_windows, apply_impairments, configure_impairments
from ..logging_config import get_logger
from ..output import atomic_save_npy, atomic_write_csv

log = get_logger("generator")


class BaseGenerator(ABC):
    """Abstract base for all signal generators.

    Subclasses must define:
        name: str               — generator name ("synthetic", "fldigi", etc.)
        required_tools: list    — CLI tools that must be in PATH
        signal_classes: list    — signal class names this generator covers
    """

    name: str = ""
    required_tools: list = []
    signal_classes: list = []

    def __init__(self, config: GeneratorConfig,
                 impairment_config: ImpairmentConfig = None,
                 fs: int = FS, window_len: int = WINDOW_LEN):
        self.config = config
        self.impairment_config = impairment_config or ImpairmentConfig()
        self.samples_per_class = config.samples_per_class
        self.fs = fs
        self.window_len = window_len

    def check_prerequisites(self):
        """Return list of missing CLI tools. Empty = ready."""
        import shutil
        missing = []
        for tool in self.required_tools:
            if shutil.which(tool) is None:
                missing.append(tool)
        return missing

    @abstractmethod
    def generate_class(self, class_name, rng=None):
        """Produce raw IQ for one class. Returns complex128 array."""

    def run(self, output_dir, seed=42):
        """Full pipeline: generate -> extract windows -> impair -> save.

        Returns a dict mapping class names to result info:
            {"FT8": {"status": "ok", "samples": 6000, "time_s": 12.3}, ...}
        Failed classes have status "failed" with a "reason" key.
        Cached classes have status "cached".
        """
        configure_impairments(self.impairment_config)

        parts_dir = os.path.join(output_dir, "parts")
        os.makedirs(parts_dir, exist_ok=True)

        classes = self._resolve_classes()
        results = {}
        if not classes:
            log.info("%s: no classes to generate", self.name)
            return results

        log.info("%s: generating %d classes, %d samples each",
                 self.name, len(classes), self.samples_per_class)

        stride = self.impairment_config.effective_stride(self.window_len)
        power_threshold = self.impairment_config.window_power_threshold

        for ci, class_name in enumerate(classes):
            if shutdown_requested():
                log.warning("Shutdown requested — stopping after "
                            "%d/%d classes", ci, len(classes))
                break

            n_samples = self._boosted_count(class_name)
            npy_path = os.path.join(parts_dir, f"{class_name}.npy")
            meta_path = os.path.join(parts_dir, f"{class_name}_meta.csv")
            hash_path = os.path.join(parts_dir, f"{class_name}.hash")
            cfg_hash = self._config_hash(class_name, n_samples)

            # Skip if checkpoint exists with matching config
            if self._check_checkpoint(npy_path, meta_path, hash_path,
                                      n_samples, cfg_hash):
                log.info("%15s: %d samples (cached)", class_name, n_samples)
                results[class_name] = {"status": "cached",
                                       "samples": n_samples}
                continue

            class_seed = seed + ci * 1000
            np.random.seed(class_seed)
            t0 = time.time()

            raw_iq = self.generate_class(class_name)
            if len(raw_iq) < self.window_len:
                log.warning("%15s: FAILED (signal too short)", class_name)
                results[class_name] = {"status": "failed",
                                       "reason": "signal too short"}
                continue

            raw_windows = extract_windows(raw_iq, window_len=self.window_len,
                                          stride=stride,
                                          power_threshold=power_threshold)
            if len(raw_windows) == 0:
                log.warning("%15s: FAILED (no valid windows)", class_name)
                results[class_name] = {"status": "failed",
                                       "reason": "no valid windows"}
                continue

            samples, meta = apply_impairments(
                raw_windows, n_samples, fs=self.fs,
                window_len=self.window_len, return_metadata=True)

            # Save checkpoint atomically
            atomic_save_npy(npy_path, samples)
            atomic_write_csv(meta_path, ["scenario"],
                             [[s] for s in meta["scenarios"]])
            self._write_hash(hash_path, cfg_hash)

            elapsed = time.time() - t0
            size_mb = os.path.getsize(npy_path) / (1024 * 1024)
            log.info("%15s: %d raw -> %d samples (%.1fs, %.0f MB)",
                     class_name, len(raw_windows), n_samples, elapsed,
                     size_mb)
            results[class_name] = {"status": "ok", "samples": n_samples,
                                   "raw_windows": len(raw_windows),
                                   "time_s": round(elapsed, 1)}

        return results

    def _resolve_classes(self):
        """Determine which classes to generate based on config."""
        if self.config.classes == "all":
            return list(self.signal_classes)
        if isinstance(self.config.classes, list):
            return [c for c in self.config.classes if c in self.signal_classes]
        return list(self.signal_classes)

    def _boosted_count(self, class_name):
        """Apply per-class boost multiplier to sample count."""
        boost = self.config.boost.get(class_name, 1.0)
        return int(self.samples_per_class * boost)

    def _config_hash(self, class_name, n_samples):
        """Compute a config hash for checkpoint staleness detection."""
        return checkpoint_config_hash(
            self.config, self.impairment_config,
            class_name, n_samples, self.fs, self.window_len)

    @staticmethod
    def _check_checkpoint(npy_path, meta_path, hash_path,
                          n_samples, expected_hash):
        """Return True if checkpoint is valid and matches current config.

        All three files (.npy, _meta.csv, .hash) must exist, the sample
        count must match, and the stored config hash must equal the
        expected hash.  Missing hash file → invalid (fail-safe).
        """
        if not (os.path.exists(npy_path) and os.path.exists(meta_path)):
            return False
        try:
            existing = np.load(npy_path, mmap_mode="r")
        except Exception:
            return False
        if existing.shape[0] != n_samples:
            return False
        if not os.path.exists(hash_path):
            return False
        try:
            with open(hash_path) as f:
                stored = f.read().strip()
            return stored == expected_hash
        except OSError:
            return False

    @staticmethod
    def _write_hash(hash_path, config_hash):
        """Write config hash sidecar file."""
        try:
            with open(hash_path, "w") as f:
                f.write(config_hash)
        except OSError as e:
            log.warning("Failed to write hash %s: %s "
                        "(checkpoint will regenerate next run)", hash_path, e)
