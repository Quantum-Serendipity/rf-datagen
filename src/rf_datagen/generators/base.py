"""BaseGenerator ABC — shared generation pipeline."""

import os
import time
from abc import ABC

import numpy as np

from .._state import shutdown_requested
from ..config import GeneratorConfig, ImpairmentConfig, checkpoint_config_hash
from ..constants import FS, WINDOW_LEN
from ..domains import DOMAINS
from ..impairments import extract_windows, apply_impairments, configure_impairments
from ..logging_config import get_logger
from ..output import atomic_save_npy, atomic_write_csv

log = get_logger("generator")


def make_gap(min_s, max_s, fs):
    """Generate a silence gap with random duration.

    Args:
        min_s: Minimum gap duration in seconds.
        max_s: Maximum gap duration in seconds.
        fs: Sample rate in Hz.

    Returns:
        Complex-zero array of random length.
    """
    return np.zeros(max(1, int(np.random.uniform(min_s, max_s) * fs)),
                    dtype=np.complex128)


def ensure_length(fn):
    """Decorator: loop synth function until output >= window_len.

    Wraps a synthesizer function with signature (*, fs, window_len)
    so it repeats until the total output length meets window_len.
    """
    def wrapper(*, fs, window_len):
        segments = []
        total = 0
        while total < window_len:
            seg = fn(fs=fs, window_len=window_len)
            segments.append(seg)
            total += len(seg)
        return np.concatenate(segments) if len(segments) > 1 else segments[0]
    wrapper.__name__ = fn.__name__
    wrapper.__doc__ = fn.__doc__
    return wrapper


class BaseGenerator(ABC):
    """Abstract base for all signal generators.

    Subclasses must define:
        name: str               — generator name ("synthetic", "fldigi", etc.)
        required_tools: list    — CLI tools that must be in PATH
        signal_classes: list    — signal class names this generator covers
        synthesizers: dict      — {CLASS_NAME: synth_fn} registry
    """

    name: str = ""
    required_tools: list = []
    signal_classes: list = []
    synthesizers: dict = {}

    def __init__(self, config: GeneratorConfig,
                 impairment_config: ImpairmentConfig = None,
                 fs: int = FS, window_len: int = WINDOW_LEN):
        self.config = config
        self.impairment_config = impairment_config or ImpairmentConfig()
        self.samples_per_class = config.samples_per_class
        self.fs = fs
        self.window_len = window_len
        # Resolve output dtype from domain registry
        self._dtype = None
        for domain in DOMAINS.values():
            if domain.sample_rate == fs and domain.window_length == window_len:
                self._dtype = domain.dtype.type
                break
        if self._dtype is None:
            import numpy as np
            self._dtype = np.complex128

    def check_prerequisites(self):
        """Return list of missing CLI tools. Empty = ready."""
        import shutil
        missing = []
        for tool in self.required_tools:
            if shutil.which(tool) is None:
                missing.append(tool)
        return missing

    def generate_class(self, class_name, rng=None):
        """Produce raw IQ for one class. Returns complex array.

        Default implementation loops the synthesizer function from
        self.synthesizers until target_samples is reached. Subclasses
        may override for special handling (e.g. CW wpm, RSID injection).
        """
        synth_fn = self.synthesizers[class_name]
        segments = []
        target_samples = max(self.window_len * 10,
                             self.samples_per_class * self.window_len // 4)
        total = 0
        while total < target_samples:
            seg = synth_fn(fs=self.fs, window_len=self.window_len)
            segments.append(seg)
            total += len(seg)
        return np.concatenate(segments)

    def generate_windows(self, class_name, n_windows):
        """Produce n_windows independent raw IQ windows for one class.

        Each window comes from a fresh synthesis call with random crop,
        maximizing signal diversity. Returns [n_windows, window_len]
        complex array with each window normalized to unit power.
        """
        from ..impairments.effects import normalize_power
        synth_fn = self.synthesizers[class_name]
        power_threshold = self.impairment_config.window_power_threshold
        windows = np.zeros((n_windows, self.window_len), dtype=np.complex128)
        i = 0
        max_retries = n_windows * 3  # safety valve
        attempts = 0
        while i < n_windows and attempts < max_retries:
            raw = synth_fn(fs=self.fs, window_len=self.window_len)
            attempts += 1
            if len(raw) < self.window_len:
                continue
            # Random crop
            start = np.random.randint(0, len(raw) - self.window_len + 1)
            w = raw[start:start + self.window_len].copy()
            # Power filter
            if np.mean(np.abs(w) ** 2) <= power_threshold:
                continue
            windows[i] = normalize_power(w)
            i += 1
        if i < n_windows:
            log.warning("%s: only got %d/%d windows after %d attempts",
                        class_name, i, n_windows, attempts)
            windows = windows[:i]
        return windows

    def run(self, output_dir, seed=42):
        """Full pipeline: generate -> extract windows -> impair -> save.

        Returns a dict mapping class names to result info:
            {"FT8": {"status": "ok", "samples": 6000, "time_s": 12.3}, ...}
        Failed classes have status "failed" with a "reason" key.
        Cached classes have status "cached".
        """
        configure_impairments(self.impairment_config)

        parts_dir = os.path.join(output_dir, "parts", self.name)
        os.makedirs(parts_dir, exist_ok=True)

        classes = self._resolve_classes()
        results = {}
        if not classes:
            log.info("%s: no classes to generate", self.name)
            return results

        log.info("%s: generating %d classes, %d samples each",
                 self.name, len(classes), self.samples_per_class)

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

            # Use per-sample synthesis if synthesizers are available,
            # otherwise fall back to bulk generate + extract_windows
            if class_name in self.synthesizers:
                raw_windows = self.generate_windows(class_name, n_samples)
            else:
                raw_iq = self.generate_class(class_name)
                if len(raw_iq) < self.window_len:
                    log.warning("%15s: FAILED (signal too short)",
                                class_name)
                    results[class_name] = {"status": "failed",
                                           "reason": "signal too short"}
                    continue
                stride = self.impairment_config.effective_stride(
                    self.window_len)
                power_thr = self.impairment_config.window_power_threshold
                raw_windows = extract_windows(
                    raw_iq, window_len=self.window_len,
                    stride=stride, power_threshold=power_thr)
            if len(raw_windows) == 0:
                log.warning("%15s: FAILED (no valid windows)", class_name)
                results[class_name] = {"status": "failed",
                                       "reason": "no valid windows"}
                continue

            samples, meta = apply_impairments(
                raw_windows, n_samples, fs=self.fs,
                window_len=self.window_len, return_metadata=True,
                dtype=self._dtype)

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
            class_name, n_samples, self.fs, self.window_len,
            generator_name=self.name)

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
