"""BaseGenerator ABC — shared generation pipeline."""

import csv
import os
import time
from abc import ABC, abstractmethod

import numpy as np

from ..config import GeneratorConfig, ImpairmentConfig
from ..constants import FS, WINDOW_LEN
from ..impairments import extract_windows, apply_impairments, configure_impairments


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
        """Full pipeline: generate -> extract windows -> impair -> save."""
        configure_impairments(self.impairment_config)

        parts_dir = os.path.join(output_dir, "parts")
        os.makedirs(parts_dir, exist_ok=True)

        classes = self._resolve_classes()
        if not classes:
            print(f"  {self.name}: no classes to generate")
            return

        print(f"  {self.name}: generating {len(classes)} classes, "
              f"{self.samples_per_class} samples each")

        stride = self.impairment_config.effective_stride(self.window_len)
        power_threshold = self.impairment_config.window_power_threshold

        for ci, class_name in enumerate(classes):
            n_samples = self._boosted_count(class_name)
            npy_path = os.path.join(parts_dir, f"{class_name}.npy")
            meta_path = os.path.join(parts_dir, f"{class_name}_meta.csv")

            # Skip if checkpoint exists with correct count
            if os.path.exists(npy_path) and os.path.exists(meta_path):
                existing = np.load(npy_path, mmap_mode="r")
                if existing.shape[0] == n_samples:
                    print(f"    {class_name:>15s}: {n_samples} samples (cached)")
                    continue

            class_seed = seed + ci * 1000
            np.random.seed(class_seed)
            t0 = time.time()

            raw_iq = self.generate_class(class_name)
            if len(raw_iq) < self.window_len:
                print(f"    {class_name:>15s}: FAILED (signal too short)")
                continue

            raw_windows = extract_windows(raw_iq, window_len=self.window_len,
                                          stride=stride,
                                          power_threshold=power_threshold)
            if len(raw_windows) == 0:
                print(f"    {class_name:>15s}: FAILED (no valid windows)")
                continue

            samples, meta = apply_impairments(
                raw_windows, n_samples, fs=self.fs,
                window_len=self.window_len, return_metadata=True)

            # Save checkpoint
            np.save(npy_path, samples)
            with open(meta_path, "w", newline="") as f:
                writer = csv.writer(f)
                writer.writerow(["scenario"])
                for s in meta["scenarios"]:
                    writer.writerow([s])

            elapsed = time.time() - t0
            size_mb = os.path.getsize(npy_path) / (1024 * 1024)
            print(f"    {class_name:>15s}: {len(raw_windows)} raw -> "
                  f"{n_samples} samples ({elapsed:.1f}s, {size_mb:.0f} MB)")

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
