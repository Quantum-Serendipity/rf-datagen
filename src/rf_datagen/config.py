"""Configuration loading from TOML files with CLI override support."""

import hashlib
import json
import tomllib
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

from .constants import FS, WINDOW_LEN, SNR_LEVELS, MAX_FREQ_OFFSET


@dataclass
class ImpairmentConfig:
    snr_levels: list[int] = field(default_factory=lambda: list(SNR_LEVELS))
    max_freq_offset: int = MAX_FREQ_OFFSET
    scenario_weights: dict[str, float] = field(default_factory=lambda: {
        "hf_clean": 0.07, "hf_good": 0.16, "hf_poor": 0.16,
        "vhf_mobile": 0.08, "uhf_urban": 0.04, "sdr_desktop": 0.08,
        "contest_crowded": 0.08, "overdriven": 0.04, "poorly_operated": 0.04,
        "vintage": 0.03, "near_far": 0.02, "auroral": 0.02,
        # Sprint 4 — multi-domain scenarios
        "indoor_multipath": 0.04, "leo_satellite": 0.03,
        "automotive": 0.02, "urban_cellular": 0.03,
        "radar_clutter": 0.02, "maritime": 0.02, "ism_congested": 0.02,
    })
    watterson_model: str = "builtin"  # "builtin" or "sdc" (scikit-dsp-comm)
    window_stride: int = 0  # 0 = auto (window_length // 2)
    window_power_threshold: float = 0.001

    def effective_stride(self, window_length=WINDOW_LEN):
        """Return stride to use, resolving 0 -> window_length // 2."""
        return self.window_stride if self.window_stride > 0 else window_length // 2

    def _hash_dict(self):
        return {
            "snr_levels": sorted(self.snr_levels),
            "max_freq_offset": self.max_freq_offset,
            "scenario_weights": dict(sorted(self.scenario_weights.items())),
            "watterson_model": self.watterson_model,
            "window_stride": self.window_stride,
            "window_power_threshold": self.window_power_threshold,
        }


@dataclass
class GeneratorConfig:
    enabled: bool = True
    samples_per_class: int = 6000
    classes: str | list[str] = "all"
    workers: int = 0
    boost: dict[str, float] = field(default_factory=dict)
    # Analog/digivoice specific
    utterances_per_class: int = 100
    voice_cache: str = "./piper-voices"
    # Synthetic generator
    rsid_probability: float = 0.35
    cw_wpm_range: list[int] = field(default_factory=lambda: [10, 30])
    # WSJT-X generator
    messages_per_mode: int = 200
    # SSTV generator
    images_per_mode: int = 10
    # Packet generator
    packets_per_baud: int = 100
    # Digivoice generator
    codec2_mode: str = "3200"
    freedv_modes: list[str] = field(default_factory=lambda: ["1600", "700C", "700D", "700E"])
    # CW CLI generator
    cw_tone_range: list[int] = field(default_factory=lambda: [400, 800])
    # Minimodem generator
    minimodem_modes: list[str] = field(default_factory=lambda: ["rtty", "bell103", "bell202"])
    # ARDOP generator
    ardop_speeds: list[int] = field(default_factory=lambda: [200, 500, 1000, 2000])

    def _hash_dict(self):
        """Fields that affect generated signal content (not format/layout)."""
        return {
            "utterances_per_class": self.utterances_per_class,
            "rsid_probability": self.rsid_probability,
            "cw_wpm_range": sorted(self.cw_wpm_range),
            "messages_per_mode": self.messages_per_mode,
            "images_per_mode": self.images_per_mode,
            "packets_per_baud": self.packets_per_baud,
            "codec2_mode": self.codec2_mode,
            "freedv_modes": sorted(self.freedv_modes),
            "cw_tone_range": sorted(self.cw_tone_range),
            "minimodem_modes": sorted(self.minimodem_modes),
            "ardop_speeds": sorted(self.ardop_speeds),
        }


@dataclass
class DatasetConfig:
    sample_rate: int = FS
    window_length: int = WINDOW_LEN
    output_dir: str = "./output"
    seed: int = 42
    workers: int = 0  # 0 = auto; used as fallback for generators without workers
    domains: list[str] = field(default_factory=lambda: ["narrowband"])


class ConfigError(ValueError):
    """Raised when configuration values are invalid."""


@dataclass
class Config:
    dataset: DatasetConfig = field(default_factory=DatasetConfig)
    impairments: ImpairmentConfig = field(default_factory=ImpairmentConfig)
    generators: dict[str, GeneratorConfig] = field(default_factory=lambda: {
        "synthetic": GeneratorConfig(samples_per_class=9600),
        "fldigi": GeneratorConfig(workers=14),
        "wsjtx": GeneratorConfig(),
        "analog": GeneratorConfig(),
        "sstv": GeneratorConfig(),
        "packet": GeneratorConfig(),
        "digivoice": GeneratorConfig(utterances_per_class=120),
        "cw": GeneratorConfig(enabled=False),
        "msk144": GeneratorConfig(enabled=False),
        "minimodem": GeneratorConfig(enabled=False),
        "sameeas": GeneratorConfig(enabled=False),
        "ardop": GeneratorConfig(enabled=False),
        "js8call": GeneratorConfig(enabled=False),
        "op25": GeneratorConfig(enabled=False),
        "hacktv": GeneratorConfig(enabled=False),
        "synthetic_moderate": GeneratorConfig(samples_per_class=2000),
        "synthetic_wideband": GeneratorConfig(samples_per_class=1000),
    })

    def validate(self):
        """Validate configuration values. Raises ConfigError on problems."""
        from .domains import DOMAINS
        errors = []
        if self.dataset.sample_rate <= 0:
            errors.append("dataset.sample_rate must be > 0")
        if self.dataset.window_length <= 0:
            errors.append("dataset.window_length must be > 0")
        for d in self.dataset.domains:
            if d not in DOMAINS:
                errors.append(
                    f"dataset.domains: unknown domain '{d}' "
                    f"(valid: {list(DOMAINS.keys())})")
        if not self.impairments.snr_levels:
            errors.append("impairments.snr_levels must not be empty")
        if self.impairments.window_stride < 0:
            errors.append("impairments.window_stride must be >= 0")
        if self.impairments.window_power_threshold < 0:
            errors.append("impairments.window_power_threshold must be >= 0")
        for name, gen in self.generators.items():
            if gen.enabled and gen.samples_per_class <= 0:
                errors.append(
                    f"generators.{name}.samples_per_class must be > 0")
        if errors:
            raise ConfigError(
                "Invalid configuration:\n  " + "\n  ".join(errors))


def checkpoint_config_hash(gen_cfg: "GeneratorConfig",
                           imp_cfg: "ImpairmentConfig",
                           class_name: str,
                           n_samples: int,
                           fs: int = FS,
                           window_len: int = WINDOW_LEN) -> str:
    """Compute a short hash of parameters that affect checkpoint contents.

    If any of these change, cached checkpoints are stale and must be
    regenerated.  Returns a 12-char hex string.
    """
    blob = {
        "class": class_name,
        "n_samples": n_samples,
        "fs": fs,
        "window_len": window_len,
        "generator": gen_cfg._hash_dict(),
        "impairments": imp_cfg._hash_dict(),
    }
    raw = json.dumps(blob, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(raw.encode()).hexdigest()[:12]


def _merge_generator(base: GeneratorConfig, toml_dict: dict) -> GeneratorConfig:
    """Merge TOML dict into a GeneratorConfig."""
    for key in ("enabled", "samples_per_class", "classes", "workers",
                "utterances_per_class", "voice_cache",
                "rsid_probability", "cw_wpm_range",
                "messages_per_mode", "images_per_mode", "packets_per_baud",
                "codec2_mode", "freedv_modes",
                "cw_tone_range", "minimodem_modes", "ardop_speeds"):
        if key in toml_dict:
            setattr(base, key, toml_dict[key])
    if "boost" in toml_dict:
        base.boost = {k: float(v) for k, v in toml_dict["boost"].items()}
    return base


_KNOWN_SECTIONS = {"dataset", "impairments", "generators"}
_KNOWN_DATASET_KEYS = {"sample_rate", "window_length", "output_dir", "seed",
                       "workers", "domains"}
_KNOWN_IMPAIRMENT_KEYS = {"snr_levels", "max_freq_offset", "watterson_model",
                          "window_stride", "window_power_threshold", "scenarios"}
_KNOWN_GENERATOR_KEYS = {
    "enabled", "samples_per_class", "classes", "workers",
    "utterances_per_class", "voice_cache",
    "rsid_probability", "cw_wpm_range",
    "messages_per_mode", "images_per_mode", "packets_per_baud",
    "codec2_mode", "freedv_modes", "boost",
    "cw_tone_range", "minimodem_modes", "ardop_speeds",
}


def _warn_unknown_keys(section_name, keys, known):
    """Log warnings for unrecognized TOML keys (likely typos)."""
    from .logging_config import get_logger
    _log = get_logger("config")
    unknown = set(keys) - known
    for k in sorted(unknown):
        _log.warning("Unknown key '%s' in [%s] (ignored — possible typo?)",
                     k, section_name)


def load_config(path: Optional[str | Path] = None) -> Config:
    """Load configuration from a TOML file.

    If path is None, returns defaults. CLI flags can override after loading.
    """
    cfg = Config()

    if path is None:
        return cfg

    path = Path(path)
    if not path.exists():
        return cfg

    with open(path, "rb") as f:
        raw = tomllib.load(f)

    # Warn on unknown top-level sections
    _warn_unknown_keys("top-level", raw.keys(), _KNOWN_SECTIONS)

    # Dataset section
    if "dataset" in raw:
        d = raw["dataset"]
        _warn_unknown_keys("dataset", d.keys(), _KNOWN_DATASET_KEYS)
        for key in ("sample_rate", "window_length", "output_dir", "seed",
                     "workers", "domains"):
            if key in d:
                setattr(cfg.dataset, key, d[key])

    # Impairments section
    if "impairments" in raw:
        imp = raw["impairments"]
        _warn_unknown_keys("impairments", imp.keys(), _KNOWN_IMPAIRMENT_KEYS)
        if "snr_levels" in imp:
            cfg.impairments.snr_levels = imp["snr_levels"]
        if "max_freq_offset" in imp:
            cfg.impairments.max_freq_offset = imp["max_freq_offset"]
        if "watterson_model" in imp:
            cfg.impairments.watterson_model = imp["watterson_model"]
        if "window_stride" in imp:
            cfg.impairments.window_stride = imp["window_stride"]
        if "window_power_threshold" in imp:
            cfg.impairments.window_power_threshold = imp["window_power_threshold"]
        if "scenarios" in imp:
            cfg.impairments.scenario_weights = {
                k: float(v) for k, v in imp["scenarios"].items()
            }

    # Generators section — TOML is declarative: only listed generators run.
    # When no [generators] section, the Config() defaults apply (all enabled).
    if "generators" in raw:
        cfg.generators = {}
        for gen_name, gen_dict in raw["generators"].items():
            _warn_unknown_keys(f"generators.{gen_name}",
                               gen_dict.keys(), _KNOWN_GENERATOR_KEYS)
            cfg.generators[gen_name] = GeneratorConfig()
            _merge_generator(cfg.generators[gen_name], gen_dict)

    cfg.validate()
    return cfg
