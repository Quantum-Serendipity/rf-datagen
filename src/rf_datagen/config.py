"""Configuration loading from TOML files with CLI override support."""

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
        "hf_clean": 0.08, "hf_good": 0.20, "hf_poor": 0.20,
        "vhf_mobile": 0.10, "uhf_urban": 0.05, "sdr_desktop": 0.10,
        "contest_crowded": 0.10, "overdriven": 0.05, "poorly_operated": 0.05,
        "vintage": 0.03, "near_far": 0.02, "auroral": 0.02,
    })
    window_stride: int = 0  # 0 = auto (window_length // 2)
    window_power_threshold: float = 0.001

    def effective_stride(self, window_length=WINDOW_LEN):
        """Return stride to use, resolving 0 -> window_length // 2."""
        return self.window_stride if self.window_stride > 0 else window_length // 2


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


@dataclass
class DatasetConfig:
    sample_rate: int = FS
    window_length: int = WINDOW_LEN
    output_dir: str = "./output"
    seed: int = 42
    workers: int = 0  # 0 = auto; used as fallback for generators without workers


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
    })


def _merge_generator(base: GeneratorConfig, toml_dict: dict) -> GeneratorConfig:
    """Merge TOML dict into a GeneratorConfig."""
    for key in ("enabled", "samples_per_class", "classes", "workers",
                "utterances_per_class", "voice_cache",
                "rsid_probability", "cw_wpm_range",
                "messages_per_mode", "images_per_mode", "packets_per_baud",
                "codec2_mode", "freedv_modes"):
        if key in toml_dict:
            setattr(base, key, toml_dict[key])
    if "boost" in toml_dict:
        base.boost = {k: float(v) for k, v in toml_dict["boost"].items()}
    return base


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

    # Dataset section
    if "dataset" in raw:
        d = raw["dataset"]
        for key in ("sample_rate", "window_length", "output_dir", "seed",
                     "workers"):
            if key in d:
                setattr(cfg.dataset, key, d[key])

    # Impairments section
    if "impairments" in raw:
        imp = raw["impairments"]
        if "snr_levels" in imp:
            cfg.impairments.snr_levels = imp["snr_levels"]
        if "max_freq_offset" in imp:
            cfg.impairments.max_freq_offset = imp["max_freq_offset"]
        if "window_stride" in imp:
            cfg.impairments.window_stride = imp["window_stride"]
        if "window_power_threshold" in imp:
            cfg.impairments.window_power_threshold = imp["window_power_threshold"]
        if "scenarios" in imp:
            cfg.impairments.scenario_weights = {
                k: float(v) for k, v in imp["scenarios"].items()
            }

    # Generators section
    if "generators" in raw:
        for gen_name, gen_dict in raw["generators"].items():
            if gen_name not in cfg.generators:
                cfg.generators[gen_name] = GeneratorConfig()
            _merge_generator(cfg.generators[gen_name], gen_dict)

    return cfg
