"""Base class and registry for round-trip validators."""

import shutil
from abc import ABC, abstractmethod
from typing import ClassVar

VALIDATORS = {}


def register(*mode_names):
    """Decorator to register a validator class for one or more mode names."""
    def decorator(cls):
        for name in mode_names:
            VALIDATORS[name] = cls
        return cls
    return decorator


class BaseRoundtripValidator(ABC):
    """Base class for round-trip validators.

    Subclasses must implement make_trial(mode) which returns a callable
    (snr_db, trial_idx, tmpdir) -> bool.
    """

    required_tools: ClassVar[list] = []
    tier: ClassVar[int] = 1        # 1=exact, 2=sync/metric, 3=spectral
    expected_fail: ClassVar[bool] = False

    @classmethod
    def available(cls) -> bool:
        """Check if all required external tools are on PATH."""
        return all(shutil.which(t) for t in cls.required_tools)

    @classmethod
    def missing_tools(cls) -> list:
        """Return list of missing tool names."""
        return [t for t in cls.required_tools if not shutil.which(t)]

    @abstractmethod
    def make_trial(self, mode: str):
        """Return a callable (snr_db, trial_idx, tmpdir) -> bool."""

    def setup(self):
        """Called once before any trials. Override for one-time init."""

    def teardown(self):
        """Called after all trials. Override for cleanup."""
