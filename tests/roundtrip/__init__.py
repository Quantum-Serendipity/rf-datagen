"""Round-trip validation package — encode → decode → verify.

Public API:
    run_validation()  — run validation for selected modes
    ALL_MODES         — list of all registered mode names
    DEFAULT_MODES     — default mode set
    VALIDATORS        — mode name → validator class registry
    main()            — CLI entry point
"""

from ._base import VALIDATORS
from ._runner import (
    run_validation,
    main,
    ALL_MODES,
    DEFAULT_MODES,
    EXTENDED_MODES,
    FLDIGI_MODES_LIST,
    FLDIGI_QUICK,
    EXPECTED_FAIL_MODES,
    DEFAULT_SNR_LEVELS,
)

__all__ = [
    "run_validation",
    "main",
    "ALL_MODES",
    "DEFAULT_MODES",
    "EXTENDED_MODES",
    "FLDIGI_MODES_LIST",
    "FLDIGI_QUICK",
    "EXPECTED_FAIL_MODES",
    "DEFAULT_SNR_LEVELS",
    "VALIDATORS",
]
