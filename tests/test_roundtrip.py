#!/usr/bin/env python3
"""Round-trip reception validation — backward-compatible shim.

All logic lives in the tests.roundtrip package. This module re-exports the
public API so existing imports continue to work:

    from tests.test_roundtrip import run_validation, ALL_MODES, main

Usage:
    python -m tests.test_roundtrip
    python -m tests.test_roundtrip --modes FT8 PACKET_1200
    rf-datagen validate-roundtrip
"""

from tests.roundtrip import run_validation, ALL_MODES, DEFAULT_MODES, main

__all__ = ["run_validation", "ALL_MODES", "DEFAULT_MODES", "main"]

if __name__ == "__main__":
    main()
