"""Shared constants for the rf-datagen package."""

from .domains import ALL_SIGNAL_LABELS

# Default narrowband parameters (used as fallback when domain is unspecified)
FS = 12_000              # IQ sample rate (Hz)
WINDOW_LEN = 2048        # Samples per training window (~170ms at 12 kHz)
MAX_FREQ_OFFSET = 500    # Hz (RTL-SDR realistic carrier offset)

SNR_LEVELS = [30, 25, 20, 15, 10, 5, 0]  # dB

# Canonical signal labels — single source of truth in domains.py
SIGNAL_LABELS = list(ALL_SIGNAL_LABELS)
NUM_CLASSES = len(SIGNAL_LABELS)

# Streaming threshold: outputs larger than this use memmap-backed
# streaming impairments instead of in-memory arrays.  500 MB is a
# safe default that avoids OOM for moderate/wideband domains while
# keeping the faster in-memory path for narrowband.
STREAMING_THRESHOLD = 500 * 1024 * 1024  # bytes
