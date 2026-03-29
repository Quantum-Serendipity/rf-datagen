"""Shared constants for the rf-datagen package."""

from .domains import ALL_SIGNAL_LABELS

# Default narrowband parameters (used as fallback when domain is unspecified)
FS = 12_000              # IQ sample rate (Hz)
WINDOW_LEN = 2048        # Samples per training window (~170ms at 12 kHz)
MAX_FREQ_OFFSET = 500    # Hz (RTL-SDR realistic carrier offset)

SNR_LEVELS = [25, 20, 15, 10, 5, 0, -5, -10]  # dB

# Canonical signal labels — single source of truth in domains.py
SIGNAL_LABELS = list(ALL_SIGNAL_LABELS)
NUM_CLASSES = len(SIGNAL_LABELS)
