"""Shared constants for the rf-datagen package."""

FS = 12_000              # IQ sample rate (Hz)
WINDOW_LEN = 2048        # Samples per training window (~170ms at 12 kHz)
MAX_FREQ_OFFSET = 500    # Hz (RTL-SDR realistic carrier offset)

SNR_LEVELS = [25, 20, 15, 10, 5, 0, -5, -10]  # dB

SIGNAL_LABELS = [
    # Original (gen1-2)
    "CW", "PSK31", "PSK63", "RTTY", "OLIVIA", "JS8", "FT8", "FT4",
    "WSPR", "JT65", "JT9", "SSB", "AM", "FM", "SSTV", "FAX",
    "NAVTEX", "NOISE", "DOMINOEX", "MT63", "HELLSCHREIBER",
    "MFSK16", "MFSK32", "CONTESTIA", "THOR", "PACKET",
    # Gen3 additions
    "QPSK", "PSK125", "8PSK", "FSQ", "IFKP", "THROB",
    # Gen4.1 additions — digital voice
    "FREEDV", "M17", "DMR", "DSTAR", "YSF", "P25", "NXDN",
]
NUM_CLASSES = len(SIGNAL_LABELS)  # 39
