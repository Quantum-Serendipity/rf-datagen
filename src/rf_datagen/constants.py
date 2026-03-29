"""Shared constants for the rf-datagen package."""

# Legacy narrowband defaults (backward compatible)
FS = 12_000              # IQ sample rate (Hz)
WINDOW_LEN = 2048        # Samples per training window (~170ms at 12 kHz)
MAX_FREQ_OFFSET = 500    # Hz (RTL-SDR realistic carrier offset)

# Per-domain sample rates and window lengths
NARROWBAND_FS = 12_000
NARROWBAND_WINDOW_LEN = 2048

MODERATE_FS = 1_000_000
MODERATE_WINDOW_LEN = 131_072

WIDEBAND_FS = 20_000_000
WIDEBAND_WINDOW_LEN = 2_097_152

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
    # Gen4.2 additions — expanded modes
    "MSK144", "EAS", "ARDOP", "BELL103", "BELL202",
    # Gen4.3 additions — moderate integration
    "ATV",
    # Gen4.4 additions — expanded synthetic + GNU Radio ecosystem
    "LORA", "POCSAG", "FLEX", "HDRADIO", "DTMF", "DRM",
    # Gen5 additions — universal RF: nav/timing, aviation, maritime, IoT,
    # industrial, public safety, EW, radar
    "WWVB", "DCF77", "NDB", "ACARS", "SELCAL", "ATIS",
    "AIS", "SIGFOX", "TPMS", "SCADA_TELEMETRY", "TETRA",
    "SPOT_JAMMER", "SWEEP_JAMMER", "NOISE_JAMMER", "BARRAGE_JAMMER",
    "PULSE_RADAR", "BARKER_RADAR",
]
NUM_CLASSES = len(SIGNAL_LABELS)  # 68
