"""Signal domain registry — multi-rate architecture for universal RF coverage.

Signals are grouped into domains by required sample rate:
  - narrowband (12 kHz): HF/VHF/UHF amateur, nav/timing, EW, narrowband radar
  - moderate   (1 MHz):  BLE, ADS-B, GSM, satellite, wider radar
  - wideband  (20 MHz):  WiFi, LTE, 5G NR, GPS, DAB, DVB-T
"""

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class SignalDomain:
    name: str            # "narrowband", "moderate", "wideband"
    sample_rate: int     # Hz
    window_length: int   # samples per training window
    dtype: np.dtype      # complex dtype for storage
    default_samples_per_class: int
    description: str


NARROWBAND = SignalDomain(
    name="narrowband",
    sample_rate=12_000,
    window_length=2048,
    dtype=np.dtype(np.complex128),
    default_samples_per_class=6000,
    description="HF/VHF/UHF narrowband (6 kHz Nyquist)",
)

MODERATE = SignalDomain(
    name="moderate",
    sample_rate=1_000_000,
    window_length=131_072,
    dtype=np.dtype(np.complex64),
    default_samples_per_class=2000,
    description="BLE/radar/ADS-B (500 kHz Nyquist)",
)

WIDEBAND = SignalDomain(
    name="wideband",
    sample_rate=20_000_000,
    window_length=2_097_152,
    dtype=np.dtype(np.complex64),
    default_samples_per_class=1000,
    description="WiFi/LTE/GPS (10 MHz Nyquist)",
)

DOMAINS = {
    "narrowband": NARROWBAND,
    "moderate": MODERATE,
    "wideband": WIDEBAND,
}

# ---------------------------------------------------------------------------
# Signal-to-domain mapping
# ---------------------------------------------------------------------------

# Narrowband signals (FS=12 kHz) — all existing 51 + 17 new from Sprint 1
_NARROWBAND_LABELS = [
    # Original (gen1-2)
    "CW", "PSK31", "PSK63", "RTTY", "OLIVIA", "JS8", "FT8", "FT4",
    "WSPR", "JT65", "JT9", "SSB", "AM", "FM", "SSTV", "FAX",
    "NAVTEX", "NOISE", "DOMINOEX", "MT63", "HELLSCHREIBER",
    "MFSK16", "MFSK32", "CONTESTIA", "THOR", "PACKET",
    # Gen3
    "QPSK", "PSK125", "8PSK", "FSQ", "IFKP", "THROB",
    # Gen4.1 — digital voice
    "FREEDV", "M17", "DMR", "DSTAR", "YSF", "P25", "NXDN",
    # Gen4.2 — expanded modes
    "MSK144", "EAS", "ARDOP", "BELL103", "BELL202",
    # Gen4.3
    "ATV",
    # Gen4.4
    "LORA", "POCSAG", "FLEX", "HDRADIO", "DTMF", "DRM",
    # Gen5 — universal RF narrowband
    "WWVB", "DCF77", "NDB", "ACARS", "SELCAL", "ATIS",
    "AIS", "SIGFOX", "TPMS", "SCADA_TELEMETRY", "TETRA",
    "SPOT_JAMMER", "SWEEP_JAMMER", "NOISE_JAMMER", "BARRAGE_JAMMER",
    "PULSE_RADAR", "BARKER_RADAR",
]

# Moderate-rate signals (FS=1 MHz)
_MODERATE_LABELS = [
    "BLE", "ZWAVE", "ADS_B", "GSM_BURST",
    "LFM_RADAR", "FMCW_RADAR", "PHASE_CODED_RADAR",
    "NOAA_APT", "COSPAS_SARSAT", "LORA_WIDE",
    "VDL2", "DRM_WIDE", "DECT", "IRIDIUM",
]

# Wideband signals (FS=20 MHz)
_WIDEBAND_LABELS = [
    "WIFI_PREAMBLE", "LTE_FRAME", "FIVEG_NR", "GPS_L1",
    "ZIGBEE", "DAB", "DVB_T", "LORAN_C_WIDE",
]


def _build_domain_map():
    """Build signal_class -> SignalDomain mapping."""
    m = {}
    for label in _NARROWBAND_LABELS:
        m[label] = NARROWBAND
    for label in _MODERATE_LABELS:
        m[label] = MODERATE
    for label in _WIDEBAND_LABELS:
        m[label] = WIDEBAND
    return m


SIGNAL_DOMAIN_MAP = _build_domain_map()


def labels_for_domain(domain_name):
    """Return ordered list of signal labels for a given domain."""
    if domain_name == "narrowband":
        return list(_NARROWBAND_LABELS)
    elif domain_name == "moderate":
        return list(_MODERATE_LABELS)
    elif domain_name == "wideband":
        return list(_WIDEBAND_LABELS)
    else:
        raise ValueError(f"Unknown domain: {domain_name}")


def all_signal_labels():
    """Return all signal labels across all domains (narrowband first)."""
    return _NARROWBAND_LABELS + _MODERATE_LABELS + _WIDEBAND_LABELS


ALL_SIGNAL_LABELS = all_signal_labels()
