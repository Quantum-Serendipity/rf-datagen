"""Hierarchical signal taxonomy for metadata, filtering, and reporting.

Every signal class maps to a (category, subcategory) pair.
"""

from .domains import SIGNAL_DOMAIN_MAP, ALL_SIGNAL_LABELS

# category / subcategory mapping
_TAXONOMY = {
    # --- Ham Radio ---
    "CW":              ("ham_radio", "morse"),
    "PSK31":           ("ham_radio", "psk"),
    "PSK63":           ("ham_radio", "psk"),
    "RTTY":            ("ham_radio", "fsk"),
    "OLIVIA":          ("ham_radio", "mfsk"),
    "JS8":             ("ham_radio", "gfsk"),
    "FT8":             ("ham_radio", "gfsk"),
    "FT4":             ("ham_radio", "gfsk"),
    "WSPR":            ("ham_radio", "fsk"),
    "JT65":            ("ham_radio", "fsk"),
    "JT9":             ("ham_radio", "fsk"),
    "SSB":             ("ham_radio", "analog"),
    "AM":              ("ham_radio", "analog"),
    "FM":              ("ham_radio", "analog"),
    "SSTV":            ("ham_radio", "image"),
    "FAX":             ("ham_radio", "image"),
    "NAVTEX":          ("ham_radio", "fsk"),
    "NOISE":           ("ham_radio", "noise"),
    "DOMINOEX":        ("ham_radio", "mfsk"),
    "MT63":            ("ham_radio", "ofdm"),
    "HELLSCHREIBER":   ("ham_radio", "ook"),
    "MFSK16":          ("ham_radio", "mfsk"),
    "MFSK32":          ("ham_radio", "mfsk"),
    "CONTESTIA":       ("ham_radio", "mfsk"),
    "THOR":            ("ham_radio", "mfsk"),
    "PACKET":          ("ham_radio", "fsk"),
    "QPSK":            ("ham_radio", "psk"),
    "PSK125":          ("ham_radio", "psk"),
    "8PSK":            ("ham_radio", "psk"),
    "FSQ":             ("ham_radio", "mfsk"),
    "IFKP":            ("ham_radio", "mfsk"),
    "THROB":           ("ham_radio", "mfsk"),
    "FREEDV":          ("ham_radio", "digital_voice"),
    "M17":             ("ham_radio", "digital_voice"),
    "DMR":             ("ham_radio", "digital_voice"),
    "DSTAR":           ("ham_radio", "digital_voice"),
    "YSF":             ("ham_radio", "digital_voice"),
    "P25":             ("ham_radio", "digital_voice"),
    "NXDN":            ("ham_radio", "digital_voice"),
    "MSK144":          ("ham_radio", "gfsk"),
    "EAS":             ("ham_radio", "fsk"),
    "ARDOP":           ("ham_radio", "ofdm"),
    "BELL103":         ("ham_radio", "fsk"),
    "BELL202":         ("ham_radio", "fsk"),
    "ATV":             ("ham_radio", "analog"),
    "LORA":            ("ham_radio", "css"),
    "POCSAG":          ("ham_radio", "fsk"),
    "FLEX":            ("ham_radio", "fsk"),
    "HDRADIO":         ("ham_radio", "ofdm"),
    "DTMF":            ("ham_radio", "tone"),
    "DRM":             ("ham_radio", "ofdm"),

    # --- Aviation ---
    "ACARS":           ("aviation", "msk"),
    "SELCAL":          ("aviation", "tone"),
    "ATIS":            ("aviation", "am_voice"),
    "ADS_B":           ("aviation", "ppm"),
    "VDL2":            ("aviation", "psk"),

    # --- Maritime ---
    "AIS":             ("maritime", "gmsk"),

    # --- Navigation / Timing ---
    "WWVB":            ("navigation", "am_pulse"),
    "DCF77":           ("navigation", "ask_pulse"),
    "NDB":             ("navigation", "am_cw"),
    "GPS_L1":          ("navigation", "dsss"),
    "LORAN_C_WIDE":    ("navigation", "pulsed"),

    # --- Cellular ---
    "GSM_BURST":       ("cellular", "gmsk"),
    "LTE_FRAME":       ("cellular", "ofdm"),
    "FIVEG_NR":        ("cellular", "ofdm"),

    # --- Commercial ---
    "BLE":             ("commercial", "gfsk"),
    "ZWAVE":           ("commercial", "fsk"),
    "DECT":            ("commercial", "gfsk"),
    "WIFI_PREAMBLE":   ("commercial", "ofdm"),
    "ZIGBEE":          ("commercial", "oqpsk"),

    # --- IoT ---
    "SIGFOX":          ("iot", "dbpsk"),
    "LORA_WIDE":       ("iot", "css"),
    "TPMS":            ("iot", "ook_fsk"),

    # --- Broadcast ---
    "DAB":             ("broadcast", "ofdm"),
    "DVB_T":           ("broadcast", "ofdm"),
    "DRM_WIDE":        ("broadcast", "ofdm"),
    "NOAA_APT":        ("broadcast", "am_fsk"),

    # --- Radar ---
    "PULSE_RADAR":     ("radar", "pulsed_cw"),
    "BARKER_RADAR":    ("radar", "bpsk_coded"),
    "LFM_RADAR":       ("radar", "chirp"),
    "FMCW_RADAR":      ("radar", "fmcw"),
    "PHASE_CODED_RADAR": ("radar", "polyphase"),

    # --- Electronic Warfare ---
    "SPOT_JAMMER":     ("ew", "narrowband_noise"),
    "SWEEP_JAMMER":    ("ew", "chirp"),
    "NOISE_JAMMER":    ("ew", "wideband_noise"),
    "BARRAGE_JAMMER":  ("ew", "modulated_noise"),

    # --- Industrial ---
    "SCADA_TELEMETRY": ("industrial", "fsk"),

    # --- Public Safety ---
    "TETRA":           ("public_safety", "pi4dqpsk"),

    # --- Satellite ---
    "COSPAS_SARSAT":   ("satellite", "bpsk"),
    "IRIDIUM":         ("satellite", "dqpsk"),
}


def get_category(signal_class):
    """Return (category, subcategory) for a signal class."""
    return _TAXONOMY.get(signal_class, ("unknown", "unknown"))


def get_all_categories():
    """Return sorted list of unique categories."""
    return sorted(set(cat for cat, _ in _TAXONOMY.values()))


def signals_in_category(category):
    """Return list of signal classes in a given category."""
    return [cls for cls, (cat, _) in _TAXONOMY.items() if cat == category]


def get_signal_metadata(signal_class):
    """Return full metadata dict for a signal class."""
    category, subcategory = get_category(signal_class)
    domain = SIGNAL_DOMAIN_MAP.get(signal_class)
    return {
        "class": signal_class,
        "category": category,
        "subcategory": subcategory,
        "domain": domain.name if domain else "unknown",
        "sample_rate": domain.sample_rate if domain else 0,
        "window_length": domain.window_length if domain else 0,
    }


# Validate at import time: every label must have a taxonomy entry
_missing = [l for l in ALL_SIGNAL_LABELS if l not in _TAXONOMY]
if _missing:
    raise RuntimeError(
        f"Taxonomy missing entries for: {_missing}. "
        f"Add them to rf_datagen/taxonomy.py:_TAXONOMY.")
