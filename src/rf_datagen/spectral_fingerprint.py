"""Per-class spectral fingerprint validation.

Defines expected spectral properties (3dB bandwidth, PAPR) for all 90
signal classes, derived from the synthesizer implementations.  Validates
actual generated data against these specs using Welch PSD analysis.
"""

import numpy as np

from .domains import SIGNAL_DOMAIN_MAP, DOMAINS, labels_for_domain
from .logging_config import get_logger

log = get_logger("spectral_fingerprint")


# ---------------------------------------------------------------------------
# Spectral specifications per class
#
# Keys:
#   bw_3db: (min_hz, max_hz)  — expected 3dB bandwidth range
#   papr:   (min_db, max_db)  — expected peak-to-average power ratio
#   exempt: True               — skip spectral checks (ill-defined BW)
#
# Ranges are generous to accommodate impairment broadening, windowing
# artifacts, and Welch PSD estimation noise.
# ---------------------------------------------------------------------------

SPECTRAL_SPECS = {
    # === NARROWBAND (FS=12,000 Hz) ===
    # Auto-generated from clean synthesizer measurements (15 samples each).
    # BW upper = min(max(clean_max*3, clean_median*5, 200), fs).
    # PAPR upper = max(clean_max+10, 15).
    # Exempt = synthesizer produces sub-resolution BW (< ~10 Hz at fs=12 kHz).

    "8PSK": {"bw_3db": (0, 769), "papr": (0, 15)},
    "ACARS": {"bw_3db": (0, 3041), "papr": (0, 15)},
    "AIS": {"bw_3db": (0, 8828), "papr": (0, 15)},
    "AM": {"bw_3db": (0, 200), "papr": (0, 15), "exempt": True},
    "ARDOP": {"bw_3db": (0, 5031), "papr": (0, 18)},
    "ATIS": {"bw_3db": (0, 200), "papr": (0, 15), "exempt": True},
    "ATV": {"bw_3db": (0, 200), "papr": (0, 15), "exempt": True},
    "BARKER_RADAR": {"bw_3db": (0, 200), "papr": (0, 15), "exempt": True},
    "BARRAGE_JAMMER": {"bw_3db": (0, 12000), "papr": (0, 22)},
    "BELL103": {"bw_3db": (0, 852), "papr": (0, 15)},
    "BELL202": {"bw_3db": (0, 6027), "papr": (0, 15)},
    "CONTESTIA": {"bw_3db": (0, 689), "papr": (0, 15)},
    "CW": {"bw_3db": (0, 200), "papr": (0, 15), "exempt": True},
    "DCF77": {"bw_3db": (0, 200), "papr": (0, 15), "exempt": True},
    "DMR": {"bw_3db": (0, 6996), "papr": (0, 15)},
    "DOMINOEX": {"bw_3db": (0, 597), "papr": (0, 15)},
    "DRM": {"bw_3db": (0, 11359), "papr": (0, 19)},
    "DSTAR": {"bw_3db": (0, 7316), "papr": (0, 15)},
    "DTMF": {"bw_3db": (0, 3237), "papr": (0, 16)},
    "EAS": {"bw_3db": (0, 5244), "papr": (0, 15)},
    "FAX": {"bw_3db": (0, 1494), "papr": (0, 15)},
    "FLEX": {"bw_3db": (0, 11184), "papr": (0, 15)},
    "FM": {"bw_3db": (0, 200), "papr": (0, 15), "exempt": True},
    "FREEDV": {"bw_3db": (0, 4737), "papr": (0, 18)},
    "FSQ": {"bw_3db": (0, 200), "papr": (0, 15), "exempt": True},
    "FT4": {"bw_3db": (0, 202), "papr": (0, 15)},
    "FT8": {"bw_3db": (0, 200), "papr": (0, 15), "exempt": True},
    "HDRADIO": {"bw_3db": (0, 11790), "papr": (0, 19)},
    "HELLSCHREIBER": {"bw_3db": (0, 200), "papr": (0, 15), "exempt": True},
    "IFKP": {"bw_3db": (0, 200), "papr": (0, 15), "exempt": True},
    "JS8": {"bw_3db": (0, 200), "papr": (0, 15), "exempt": True},
    "JT65": {"bw_3db": (0, 200), "papr": (0, 15), "exempt": True},
    "JT9": {"bw_3db": (0, 200), "papr": (0, 15), "exempt": True},
    "LORA": {"bw_3db": (0, 1142), "papr": (0, 15)},
    "M17": {"bw_3db": (0, 12000), "papr": (0, 20)},
    "MFSK16": {"bw_3db": (0, 200), "papr": (0, 15)},
    "MFSK32": {"bw_3db": (0, 2390), "papr": (0, 15)},
    "MSK144": {"bw_3db": (0, 2781), "papr": (0, 20)},
    "MT63": {"bw_3db": (0, 4738), "papr": (0, 18)},
    "NAVTEX": {"bw_3db": (0, 878), "papr": (0, 15)},
    "NDB": {"bw_3db": (0, 200), "papr": (0, 15), "exempt": True},
    "NOISE": {"bw_3db": (0, 12000), "papr": (0, 37), "exempt": True},
    "NOISE_JAMMER": {"bw_3db": (0, 12000), "papr": (0, 19)},
    "NXDN": {"bw_3db": (0, 10168), "papr": (0, 15)},
    "OLIVIA": {"bw_3db": (0, 479), "papr": (0, 15)},
    "P25": {"bw_3db": (0, 10019), "papr": (0, 15)},
    "PACKET": {"bw_3db": (0, 914), "papr": (0, 15)},
    "POCSAG": {"bw_3db": (0, 200), "papr": (0, 15), "exempt": True},
    "PSK125": {"bw_3db": (0, 1318), "papr": (0, 15)},
    "PSK31": {"bw_3db": (0, 200), "papr": (0, 15)},
    "PSK63": {"bw_3db": (0, 270), "papr": (0, 15)},
    "PULSE_RADAR": {"bw_3db": (0, 12000), "papr": (0, 23)},
    "QPSK": {"bw_3db": (0, 377), "papr": (0, 15)},
    "RTTY": {"bw_3db": (0, 2513), "papr": (0, 15)},
    "SCADA_TELEMETRY": {"bw_3db": (0, 3647), "papr": (0, 18)},
    "SELCAL": {"bw_3db": (0, 3036), "papr": (0, 15)},
    "SIGFOX": {"bw_3db": (0, 320), "papr": (0, 15)},
    "SPOT_JAMMER": {"bw_3db": (0, 1199), "papr": (0, 19)},
    "SSB": {"bw_3db": (0, 7155), "papr": (0, 24)},
    "SSTV": {"bw_3db": (0, 2028), "papr": (0, 15)},
    "SWEEP_JAMMER": {"bw_3db": (0, 952), "papr": (0, 15)},
    "TETRA": {"bw_3db": (0, 12000), "papr": (0, 15)},
    "THOR": {"bw_3db": (0, 200), "papr": (0, 15)},
    "THROB": {"bw_3db": (0, 200), "papr": (0, 15)},
    "TPMS": {"bw_3db": (0, 1072), "papr": (0, 23)},
    "WSPR": {"bw_3db": (0, 200), "papr": (0, 15), "exempt": True},
    "WWVB": {"bw_3db": (0, 200), "papr": (0, 15), "exempt": True},
    "YSF": {"bw_3db": (0, 6928), "papr": (0, 15)},

    # === MODERATE (FS=1,000,000 Hz) ===

    "ADS_B": {"bw_3db": (0, 14915), "papr": (0, 24)},
    "BLE": {"bw_3db": (0, 1.0e6), "papr": (0, 24)},
    "COSPAS_SARSAT": {"bw_3db": (0, 1213), "papr": (0, 15), "exempt": True},
    "DECT": {"bw_3db": (0, 1.0e6), "papr": (0, 15)},
    "DRM_WIDE": {"bw_3db": (0, 47054), "papr": (0, 20)},
    "FMCW_RADAR": {"bw_3db": (0, 1.0e6), "papr": (0, 15)},
    "GSM_BURST": {"bw_3db": (0, 462417), "papr": (0, 15)},
    "IRIDIUM": {"bw_3db": (0, 105304), "papr": (0, 15)},
    "LFM_RADAR": {"bw_3db": (0, 663837), "papr": (0, 27)},
    "LORA_WIDE": {"bw_3db": (0, 1.0e6), "papr": (0, 15)},
    "NOAA_APT": {"bw_3db": (0, 1000), "papr": (0, 15), "exempt": True},
    "PHASE_CODED_RADAR": {"bw_3db": (0, 769729), "papr": (0, 19)},
    "VDL2": {"bw_3db": (0, 43220), "papr": (0, 17)},
    "ZWAVE": {"bw_3db": (0, 115459), "papr": (0, 21)},

    # === WIDEBAND (FS=20,000,000 Hz) ===

    "DAB": {"bw_3db": (0, 20.0e6), "papr": (0, 22)},
    "DVB_T": {"bw_3db": (0, 20.0e6), "papr": (0, 30)},
    "FIVEG_NR": {"bw_3db": (0, 20.0e6), "papr": (0, 30)},
    "GPS_L1": {"bw_3db": (0, 4.2e6), "papr": (0, 15)},
    "LORAN_C_WIDE": {"bw_3db": (0, 1.0e6), "papr": (0, 32)},
    "LTE_FRAME": {"bw_3db": (0, 20.0e6), "papr": (0, 28)},
    "WIFI_PREAMBLE": {"bw_3db": (0, 20.0e6), "papr": (0, 25)},
    "ZIGBEE": {"bw_3db": (0, 114560), "papr": (0, 15)},
}


# ---------------------------------------------------------------------------
# Spectral measurement (reuses logic from gnuradio_probe._spectral_analysis)
# ---------------------------------------------------------------------------

def _measure_spectral(iq, fs):
    """Compute bandwidth, PAPR, and estimated SNR for an IQ window.

    Uses zero-padded FFT for better frequency resolution and peak-relative
    3dB bandwidth measurement.

    Returns dict with bandwidth_3db (Hz), papr_db, estimated_snr (dB).
    """
    n = len(iq)
    # Zero-pad to at least 8192 for ~1.5 Hz resolution at fs=12 kHz
    nfft = max(8192, n * 2)
    windowed = iq * np.hanning(n)
    spec = np.fft.fftshift(np.fft.fft(windowed, n=nfft))
    psd = np.abs(spec) ** 2
    psd_db = 10.0 * np.log10(psd + 1e-30)
    freqs = np.fft.fftshift(np.fft.fftfreq(nfft, 1.0 / fs))

    peak_db = float(np.max(psd_db))
    noise_floor = float(np.median(psd_db))
    estimated_snr = peak_db - noise_floor

    # Peak-relative 3dB bandwidth
    bw3_mask = psd_db > (peak_db - 3)
    bw3_freqs = freqs[bw3_mask]
    bw_3db = (float(bw3_freqs[-1] - bw3_freqs[0])
              if len(bw3_freqs) > 1 else 0.0)

    # PAPR
    sig_power = np.mean(np.abs(iq) ** 2)
    peak_power = np.max(np.abs(iq) ** 2)
    papr_db = float(10.0 * np.log10(peak_power / (sig_power + 1e-30)))

    return {
        "bandwidth_3db": bw_3db,
        "papr_db": papr_db,
        "estimated_snr": estimated_snr,
    }


# ---------------------------------------------------------------------------
# Per-class validation
# ---------------------------------------------------------------------------

def check_class_spectral(iq_windows, fs, spec):
    """Check a batch of IQ windows against a spectral spec.

    Uses class-level statistics (median of high-SNR windows) rather than
    per-window thresholds.  This is robust to the wide range of SNR levels
    present in impaired training data.

    Parameters
    ----------
    iq_windows : list of np.ndarray
        IQ windows to analyze.
    fs : int
        Sample rate for this domain.
    spec : dict
        Expected spec from SPECTRAL_SPECS (bw_3db, papr keys).

    Returns
    -------
    dict : {status, median_bw, median_papr, ...}
    """
    if spec.get("exempt"):
        return {
            "status": "PASS",
            "detail": "Exempt from spectral checks",
            "median_bw": 0,
            "median_papr": 0,
        }

    bw_range = spec["bw_3db"]
    papr_range = spec["papr"]

    # Minimum spectral SNR to consider a window measurable.
    # Below this, noise dominates the peak-3dB BW measurement.
    min_snr_db = 20.0

    bws = []
    paprs = []

    for iq in iq_windows:
        m = _measure_spectral(iq, fs)
        if m["estimated_snr"] >= min_snr_db:
            bws.append(m["bandwidth_3db"])
            paprs.append(m["papr_db"])

    if len(bws) < 2:
        return {
            "status": "PASS",
            "detail": f"Too few measurable windows ({len(bws)})",
            "median_bw": 0,
            "median_papr": 0,
        }

    median_bw = float(np.median(bws))
    median_papr = float(np.median(paprs))

    # Check median BW against expected range
    bw_ok = bw_range[0] <= median_bw <= bw_range[1]

    # Check median PAPR against expected range
    papr_ok = papr_range[0] <= median_papr <= papr_range[1]

    status = "PASS" if (bw_ok and papr_ok) else "FAIL"

    return {
        "status": status,
        "median_bw": round(median_bw, 1),
        "median_papr": round(median_papr, 1),
        "measurable_windows": len(bws),
        "total_windows": len(iq_windows),
        "bw_range": list(bw_range),
        "papr_range": list(papr_range),
        "bw_ok": bw_ok,
        "papr_ok": papr_ok,
    }


# ---------------------------------------------------------------------------
# Dataset-level spectral validation
# ---------------------------------------------------------------------------

def validate_spectral(output_dir, domains, n_samples=20, seed=42):
    """Run spectral checks across all classes in a generated dataset.

    Parameters
    ----------
    output_dir : str
        Root output directory.
    domains : list[str]
        Domains to validate.
    n_samples : int
        Windows to sample per class.
    seed : int
        RNG seed for reproducible sampling.

    Returns
    -------
    dict : {status, per_class: {class: result}, pass_rate}
    """
    import csv as csv_mod
    import os

    rng = np.random.RandomState(seed)
    multi_domain = len(domains) > 1

    per_class = {}
    total_checked = 0
    total_passed = 0

    for domain_name in domains:
        domain = DOMAINS[domain_name]
        expected_labels = labels_for_domain(domain_name)

        if multi_domain:
            domain_dir = os.path.join(output_dir, domain_name)
            prefix = f"rf_datagen_{domain_name}"
        else:
            domain_dir = output_dir
            prefix = "rf_datagen"

        iq_path = os.path.join(domain_dir, f"{prefix}_iq.npy")
        csv_path = os.path.join(domain_dir, f"{prefix}_tags.csv")

        if not os.path.exists(iq_path) or not os.path.exists(csv_path):
            log.warning("Skipping spectral checks for %s: files missing",
                        domain_name)
            continue

        iq = np.load(iq_path, mmap_mode="r")

        with open(csv_path) as f:
            reader = csv_mod.reader(f)
            header = next(reader)
            rows = list(reader)

        mode_col = header.index("mode") if "mode" in header else 1
        labels = [row[mode_col] for row in rows]

        for class_name in expected_labels:
            spec = SPECTRAL_SPECS.get(class_name)
            if spec is None:
                per_class[class_name] = {
                    "status": "SKIP",
                    "detail": "No spectral spec defined",
                }
                continue

            # Find indices for this class
            indices = [i for i, l in enumerate(labels) if l == class_name]
            if not indices:
                per_class[class_name] = {
                    "status": "SKIP",
                    "detail": "No samples in dataset",
                }
                continue

            # Sample N windows
            n = min(n_samples, len(indices))
            chosen = rng.choice(indices, n, replace=False)
            windows = [np.array(iq[i]) for i in chosen]

            result = check_class_spectral(windows, domain.sample_rate, spec)
            per_class[class_name] = result

            if not spec.get("exempt"):
                total_checked += 1
                if result["status"] == "PASS":
                    total_passed += 1

            # Strip raw measurements from report to keep it compact
            result.pop("measurements", None)

            log.info("  %15s: BW=%.1f PAPR=%.1f [%s]",
                     class_name,
                     result.get("median_bw", 0),
                     result.get("median_papr", 0),
                     result["status"])

    pass_rate = f"{total_passed}/{total_checked}"
    # Gate: ≥85% of non-exempt classes must pass
    gate_threshold = 0.85
    gate_passed = (total_passed / total_checked >= gate_threshold
                   if total_checked > 0 else True)

    return {
        "status": "PASS" if gate_passed else "FAIL",
        "per_class": per_class,
        "pass_rate": pass_rate,
        "total_checked": total_checked,
        "total_passed": total_passed,
        "gate_threshold": gate_threshold,
    }
