"""Scenario-based impairment profiles and the high-level pipeline."""

import numpy as np

from ..config import ImpairmentConfig
from ..constants import FS, WINDOW_LEN
from .effects import (
    normalize_power, add_awgn, freq_shift,
    apply_watterson, apply_watterson_sdc,
    apply_rayleigh, apply_rician,
    apply_qsb, apply_atmospheric_noise,
    apply_iq_imbalance, apply_phase_noise, apply_dc_offset,
    apply_adc_quantization, apply_clock_jitter,
    apply_image_rejection,
    apply_impulse_noise, apply_adjacent_signal,
    apply_powerline_hum, apply_narrowband_birdie,
    apply_doppler_rate, apply_tapped_delay_line,
    apply_clutter, apply_ism_interference,
)
from .transmitter import TransmitterModel


# --- Module-level impairment configuration ---

_config = ImpairmentConfig()


def configure(cfg: ImpairmentConfig):
    """Set impairment config for the pipeline. Call before apply_impairments()."""
    global _config
    _config = cfg


def _watterson(sig, fs=FS):
    """Dispatch to the configured Watterson model."""
    if _config.watterson_model == "sdc":
        return apply_watterson_sdc(sig, fs)
    return apply_watterson(sig, fs)


# Real multi-signal interference pool (optional)
_INTERFERER_POOL = None


def load_interferer_pool(data_dir):
    """Load first 50 clean windows per class from the synthetic dataset."""
    global _INTERFERER_POOL
    import os

    iq_path = os.path.join(data_dir, "synthetic_iq.npy")
    tags_path = os.path.join(data_dir, "synthetic_tags.csv")

    if not os.path.exists(iq_path) or not os.path.exists(tags_path):
        return None

    iq_data = np.load(iq_path)
    with open(tags_path, "r") as f:
        tags = [line.strip() for line in f if line.strip()]

    pool = {}
    for i, tag in enumerate(tags):
        if i >= len(iq_data):
            break
        if tag not in pool:
            pool[tag] = []
        if len(pool[tag]) < 50:
            pool[tag].append(iq_data[i])

    for k in pool:
        pool[k] = np.array(pool[k])

    _INTERFERER_POOL = pool
    return pool


def apply_real_interferer(sig, pool=None, n_interferers=None, fs=FS):
    """Mix real signal windows as adjacent-channel interferers."""
    if pool is None:
        pool = _INTERFERER_POOL
    if pool is None or len(pool) == 0:
        return apply_adjacent_signal(sig, fs)

    if n_interferers is None:
        n_interferers = np.random.randint(1, 4)

    classes = list(pool.keys())
    result = sig.copy()
    sig_power = np.mean(np.abs(sig) ** 2)
    if sig_power < 1e-20:
        return sig

    for _ in range(n_interferers):
        cls = classes[np.random.randint(len(classes))]
        windows = pool[cls]
        w = windows[np.random.randint(len(windows))].copy()
        if len(w) < len(sig):
            w = np.pad(w, (0, len(sig) - len(w)))
        elif len(w) > len(sig):
            start = np.random.randint(0, len(w) - len(sig))
            w = w[start:start + len(sig)]
        offset_hz = np.random.uniform(-2500, 2500)
        w = freq_shift(w, offset_hz, fs)
        rel_db = np.random.uniform(-10, -3)
        w_power = np.mean(np.abs(w) ** 2)
        if w_power > 1e-20:
            scale = np.sqrt(sig_power * 10 ** (rel_db / 10) / w_power)
            w *= scale
        result += w
    return result


def apply_agc_compression(sig, strong_signal_db, fs=FS):
    """AGC gain reduction from a strong adjacent signal."""
    reduction_db = max(0.0, strong_signal_db - 6.0) * 0.8
    gain = 10 ** (-reduction_db / 20)
    return sig * gain


def apply_receiver_intermod(sig, fs=FS):
    """3rd-order intermodulation from strong adjacent signal."""
    n = len(sig)
    t = np.arange(n) / fs
    sig_power = np.mean(np.abs(sig) ** 2)
    if sig_power < 1e-20:
        return sig
    strong_db = np.random.uniform(0, 20)
    strong_amp = np.sqrt(sig_power) * 10 ** (strong_db / 20)
    f1 = np.random.uniform(500, 2000)
    f2 = f1 + np.random.choice([-1, 1]) * np.random.uniform(500, 2000)
    imd_db = np.random.uniform(-40, -20)
    imd_amp = strong_amp * 10 ** (imd_db / 20)
    f_imd1 = 2 * f1 - f2
    f_imd2 = 2 * f2 - f1
    phase1 = np.random.uniform(0, 2 * np.pi)
    phase2 = np.random.uniform(0, 2 * np.pi)
    imd = (imd_amp * np.exp(2j * np.pi * f_imd1 * t + 1j * phase1) +
           imd_amp * np.exp(2j * np.pi * f_imd2 * t + 1j * phase2))
    return sig + imd


def apply_auroral_scatter(sig, fs=FS):
    """Auroral scatter propagation — high Doppler spread fading."""
    n = len(sig)
    doppler_hz = np.random.uniform(50, 500)
    delay_s = np.random.uniform(1e-3, 5e-3)
    delay_samples = max(1, int(delay_s * fs))

    def _auroral_fading_tap(n, doppler, fs):
        noise = (np.random.randn(n) + 1j * np.random.randn(n)) / np.sqrt(2)
        freqs = np.fft.fftfreq(n, 1.0 / fs)
        doppler_filter = np.exp(-0.5 * (freqs / doppler) ** 2)
        filtered = np.fft.ifft(np.fft.fft(noise) * doppler_filter)
        pwr = np.sqrt(np.mean(np.abs(filtered) ** 2))
        return filtered / pwr if pwr > 1e-10 else filtered

    g1 = _auroral_fading_tap(n, doppler_hz, fs)
    g2 = _auroral_fading_tap(n, doppler_hz, fs)
    delayed = np.zeros_like(sig)
    if delay_samples < n:
        delayed[delay_samples:] = sig[:-delay_samples]
    second_path_gain = np.random.uniform(0.5, 0.8)
    return g1 * sig + second_path_gain * g2 * delayed


# --- 12 scenario profiles ---
# Each reads _config.max_freq_offset instead of a constant.

def _apply_scenario_hf_clean(sig, snr_db, fs=FS):
    mfo = _config.max_freq_offset
    sig = freq_shift(sig, np.random.uniform(-mfo, mfo), fs)
    if np.random.random() < 0.25:
        sig = apply_dc_offset(sig)
    sig = add_awgn(sig, snr_db)
    return normalize_power(sig)


def _apply_scenario_hf_good(sig, snr_db, fs=FS):
    mfo = _config.max_freq_offset
    sig = _watterson(sig, fs)
    sig = freq_shift(sig, np.random.uniform(-mfo, mfo), fs)
    if np.random.random() < 0.2:
        sig = apply_qsb(sig, fs)
    if np.random.random() < 0.25:
        sig = apply_iq_imbalance(sig)
    if np.random.random() < 0.2:
        sig = apply_phase_noise(sig, fs)
    if np.random.random() < 0.15:
        sig = apply_atmospheric_noise(sig, fs)
    if np.random.random() < 0.2:
        sig = apply_dc_offset(sig)
    sig = add_awgn(sig, snr_db)
    return normalize_power(sig)


def _apply_scenario_hf_poor(sig, snr_db, fs=FS):
    mfo = _config.max_freq_offset
    sig = _watterson(sig, fs)
    if np.random.random() < 0.4:
        sig = apply_qsb(sig, fs)
    sig = freq_shift(sig, np.random.uniform(-mfo, mfo), fs)
    if np.random.random() < 0.3:
        sig = apply_iq_imbalance(sig)
    if np.random.random() < 0.3:
        sig = apply_phase_noise(sig, fs)
    sig = apply_atmospheric_noise(sig, fs)
    if np.random.random() < 0.4:
        sig = apply_impulse_noise(sig, fs)
    if np.random.random() < 0.2:
        sig = apply_adjacent_signal(sig, fs)
    if np.random.random() < 0.15:
        sig = apply_powerline_hum(sig, fs)
    if np.random.random() < 0.25:
        sig = apply_dc_offset(sig)
    sig = add_awgn(sig, snr_db)
    return normalize_power(sig)


def _apply_scenario_vhf_mobile(sig, snr_db, fs=FS):
    mfo = _config.max_freq_offset
    sig = apply_rician(sig, fs, k_db=np.random.uniform(3, 10))
    sig = freq_shift(sig, np.random.uniform(-mfo, mfo), fs)
    if np.random.random() < 0.3:
        sig = apply_qsb(sig, fs)
    if np.random.random() < 0.25:
        sig = apply_iq_imbalance(sig)
    if np.random.random() < 0.2:
        sig = apply_phase_noise(sig, fs)
    if np.random.random() < 0.3:
        sig = apply_dc_offset(sig)
    sig = add_awgn(sig, snr_db)
    return normalize_power(sig)


def _apply_scenario_uhf_urban(sig, snr_db, fs=FS):
    mfo = _config.max_freq_offset
    sig = apply_rician(sig, fs, k_db=np.random.uniform(-3, 3))
    sig = freq_shift(sig, np.random.uniform(-mfo, mfo), fs)
    if np.random.random() < 0.3:
        sig = apply_iq_imbalance(sig)
    if np.random.random() < 0.2:
        sig = apply_adjacent_signal(sig, fs)
    if np.random.random() < 0.3:
        sig = apply_dc_offset(sig)
    sig = add_awgn(sig, snr_db)
    return normalize_power(sig)


def _apply_scenario_sdr_desktop(sig, snr_db, fs=FS):
    mfo = _config.max_freq_offset
    if np.random.random() < 0.3:
        sig = apply_rayleigh(sig)
    sig = freq_shift(sig, np.random.uniform(-mfo, mfo), fs)
    sig = apply_dc_offset(sig)
    sig = apply_iq_imbalance(sig)
    if np.random.random() < 0.3:
        sig = apply_image_rejection(sig)
    if np.random.random() < 0.2:
        sig = apply_phase_noise(sig, fs)
    if np.random.random() < 0.15:
        sig = apply_narrowband_birdie(sig, fs)
    if snr_db > 5:
        sig = apply_adc_quantization(sig, bits=np.random.choice([8, 10]))
    if np.random.random() < 0.3:
        sig = apply_clock_jitter(sig, fs)
    sig = add_awgn(sig, snr_db)
    return normalize_power(sig)


def _apply_scenario_contest_crowded(sig, snr_db, fs=FS):
    mfo = _config.max_freq_offset
    if np.random.random() < 0.5:
        sig = _watterson(sig, fs)
    else:
        sig = apply_rayleigh(sig)
    sig = freq_shift(sig, np.random.uniform(-mfo, mfo), fs)
    if _INTERFERER_POOL is not None:
        sig = apply_real_interferer(sig, n_interferers=np.random.randint(1, 4), fs=fs)
    else:
        for _ in range(np.random.randint(1, 4)):
            sig = apply_adjacent_signal(sig, fs)
    if np.random.random() < 0.3:
        sig = apply_powerline_hum(sig, fs)
    if np.random.random() < 0.25:
        sig = apply_iq_imbalance(sig)
    if np.random.random() < 0.3:
        sig = apply_dc_offset(sig)
    sig = add_awgn(sig, snr_db)
    return normalize_power(sig)


def _apply_scenario_overdriven(sig, snr_db, fs=FS):
    mfo = _config.max_freq_offset
    profile = np.random.choice(["POORLY_OPERATED", "CASUAL"])
    tx = TransmitterModel(profile)
    sig = tx.apply(sig, fs)
    if np.random.random() < 0.5:
        sig = _watterson(sig, fs)
    else:
        sig = apply_rayleigh(sig)
    sig = freq_shift(sig, np.random.uniform(-mfo, mfo), fs)
    if np.random.random() < 0.2:
        sig = apply_iq_imbalance(sig)
    if np.random.random() < 0.2:
        sig = apply_phase_noise(sig, fs)
    if np.random.random() < 0.3:
        sig = apply_dc_offset(sig)
    sig = add_awgn(sig, snr_db)
    return normalize_power(sig)


def _apply_scenario_poorly_operated(sig, snr_db, fs=FS):
    mfo = _config.max_freq_offset
    tx = TransmitterModel("POORLY_OPERATED")
    sig = tx.apply(sig, fs)
    if np.random.random() < 0.5:
        sig = _watterson(sig, fs)
    else:
        sig = apply_rayleigh(sig)
    sig = freq_shift(sig, np.random.uniform(-mfo, mfo), fs)
    if np.random.random() < 0.25:
        sig = apply_iq_imbalance(sig)
    if np.random.random() < 0.3:
        sig = apply_dc_offset(sig)
    sig = add_awgn(sig, snr_db)
    return normalize_power(sig)


def _apply_scenario_vintage(sig, snr_db, fs=FS):
    mfo = _config.max_freq_offset
    tx = TransmitterModel("VINTAGE")
    sig = tx.apply(sig, fs)
    if np.random.random() < 0.5:
        sig = _watterson(sig, fs)
    else:
        sig = apply_rayleigh(sig)
    sig = freq_shift(sig, np.random.uniform(-mfo, mfo), fs)
    if np.random.random() < 0.2:
        sig = apply_atmospheric_noise(sig, fs)
    if np.random.random() < 0.3:
        sig = apply_dc_offset(sig)
    sig = add_awgn(sig, snr_db)
    return normalize_power(sig)


def _apply_scenario_near_far(sig, snr_db, fs=FS):
    mfo = _config.max_freq_offset
    strong_db = np.random.uniform(10, 30)
    sig = apply_agc_compression(sig, strong_db, fs)
    if np.random.random() < 0.7:
        sig = apply_receiver_intermod(sig, fs)
    if np.random.random() < 0.5:
        sig = _watterson(sig, fs)
    else:
        sig = apply_rayleigh(sig)
    sig = freq_shift(sig, np.random.uniform(-mfo, mfo), fs)
    if np.random.random() < 0.3:
        sig = apply_dc_offset(sig)
    sig = add_awgn(sig, snr_db)
    return normalize_power(sig)


def _apply_scenario_auroral(sig, snr_db, fs=FS):
    mfo = _config.max_freq_offset
    sig = apply_auroral_scatter(sig, fs)
    sig = freq_shift(sig, np.random.uniform(-mfo, mfo), fs)
    if np.random.random() < 0.6:
        sig = apply_atmospheric_noise(sig, fs)
    if np.random.random() < 0.3:
        sig = apply_impulse_noise(sig, fs)
    if np.random.random() < 0.2:
        sig = apply_iq_imbalance(sig)
    if np.random.random() < 0.25:
        sig = apply_dc_offset(sig)
    sig = add_awgn(sig, snr_db)
    return normalize_power(sig)


# --- 7 new scenario profiles for multi-domain coverage ---

def _apply_scenario_indoor_multipath(sig, snr_db, fs=FS):
    """Indoor multipath: WiFi, BLE, Zigbee, DECT."""
    mfo = _config.max_freq_offset
    # Rician K=0-10 dB (LOS often present indoors)
    sig = apply_rician(sig, fs, k_db=np.random.uniform(0, 10))
    # Short multipath delays (< 100 ns in indoor)
    if np.random.random() < 0.6:
        delays = [0, 50e-9, 100e-9]
        powers = [0, np.random.uniform(-6, -3), np.random.uniform(-12, -6)]
        doppler = np.random.uniform(1, 10)
        sig = apply_tapped_delay_line(sig, delays, powers, doppler, fs)
    sig = freq_shift(sig, np.random.uniform(-mfo, mfo), fs)
    if np.random.random() < 0.3:
        sig = apply_iq_imbalance(sig)
    if np.random.random() < 0.2:
        sig = apply_dc_offset(sig)
    sig = add_awgn(sig, snr_db)
    return normalize_power(sig)


def _apply_scenario_leo_satellite(sig, snr_db, fs=FS):
    """LEO satellite: Iridium, NOAA APT, COSPAS-SARSAT."""
    mfo = _config.max_freq_offset
    # Large Doppler shift/rate from LEO pass
    doppler_rate = np.random.uniform(10, 100)  # Hz/s
    sig = apply_doppler_rate(sig, doppler_rate, fs)
    # Free-space path loss variation (slow amplitude fading)
    if np.random.random() < 0.4:
        sig = apply_qsb(sig, fs)
    # Scintillation (ionospheric)
    if np.random.random() < 0.3:
        sig = apply_rician(sig, fs, k_db=np.random.uniform(5, 15))
    sig = freq_shift(sig, np.random.uniform(-mfo * 2, mfo * 2), fs)
    if np.random.random() < 0.2:
        sig = apply_phase_noise(sig, fs)
    if np.random.random() < 0.2:
        sig = apply_dc_offset(sig)
    sig = add_awgn(sig, snr_db)
    return normalize_power(sig)


def _apply_scenario_automotive(sig, snr_db, fs=FS):
    """Automotive: TPMS, BLE beacon, V2X."""
    mfo = _config.max_freq_offset
    # Rayleigh with high Doppler (100-500 Hz for highway speeds)
    doppler = np.random.uniform(100, 500)
    sig = apply_rician(sig, fs, k_db=np.random.uniform(-5, 3))
    sig = freq_shift(sig, np.random.uniform(-mfo, mfo), fs)
    # Ignition impulse noise
    if np.random.random() < 0.4:
        sig = apply_impulse_noise(sig, fs)
    if np.random.random() < 0.3:
        sig = apply_iq_imbalance(sig)
    if np.random.random() < 0.2:
        sig = apply_dc_offset(sig)
    sig = add_awgn(sig, snr_db)
    return normalize_power(sig)


def _apply_scenario_urban_cellular(sig, snr_db, fs=FS):
    """Urban cellular: GSM, LTE, 5G NR."""
    mfo = _config.max_freq_offset
    # ITU Pedestrian or Vehicular delay profile
    profile = np.random.choice(["ped_a", "ped_b", "veh_a"])
    if profile == "ped_a":
        delays = [0, 110e-9, 190e-9, 410e-9]
        powers = [0, -9.7, -19.2, -22.8]
        doppler = np.random.uniform(1, 10)
    elif profile == "ped_b":
        delays = [0, 200e-9, 800e-9, 1200e-9, 2300e-9, 3700e-9]
        powers = [0, -0.9, -4.9, -8.0, -7.8, -23.9]
        doppler = np.random.uniform(1, 10)
    else:  # veh_a
        delays = [0, 310e-9, 710e-9, 1090e-9, 1730e-9, 2510e-9]
        powers = [0, -1.0, -9.0, -10.0, -15.0, -20.0]
        doppler = np.random.uniform(50, 200)

    sig = apply_tapped_delay_line(sig, delays, powers, doppler, fs)
    sig = freq_shift(sig, np.random.uniform(-mfo, mfo), fs)
    # Co-channel interference
    if np.random.random() < 0.3:
        sig = apply_adjacent_signal(sig, fs)
    if np.random.random() < 0.2:
        sig = apply_iq_imbalance(sig)
    if np.random.random() < 0.2:
        sig = apply_dc_offset(sig)
    sig = add_awgn(sig, snr_db)
    return normalize_power(sig)


def _apply_scenario_radar_clutter(sig, snr_db, fs=FS):
    """Radar clutter: all radar types."""
    mfo = _config.max_freq_offset
    # Ground/sea clutter
    clutter_type = np.random.choice(["gaussian", "weibull", "k"])
    scr_db = np.random.uniform(5, 25)
    sig = apply_clutter(sig, clutter_type, scr_db, fs)
    sig = freq_shift(sig, np.random.uniform(-mfo, mfo), fs)
    if np.random.random() < 0.3:
        sig = apply_phase_noise(sig, fs)
    if np.random.random() < 0.2:
        sig = apply_dc_offset(sig)
    sig = add_awgn(sig, snr_db)
    return normalize_power(sig)


def _apply_scenario_maritime(sig, snr_db, fs=FS):
    """Maritime: AIS, maritime VHF."""
    mfo = _config.max_freq_offset
    # 2-ray sea-surface multipath
    delays = [0, np.random.uniform(0.5e-3, 5e-3)]
    powers = [0, np.random.uniform(-10, -3)]
    doppler = np.random.uniform(0.5, 5)
    sig = apply_tapped_delay_line(sig, delays, powers, doppler, fs)
    sig = freq_shift(sig, np.random.uniform(-mfo, mfo), fs)
    # High atmospheric noise (maritime environment)
    if np.random.random() < 0.4:
        sig = apply_atmospheric_noise(sig, fs)
    if np.random.random() < 0.2:
        sig = apply_dc_offset(sig)
    sig = add_awgn(sig, snr_db)
    return normalize_power(sig)


def _apply_scenario_ism_congested(sig, snr_db, fs=FS):
    """ISM congested: BLE, WiFi, Zigbee, LoRa."""
    mfo = _config.max_freq_offset
    # Multi-source co-channel interference
    n_interferers = np.random.randint(1, 5)
    sig = apply_ism_interference(sig, n_interferers, fs)
    sig = freq_shift(sig, np.random.uniform(-mfo, mfo), fs)
    if np.random.random() < 0.3:
        sig = apply_iq_imbalance(sig)
    if np.random.random() < 0.2:
        sig = apply_dc_offset(sig)
    sig = add_awgn(sig, snr_db)
    return normalize_power(sig)


# --- Scenario registry ---

_SCENARIO_FUNCS = {
    "hf_clean": _apply_scenario_hf_clean,
    "hf_good": _apply_scenario_hf_good,
    "hf_poor": _apply_scenario_hf_poor,
    "vhf_mobile": _apply_scenario_vhf_mobile,
    "uhf_urban": _apply_scenario_uhf_urban,
    "sdr_desktop": _apply_scenario_sdr_desktop,
    "contest_crowded": _apply_scenario_contest_crowded,
    "overdriven": _apply_scenario_overdriven,
    "poorly_operated": _apply_scenario_poorly_operated,
    "vintage": _apply_scenario_vintage,
    "near_far": _apply_scenario_near_far,
    "auroral": _apply_scenario_auroral,
    # Sprint 4 — multi-domain scenarios
    "indoor_multipath": _apply_scenario_indoor_multipath,
    "leo_satellite": _apply_scenario_leo_satellite,
    "automotive": _apply_scenario_automotive,
    "urban_cellular": _apply_scenario_urban_cellular,
    "radar_clutter": _apply_scenario_radar_clutter,
    "maritime": _apply_scenario_maritime,
    "ism_congested": _apply_scenario_ism_congested,
}

SCENARIO_NAMES = list(_SCENARIO_FUNCS.keys())


def _build_weights():
    """Build normalized weight array from config, matching _SCENARIO_FUNCS order."""
    names = list(_SCENARIO_FUNCS.keys())
    weights = np.array([_config.scenario_weights.get(n, 0.0) for n in names])
    total = weights.sum()
    if total <= 0:
        weights = np.ones(len(names))
    weights /= weights.sum()
    return weights


def apply_scenario(sig, snr_db, fs=FS):
    """Apply a randomly-selected scenario to one independent training window."""
    names = list(_SCENARIO_FUNCS.keys())
    funcs = list(_SCENARIO_FUNCS.values())
    weights = _build_weights()
    idx = np.random.choice(len(funcs), p=weights)
    return funcs[idx](sig, snr_db, fs), names[idx]


def apply_scenario_continuous(sig, snr_db, fs=FS, scenario=None):
    """Apply a single coherent scenario across a continuous signal.

    Use this instead of apply_scenario when the signal spans multiple windows
    concatenated before impairment (e.g. round-trip validation).
    """
    if scenario is not None:
        if scenario not in _SCENARIO_FUNCS:
            raise ValueError(
                f"Unknown scenario '{scenario}'. "
                f"Available: {SCENARIO_NAMES}")
        idx = SCENARIO_NAMES.index(scenario)
    else:
        weights = _build_weights()
        idx = np.random.choice(len(SCENARIO_NAMES), p=weights)

    func = _SCENARIO_FUNCS[SCENARIO_NAMES[idx]]
    return func(sig, snr_db, fs), SCENARIO_NAMES[idx]


def apply_impairments(raw_windows, target_count, fs=FS, window_len=WINDOW_LEN,
                      interferer_pool=None, return_metadata=False):
    """Produce target_count impaired samples from raw clean windows.

    Distributes samples evenly across SNR levels (from config), applying
    scenario-based impairment profiles for physically coherent augmentation.
    """
    global _INTERFERER_POOL
    if interferer_pool is not None:
        _INTERFERER_POOL = interferer_pool

    if len(raw_windows) == 0:
        empty = np.zeros((0, window_len), dtype=np.complex128)
        return (empty, {"scenarios": []}) if return_metadata else empty

    snr_levels = _config.snr_levels

    samples = np.zeros((target_count, window_len), dtype=np.complex128)
    scenarios = [] if return_metadata else None
    snr_per = target_count // len(snr_levels)
    idx = 0

    for snr_idx, snr_db in enumerate(snr_levels):
        count = (snr_per if snr_idx < len(snr_levels) - 1
                 else target_count - idx)
        for _ in range(count):
            w = raw_windows[np.random.randint(len(raw_windows))].copy()
            impaired, scenario_name = apply_scenario(w, snr_db, fs)
            samples[idx] = impaired
            if scenarios is not None:
                scenarios.append(scenario_name)
            idx += 1

    if return_metadata:
        return samples, {"scenarios": scenarios}
    return samples
