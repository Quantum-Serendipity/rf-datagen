"""Transmitter chain impairment models."""

import numpy as np

from ..constants import FS


def apply_alc_compression(sig, attack_ms, release_ms, ratio, threshold_db,
                          fs=FS):
    """Envelope compressor modelling automatic level control (ALC)."""
    n = len(sig)
    envelope = np.abs(sig)
    threshold = 10 ** (threshold_db / 20)
    attack_coeff = 1.0 - np.exp(-1.0 / (attack_ms * 1e-3 * fs))
    release_coeff = 1.0 - np.exp(-1.0 / (release_ms * 1e-3 * fs))
    gain = np.ones(n)
    env_state = 0.0
    for i in range(n):
        coeff = attack_coeff if envelope[i] > env_state else release_coeff
        env_state += coeff * (envelope[i] - env_state)
        if env_state > threshold:
            reduction_db = (20 * np.log10(env_state / threshold)) * (1 - 1 / ratio)
            gain[i] = 10 ** (-reduction_db / 20)
    return sig * gain


def apply_rf_clipping(sig, clip_db, fs=FS):
    """Hard-clip at threshold then bandpass to limit splatter."""
    threshold = 10 ** (clip_db / 20)
    mag = np.abs(sig)
    phase = np.angle(sig)
    clipped_mag = np.minimum(mag, threshold)
    clipped = clipped_mag * np.exp(1j * phase)
    n = len(clipped)
    freqs = np.fft.fftfreq(n, 1.0 / fs)
    bp_mask = np.zeros(n)
    bp_mask[(np.abs(freqs) >= 300) & (np.abs(freqs) <= 3000)] = 1.0
    for f_edge, bw in [(300, 50), (3000, 50)]:
        lower = np.exp(-0.5 * ((np.abs(freqs) - f_edge) / bw) ** 2)
        bp_mask = np.maximum(bp_mask, lower * (np.abs(freqs) < f_edge + bw))
    bp_mask = np.clip(bp_mask, 0, 1)
    return np.fft.ifft(np.fft.fft(clipped) * bp_mask)


def apply_tx_hum(sig, hum_freq, hum_level_db, fs=FS):
    """Multiplicative TX power-supply hum."""
    hum_level = 10 ** (hum_level_db / 20)
    t = np.arange(len(sig)) / fs
    return sig * (1 + hum_level * np.sin(2 * np.pi * hum_freq * t))


def apply_key_clicks(sig, rise_fall_ms, fs=FS):
    """Apply finite rise/fall time to keying transitions."""
    n = len(sig)
    mag = np.abs(sig)
    mean_mag = np.mean(mag)
    if mean_mag < 1e-10:
        return sig
    transition_samples = max(1, int(rise_fall_ms * 1e-3 * fs))
    envelope = np.ones(n, dtype=float)
    mag_smooth = np.convolve(mag, np.ones(3) / 3, mode='same')
    dmag = np.abs(np.diff(mag_smooth, prepend=mag_smooth[0]))
    threshold = 0.3 * mean_mag
    transition_idx = np.where(dmag > threshold)[0]
    if len(transition_idx) == 0:
        return sig
    half_win = transition_samples
    for idx in transition_idx:
        start = max(0, idx - half_win)
        end = min(n, idx + half_win)
        win_len = end - start
        if win_len < 2:
            continue
        ramp = 0.5 * (1 - np.cos(np.pi * np.arange(win_len) / win_len))
        if mag_smooth[min(idx, n - 1)] > mag_smooth[max(idx - 1, 0)]:
            envelope[start:end] = np.minimum(envelope[start:end], ramp)
        else:
            envelope[start:end] = np.minimum(envelope[start:end], ramp[::-1])
    return sig * envelope


def apply_tx_drift(sig, drift_hz, fs=FS):
    """Slow linear frequency drift (e.g. tube VFO thermal drift)."""
    n = len(sig)
    t = np.arange(n) / fs
    duration = n / fs
    drift_rate = drift_hz / duration
    return sig * np.exp(2j * np.pi * drift_rate * t ** 2 / 2)


def apply_switching_noise(sig, fs=FS):
    """Switch-mode PSU spurs — 2-4 narrowband tones aliased into 12 kHz BW."""
    n = len(sig)
    t = np.arange(n) / fs
    sig_power = np.mean(np.abs(sig) ** 2)
    if sig_power < 1e-20:
        return sig
    n_spurs = np.random.randint(2, 5)
    result = sig.copy()
    for _ in range(n_spurs):
        spur_freq = np.random.uniform(-fs / 2, fs / 2)
        spur_db = np.random.uniform(-40, -20)
        spur_amp = np.sqrt(sig_power) * 10 ** (spur_db / 20)
        phase = np.random.uniform(0, 2 * np.pi)
        result += spur_amp * np.exp(2j * np.pi * spur_freq * t + 1j * phase)
    return result


class TransmitterModel:
    """Transmitter profile applying a physically-coherent chain of TX impairments."""

    PROFILES = {
        "WELL_OPERATED": dict(
            alc_ratio=2.0, alc_threshold_db=-6.0,
            alc_attack_ms=10.0, alc_release_ms=100.0,
            clip_db=None, hum_db=None, hum_freq=None,
            key_click_ms=None, drift_hz=0.0, switching_noise=False,
        ),
        "CASUAL": dict(
            alc_ratio=4.0, alc_threshold_db=-6.0,
            alc_attack_ms=5.0, alc_release_ms=80.0,
            clip_db=None, hum_db=-25.0, hum_freq=120.0,
            key_click_ms=3.0, drift_hz=0.0, switching_noise=False,
        ),
        "POORLY_OPERATED": dict(
            alc_ratio=8.0, alc_threshold_db=-3.0,
            alc_attack_ms=2.0, alc_release_ms=50.0,
            clip_db=-3.0, hum_db=-15.0, hum_freq=100.0,
            key_click_ms=1.0, drift_hz=0.0, switching_noise=True,
        ),
        "VINTAGE": dict(
            alc_ratio=1.5, alc_threshold_db=-3.0,
            alc_attack_ms=20.0, alc_release_ms=200.0,
            clip_db=-1.0, hum_db=None, hum_freq=None,
            key_click_ms=0.5, drift_hz=None, switching_noise=False,
        ),
    }

    WEIGHTS = {
        "WELL_OPERATED": 0.50,
        "CASUAL": 0.25,
        "POORLY_OPERATED": 0.15,
        "VINTAGE": 0.10,
    }

    def __init__(self, profile=None):
        if profile is None:
            names = list(self.WEIGHTS.keys())
            weights = np.array([self.WEIGHTS[n] for n in names])
            weights /= weights.sum()
            profile = np.random.choice(names, p=weights)
        self.profile_name = profile
        self.params = dict(self.PROFILES[profile])
        if self.params["key_click_ms"] is None:
            self.params["key_click_ms"] = np.random.uniform(5, 8)

    def apply(self, sig, fs=FS):
        """Apply the profile's transmitter effects to sig."""
        p = self.params
        sig = apply_alc_compression(
            sig, p["alc_attack_ms"], p["alc_release_ms"],
            p["alc_ratio"], p["alc_threshold_db"], fs)
        if p["clip_db"] is not None:
            sig = apply_rf_clipping(sig, p["clip_db"], fs)
        if p["hum_db"] is not None and p["hum_freq"] is not None:
            sig = apply_tx_hum(sig, p["hum_freq"], p["hum_db"], fs)
        sig = apply_key_clicks(sig, p["key_click_ms"], fs)
        drift = p["drift_hz"]
        if drift is None:
            drift = np.random.uniform(10, 50)
        if drift > 0:
            sig = apply_tx_drift(sig, drift, fs)
        if p["switching_noise"]:
            sig = apply_switching_noise(sig, fs)
        return sig
