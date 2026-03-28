"""Individual RF channel effect functions.

References:
    ITU-R F.1487  — HF Watterson channel model
    ITU-R P.1411  — VHF/UHF propagation, Rician K-factors
    ITU-R M.1225  — Mobile channel delay profiles
    ITU-R P.372   — Man-made noise levels
    Rapp (1991)   — Solid-state PA non-linearity model
    Park et al.   — SpecAugment (time/frequency masking)
"""

import numpy as np

from ..constants import FS, WINDOW_LEN


def normalize_power(sig):
    """Normalize to unit average power."""
    p = np.sqrt(np.mean(np.abs(sig) ** 2))
    return sig / p if p > 1e-10 else sig


def add_awgn(sig, snr_db):
    """Add complex AWGN at specified SNR (dB)."""
    sig_power = np.mean(np.abs(sig) ** 2)
    if sig_power < 1e-20:
        sig_power = 1.0
    noise_power = sig_power * 10 ** (-snr_db / 10)
    noise = np.sqrt(noise_power / 2) * (
        np.random.randn(len(sig)) + 1j * np.random.randn(len(sig)))
    return sig + noise


def freq_shift(sig, offset_hz, fs=FS):
    """Apply frequency offset to complex signal."""
    t = np.arange(len(sig)) / fs
    return sig * np.exp(2j * np.pi * offset_hz * t)


def apply_watterson(sig, fs=FS):
    """ITU-R F.1487 Watterson HF channel model.

    2-tap Rayleigh with Gaussian Doppler spectrum.
    """
    profiles = [(0.5e-3, 0.1), (1.0e-3, 0.5), (2.0e-3, 1.0)]
    delay_s, doppler_hz = profiles[np.random.randint(0, 3)]
    delay_samples = max(1, int(delay_s * fs))
    n = len(sig)

    def _fading_tap(n, doppler, fs):
        noise = (np.random.randn(n) + 1j * np.random.randn(n)) / np.sqrt(2)
        if doppler < 0.01:
            return noise * np.abs(noise[0])
        freqs = np.fft.fftfreq(n, 1.0 / fs)
        doppler_filter = np.exp(-0.5 * (freqs / doppler) ** 2)
        filtered = np.fft.ifft(np.fft.fft(noise) * doppler_filter)
        pwr = np.sqrt(np.mean(np.abs(filtered) ** 2))
        return filtered / pwr if pwr > 1e-10 else filtered

    g1 = _fading_tap(n, doppler_hz, fs)
    g2 = _fading_tap(n, doppler_hz, fs)
    delayed = np.zeros_like(sig)
    if delay_samples < n:
        delayed[delay_samples:] = sig[:-delay_samples]
    return g1 * sig + 0.5 * g2 * delayed


def apply_watterson_sdc(sig, fs=FS):
    """Watterson HF channel using scikit-dsp-comm for higher-fidelity simulation.

    Uses sk_dsp_comm.multipath_fading.FadingModel with ITU-R F.1487 profiles.
    Falls back to the built-in apply_watterson() if scikit-dsp-comm is unavailable.
    """
    try:
        from sk_dsp_comm.multipath_fading import FadingModel
    except ImportError:
        return apply_watterson(sig, fs)

    profiles = [
        # (delay_spread_ms, doppler_hz, N_taps, description)
        (0.5, 0.1, 2, "good"),
        (1.0, 0.5, 2, "moderate"),
        (2.0, 1.0, 2, "poor"),
        (2.0, 2.0, 3, "disturbed"),
    ]
    delay_ms, doppler_hz, n_taps, _ = profiles[np.random.randint(0, len(profiles))]

    try:
        chan = FadingModel(
            N_taps=n_taps,
            fd=doppler_hz,
            fs=fs,
            power_profile=tuple(1.0 / n_taps for _ in range(n_taps)),
            delay_profile=tuple(i * delay_ms * 1e-3 for i in range(n_taps)),
        )
        out = np.zeros_like(sig)
        for i in range(len(sig)):
            out[i] = chan.model_step(sig[i])
        return out
    except Exception:
        return apply_watterson(sig, fs)


def apply_rayleigh(sig):
    """Simplified 2-path Rayleigh flat fading (HF propagation)."""
    g1 = (np.random.randn() + 1j * np.random.randn()) / np.sqrt(2)
    g2 = (np.random.randn() + 1j * np.random.randn()) / np.sqrt(2)
    delay = np.random.randint(1, 4)
    delayed = np.zeros_like(sig)
    if delay < len(sig):
        delayed[delay:] = sig[:-delay]
    return g1 * sig + 0.3 * g2 * delayed


def apply_rician(sig, fs=FS, k_db=None):
    """Rician fading for VHF/UHF paths with line-of-sight component."""
    if k_db is None:
        k_db = np.random.uniform(-3, 15)
    k_linear = 10 ** (k_db / 10)
    n = len(sig)
    los_amp = np.sqrt(k_linear / (k_linear + 1))
    scatter_amp = np.sqrt(1.0 / (k_linear + 1))
    doppler_hz = np.random.uniform(1.0, 30.0)
    freqs = np.fft.fftfreq(n, 1.0 / fs)
    with np.errstate(divide='ignore', invalid='ignore'):
        jakes = np.where(
            np.abs(freqs) < doppler_hz,
            1.0 / np.sqrt(np.maximum(1e-10, 1 - (freqs / doppler_hz) ** 2)),
            0.0)
    jakes[0] = 0.0
    jakes_energy = np.sqrt(np.sum(jakes ** 2))
    if jakes_energy > 1e-10:
        jakes /= jakes_energy
    scatter = (np.random.randn(n) + 1j * np.random.randn(n)) / np.sqrt(2)
    scatter = np.fft.ifft(np.fft.fft(scatter) * jakes * np.sqrt(n))
    pwr = np.sqrt(np.mean(np.abs(scatter) ** 2))
    if pwr > 1e-10:
        scatter /= pwr
    los_phase = np.random.uniform(0, 2 * np.pi)
    channel = los_amp * np.exp(1j * los_phase) + scatter_amp * scatter
    if np.random.random() < 0.5:
        delay = np.random.randint(1, 6)
        delayed = np.zeros_like(sig)
        if delay < n:
            delayed[delay:] = sig[:-delay]
        g2 = (np.random.randn() + 1j * np.random.randn()) / np.sqrt(2)
        scatter_power = 0.2 * scatter_amp
        return channel * sig + scatter_power * g2 * delayed
    return channel * sig


def apply_qsb(sig, fs=FS):
    """Slow amplitude fading (QSB) — sinusoidal power variation."""
    fade_hz = np.random.uniform(0.2, 2.0)
    depth_db = np.random.uniform(3.0, 15.0)
    depth_linear = 10 ** (-depth_db / 20)
    t = np.arange(len(sig)) / fs
    phase = np.random.uniform(0, 2 * np.pi)
    envelope = 1.0 - (1.0 - depth_linear) * 0.5 * (
        1 + np.sin(2 * np.pi * fade_hz * t + phase))
    return sig * envelope


def apply_atmospheric_noise(sig, fs=FS):
    """Band-specific HF atmospheric noise — colored 1/f^a + white."""
    n = len(sig)
    alpha = np.random.uniform(0.5, 1.5)
    freqs = np.fft.fftfreq(n, 1.0 / fs)
    freqs[0] = 1.0
    spectral_shape = 1.0 / np.abs(freqs) ** (alpha / 2)
    spectral_shape[0] = 0.0
    white = (np.random.randn(n) + 1j * np.random.randn(n)) / np.sqrt(2)
    colored = np.fft.ifft(np.fft.fft(white) * spectral_shape)
    pwr = np.sqrt(np.mean(np.abs(colored) ** 2))
    if pwr > 1e-10:
        colored /= pwr
    noise = 0.7 * colored + 0.3 * white
    sig_rms = np.sqrt(np.mean(np.abs(sig) ** 2))
    if sig_rms < 1e-10:
        sig_rms = 1.0
    noise_db = np.random.uniform(-15, -3)
    noise_level = sig_rms * 10 ** (noise_db / 20)
    return sig + noise_level * noise


def apply_clock_drift(sig, fs=FS):
    """Oscillator frequency drift from crystal ppm error."""
    drift_rate = np.random.uniform(0.1, 5.0)
    if np.random.random() < 0.5:
        drift_rate = -drift_rate
    n = len(sig)
    t = np.arange(n) / fs
    phase = 2 * np.pi * drift_rate * t * t / 2
    return sig * np.exp(1j * phase)


def apply_iq_imbalance(sig):
    """IQ gain and phase imbalance from real SDR hardware."""
    gain_db = np.random.uniform(0.5, 3.0)
    phase_deg = np.random.uniform(1.0, 5.0)
    g = 10 ** (gain_db / 20)
    phi = np.radians(phase_deg)
    i_in, q_in = sig.real, sig.imag
    i_out = i_in * np.cos(phi / 2) + q_in * np.sin(phi / 2)
    q_out = g * (i_in * np.sin(phi / 2) + q_in * np.cos(phi / 2))
    return i_out + 1j * q_out


def apply_phase_noise(sig, fs=FS):
    """Oscillator phase noise — colored 1/f + white noise model."""
    n = len(sig)
    white = (np.random.randn(n) + 1j * np.random.randn(n)) / np.sqrt(2)
    W = np.fft.fft(white)
    freqs = np.fft.fftfreq(n, 1.0 / fs)
    with np.errstate(divide='ignore', invalid='ignore'):
        flicker_shape = np.where(
            np.abs(freqs) > 0.1,
            1.0 / np.sqrt(np.abs(freqs)),
            0.0)
    flicker_shape[0] = 0.0
    combined = 0.7 * np.fft.ifft(W * flicker_shape).real + 0.3 * white.real
    phase_std = np.random.uniform(0.01, 0.12)
    rms = np.sqrt(np.mean(combined ** 2))
    if rms > 1e-10:
        combined *= phase_std / rms
    phase = np.cumsum(combined)
    return sig * np.exp(1j * phase)


def apply_dc_offset(sig):
    """SDR DC offset / LO leakage spike at center frequency."""
    dc_i = np.random.uniform(-0.05, 0.05)
    dc_q = np.random.uniform(-0.05, 0.05)
    return sig + (dc_i + 1j * dc_q)


def apply_adc_quantization(sig, bits=None):
    """ADC quantization noise from finite bit-depth digitization."""
    if bits is None:
        bits = np.random.choice([8, 10, 12])
    levels = 2 ** (bits - 1)
    peak = np.abs(sig).max()
    if peak < 1e-10:
        return sig
    scaled = sig / peak
    i_q = np.round(scaled.real * levels) / levels
    q_q = np.round(scaled.imag * levels) / levels
    return (i_q + 1j * q_q) * peak


def apply_clock_jitter(sig, fs=FS):
    """Sampling clock jitter from SDR crystal oscillator imperfections."""
    n = len(sig)
    jitter_ns = np.random.uniform(10, 100)
    jitter_samples = jitter_ns * 1e-9 * fs
    dsig = np.diff(sig, prepend=sig[0]) * fs
    dt = np.random.randn(n) * jitter_samples / fs
    jitter_noise = dsig * dt
    return sig + jitter_noise


def apply_nonlinear_distortion(sig):
    """Polynomial non-linearity model for TX/RX chain."""
    a3 = np.random.uniform(-0.3, -0.01)
    a5 = a3 * np.random.uniform(0.01, 0.1)
    peak = np.abs(sig).max()
    if peak < 1e-10:
        return sig
    x = sig / peak
    abs_sq = np.abs(x) ** 2
    y = x + a3 * x * abs_sq + a5 * x * abs_sq ** 2
    return y * peak


def apply_image_rejection(sig):
    """SDR image response — mirror-frequency leakage."""
    rejection_db = np.random.uniform(25, 50)
    image_level = 10 ** (-rejection_db / 20)
    return sig + image_level * np.conj(sig)


def apply_impulse_noise(sig, fs=FS):
    """Atmospheric/ignition impulse noise — short broadband bursts."""
    n_bursts = np.random.randint(1, 6)
    result = sig.copy()
    sig_rms = np.sqrt(np.mean(np.abs(sig) ** 2))
    if sig_rms < 1e-10:
        sig_rms = 1.0
    for _ in range(n_bursts):
        dur = np.random.randint(max(1, int(0.0001 * fs)),
                                max(2, int(0.002 * fs)))
        start = np.random.randint(0, max(1, len(sig) - dur))
        amp = sig_rms * np.random.uniform(2.0, 20.0)
        burst = amp * (np.random.randn(dur)
                       + 1j * np.random.randn(dur)) / np.sqrt(2)
        result[start:start + dur] += burst
    return result


def apply_adjacent_signal(sig, fs=FS):
    """Adjacent-channel interference — nearby narrowband signal bleed."""
    sig_power = np.mean(np.abs(sig) ** 2)
    if sig_power < 1e-20:
        return sig
    offset_hz = np.random.choice([-1, 1]) * np.random.uniform(500, 2500)
    rel_db = np.random.uniform(-20, -3)
    adj_power = sig_power * 10 ** (rel_db / 10)
    t = np.arange(len(sig)) / fs
    if np.random.random() < 0.5:
        adj = np.sqrt(adj_power) * np.exp(2j * np.pi * offset_hz * t)
    else:
        bw = np.random.uniform(50, 200)
        noise = (np.random.randn(len(sig))
                 + 1j * np.random.randn(len(sig))) / np.sqrt(2)
        freqs = np.fft.fftfreq(len(sig), 1.0 / fs)
        mask = np.exp(-0.5 * ((freqs - offset_hz) / (bw / 2)) ** 2)
        adj = np.fft.ifft(np.fft.fft(noise) * mask)
        adj *= np.sqrt(adj_power / max(1e-20, np.mean(np.abs(adj) ** 2)))
    return sig + adj


def apply_powerline_hum(sig, fs=FS):
    """Power line interference — harmonic comb at 100/120 Hz."""
    comb_hz = np.random.choice([100, 120])
    n = len(sig)
    t = np.arange(n) / fs
    sig_rms = np.sqrt(np.mean(np.abs(sig) ** 2))
    if sig_rms < 1e-10:
        sig_rms = 1.0
    hum_db = np.random.uniform(0, 20)
    hum_level = sig_rms * 10 ** (hum_db / 20) * 0.1
    n_harmonics = int(fs / 2 / comb_hz)
    hum = np.zeros(n, dtype=np.complex128)
    for k in range(1, n_harmonics + 1):
        freq = k * comb_hz
        amp = hum_level / np.sqrt(k)
        phase = np.random.uniform(0, 2 * np.pi)
        hum += amp * np.exp(2j * np.pi * freq * t + 1j * phase)
    return sig + hum


def apply_narrowband_birdie(sig, fs=FS):
    """SDR LO spur — fixed-frequency narrowband interference."""
    sig_power = np.mean(np.abs(sig) ** 2)
    if sig_power < 1e-20:
        return sig
    n_birdies = np.random.randint(1, 4)
    t = np.arange(len(sig)) / fs
    result = sig.copy()
    for _ in range(n_birdies):
        freq = np.random.uniform(-fs / 2, fs / 2)
        rel_db = np.random.uniform(-30, -10)
        birdie_power = sig_power * 10 ** (rel_db / 10)
        phase = np.random.uniform(0, 2 * np.pi)
        result += np.sqrt(birdie_power) * np.exp(
            2j * np.pi * freq * t + 1j * phase)
    return result


def apply_time_mask(sig, n_masks=None, max_width=200):
    """Zero out random time segments (SpecAugment time masking)."""
    if n_masks is None:
        n_masks = np.random.randint(1, 4)
    result = sig.copy()
    for _ in range(n_masks):
        width = np.random.randint(50, max_width + 1)
        start = np.random.randint(0, max(1, len(result) - width))
        result[start:start + width] = 0
    return result


def apply_freq_mask(sig, n_masks=None, max_width_frac=0.15, fs=FS):
    """Zero out random frequency bands (SpecAugment frequency masking)."""
    if n_masks is None:
        n_masks = np.random.randint(1, 3)
    n = len(sig)
    X = np.fft.fft(sig)
    max_bins = int(n * max_width_frac)
    for _ in range(n_masks):
        width = np.random.randint(1, max(2, max_bins))
        start = np.random.randint(0, max(1, n - width))
        X[start:start + width] = 0
    return np.fft.ifft(X)


def extract_windows(iq_signal, window_len=WINDOW_LEN, stride=None,
                     power_threshold=None):
    """Extract non-silent training windows from a continuous IQ signal."""
    if stride is None:
        stride = window_len // 2
    if power_threshold is None:
        power_threshold = 0.001
    if len(iq_signal) < window_len:
        return np.zeros((0, window_len), dtype=np.complex128)
    windows = []
    for start in range(0, len(iq_signal) - window_len + 1, stride):
        w = iq_signal[start:start + window_len]
        if np.mean(np.abs(w) ** 2) > power_threshold:
            windows.append(normalize_power(w))
    if not windows:
        return np.zeros((0, window_len), dtype=np.complex128)
    return np.array(windows)
