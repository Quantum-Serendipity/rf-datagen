"""Filter primitives — bandpass, RRC, Gaussian."""

import numpy as np
from scipy.signal import butter, filtfilt


def bandpass_filter(audio, fs, low=300, high=3000, order=4):
    """Bandpass filter for radio voice channel."""
    nyq = fs / 2
    low_norm = max(low / nyq, 0.001)
    high_norm = min(high / nyq, 0.999)
    if low_norm >= high_norm:
        return audio
    b, a = butter(order, [low_norm, high_norm], btype='band')
    return filtfilt(b, a, audio)


def rrc_filter(n_taps, rolloff, sps):
    """Root Raised Cosine filter."""
    t = (np.arange(n_taps) - n_taps // 2) / sps
    h = np.zeros(n_taps)
    for i in range(n_taps):
        if t[i] == 0:
            h[i] = 1.0 + rolloff * (4 / np.pi - 1)
        elif abs(abs(t[i]) - 1 / (4 * rolloff)) < 1e-8 and rolloff > 0:
            h[i] = (rolloff / np.sqrt(2)) * (
                (1 + 2 / np.pi) * np.sin(np.pi / (4 * rolloff)) +
                (1 - 2 / np.pi) * np.cos(np.pi / (4 * rolloff)))
        else:
            num = np.sin(np.pi * t[i] * (1 - rolloff)) + \
                  4 * rolloff * t[i] * np.cos(np.pi * t[i] * (1 + rolloff))
            den = np.pi * t[i] * (1 - (4 * rolloff * t[i]) ** 2)
            if abs(den) > 1e-12:
                h[i] = num / den
            else:
                h[i] = 0.0
    h /= np.sqrt(np.sum(h ** 2))
    return h


def gaussian_filter(bt, sps, n_taps=None):
    """Gaussian pulse shaping filter for GMSK/GFSK.

    Args:
        bt: Bandwidth-time product.
        sps: Samples per symbol.
        n_taps: Filter length (default: 4*sps).
    """
    if n_taps is None:
        n_taps = 4 * sps
    t = (np.arange(n_taps) - n_taps // 2) / sps
    alpha = np.pi * bt / np.sqrt(np.log(2))
    h = alpha * np.exp(-(alpha * t) ** 2)
    h /= h.sum()
    return h


def gaussian_filter_sigma(sigma_samples, n_taps):
    """Gaussian filter parameterized by sigma in samples.

    Args:
        sigma_samples: Standard deviation in samples.
        n_taps: Filter length.

    Returns:
        Normalized Gaussian kernel (sums to 1).
    """
    t = np.arange(n_taps) - n_taps // 2
    h = np.exp(-t ** 2 / (2 * sigma_samples ** 2))
    h /= h.sum()
    return h
