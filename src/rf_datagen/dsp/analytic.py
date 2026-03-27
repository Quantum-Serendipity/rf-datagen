"""Analytic signal conversion via Hilbert transform."""

import numpy as np
from scipy.signal import resample

from ..constants import FS


def hilbert_analytic(x):
    """Convert real-valued signal to analytic (complex IQ) via Hilbert transform."""
    N = len(x)
    X = np.fft.fft(x)
    h = np.zeros(N)
    h[0] = 1
    h[1:(N + 1) // 2] = 2
    if N % 2 == 0:
        h[N // 2] = 1
    return np.fft.ifft(X * h)


def audio_to_iq(audio, source_fs, target_fs=FS):
    """Real audio -> analytic signal -> resample to target_fs -> complex IQ.

    Accepts either raw int16 PCM or float64 audio. Int16 is normalized to
    [-1, 1] automatically.
    """
    if len(audio) == 0:
        return np.array([], dtype=np.complex128)

    # Normalize int16 to float64 if needed
    if audio.dtype == np.int16:
        audio = audio.astype(np.float64) / 32768.0

    # Resample to target sample rate
    target_len = int(len(audio) * target_fs / source_fs)
    if target_len < 1:
        return np.array([], dtype=np.complex128)
    audio = resample(audio, target_len)

    # Hilbert transform -> complex analytic signal
    return hilbert_analytic(audio)
