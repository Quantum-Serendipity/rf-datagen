"""IQ format adapters — reshape/resample IQ data for each model family."""

import numpy as np
from scipy.signal import resample


def adapt_torchsig(iq, target_len=4096):
    """Adapt IQ for TorchSig XCiT — (2, target_len) float32.

    Input: complex128/64 array of any length.
    Output: (2, target_len) float32 with I/Q as separate channels, normalized.
    """
    if len(iq) == 0:
        return np.zeros((2, target_len), dtype=np.float32)

    # Resample to target length
    if len(iq) != target_len:
        iq = resample(iq, target_len)

    i_data = np.real(iq).astype(np.float32)
    q_data = np.imag(iq).astype(np.float32)

    # Normalize to unit energy
    energy = np.sqrt(np.mean(i_data**2 + q_data**2))
    if energy > 1e-10:
        i_data /= energy
        q_data /= energy

    return np.stack([i_data, q_data], axis=0)


def adapt_radioml_2016(iq, target_len=128):
    """Adapt IQ for RadioML 2016.10a models — (2, 128) float32.

    Center-extracts or resamples to 128 samples.
    """
    if len(iq) == 0:
        return np.zeros((2, target_len), dtype=np.float32)

    if len(iq) > target_len:
        # Center extract
        start = (len(iq) - target_len) // 2
        iq = iq[start:start + target_len]
    elif len(iq) < target_len:
        iq = resample(iq, target_len)

    i_data = np.real(iq).astype(np.float32)
    q_data = np.imag(iq).astype(np.float32)

    energy = np.sqrt(np.mean(i_data**2 + q_data**2))
    if energy > 1e-10:
        i_data /= energy
        q_data /= energy

    return np.stack([i_data, q_data], axis=0)


def adapt_radioml_2018(iq, target_len=1024):
    """Adapt IQ for RadioML 2018 models — (2, 1024) float32."""
    if len(iq) == 0:
        return np.zeros((2, target_len), dtype=np.float32)

    if len(iq) > target_len:
        start = (len(iq) - target_len) // 2
        iq = iq[start:start + target_len]
    elif len(iq) < target_len:
        iq = resample(iq, target_len)

    i_data = np.real(iq).astype(np.float32)
    q_data = np.imag(iq).astype(np.float32)

    energy = np.sqrt(np.mean(i_data**2 + q_data**2))
    if energy > 1e-10:
        i_data /= energy
        q_data /= energy

    return np.stack([i_data, q_data], axis=0)


def adapt_torchsig_wideband(iq, nfft=1024, hop=256):
    """Adapt IQ for TorchSig DETR wideband — STFT spectrogram tensor.

    Returns (1, freq_bins, time_bins) float32 power spectrogram.
    """
    if len(iq) < nfft:
        padded = np.zeros(nfft, dtype=iq.dtype)
        padded[:len(iq)] = iq
        iq = padded

    # Compute STFT
    n_frames = 1 + (len(iq) - nfft) // hop
    frames = np.zeros((nfft, n_frames), dtype=np.complex128)
    window = np.hanning(nfft)
    for i in range(n_frames):
        start = i * hop
        frames[:, i] = np.fft.fft(iq[start:start + nfft] * window)

    # Power spectrogram in dB
    power = np.abs(frames) ** 2
    power_db = 10 * np.log10(power + 1e-12)

    # Normalize to [0, 1]
    pmin, pmax = power_db.min(), power_db.max()
    if pmax - pmin > 1e-6:
        power_db = (power_db - pmin) / (pmax - pmin)
    else:
        power_db = np.zeros_like(power_db)

    return power_db.astype(np.float32)[np.newaxis, :, :]


def adapt_batch(iq_batch, adapter_fn, **kwargs):
    """Apply an adapter to a batch of IQ samples.

    iq_batch: list or array of complex IQ arrays
    Returns: numpy array of adapted samples
    """
    adapted = [adapter_fn(iq, **kwargs) for iq in iq_batch]
    return np.array(adapted)
