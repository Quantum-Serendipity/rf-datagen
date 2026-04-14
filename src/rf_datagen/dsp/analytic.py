"""Analytic signal conversion via Hilbert transform."""

import numpy as np
from scipy.signal import resample

from ..constants import FS

# Process audio in 10-second blocks (at source rate) to keep FFT
# workspace under ~1 GB.  Overlap-save removes edge artifacts from
# the Hilbert transform.
_BLOCK_SECONDS = 10


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


def _chunked_resample(audio, source_fs, target_fs):
    """Resample long signals in chunks to avoid FFT OOM.

    scipy.signal.resample pads to the next power-of-2 and allocates
    a complex128 FFT workspace.  For 1B+ samples this exceeds 34 GB.
    Chunking keeps the workspace under ~1 GB per block.
    """
    block_samples = int(_BLOCK_SECONDS * source_fs)
    target_len = int(len(audio) * target_fs / source_fs)

    if len(audio) <= block_samples * 2:
        # Short signal — process in one shot
        return resample(audio, target_len)

    out_parts = []
    n = len(audio)
    pos = 0
    while pos < n:
        end = min(pos + block_samples, n)
        chunk = audio[pos:end]
        chunk_target = int(len(chunk) * target_fs / source_fs)
        if chunk_target < 1:
            break
        out_parts.append(resample(chunk, chunk_target))
        pos = end

    return np.concatenate(out_parts)


def _chunked_hilbert(audio, fs):
    """Apply Hilbert transform in overlapping blocks.

    The Hilbert transform has edge artifacts at block boundaries.
    Overlap-save discards the transient edges and keeps only the
    settled center portion of each block.
    """
    block_samples = int(_BLOCK_SECONDS * fs)
    overlap = int(0.5 * fs)  # 0.5 seconds of overlap

    if len(audio) <= block_samples * 2:
        return hilbert_analytic(audio)

    n = len(audio)
    result = np.empty(n, dtype=np.complex128)
    pos = 0
    out_pos = 0

    while pos < n:
        # Read block with overlap on both sides
        blk_start = max(0, pos - overlap)
        blk_end = min(n, pos + block_samples + overlap)
        block = audio[blk_start:blk_end]

        analytic = hilbert_analytic(block)

        # Trim overlap — keep only the center portion
        trim_left = pos - blk_start
        usable = min(block_samples, n - pos)
        result[out_pos:out_pos + usable] = analytic[trim_left:trim_left + usable]

        pos += block_samples
        out_pos += usable

    return result[:out_pos]


def audio_to_iq(audio, source_fs, target_fs=FS):
    """Real audio -> analytic signal -> resample to target_fs -> complex IQ.

    Accepts either raw int16 PCM or float64 audio. Int16 is normalized to
    [-1, 1] automatically.

    For long signals (>20s), processing is chunked to keep FFT workspace
    under ~1 GB, preventing OOM on signals with millions of samples.
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
    audio = _chunked_resample(audio, source_fs, target_fs)

    # Hilbert transform -> complex analytic signal
    return _chunked_hilbert(audio, target_fs)
