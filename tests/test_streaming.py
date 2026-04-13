"""Tests for streaming pipeline: extract_windows two-pass and streaming impairments."""

import os

import numpy as np
import pytest

from rf_datagen.impairments.effects import extract_windows, normalize_power


FS = 12_000


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def long_iq_signal():
    """16384-sample complex signal with mixed loud and quiet regions."""
    n = 16384
    t = np.arange(n) / FS
    loud = np.exp(2j * np.pi * 1000 * t[:8192])
    quiet = np.zeros(4096, dtype=complex)
    loud2 = np.exp(2j * np.pi * 500 * t[:4096]) * 0.8
    return np.concatenate([loud, quiet, loud2])


@pytest.fixture
def small_raw_windows():
    """50 pre-extracted windows at window_len=512."""
    windows = np.zeros((50, 512), dtype=np.complex64)
    for i in range(50):
        t = np.arange(512) / FS
        freq = 200 + i * 40
        w = np.exp(2j * np.pi * freq * t)
        w /= np.sqrt(np.mean(np.abs(w) ** 2))
        windows[i] = w.astype(np.complex64)
    return windows


@pytest.fixture
def memmap_raw_windows(tmp_path, small_raw_windows):
    """Write small_raw_windows to disk and return path."""
    path = str(tmp_path / "raw_windows.npy")
    np.save(path, small_raw_windows)
    return path


# ---------------------------------------------------------------------------
# extract_windows two-pass tests
# ---------------------------------------------------------------------------

class TestExtractWindowsTwoPass:
    """Verify the two-pass pre-allocate version produces correct output."""

    def test_shape_basic(self, long_iq_signal):
        """Output shape is (N, window_len) with N > 0 for a loud signal."""
        result = extract_windows(long_iq_signal, window_len=2048)
        assert result.ndim == 2
        assert result.shape[1] == 2048
        assert result.shape[0] > 0

    def test_empty_signal(self):
        """All-zeros signal returns empty array."""
        sig = np.zeros(4096, dtype=complex)
        result = extract_windows(sig, window_len=512)
        assert result.shape == (0, 512)

    def test_short_signal(self):
        """Signal shorter than window_len returns empty."""
        sig = np.exp(2j * np.pi * 1000 * np.arange(100) / FS)
        result = extract_windows(sig, window_len=512)
        assert result.shape == (0, 512)

    def test_max_windows_exact(self, long_iq_signal):
        """max_windows limits output to exactly that many rows."""
        result = extract_windows(long_iq_signal, window_len=512,
                                 max_windows=5)
        assert result.shape[0] == 5

    def test_dtype_complex64(self, long_iq_signal):
        """Output dtype matches requested dtype."""
        result = extract_windows(long_iq_signal, window_len=512,
                                 dtype=np.complex64)
        assert result.dtype == np.complex64

    def test_dtype_complex128(self, long_iq_signal):
        """Default dtype is complex128."""
        result = extract_windows(long_iq_signal, window_len=512)
        assert result.dtype == np.complex128

    def test_windows_are_unit_power(self, long_iq_signal):
        """Each returned window should be normalized to unit power."""
        result = extract_windows(long_iq_signal, window_len=512,
                                 max_windows=10)
        for i in range(len(result)):
            power = np.mean(np.abs(result[i]) ** 2)
            np.testing.assert_allclose(power, 1.0, atol=0.01,
                                       err_msg=f"Window {i} not unit power")

    def test_deterministic(self, long_iq_signal):
        """Same input produces identical output."""
        a = extract_windows(long_iq_signal, window_len=512, max_windows=10)
        b = extract_windows(long_iq_signal, window_len=512, max_windows=10)
        np.testing.assert_array_equal(a, b)

    def test_power_threshold_filters_silence(self):
        """Windows below power threshold are excluded."""
        # Signal with loud start, silent end
        loud = np.exp(2j * np.pi * 1000 * np.arange(2048) / FS)
        silent = np.zeros(2048, dtype=complex)
        sig = np.concatenate([loud, silent])
        result = extract_windows(sig, window_len=1024,
                                 stride=1024, power_threshold=0.001)
        # Should only get windows from the loud region
        assert result.shape[0] <= 2  # at most 2 from the loud 2048 samples


# ---------------------------------------------------------------------------
# apply_impairments_streaming tests
# ---------------------------------------------------------------------------

class TestApplyImpairmentStreaming:
    """Test the memmap-based streaming impairment function."""

    def test_output_shape_and_dtype(self, tmp_path, memmap_raw_windows):
        """Output memmap has correct shape and dtype."""
        from rf_datagen.impairments.scenarios import (
            apply_impairments_streaming, configure)
        from rf_datagen.config import ImpairmentConfig
        configure(ImpairmentConfig(snr_levels=[20, 10]))

        out_path = str(tmp_path / "output.npy")
        scenarios, snrs = apply_impairments_streaming(
            memmap_raw_windows, out_path, target_count=20,
            window_len=512, dtype=np.complex64)

        result = np.load(out_path, mmap_mode='r')
        assert result.shape == (20, 512)
        assert result.dtype == np.complex64

    def test_metadata_length(self, tmp_path, memmap_raw_windows):
        """Returned scenarios and snrs lists match target_count."""
        from rf_datagen.impairments.scenarios import (
            apply_impairments_streaming, configure)
        from rf_datagen.config import ImpairmentConfig
        configure(ImpairmentConfig(snr_levels=[20, 10]))

        out_path = str(tmp_path / "output.npy")
        scenarios, snrs = apply_impairments_streaming(
            memmap_raw_windows, out_path, target_count=30,
            window_len=512)

        assert len(scenarios) == 30
        assert len(snrs) == 30

    def test_snr_distribution(self, tmp_path, memmap_raw_windows):
        """SNR values are evenly distributed across configured levels."""
        from collections import Counter
        from rf_datagen.impairments.scenarios import (
            apply_impairments_streaming, configure)
        from rf_datagen.config import ImpairmentConfig
        configure(ImpairmentConfig(snr_levels=[20, 10]))

        out_path = str(tmp_path / "output.npy")
        _, snrs = apply_impairments_streaming(
            memmap_raw_windows, out_path, target_count=20,
            window_len=512)

        counts = Counter(snrs)
        assert counts[20] == 10
        assert counts[10] == 10

    def test_output_has_no_nan(self, tmp_path, memmap_raw_windows):
        """Output should not contain NaN or Inf values."""
        from rf_datagen.impairments.scenarios import (
            apply_impairments_streaming, configure)
        from rf_datagen.config import ImpairmentConfig
        configure(ImpairmentConfig(snr_levels=[20]))

        out_path = str(tmp_path / "output.npy")
        apply_impairments_streaming(
            memmap_raw_windows, out_path, target_count=10,
            window_len=512)

        result = np.load(out_path)
        assert not np.any(np.isnan(result))
        assert not np.any(np.isinf(result))

    def test_empty_raw_windows(self, tmp_path):
        """Empty raw windows returns empty lists."""
        from rf_datagen.impairments.scenarios import (
            apply_impairments_streaming, configure)
        from rf_datagen.config import ImpairmentConfig
        configure(ImpairmentConfig(snr_levels=[20]))

        empty_path = str(tmp_path / "empty.npy")
        np.save(empty_path, np.zeros((0, 512), dtype=np.complex64))

        out_path = str(tmp_path / "output.npy")
        scenarios, snrs = apply_impairments_streaming(
            empty_path, out_path, target_count=10,
            window_len=512)
        assert scenarios == []
        assert snrs == []


# ---------------------------------------------------------------------------
# Chunk PCM roundtrip test
# ---------------------------------------------------------------------------

def test_chunk_pcm_save_load_roundtrip(tmp_path):
    """int16 PCM data survives save/load via tofile/fromfile."""
    n = 24000  # 0.5s at 48kHz
    t = np.arange(n) / 48000
    audio = (16000 * np.sin(2 * np.pi * 1000 * t)).astype(np.int16)

    path = str(tmp_path / "chunk.pcm")
    audio.tofile(path)
    loaded = np.fromfile(path, dtype=np.int16)
    np.testing.assert_array_equal(audio, loaded)


# ---------------------------------------------------------------------------
# clear_interferer_pool test
# ---------------------------------------------------------------------------

def test_clear_interferer_pool():
    """clear_interferer_pool sets pool to None."""
    from rf_datagen.impairments.scenarios import (
        clear_interferer_pool, _INTERFERER_POOL)
    import rf_datagen.impairments.scenarios as sc
    sc._INTERFERER_POOL = {"fake": "data"}
    clear_interferer_pool()
    assert sc._INTERFERER_POOL is None


# ---------------------------------------------------------------------------
# STREAMING_THRESHOLD constant test
# ---------------------------------------------------------------------------

def test_streaming_threshold_defined():
    """STREAMING_THRESHOLD is defined and is 500 MB."""
    from rf_datagen.constants import STREAMING_THRESHOLD
    assert STREAMING_THRESHOLD == 500 * 1024 * 1024
