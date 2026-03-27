"""Unit tests for protocol modules (coding, framers, encoders)."""

import numpy as np
import pytest

from rf_datagen.protocols.coding import convolutional_encode, interleave_block
from rf_datagen.protocols.dmr import frame_dmr, _golay_20_8_encode, _trellis_34_encode
from rf_datagen.protocols.dstar import frame_dstar
from rf_datagen.protocols.ysf import frame_ysf
from rf_datagen.protocols.p25 import frame_p25, _p25_golay_24_12_encode
from rf_datagen.protocols.nxdn import frame_nxdn
from rf_datagen.protocols.sync_words import (
    DMR_BS_VOICE_SYNC,
    DMR_BS_DATA_SYNC,
    DSTAR_FRAME_SYNC,
    YSF_SYNC,
    P25_FRAME_SYNC,
    NXDN_FRAME_SYNC_RDCH,
    NXDN_FRAME_SYNC_TDCH,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _find_sync(stream, sync_pattern):
    """Sliding window search for sync pattern in stream."""
    n = len(sync_pattern)
    for i in range(len(stream) - n + 1):
        if np.array_equal(stream[i : i + n], sync_pattern):
            return True
    return False


def _bits_to_dibits(bits):
    """Convert bit array to dibit array."""
    assert len(bits) % 2 == 0
    return np.array(
        [bits[i] * 2 + bits[i + 1] for i in range(0, len(bits), 2)],
        dtype=np.uint8,
    )


# ===================================================================
# coding.py — convolutional_encode
# ===================================================================

class TestConvolutionalEncode:
    def test_conv_rate_half(self):
        inp = np.array([1, 0, 1, 1], dtype=np.uint8)
        out = convolutional_encode(inp)
        assert len(out) == 2 * len(inp)

    def test_conv_output_binary(self):
        inp = np.array([1, 0, 1, 1], dtype=np.uint8)
        out = convolutional_encode(inp)
        assert set(out.tolist()).issubset({0, 1})

    def test_conv_deterministic(self):
        inp = np.array([1, 0, 1, 1], dtype=np.uint8)
        out1 = convolutional_encode(inp)
        out2 = convolutional_encode(inp)
        np.testing.assert_array_equal(out1, out2)

    def test_conv_known_vector(self):
        inp = np.array([1, 0, 1, 1], dtype=np.uint8)
        out = convolutional_encode(inp, g1=0x19, g2=0x17)
        expected = np.array([1, 1, 0, 1, 1, 0, 0, 0], dtype=np.uint8)
        np.testing.assert_array_equal(out, expected)


# ===================================================================
# coding.py — interleave_block
# ===================================================================

class TestInterleaveBlock:
    def test_interleave_output_length(self):
        inp = np.arange(9, dtype=np.uint8)
        out = interleave_block(inp, rows=3, cols=3)
        assert len(out) == 9

    def test_interleave_preserves_bits(self):
        inp = np.arange(9, dtype=np.uint8)
        out = interleave_block(inp, rows=3, cols=3)
        assert sorted(out.tolist()) == sorted(inp.tolist())

    def test_interleave_known_3x3(self):
        inp = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9], dtype=np.uint8)
        out = interleave_block(inp, rows=3, cols=3)
        expected = np.array([1, 4, 7, 2, 5, 8, 3, 6, 9], dtype=np.uint8)
        np.testing.assert_array_equal(out, expected)


# ===================================================================
# Protocol framers — DMR
# ===================================================================

class TestFrameDMR:
    @pytest.fixture()
    def dmr_output(self):
        codec_bits = np.random.randint(0, 2, 5000).astype(np.uint8)
        return frame_dmr(codec_bits)

    def test_dmr_output_value_range(self, dmr_output):
        assert set(dmr_output.tolist()).issubset({0, 1, 2, 3})

    def test_dmr_output_nonempty(self, dmr_output):
        assert len(dmr_output) > 0

    def test_dmr_contains_sync(self, dmr_output):
        sync_dibits = _bits_to_dibits(DMR_BS_VOICE_SYNC)
        assert _find_sync(dmr_output, sync_dibits)


# ===================================================================
# Protocol framers — D-STAR
# ===================================================================

class TestFrameDSTAR:
    @pytest.fixture()
    def dstar_output(self):
        codec_bits = np.random.randint(0, 2, 5000).astype(np.uint8)
        return frame_dstar(codec_bits)

    def test_dstar_output_value_range(self, dstar_output):
        assert set(dstar_output.tolist()).issubset({0, 1})

    def test_dstar_output_nonempty(self, dstar_output):
        assert len(dstar_output) > 0

    def test_dstar_contains_sync(self, dstar_output):
        assert _find_sync(dstar_output, DSTAR_FRAME_SYNC)


# ===================================================================
# Protocol framers — YSF
# ===================================================================

class TestFrameYSF:
    @pytest.fixture()
    def ysf_output(self):
        codec_bits = np.random.randint(0, 2, 5000).astype(np.uint8)
        return frame_ysf(codec_bits)

    def test_ysf_output_value_range(self, ysf_output):
        assert set(ysf_output.tolist()).issubset({0, 1, 2, 3})

    def test_ysf_output_nonempty(self, ysf_output):
        assert len(ysf_output) > 0

    def test_ysf_contains_sync(self, ysf_output):
        assert _find_sync(ysf_output, YSF_SYNC)


# ===================================================================
# Protocol framers — P25
# ===================================================================

class TestFrameP25:
    @pytest.fixture()
    def p25_output(self):
        codec_bits = np.random.randint(0, 2, 5000).astype(np.uint8)
        return frame_p25(codec_bits)

    def test_p25_output_value_range(self, p25_output):
        assert set(p25_output.tolist()).issubset({0, 1, 2, 3})

    def test_p25_output_nonempty(self, p25_output):
        assert len(p25_output) > 0

    def test_p25_contains_sync(self, p25_output):
        assert _find_sync(p25_output, P25_FRAME_SYNC)


# ===================================================================
# Protocol framers — NXDN
# ===================================================================

class TestFrameNXDN:
    @pytest.fixture()
    def nxdn_output(self):
        codec_bits = np.random.randint(0, 2, 5000).astype(np.uint8)
        return frame_nxdn(codec_bits)

    def test_nxdn_output_value_range(self, nxdn_output):
        assert set(nxdn_output.tolist()).issubset({0, 1, 2, 3})

    def test_nxdn_output_nonempty(self, nxdn_output):
        assert len(nxdn_output) > 0

    def test_nxdn_contains_sync(self, nxdn_output):
        found = _find_sync(nxdn_output, NXDN_FRAME_SYNC_RDCH) or _find_sync(
            nxdn_output, NXDN_FRAME_SYNC_TDCH
        )
        assert found


# ===================================================================
# Protocol-specific encoders
# ===================================================================

class TestProtocolEncoders:
    def test_golay_20_output_length(self):
        out = _golay_20_8_encode(42)
        assert len(out) == 20

    def test_golay_20_output_binary(self):
        out = _golay_20_8_encode(42)
        assert set(out.tolist()).issubset({0, 1})

    def test_trellis_dibit_valued(self):
        inp = np.array([0, 1, 2, 3] * 5, dtype=np.uint8)
        out = _trellis_34_encode(inp)
        assert set(out.tolist()).issubset({0, 1, 2, 3})

    def test_p25_golay_24_length(self):
        inp = np.array([1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0], dtype=np.uint8)
        out = _p25_golay_24_12_encode(inp)
        assert len(out) == 24

    def test_p25_golay_systematic(self):
        inp = np.array([1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0], dtype=np.uint8)
        out = _p25_golay_24_12_encode(inp)
        np.testing.assert_array_equal(out[:12], inp)
