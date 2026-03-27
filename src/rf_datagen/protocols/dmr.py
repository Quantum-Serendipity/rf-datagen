"""DMR burst/superframe framing per ETSI TS 102 361-1."""

import numpy as np

from .sync_words import DMR_BS_VOICE_SYNC, DMR_BS_DATA_SYNC


def _golay_20_8_encode(byte_val):
    """Golay(20,8) encoder for DMR CACH/embedded signaling."""
    g = np.array([
        [1,0,0,1,1,1,0,0,1,1,0,1],
        [0,1,0,0,1,1,1,0,0,1,1,1],
        [1,0,1,1,1,0,1,1,1,1,0,0],
        [0,1,0,1,1,1,0,1,1,1,1,0],
        [1,0,1,1,0,0,1,0,0,0,0,1],
        [1,1,0,0,0,1,0,1,1,1,1,1],
        [1,1,1,1,1,1,1,0,0,0,1,0],
        [1,1,1,0,0,0,1,1,1,0,1,1],
    ], dtype=np.uint8)
    data_bits = np.array([(byte_val >> (7 - i)) & 1 for i in range(8)], dtype=np.uint8)
    parity = np.mod(data_bits @ g, 2).astype(np.uint8)
    return np.concatenate([parity, data_bits])


def _trellis_34_encode(dibits):
    """Rate 3/4 trellis encoder for DMR voice."""
    coded = []
    state = 0
    for d in dibits:
        out1 = (d ^ (state & 0x03)) & 0x03
        out2 = ((d + state) ^ ((state >> 1) & 0x01)) & 0x03
        coded.append(out1)
        coded.append(out2)
        state = ((state << 2) | d) & 0x0F
    result = []
    for i, c in enumerate(coded):
        if i % 4 != 3:
            result.append(c)
    return np.array(result, dtype=np.uint8) if result else np.array(dibits, dtype=np.uint8)


def frame_dmr(codec_bits):
    """DMR framing per ETSI TS 102 361-1.

    Returns dibit stream for two TDMA slots.
    """
    dibits = []
    pos = 0
    burst_count = 0

    while pos < len(codec_bits) - 1:
        for slot in range(2):
            # CACH: 12 dibits
            cach_data = np.random.randint(0, 256) & 0xFF
            cach_coded = _golay_20_8_encode(cach_data)
            cach_dibits = []
            for j in range(0, len(cach_coded) - 1, 2):
                cach_dibits.append((cach_coded[j] << 1) | cach_coded[j + 1])
            while len(cach_dibits) < 12:
                cach_dibits.append(0)
            dibits.extend(cach_dibits[:12])

            # Info1: 49 dibits of voice data (trellis-coded)
            info1_raw = []
            for _ in range(49):
                if pos + 1 < len(codec_bits):
                    info1_raw.append((codec_bits[pos] << 1) | codec_bits[pos + 1])
                    pos += 2
                else:
                    info1_raw.append(np.random.randint(0, 4))
            info1_coded = _trellis_34_encode(np.array(info1_raw, dtype=np.uint8))
            dibits.extend(info1_coded[:49].tolist())

            # Sync pattern: 24 dibits
            if burst_count % 6 == 0:
                sync_bits = DMR_BS_DATA_SYNC
            else:
                sync_bits = DMR_BS_VOICE_SYNC
            sync_dibits = []
            for j in range(0, len(sync_bits) - 1, 2):
                sync_dibits.append((sync_bits[j] << 1) | sync_bits[j + 1])
            dibits.extend(sync_dibits[:24])

            # Info2: 49 dibits of voice data (trellis-coded)
            info2_raw = []
            for _ in range(49):
                if pos + 1 < len(codec_bits):
                    info2_raw.append((codec_bits[pos] << 1) | codec_bits[pos + 1])
                    pos += 2
                else:
                    info2_raw.append(np.random.randint(0, 4))
            info2_coded = _trellis_34_encode(np.array(info2_raw, dtype=np.uint8))
            dibits.extend(info2_coded[:49].tolist())

            burst_count += 1

    return np.array(dibits, dtype=np.uint8)
