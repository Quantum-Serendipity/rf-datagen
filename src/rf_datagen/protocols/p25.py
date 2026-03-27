"""P25 Phase 1 LDU framing per TIA-102.BAAA."""

import numpy as np

from .coding import convolutional_encode
from .sync_words import P25_FRAME_SYNC


def _p25_golay_24_12_encode(data_12):
    """Golay(24,12) encoder for P25 NID."""
    g_poly = 0xC75
    val = 0
    for b in data_12[:12]:
        val = (val << 1) | int(b)
    remainder = val << 12
    for i in range(11, -1, -1):
        if remainder & (1 << (i + 12)):
            remainder ^= g_poly << i
    parity = remainder & 0xFFF
    coded = np.zeros(24, dtype=np.uint8)
    for i in range(12):
        coded[i] = (val >> (11 - i)) & 1
        coded[12 + i] = (parity >> (11 - i)) & 1
    return coded


def frame_p25(codec_bits):
    """P25 Phase 1 framing. Returns dibit stream."""
    dibits = []
    pos = 0
    ldu_count = 0

    while pos < len(codec_bits) - 1:
        dibits.extend(P25_FRAME_SYNC.tolist())

        # NID: 32 dibits
        nac_bits = np.random.randint(0, 2, 12).astype(np.uint8)
        nid_coded = _p25_golay_24_12_encode(nac_bits)
        duid = np.array([0,1,0,1], dtype=np.uint8) if ldu_count % 2 == 0 \
            else np.array([1,0,1,0], dtype=np.uint8)
        nid_bits = np.concatenate([nid_coded, duid, np.zeros(36, dtype=np.uint8)])
        nid_dibits = []
        for j in range(0, min(len(nid_bits) - 1, 63), 2):
            nid_dibits.append((nid_bits[j] << 1) | nid_bits[j + 1])
        while len(nid_dibits) < 32:
            nid_dibits.append(np.random.randint(0, 4))
        dibits.extend(nid_dibits[:32])

        # 9 IMBE voice code word blocks
        dibit_count_in_ldu = 0
        for _imbe in range(9):
            voice_bits = []
            for _ in range(88):
                if pos < len(codec_bits):
                    voice_bits.append(codec_bits[pos])
                    pos += 1
                else:
                    voice_bits.append(np.random.randint(0, 2))
            voice_arr = np.array(voice_bits, dtype=np.uint8)
            golay_part = _p25_golay_24_12_encode(voice_arr[:12])
            remaining = voice_arr[12:]
            coded_remaining = convolutional_encode(remaining)[:120]
            coded_voice = np.concatenate([golay_part, coded_remaining])
            for j in range(0, len(coded_voice) - 1, 2):
                dibits.append((coded_voice[j] << 1) | coded_voice[j + 1])
                dibit_count_in_ldu += 1
                if dibit_count_in_ldu % 35 == 0:
                    dibits.append(0b01)

        ldu_count += 1

    return np.array(dibits, dtype=np.uint8)
