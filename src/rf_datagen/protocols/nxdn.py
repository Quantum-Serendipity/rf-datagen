"""NXDN framing per NXDN-TS 1-A."""

import numpy as np

from .coding import convolutional_encode, interleave_block
from .sync_words import NXDN_FRAME_SYNC_RDCH, NXDN_FRAME_SYNC_TDCH


def frame_nxdn(codec_bits):
    """NXDN framing. Returns dibit stream."""
    dibits = []
    pos = 0
    frame_count = 0

    while pos < len(codec_bits) - 1:
        if frame_count % 2 == 0:
            dibits.extend(NXDN_FRAME_SYNC_RDCH.tolist())
        else:
            dibits.extend(NXDN_FRAME_SYNC_TDCH.tolist())

        # LICH: 8 dibits
        lich_data = np.random.randint(0, 2, 8).astype(np.uint8)
        lich_coded = convolutional_encode(lich_data)
        lich_dibits = []
        for j in range(0, len(lich_coded) - 1, 2):
            lich_dibits.append((lich_coded[j] << 1) | lich_coded[j + 1])
        while len(lich_dibits) < 8:
            lich_dibits.append(0)
        dibits.extend(lich_dibits[:8])

        # SACCH: 30 dibits
        sacch_data = np.random.randint(0, 2, 26).astype(np.uint8)
        sacch_coded = convolutional_encode(sacch_data)
        sacch_interleaved = interleave_block(sacch_coded, 6, 10)
        sacch_dibits = []
        for j in range(0, len(sacch_interleaved) - 1, 2):
            sacch_dibits.append((sacch_interleaved[j] << 1) | sacch_interleaved[j + 1])
        while len(sacch_dibits) < 30:
            sacch_dibits.append(np.random.randint(0, 4))
        dibits.extend(sacch_dibits[:30])

        # 2 voice blocks, each 72 dibits
        for _vb in range(2):
            vb_bits = []
            for _ in range(72):
                if pos < len(codec_bits):
                    vb_bits.append(codec_bits[pos])
                    pos += 1
                else:
                    vb_bits.append(np.random.randint(0, 2))
            vb_coded = convolutional_encode(np.array(vb_bits, dtype=np.uint8))
            vb_interleaved = interleave_block(vb_coded, 12, 12)
            vb_dibits = []
            for j in range(0, len(vb_interleaved) - 1, 2):
                vb_dibits.append((vb_interleaved[j] << 1) | vb_interleaved[j + 1])
            while len(vb_dibits) < 72:
                vb_dibits.append(np.random.randint(0, 4))
            dibits.extend(vb_dibits[:72])

        frame_count += 1

    return np.array(dibits, dtype=np.uint8)
