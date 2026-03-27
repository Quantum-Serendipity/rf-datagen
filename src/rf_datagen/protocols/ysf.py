"""YSF (Yaesu System Fusion) framing."""

import numpy as np

from .coding import convolutional_encode, interleave_block
from .sync_words import YSF_SYNC


def frame_ysf(codec_bits):
    """YSF framing. Returns dibit stream."""
    dibits = []
    pos = 0

    while pos < len(codec_bits) - 1:
        dibits.extend(YSF_SYNC.tolist())

        # FICH: 100 dibits
        fich_data = np.random.randint(0, 2, 48).astype(np.uint8)
        fich_coded = convolutional_encode(fich_data)
        fich_interleaved = interleave_block(fich_coded, 10, 10)
        fich_dibits = []
        for j in range(0, len(fich_interleaved) - 1, 2):
            fich_dibits.append((fich_interleaved[j] << 1) | fich_interleaved[j + 1])
        while len(fich_dibits) < 100:
            fich_dibits.append(np.random.randint(0, 4))
        dibits.extend(fich_dibits[:100])

        # 5 voice channel blocks, each 72 dibits
        for _vch in range(5):
            vch_bits = []
            for _ in range(72):
                if pos < len(codec_bits):
                    vch_bits.append(codec_bits[pos])
                    pos += 1
                else:
                    vch_bits.append(np.random.randint(0, 2))
            vch_coded = convolutional_encode(np.array(vch_bits, dtype=np.uint8))
            vch_interleaved = interleave_block(vch_coded, 12, 12)
            for j in range(0, min(len(vch_interleaved) - 1, 143), 2):
                dibits.append((vch_interleaved[j] << 1) | vch_interleaved[j + 1])
            while len(dibits) % 72 != 0:
                dibits.append(np.random.randint(0, 4))

    return np.array(dibits, dtype=np.uint8)
