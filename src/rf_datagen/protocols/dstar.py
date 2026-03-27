"""D-STAR frame structure per JARL specification."""

import numpy as np

from .sync_words import DSTAR_FRAME_SYNC


def frame_dstar(codec_bits):
    """D-STAR framing per JARL specification.

    Voice superframe: 21 frames, each 96 bits (72 voice + 24 slow data).
    Returns bit stream.
    """
    bits = []
    frame_count = 0
    pos = 0

    lfsr = 0x1FF

    while pos < len(codec_bits):
        if frame_count % 21 == 0:
            bits.extend(DSTAR_FRAME_SYNC.tolist())
            lfsr = 0x1FF

        # 72 voice bits
        for _ in range(72):
            if pos < len(codec_bits):
                bits.append(codec_bits[pos])
                pos += 1
            else:
                bits.append(np.random.randint(0, 2))

        # 24 slow data bits (scrambled with LFSR)
        for _ in range(24):
            data_bit = np.random.randint(0, 2)
            fb = ((lfsr >> 8) ^ (lfsr >> 4)) & 1
            scrambled = data_bit ^ fb
            bits.append(scrambled)
            lfsr = ((lfsr << 1) | fb) & 0x1FF

        frame_count += 1

    return np.array(bits, dtype=np.uint8)
