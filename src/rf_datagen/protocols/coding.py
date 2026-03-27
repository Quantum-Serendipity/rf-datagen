"""Shared FEC primitives for digital voice protocol framing."""

import numpy as np


def convolutional_encode(bits, g1=0x19, g2=0x17):
    """Rate-1/2 convolutional encoder (K=5)."""
    state = 0
    output = []
    for b in bits:
        state = ((state << 1) | int(b)) & 0x1F
        p1 = bin(state & g1).count('1') % 2
        p2 = bin(state & g2).count('1') % 2
        output.append(p1)
        output.append(p2)
    return np.array(output, dtype=np.uint8)


def interleave_block(bits, rows, cols):
    """Block interleaver: write by rows, read by columns."""
    n = rows * cols
    padded = np.zeros(n, dtype=np.uint8)
    padded[:min(len(bits), n)] = bits[:n]
    matrix = padded.reshape(rows, cols)
    return matrix.T.flatten()
