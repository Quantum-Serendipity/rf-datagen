"""Modulation primitives — GFSK, FSK, OOK, PSK, 4FSK, GMSK, OFDM."""

import numpy as np
from scipy.signal import fftconvolve

from ..constants import FS


def gfsk_mod(symbols, num_tones, tone_spacing, symbol_dur, fs=FS, bt=2.0):
    """Gaussian Frequency Shift Keying (FT8, FT4, JS8).

    Tones are centered around DC (baseband).
    """
    sps = max(1, int(symbol_dur * fs))
    n = len(symbols) * sps

    freq = np.zeros(n)
    for i, s in enumerate(symbols):
        freq[i * sps : (i + 1) * sps] = (s - (num_tones - 1) / 2.0) * tone_spacing

    # Gaussian smoothing
    filt_len = min(3 * sps, n)
    if filt_len > 1:
        t_f = (np.arange(filt_len) - filt_len // 2) / fs
        sigma = symbol_dur / (2 * np.pi * bt)
        gauss = np.exp(-t_f ** 2 / (2 * sigma ** 2))
        gauss /= gauss.sum()
        freq = fftconvolve(freq, gauss, mode="same")

    phase = 2 * np.pi * np.cumsum(freq) / fs
    phase += np.random.uniform(0, 2 * np.pi)
    return np.exp(1j * phase)


def fsk_mod(symbols, num_tones, tone_spacing, symbol_dur, fs=FS):
    """Continuous-phase FSK (WSPR, JT65, JT9, MFSK, Contestia, Packet)."""
    sps = max(1, int(symbol_dur * fs))
    n = len(symbols) * sps

    freq = np.zeros(n)
    for i, s in enumerate(symbols):
        freq[i * sps : (i + 1) * sps] = (s - (num_tones - 1) / 2.0) * tone_spacing

    phase = 2 * np.pi * np.cumsum(freq) / fs
    phase += np.random.uniform(0, 2 * np.pi)
    return np.exp(1j * phase)


def ook_mod(bits, tone_freq, bit_dur, fs=FS):
    """On-off keying of a single tone (Hellschreiber)."""
    sps = max(1, int(bit_dur * fs))
    n = len(bits) * sps
    t = np.arange(n) / fs

    envelope = np.zeros(n)
    for i, b in enumerate(bits):
        if b:
            envelope[i * sps : (i + 1) * sps] = 1.0

    phase0 = np.random.uniform(0, 2 * np.pi)
    return envelope * np.exp(1j * (2 * np.pi * tone_freq * t + phase0))


def psk_mod(phase_bits, baud, fs=FS, order=2):
    """PSK modulator with raised cosine amplitude shaping.

    Args:
        phase_bits: Array of symbol indices (0 to order-1).
        baud: Symbol rate in baud.
        fs: Sample rate.
        order: PSK order (2=BPSK, 4=QPSK, 8=8PSK).

    Returns:
        Complex IQ signal.
    """
    sps = max(1, int(fs / baud))
    n_sym = len(phase_bits)
    n = n_sym * sps
    t = np.arange(n) / fs

    phases = np.array(phase_bits, dtype=float) * (2 * np.pi / order)

    phase_sig = np.zeros(n)
    for i, p in enumerate(phases):
        phase_sig[i * sps:(i + 1) * sps] = p

    # Raised cosine envelope at transitions (for BPSK)
    envelope = np.ones(n)
    if order == 2:
        ramp_len = sps // 4
        if ramp_len > 1:
            for i in range(1, n_sym):
                if phase_bits[i] != phase_bits[i - 1]:
                    center = i * sps
                    start = max(0, center - ramp_len)
                    end = min(n, center + ramp_len)
                    mid = (start + end) // 2
                    ramp = 0.5 * (1 - np.cos(np.pi * np.arange(ramp_len) / ramp_len))
                    envelope[start:mid] = np.minimum(
                        envelope[start:mid], ramp[:mid - start])
                    envelope[mid:end] = np.minimum(
                        envelope[mid:end], ramp[:end - mid][::-1])

    carrier_freq = np.random.uniform(-200, 200)
    return envelope * np.exp(1j * (2 * np.pi * carrier_freq * t + phase_sig))


def _4fsk_mod(dibits, sym_rate, dev_outer, dev_inner, fs=FS, smooth=True):
    """Generic 4-level FSK modulator.

    Dibit mapping: 01 -> +dev_outer, 00 -> +dev_inner,
                   10 -> -dev_inner, 11 -> -dev_outer.
    """
    dibit_to_dev = {
        0b01: dev_outer,
        0b00: dev_inner,
        0b10: -dev_inner,
        0b11: -dev_outer,
    }
    sps = max(1, int(fs / sym_rate))
    n = len(dibits) * sps
    freq = np.zeros(n)
    for i, d in enumerate(dibits):
        freq[i * sps:(i + 1) * sps] = dibit_to_dev.get(d & 0x03, 0.0)

    if smooth:
        filt_len = min(4 * sps, n)
        if filt_len > 1:
            sigma = sps / (2 * np.pi * 0.3)
            t_f = (np.arange(filt_len) - filt_len // 2)
            gauss = np.exp(-t_f ** 2 / (2 * sigma ** 2))
            gauss /= gauss.sum()
            freq = fftconvolve(freq, gauss, mode="same")

    phase = 2 * np.pi * np.cumsum(freq) / fs
    phase += np.random.uniform(0, 2 * np.pi)
    return np.exp(1j * phase)


def _gmsk_mod(bits, bit_rate, bt=0.5, fs=FS):
    """GMSK modulator with configurable BT product."""
    sps = max(1, int(fs / bit_rate))
    n = len(bits) * sps
    # NRZ: 0 -> -1, 1 -> +1
    freq = np.zeros(n)
    deviation = bit_rate / 4.0  # MSK deviation
    for i, b in enumerate(bits):
        freq[i * sps:(i + 1) * sps] = deviation if b else -deviation

    # Gaussian filter
    filt_len = min(4 * sps, n)
    if filt_len > 1:
        sigma = sps / (2 * np.pi * bt)
        t_f = (np.arange(filt_len) - filt_len // 2)
        gauss = np.exp(-t_f ** 2 / (2 * sigma ** 2))
        gauss /= gauss.sum()
        freq = fftconvolve(freq, gauss, mode="same")

    phase = 2 * np.pi * np.cumsum(freq) / fs
    phase += np.random.uniform(0, 2 * np.pi)
    return np.exp(1j * phase)


def ofdm_carriers(n_carriers, carrier_spacing, symbol_dur, n_symbols, fs=FS):
    """Multi-carrier QPSK (OFDM-like) for FreeDV synthetic mode."""
    sps = max(1, int(symbol_dur * fs))
    n = n_symbols * sps
    t = np.arange(n) / fs
    sig = np.zeros(n, dtype=np.complex128)

    bw = n_carriers * carrier_spacing
    f_start = -bw / 2

    for c in range(n_carriers):
        f_c = f_start + c * carrier_spacing
        phases = np.random.choice([0, np.pi/2, np.pi, 3*np.pi/2], n_symbols)
        carrier = np.zeros(n, dtype=np.complex128)
        for i in range(n_symbols):
            carrier[i * sps:(i + 1) * sps] = np.exp(1j * phases[i])
        sig += carrier * np.exp(2j * np.pi * f_c * t)

    sig /= np.sqrt(n_carriers)
    return sig
