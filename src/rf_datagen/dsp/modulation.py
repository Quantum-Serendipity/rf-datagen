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


def _pi4dqpsk_mod(dibits, sym_rate, fs=FS):
    """π/4-DQPSK modulator with root raised cosine pulse shaping.

    Each dibit rotates phase by one of {π/4, 3π/4, −3π/4, −π/4}.
    Used by TETRA (18 ksym/s), reusable for IS-136 cellular.
    """
    from .filters import rrc_filter

    phase_map = {
        0b00: np.pi / 4,
        0b01: 3 * np.pi / 4,
        0b10: -3 * np.pi / 4,
        0b11: -np.pi / 4,
    }
    sps = max(1, int(fs / sym_rate))
    n_sym = len(dibits)

    # Differential encoding: accumulate phase rotations
    phases = np.zeros(n_sym)
    phase_acc = np.random.uniform(0, 2 * np.pi)
    for i, d in enumerate(dibits):
        phase_acc += phase_map.get(d & 0x03, np.pi / 4)
        phases[i] = phase_acc

    # Upsample symbols
    n = n_sym * sps
    i_up = np.zeros(n)
    q_up = np.zeros(n)
    for k in range(n_sym):
        i_up[k * sps] = np.cos(phases[k])
        q_up[k * sps] = np.sin(phases[k])

    # RRC pulse shaping
    n_taps = min(6 * sps + 1, n)
    if n_taps > 1:
        rrc = rrc_filter(n_taps, rolloff=0.35, sps=sps)
        i_shaped = fftconvolve(i_up, rrc, mode="same")
        q_shaped = fftconvolve(q_up, rrc, mode="same")
    else:
        i_shaped, q_shaped = i_up, q_up

    return i_shaped + 1j * q_shaped


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


def chirp_mod(bw, pulse_dur, prf, n_pulses, fs=FS):
    """Linear FM chirp for radar signals (LFM, FMCW).

    Args:
        bw: Chirp bandwidth in Hz.
        pulse_dur: Pulse duration in seconds.
        prf: Pulse repetition frequency in Hz.
        n_pulses: Number of pulses to generate.
        fs: Sample rate.

    Returns:
        Complex IQ signal with chirp pulses.
    """
    pri = 1.0 / prf  # pulse repetition interval
    pri_samples = max(1, int(pri * fs))
    pulse_samples = max(1, int(pulse_dur * fs))
    n = n_pulses * pri_samples

    sig = np.zeros(n, dtype=np.complex128)
    for i in range(n_pulses):
        start = i * pri_samples
        end = min(start + pulse_samples, n)
        actual = end - start
        if actual < 1:
            break
        t = np.arange(actual) / fs
        # Linear frequency sweep from -bw/2 to +bw/2
        k = bw / pulse_dur
        freq = -bw / 2 + k * t
        phase = 2 * np.pi * np.cumsum(freq) / fs
        sig[start:end] = np.exp(1j * phase)

    return sig


def dsss_mod(data_bits, chip_code, chips_per_bit, fs=FS, chip_rate=None):
    """Direct Sequence Spread Spectrum modulator.

    Args:
        data_bits: Data bit array (0/1).
        chip_code: Spreading code (e.g., Gold code for GPS).
        chips_per_bit: Number of chips per data bit.
        fs: Sample rate.
        chip_rate: Chip rate in Hz (default: fs/2).

    Returns:
        Complex BPSK-DSSS signal.
    """
    if chip_rate is None:
        chip_rate = fs / 2

    spc = max(1, int(fs / chip_rate))  # samples per chip
    n_chips = len(data_bits) * chips_per_bit
    code_len = len(chip_code)

    # Spread data with chip code
    chips = np.zeros(n_chips)
    for i, bit in enumerate(data_bits):
        data_val = 1 if bit else -1
        for j in range(chips_per_bit):
            chip_idx = (i * chips_per_bit + j) % code_len
            chip_val = 1 if chip_code[chip_idx] else -1
            chips[i * chips_per_bit + j] = data_val * chip_val

    # Upsample to sample rate
    n = n_chips * spc
    sig = np.zeros(n, dtype=np.complex128)
    for i, c in enumerate(chips):
        sig[i * spc:(i + 1) * spc] = c

    # Random carrier phase
    phase0 = np.random.uniform(0, 2 * np.pi)
    return sig * np.exp(1j * phase0)


def oqpsk_mod(symbols, sym_rate, fs=FS):
    """Offset QPSK with half-sine pulse shaping (IEEE 802.15.4 / Zigbee).

    Args:
        symbols: Array of symbol indices (0-3).
        sym_rate: Symbol rate in symbols/s.
        fs: Sample rate.

    Returns:
        Complex O-QPSK signal.
    """
    sps = max(1, int(fs / sym_rate))
    n_sym = len(symbols)
    n = n_sym * sps * 2  # *2 because I and Q are offset by half-symbol

    # Map symbols to I/Q
    i_bits = np.array([(s >> 1) & 1 for s in symbols]) * 2 - 1  # MSB
    q_bits = np.array([s & 1 for s in symbols]) * 2 - 1          # LSB

    # Half-sine pulse shape
    pulse = np.sin(np.pi * np.arange(2 * sps) / (2 * sps))

    i_sig = np.zeros(n)
    q_sig = np.zeros(n)

    for k in range(n_sym):
        # I channel: starts at k * 2*sps
        i_start = k * 2 * sps
        i_end = min(i_start + 2 * sps, n)
        plen = i_end - i_start
        i_sig[i_start:i_end] = i_bits[k] * pulse[:plen]

        # Q channel: offset by sps (half-symbol)
        q_start = k * 2 * sps + sps
        q_end = min(q_start + 2 * sps, n)
        plen = q_end - q_start
        if plen > 0:
            q_sig[q_start:q_end] = q_bits[k] * pulse[:plen]

    phase0 = np.random.uniform(0, 2 * np.pi)
    return (i_sig + 1j * q_sig) * np.exp(1j * phase0)


def ppm_mod(bits, slot_dur, fs=FS):
    """Pulse Position Modulation (ADS-B / Mode S).

    Each bit is encoded as a pulse in one of two half-slots:
    bit=1 → pulse in first half, bit=0 → pulse in second half.

    Args:
        bits: Binary data (0/1 array).
        slot_dur: Duration of one bit slot in seconds.
        fs: Sample rate.

    Returns:
        Complex IQ signal with PPM pulses.
    """
    sps = max(2, int(slot_dur * fs))  # samples per slot
    half = sps // 2
    n = len(bits) * sps

    sig = np.zeros(n, dtype=np.complex128)
    for i, bit in enumerate(bits):
        start = i * sps
        if bit:
            sig[start:start + half] = 1.0
        else:
            sig[start + half:start + sps] = 1.0

    phase0 = np.random.uniform(0, 2 * np.pi)
    return sig * np.exp(1j * phase0)


def ofdm_full(n_subcarriers, subcarrier_spacing, cp_length,
              constellation, n_symbols, fs=FS, pilot_spacing=0):
    """Full OFDM with cyclic prefix, pilot insertion, multiple constellations.

    Args:
        n_subcarriers: Number of active subcarriers (FFT size).
        subcarrier_spacing: Subcarrier spacing in Hz.
        cp_length: Cyclic prefix length in samples.
        constellation: "bpsk", "qpsk", "16qam", or "64qam".
        n_symbols: Number of OFDM symbols to generate.
        fs: Sample rate.
        pilot_spacing: Insert pilot every N subcarriers (0 = no pilots).

    Returns:
        Complex IQ signal.
    """
    fft_size = n_subcarriers

    # Constellation mapping
    if constellation == "bpsk":
        points = np.array([1 + 0j, -1 + 0j])
    elif constellation == "qpsk":
        points = np.array([1 + 1j, -1 + 1j, 1 - 1j, -1 - 1j]) / np.sqrt(2)
    elif constellation == "16qam":
        levels = [-3, -1, 1, 3]
        points = np.array([i + 1j * q for i in levels for q in levels])
        points /= np.sqrt(np.mean(np.abs(points) ** 2))
    else:  # 64qam
        levels = [-7, -5, -3, -1, 1, 3, 5, 7]
        points = np.array([i + 1j * q for i in levels for q in levels])
        points /= np.sqrt(np.mean(np.abs(points) ** 2))

    symbol_samples = fft_size + cp_length
    n = n_symbols * symbol_samples
    sig = np.zeros(n, dtype=np.complex128)

    for sym_idx in range(n_symbols):
        # Allocate subcarriers
        freq_domain = np.zeros(fft_size, dtype=np.complex128)

        for k in range(fft_size):
            # DC null
            if k == 0 or k == fft_size // 2:
                continue
            # Pilot subcarriers
            if pilot_spacing > 0 and k % pilot_spacing == 0:
                freq_domain[k] = 1.0  # BPSK pilot
                continue
            # Data subcarrier
            freq_domain[k] = points[np.random.randint(len(points))]

        # IFFT to time domain
        time_domain = np.fft.ifft(freq_domain) * np.sqrt(fft_size)

        # Add cyclic prefix
        ofdm_symbol = np.concatenate([time_domain[-cp_length:], time_domain])

        start = sym_idx * symbol_samples
        end = start + symbol_samples
        if end <= n:
            sig[start:end] = ofdm_symbol

    return sig
