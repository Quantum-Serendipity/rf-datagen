"""Wideband signal synthesis (FS=20 MHz) — 8 signal classes."""

import numpy as np

from ..domains import WIDEBAND
from ..dsp import dsss_mod, oqpsk_mod, ofdm_full
from .base import BaseGenerator, ensure_length, make_gap


_FS = WIDEBAND.sample_rate
_WL = WIDEBAND.window_length


# ---------------------------------------------------------------------------
# Mode synthesizers
# ---------------------------------------------------------------------------

@ensure_length
def synth_wifi_preamble(*, fs=_FS, window_len=_WL):
    """WiFi 802.11a/g — STF + LTF + SIGNAL + data OFDM symbols."""
    fft_size = 64
    subcarrier_spacing = 312.5e3  # 20 MHz / 64
    cp_length = 16  # 0.8 μs CP

    # Short Training Field: 10 × 16-sample short symbols
    stf_freq = np.zeros(64, dtype=np.complex128)
    stf_subcarriers = [-24, -20, -16, -12, -8, -4, 4, 8, 12, 16, 20, 24]
    stf_values = [1+1j, -1-1j, 1+1j, -1-1j, -1-1j, 1+1j,
                  -1-1j, -1-1j, 1+1j, 1+1j, 1+1j, 1+1j]
    stf_scale = np.sqrt(13 / 6)
    for k, val in zip(stf_subcarriers, stf_values):
        stf_freq[k % 64] = val * stf_scale
    stf_time = np.fft.ifft(stf_freq) * np.sqrt(64)
    stf = np.tile(stf_time[:16], 10)

    # Long Training Field: GI2 (32 samples) + 2 × LTF (64 samples each)
    ltf_freq = np.zeros(64, dtype=np.complex128)
    ltf_known = [1, 1, -1, -1, 1, 1, -1, 1, -1, 1, 1, 1, 1, 1, 1, -1, -1,
                 1, 1, -1, 1, -1, 1, 1, 1, 1,                   # -26 to -1
                 1, -1, -1, 1, 1, -1, 1, -1, 1, -1, -1, -1, -1, -1, 1, 1,
                 -1, -1, 1, -1, 1, -1, 1, 1, 1, 1]              # +1 to +26
    ltf_idx = 0
    for k in range(-26, 27):
        if k == 0:
            continue
        ltf_freq[k % 64] = ltf_known[ltf_idx]
        ltf_idx += 1
    ltf_time = np.fft.ifft(ltf_freq) * np.sqrt(64)
    gi2 = ltf_time[-32:]
    ltf = np.concatenate([gi2, ltf_time, ltf_time])

    # SIGNAL + data OFDM symbols
    n_data_symbols = np.random.randint(5, 30)
    data = ofdm_full(fft_size, subcarrier_spacing, cp_length,
                     np.random.choice(["qpsk", "16qam", "64qam"]),
                     n_data_symbols, fs=fs, pilot_spacing=7)

    frame = np.concatenate([stf, ltf, data])

    # Add inter-frame gap
    gap = make_gap(20e-6, 100e-6, fs)
    return np.concatenate([frame, gap])


@ensure_length
def synth_lte_frame(*, fs=_FS, window_len=_WL):
    """LTE — PSS/SSS sync signals + resource elements, 15 kHz SCS."""
    fft_size = 1024  # ~15 kHz SCS at 15.36 MHz, close enough at 20 MHz
    subcarrier_spacing = fs / fft_size
    cp_length = fft_size // 8  # Normal CP

    # PSS (Primary Sync Signal) — 62 subcarriers, Zadoff-Chu sequence
    pss_len = int((fft_size + cp_length) * 1)
    pss = np.zeros(pss_len, dtype=np.complex128)
    t = np.arange(pss_len) / fs
    # Simplified PSS: ZC sequence at center frequency
    root = np.random.choice([25, 29, 34])
    for k in range(62):
        phase = -np.pi * root * k * (k + 1) / 63
        pss += np.exp(1j * (2 * np.pi * (k - 31) * subcarrier_spacing * t +
                            phase)) / np.sqrt(62)

    # SSS (Secondary Sync Signal)
    sss = np.zeros(pss_len, dtype=np.complex128)
    # SSS from m-sequence pair (simplified spec-compliant generation)
    cell_id_group = np.random.randint(0, 168)
    # Generate m-sequences using x^5 + x^2 + 1 polynomial
    m0 = np.ones(31, dtype=int)  # length-31 m-sequence
    reg = np.array([0, 0, 0, 0, 1])  # initial state
    for i in range(31):
        m0[i] = reg[4]
        fb = reg[4] ^ reg[1]  # taps at positions 5 and 2
        reg = np.roll(reg, 1)
        reg[0] = fb
    # Second m-sequence with x^5 + x^4 + x^2 + x + 1
    m1 = np.ones(31, dtype=int)
    reg = np.array([0, 0, 0, 0, 1])
    for i in range(31):
        m1[i] = reg[4]
        fb = reg[4] ^ reg[3] ^ reg[1] ^ reg[0]
        reg = np.roll(reg, 1)
        reg[0] = fb
    # Cyclic shifts based on cell_id_group
    q_prime = cell_id_group // 30
    q = cell_id_group // (q_prime + 1) if q_prime > 0 else cell_id_group
    m_prime = cell_id_group + q * (q + 1) // 2
    s0 = np.array([1 - 2 * m0[(i + (m_prime % 31)) % 31] for i in range(31)])
    s1 = np.array([1 - 2 * m1[(i + (m_prime % 31)) % 31] for i in range(31)])
    sss_seq = np.zeros(62)
    sss_seq[0::2] = s0  # even indices
    sss_seq[1::2] = s1  # odd indices
    for k in range(62):
        sss += sss_seq[k] * np.exp(
            1j * 2 * np.pi * (k - 31) * subcarrier_spacing * t) / np.sqrt(62)

    # Data OFDM symbols (1 subframe = 14 symbols)
    n_subframes = np.random.randint(1, 5)
    data = ofdm_full(fft_size, subcarrier_spacing, cp_length,
                     np.random.choice(["qpsk", "16qam"]),
                     n_subframes * 14, fs=fs, pilot_spacing=6)

    return np.concatenate([pss, sss, data])


@ensure_length
def synth_fiveg_nr(*, fs=_FS, window_len=_WL):
    """5G NR — SSB (Synchronization Signal Block) + PDSCH-like OFDM."""
    # NR can use 15/30/60/120 kHz SCS; randomly pick 15 or 30 kHz
    if np.random.random() < 0.5:
        fft_size = 1024   # ~19.5 kHz SCS ≈ 15 kHz
    else:
        fft_size = 672    # ~29.76 kHz SCS ≈ 30 kHz
    subcarrier_spacing = fs / fft_size
    cp_length = fft_size // 8

    # SSB: 4 OFDM symbols (PSS + PBCH + SSS + PBCH)
    ssb = ofdm_full(fft_size, subcarrier_spacing, cp_length,
                    "qpsk", 4, fs=fs, pilot_spacing=4)

    # PDSCH data
    n_symbols = np.random.randint(10, 56)  # 1-4 slots
    data = ofdm_full(fft_size, subcarrier_spacing, cp_length,
                     np.random.choice(["qpsk", "16qam", "64qam"]),
                     n_symbols, fs=fs, pilot_spacing=4)

    gap = make_gap(10e-6, 50e-6, fs)
    return np.concatenate([ssb, data, gap])


@ensure_length
def synth_gps_l1(*, fs=_FS, window_len=_WL):
    """GPS L1 — Gold code (PRN 1-32) at 1.023 Mchip/s, 50 bps nav data."""
    chip_rate = 1.023e6
    # Generate Gold code from G1/G2 LFSRs per GPS ICD
    prn = np.random.randint(1, 33)
    # G1 LFSR: x^10 + x^3 + 1 (taps at bits 3 and 10)
    g1 = np.zeros(1023, dtype=int)
    reg = np.ones(10, dtype=int)
    for i in range(1023):
        g1[i] = reg[9]
        fb = reg[2] ^ reg[9]
        reg = np.roll(reg, 1)
        reg[0] = fb

    # G2 LFSR: x^10 + x^9 + x^8 + x^6 + x^3 + x^2 + 1
    # Tap pairs per PRN (GPS ICD table 3-I, selected entries)
    tap_pairs = {
        1: (2,6), 2: (3,7), 3: (4,8), 4: (5,9), 5: (1,9),
        6: (2,10), 7: (1,8), 8: (2,9), 9: (3,10), 10: (2,3),
        11: (3,4), 12: (5,6), 13: (6,7), 14: (7,8), 15: (8,9),
        16: (9,10), 17: (1,4), 18: (2,5), 19: (3,6), 20: (4,7),
        21: (5,8), 22: (6,9), 23: (1,3), 24: (4,6), 25: (5,7),
        26: (6,8), 27: (7,9), 28: (8,10), 29: (1,6), 30: (2,7),
        31: (3,8), 32: (4,9),
    }
    g2 = np.zeros(1023, dtype=int)
    reg = np.ones(10, dtype=int)
    tap1, tap2 = tap_pairs[prn]
    for i in range(1023):
        g2[i] = reg[tap1 - 1] ^ reg[tap2 - 1]
        fb = reg[1] ^ reg[2] ^ reg[5] ^ reg[7] ^ reg[8] ^ reg[9]
        reg = np.roll(reg, 1)
        reg[0] = fb

    gold_code = g1 ^ g2

    # Navigation data at 50 bps
    n_data_bits = np.random.randint(10, 50)
    data_bits = np.random.randint(0, 2, n_data_bits)

    # Each data bit spans 20 code epochs (20 ms)
    chips_per_bit = 20 * 1023
    sig = dsss_mod(data_bits, gold_code, chips_per_bit,
                   fs=fs, chip_rate=chip_rate)
    return sig


@ensure_length
def synth_zigbee(*, fs=_FS, window_len=_WL):
    """Zigbee 802.15.4 — O-QPSK with PN chip-to-symbol mapping."""
    # 2.4 GHz band: 2 Msym/s, 250 kbps
    sym_rate = 62500  # O-QPSK symbol rate (250 kbps / 4 bits per symbol)

    # Generate data symbols
    n_frames = np.random.randint(3, 10)
    segments = []

    for _ in range(n_frames):
        # 802.15.4 SHR: 32 zero bits (preamble) → 8 zero symbols
        preamble_symbols = np.zeros(8, dtype=int)
        # SFD (Start of Frame Delimiter): 0xA7 → mapped to symbols
        sfd_bits = np.array([1,1,1,0,0,1,0,1], dtype=int)  # 0xA7 LSB first
        sfd_symbols = np.array([sfd_bits[i*2]*2 + sfd_bits[i*2+1] for i in range(4)], dtype=int)
        # PHR (PHY Header): 1 byte frame length
        phr_symbols = np.random.randint(0, 4, 2)
        # PSDU (payload)
        n_payload = np.random.randint(40, 190)
        payload_symbols = np.random.randint(0, 4, n_payload)
        symbols = np.concatenate([preamble_symbols, sfd_symbols, phr_symbols, payload_symbols])
        seg = oqpsk_mod(symbols, sym_rate, fs=fs)
        segments.append(seg)
        # Inter-frame spacing
        gap = make_gap(0.001, 0.01, fs)
        segments.append(gap)

    return np.concatenate(segments)


@ensure_length
def synth_dab(*, fs=_FS, window_len=_WL):
    """DAB — 1536 subcarriers, 1 kHz spacing, with phase reference symbol."""
    fft_size = 2048  # Mode I: 2048-point FFT
    subcarrier_spacing = 1000.0  # 1 kHz
    cp_length = 504  # Guard interval for Mode I

    # Phase reference symbol (known pattern)
    prs = ofdm_full(fft_size, subcarrier_spacing, cp_length,
                    "qpsk", 1, fs=fs)

    # Data symbols (D-QPSK across 1536 active subcarriers)
    n_symbols = np.random.randint(10, 76)  # Up to 76 symbols per frame
    data = ofdm_full(fft_size, subcarrier_spacing, cp_length,
                     "qpsk", n_symbols, fs=fs)

    return np.concatenate([prs, data])


@ensure_length
def synth_dvb_t(*, fs=_FS, window_len=_WL):
    """DVB-T — 2K/8K mode OFDM with scattered/continual pilots."""
    mode = np.random.choice(["2k", "8k"])
    if mode == "2k":
        fft_size = 2048
        cp_length = fft_size // 4  # 1/4 guard interval
    else:
        fft_size = 8192
        cp_length = fft_size // 8  # 1/8 guard interval

    subcarrier_spacing = fs / fft_size
    constellation = np.random.choice(["qpsk", "16qam", "64qam"])
    n_symbols = np.random.randint(4, 20)

    # Continual pilot spacing ~ every 12 subcarriers
    return ofdm_full(fft_size, subcarrier_spacing, cp_length,
                     constellation, n_symbols, fs=fs, pilot_spacing=12)


@ensure_length
def synth_loran_c_wide(*, fs=_FS, window_len=_WL):
    """Loran-C — native-rate pulse envelope (8-9 pulses per GRI).

    At 20 MHz we can capture the full Loran-C pulse envelope structure.
    Each pulse: 250 μs rise, ~70 μs cycle period.
    """
    gri = np.random.choice([49900, 59900, 79300, 99600])  # μs
    gri_samples = int(gri * 1e-6 * fs)
    n_gris = max(1, window_len // gri_samples + 1)

    segments = []
    for _ in range(n_gris):
        gri_sig = np.zeros(gri_samples, dtype=np.complex128)

        # Master station: 9 pulses (8 regular + 1 extra for identification)
        n_pulses = np.random.choice([8, 9])
        pulse_spacing_us = 1000  # 1 ms between pulses
        pulse_dur_us = 250  # ~250 μs envelope

        for pi in range(n_pulses):
            pulse_start = int(pi * pulse_spacing_us * 1e-6 * fs)
            pulse_len = int(pulse_dur_us * 1e-6 * fs)
            if pulse_start + pulse_len > gri_samples:
                break

            # Loran-C pulse shape: t^2 * exp(-2t/65μs) * cos(2π*100kHz*t)
            t_pulse = np.arange(pulse_len) / fs
            tau = 65e-6
            envelope = (t_pulse / tau) ** 2 * np.exp(-2 * t_pulse / tau)
            envelope /= np.max(envelope) + 1e-10
            carrier = np.cos(2 * np.pi * 100e3 * t_pulse)

            # Phase coding (Loran-C uses specific patterns)
            phase_code = np.random.choice([0, np.pi])
            pulse = envelope * np.exp(1j * phase_code) * carrier

            end = min(pulse_start + pulse_len, gri_samples)
            gri_sig[pulse_start:end] = pulse[:end - pulse_start]

        segments.append(gri_sig)

    return np.concatenate(segments)


# ---------------------------------------------------------------------------
# Synthesizer registry
# ---------------------------------------------------------------------------

WIDEBAND_SYNTHESIZERS = {
    "WIFI_PREAMBLE": synth_wifi_preamble,
    "LTE_FRAME": synth_lte_frame,
    "FIVEG_NR": synth_fiveg_nr,
    "GPS_L1": synth_gps_l1,
    "ZIGBEE": synth_zigbee,
    "DAB": synth_dab,
    "DVB_T": synth_dvb_t,
    "LORAN_C_WIDE": synth_loran_c_wide,
}

WIDEBAND_CLASSES = list(WIDEBAND_SYNTHESIZERS.keys())


class SyntheticWidebandGenerator(BaseGenerator):
    name = "synthetic_wideband"
    required_tools = []
    signal_classes = WIDEBAND_CLASSES
    synthesizers = WIDEBAND_SYNTHESIZERS
