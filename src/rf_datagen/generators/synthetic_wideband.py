"""Wideband signal synthesis (FS=20 MHz) — 8 signal classes."""

import numpy as np

from ..constants import WIDEBAND_FS, WIDEBAND_WINDOW_LEN
from ..dsp import dsss_mod, oqpsk_mod, ofdm_full
from .base import BaseGenerator


_FS = WIDEBAND_FS
_WL = WIDEBAND_WINDOW_LEN


def _ensure_length(fn):
    """Decorator: loop synth until output >= window_len."""
    def wrapper(*, fs=_FS, window_len=_WL):
        segments = []
        total = 0
        while total < window_len:
            seg = fn(fs=fs, window_len=window_len)
            segments.append(seg)
            total += len(seg)
        return np.concatenate(segments) if len(segments) > 1 else segments[0]
    wrapper.__name__ = fn.__name__
    wrapper.__doc__ = fn.__doc__
    return wrapper


# ---------------------------------------------------------------------------
# Mode synthesizers
# ---------------------------------------------------------------------------

@_ensure_length
def synth_wifi_preamble(*, fs=_FS, window_len=_WL):
    """WiFi 802.11a/g — STF + LTF + SIGNAL + data OFDM symbols."""
    fft_size = 64
    subcarrier_spacing = 312.5e3  # 20 MHz / 64
    cp_length = 16  # 0.8 μs CP

    # Short Training Field: 10 × 16-sample short symbols
    stf_freq = np.zeros(64, dtype=np.complex128)
    stf_subcarriers = [-24, -20, -16, -12, -8, -4, 4, 8, 12, 16, 20, 24]
    for k in stf_subcarriers:
        stf_freq[k % 64] = np.random.choice([1 + 1j, 1 - 1j,
                                              -1 + 1j, -1 - 1j]) / np.sqrt(2)
    stf_time = np.fft.ifft(stf_freq) * np.sqrt(64)
    stf = np.tile(stf_time[:16], 10)

    # Long Training Field: GI2 (32 samples) + 2 × LTF (64 samples each)
    ltf_freq = np.zeros(64, dtype=np.complex128)
    for k in range(-26, 27):
        if k == 0:
            continue
        ltf_freq[k % 64] = np.random.choice([1, -1])
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
    gap = np.zeros(int(np.random.uniform(20e-6, 100e-6) * fs),
                   dtype=np.complex128)
    return np.concatenate([frame, gap])


@_ensure_length
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
    sss_seq = np.random.choice([1, -1], 62)
    for k in range(62):
        sss += sss_seq[k] * np.exp(
            1j * 2 * np.pi * (k - 31) * subcarrier_spacing * t) / np.sqrt(62)

    # Data OFDM symbols (1 subframe = 14 symbols)
    n_subframes = np.random.randint(1, 5)
    data = ofdm_full(fft_size, subcarrier_spacing, cp_length,
                     np.random.choice(["qpsk", "16qam"]),
                     n_subframes * 14, fs=fs, pilot_spacing=6)

    return np.concatenate([pss, sss, data])


@_ensure_length
def synth_fiveg_nr(*, fs=_FS, window_len=_WL):
    """5G NR — SSB (Synchronization Signal Block) + PDSCH-like OFDM."""
    # NR can use 15/30/60/120 kHz SCS; use 30 kHz at 20 MHz
    fft_size = 512
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

    gap = np.zeros(int(np.random.uniform(10e-6, 50e-6) * fs),
                   dtype=np.complex128)
    return np.concatenate([ssb, data, gap])


@_ensure_length
def synth_gps_l1(*, fs=_FS, window_len=_WL):
    """GPS L1 — Gold code (PRN 1-32) at 1.023 Mchip/s, 50 bps nav data."""
    chip_rate = 1.023e6
    # Generate a Gold code (simplified: random PN code of length 1023)
    prn = np.random.randint(1, 33)
    np.random.seed(prn + 1000)  # Deterministic per PRN
    gold_code = np.random.randint(0, 2, 1023)
    np.random.seed(None)

    # Navigation data at 50 bps
    n_data_bits = np.random.randint(10, 50)
    data_bits = np.random.randint(0, 2, n_data_bits)

    # Each data bit spans 20 code epochs (20 ms)
    chips_per_bit = 20 * 1023
    sig = dsss_mod(data_bits, gold_code, chips_per_bit,
                   fs=fs, chip_rate=chip_rate)
    return sig


@_ensure_length
def synth_zigbee(*, fs=_FS, window_len=_WL):
    """Zigbee 802.15.4 — O-QPSK with PN chip-to-symbol mapping."""
    # 2.4 GHz band: 2 Msym/s, 250 kbps
    sym_rate = 62500  # O-QPSK symbol rate (250 kbps / 4 bits per symbol)

    # Generate data symbols
    n_frames = np.random.randint(3, 10)
    segments = []

    for _ in range(n_frames):
        # SHR (preamble + SFD) + PHR + PSDU
        n_symbols = np.random.randint(50, 200)
        symbols = np.random.randint(0, 4, n_symbols)
        seg = oqpsk_mod(symbols, sym_rate, fs=fs)
        segments.append(seg)
        # Inter-frame spacing
        gap = np.zeros(int(np.random.uniform(0.001, 0.01) * fs),
                       dtype=np.complex128)
        segments.append(gap)

    return np.concatenate(segments)


@_ensure_length
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


@_ensure_length
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


@_ensure_length
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

    def generate_class(self, class_name, rng=None):
        synth_fn = WIDEBAND_SYNTHESIZERS[class_name]
        segments = []
        target_samples = max(self.window_len * 10,
                             self.samples_per_class * self.window_len // 4)
        total = 0
        while total < target_samples:
            seg = synth_fn(fs=self.fs, window_len=self.window_len)
            segments.append(seg)
            total += len(seg)
        return np.concatenate(segments)
