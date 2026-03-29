"""Moderate-rate signal synthesis (FS=1 MHz) — 14 signal classes."""

import numpy as np
from scipy.signal import fftconvolve

from ..domains import MODERATE
from ..dsp import (gfsk_mod, fsk_mod, psk_mod, _gmsk_mod, ofdm_carriers,
                    chirp_mod, dsss_mod, oqpsk_mod, ppm_mod)
from .base import BaseGenerator, ensure_length, make_gap


_FS = MODERATE.sample_rate
_WL = MODERATE.window_length


# ---------------------------------------------------------------------------
# Mode synthesizers — all accept *, fs=_FS, window_len=_WL
# ---------------------------------------------------------------------------


@ensure_length
def synth_ble(*, fs=_FS, window_len=_WL):
    """BLE — GFSK at 1 Msym/s, BT=0.5, advertising PDU framing."""
    n_packets = np.random.randint(3, 10)
    segments = []

    for _ in range(n_packets):
        # Preamble: alternating 01010101 (1 byte)
        preamble = np.array([0, 1] * 4)
        # Access address: 0x8E89BED6 for advertising, random for data
        if np.random.random() < 0.4:  # ~40% advertising packets
            aa_val = 0x8E89BED6
        else:
            aa_val = np.random.randint(0, 2**32)
        aa_bits = np.array([(aa_val >> i) & 1 for i in range(32)], dtype=int)
        # PDU: 2-39 bytes
        pdu_bits = np.random.randint(0, 2, np.random.randint(16, 312))
        # CRC: 3 bytes
        crc_bits = np.random.randint(0, 2, 24)

        bits = np.concatenate([preamble, aa_bits, pdu_bits, crc_bits])
        # BLE uses 2-GFSK: map bits to ±1 symbols
        symbols = bits.astype(float)
        seg = gfsk_mod(symbols, 2, 500e3, 1e-6, fs=fs, bt=0.5)
        segments.append(seg)
        # Inter-packet interval
        gap = make_gap(150e-6, 10e-3, fs)
        segments.append(gap)

    return np.concatenate(segments)


@ensure_length
def synth_zwave(*, fs=_FS, window_len=_WL):
    """Z-Wave — FSK at 9.6/40/100 kbps, ±40 kHz deviation."""
    baud = np.random.choice([9600, 40000, 100000])
    deviation = 40000.0  # ±40 kHz for Z-Wave

    n_frames = np.random.randint(3, 8)
    segments = []

    for _ in range(n_frames):
        # Preamble + sync + data
        preamble = np.array([0, 1] * 40)  # 80-bit preamble
        sync = np.array([0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 1, 1, 0, 1, 0, 1])
        data = np.random.randint(0, 2, np.random.randint(50, 300))
        bits = np.concatenate([preamble, sync, data])
        # Manchester encoding: 1→[1,0], 0→[0,1]
        manchester = np.zeros(len(bits) * 2, dtype=int)
        manchester[0::2] = bits
        manchester[1::2] = 1 - bits
        # FSK at 2× baud rate (Manchester doubles symbol rate)
        seg = fsk_mod(manchester, 2, deviation, 1.0 / (2 * baud), fs=fs)
        segments.append(seg)
        gap = make_gap(0.01, 0.05, fs)
        segments.append(gap)

    return np.concatenate(segments)


@ensure_length
def synth_ads_b(*, fs=_FS, window_len=_WL):
    """ADS-B — PPM 1090ES format: 8μs preamble + 112-bit message."""
    n_msgs = np.random.randint(5, 20)
    segments = []

    for _ in range(n_msgs):
        # 8μs preamble: pulses at 0, 1, 3.5, 4.5 μs
        preamble_us = 8.0
        preamble_samples = max(1, int(preamble_us * 1e-6 * fs))
        preamble = np.zeros(preamble_samples, dtype=np.complex128)
        for pulse_us in [0, 1.0, 3.5, 4.5]:
            ps = int(pulse_us * 1e-6 * fs)
            pe = min(ps + int(0.5e-6 * fs), preamble_samples)
            preamble[ps:pe] = 1.0

        # 112-bit PPM data (1 μs per bit slot)
        data_bits = np.random.randint(0, 2, 112)
        data_sig = ppm_mod(data_bits, 1e-6, fs=fs)

        msg = np.concatenate([preamble, data_sig])
        segments.append(msg)
        # Random inter-message gap
        gap = make_gap(0.0005, 0.005, fs)
        segments.append(gap)

    return np.concatenate(segments)


@ensure_length
def synth_gsm_burst(*, fs=_FS, window_len=_WL):
    """GSM — GMSK at 270.833 kbps, BT=0.3, normal burst structure."""
    baud = 270833
    n_bursts = np.random.randint(5, 20)
    segments = []

    for _ in range(n_bursts):
        # Normal burst: 3 tail + 57 data + 1 flag + 26 training + 1 flag + 57 data + 3 tail + 8.25 guard
        tail = np.zeros(3)
        training = np.array([0,0,1,0,0,1,0,1,1,1,0,0,0,0,1,0,0,0,1,0,0,1,0,1,1,1])
        flag = np.array([np.random.randint(0, 2)])
        data1 = np.random.randint(0, 2, 57)
        data2 = np.random.randint(0, 2, 57)
        guard = np.zeros(8)
        bits = np.concatenate([tail, data1, flag, training, flag, data2, tail, guard])
        seg = _gmsk_mod(bits, baud, bt=0.3, fs=fs)
        segments.append(seg)
        # TDMA gap (577 μs per timeslot, 8 slots per frame)
        gap = make_gap(0.0001, 0.001, fs)
        segments.append(gap)

    return np.concatenate(segments)


@ensure_length
def synth_lfm_radar(*, fs=_FS, window_len=_WL):
    """LFM radar — linear FM chirp with variable bandwidth/PRF."""
    bw = np.random.uniform(50e3, 500e3)
    pulse_dur = np.random.uniform(1e-6, 50e-6)
    prf = np.random.uniform(500, 5000)
    n_pulses = np.random.randint(10, 50)
    return chirp_mod(bw, pulse_dur, prf, n_pulses, fs=fs)


@ensure_length
def synth_fmcw_radar(*, fs=_FS, window_len=_WL):
    """FMCW radar — continuous triangular frequency sweep."""
    bw = np.random.uniform(100e3, 500e3)
    sweep_dur = np.random.uniform(0.5e-3, 5e-3)
    n_sweeps = np.random.randint(10, 50)

    samples_per_sweep = max(1, int(sweep_dur * fs))
    n = n_sweeps * samples_per_sweep
    sig = np.zeros(n, dtype=np.complex128)

    for i in range(n_sweeps):
        start = i * samples_per_sweep
        end = min(start + samples_per_sweep, n)
        actual = end - start
        t = np.arange(actual) / fs
        # Triangular sweep: up then down
        if i % 2 == 0:
            freq = -bw / 2 + bw * t / sweep_dur
        else:
            freq = bw / 2 - bw * t / sweep_dur
        phase = 2 * np.pi * np.cumsum(freq) / fs
        sig[start:end] = np.exp(1j * phase)

    return sig


@ensure_length
def synth_phase_coded_radar(*, fs=_FS, window_len=_WL):
    """Phase-coded radar — Frank/P4 polyphase codes on carrier."""
    code_type = np.random.choice(["frank", "p4"])
    code_len = np.random.choice([16, 25, 36, 64])
    side = int(np.sqrt(code_len))
    chip_dur = np.random.uniform(1e-6, 10e-6)
    prf = np.random.uniform(500, 3000)
    n_pulses = np.random.randint(10, 40)

    # Generate phase code
    if code_type == "frank":
        phases = np.array([2 * np.pi * i * j / side
                           for i in range(side) for j in range(side)])
    else:
        # P4 code
        phases = np.array([np.pi / code_len * (i - 1) * (i - 1)
                           for i in range(1, code_len + 1)])

    chip_samples = max(1, int(chip_dur * fs))
    pulse_samples = code_len * chip_samples
    pri_samples = max(pulse_samples + 1, int(fs / prf))
    n = n_pulses * pri_samples

    sig = np.zeros(n, dtype=np.complex128)
    carrier_freq = np.random.uniform(-100e3, 100e3)

    for p in range(n_pulses):
        for ci, phase in enumerate(phases):
            cs = p * pri_samples + ci * chip_samples
            ce = min(cs + chip_samples, n)
            if cs >= n:
                break
            t = np.arange(ce - cs) / fs
            sig[cs:ce] = np.exp(1j * (2 * np.pi * carrier_freq * t + phase))

    return sig


@ensure_length
def synth_noaa_apt(*, fs=_FS, window_len=_WL):
    """NOAA APT — 2400 Hz AM subcarrier with sync tones and image data."""
    line_dur = 0.5  # 0.5s per line, 4160 samples
    n_lines = np.random.randint(20, 60)
    n = int(n_lines * line_dur * fs)
    t = np.arange(n) / fs

    # 2400 Hz subcarrier
    subcarrier_freq = 2400.0
    sig = np.zeros(n, dtype=np.complex128)

    pos = 0
    for line in range(n_lines):
        line_samples = int(line_dur * fs)
        if pos + line_samples > n:
            break

        line_t = np.arange(line_samples) / fs

        # Sync A (1040 Hz tone, ~0.09s) or Sync B (832 Hz)
        sync_dur = int(0.09 * fs)
        sync_freq = 1040.0 if line % 2 == 0 else 832.0
        sync_t = np.arange(sync_dur) / fs
        sync = np.sin(2 * np.pi * sync_freq * sync_t)

        # Image data: amplitude-modulated subcarrier
        data_samples = line_samples - sync_dur
        data_t = np.arange(data_samples) / fs
        # Simulate image luminance
        luminance = np.random.uniform(0.1, 1.0, max(1, data_samples // 100))
        luminance = np.interp(np.arange(data_samples),
                              np.linspace(0, data_samples, len(luminance)),
                              luminance)
        data = luminance * np.sin(2 * np.pi * subcarrier_freq * data_t)

        # Combine
        line_sig = np.concatenate([sync, data])[:line_samples]
        # Convert to complex
        analytic = np.fft.ifft(
            2 * np.fft.fft(line_sig) *
            (np.arange(len(line_sig)) < len(line_sig) // 2))
        sig[pos:pos + len(analytic)] = analytic
        pos += line_samples

    return sig


@ensure_length
def synth_cospas_sarsat(*, fs=_FS, window_len=_WL):
    """COSPAS-SARSAT — BPSK at 400 bps, 406 MHz beacon frame."""
    baud = 400.0
    n_bursts = np.random.randint(3, 8)
    segments = []

    for _ in range(n_bursts):
        # Beacon message: 144 bits (15 bit sync + 129 bit message)
        sync = np.array([1, 1, 1, 1, 1, 1, 1, 0, 0, 1, 0, 0, 1, 0, 1])
        msg = np.random.randint(0, 2, 129)
        bits = np.concatenate([sync, msg])
        seg = psk_mod(bits, baud, fs=fs, order=2)
        segments.append(seg)
        # ~50s repetition interval (scaled down for dataset)
        gap = make_gap(0.1, 0.5, fs)
        segments.append(gap)

    return np.concatenate(segments)


@ensure_length
def synth_lora_wide(*, fs=_FS, window_len=_WL):
    """LoRa wide — full-bandwidth chirp spread spectrum at native rate."""
    sf = np.random.randint(7, 13)
    bw = np.random.choice([125e3, 250e3, 500e3])
    n_symbols = 2 ** sf
    n_chirps = np.random.randint(8, 30)
    chirp_samples = max(n_symbols, int(n_symbols / bw * fs))

    sig = np.zeros(n_chirps * chirp_samples, dtype=np.complex128)
    n = len(sig)

    # Preamble (8 unmodulated up-chirps)
    preamble_chirps = min(8, n_chirps)
    for i in range(preamble_chirps):
        start = i * chirp_samples
        end = min(start + chirp_samples, n)
        actual = end - start
        t = np.arange(actual) / fs
        k = bw / (chirp_samples / fs)
        freq = -bw / 2 + k * t
        phase = 2 * np.pi * np.cumsum(freq) / fs
        sig[start:end] = np.exp(1j * phase)

    # Data symbols
    for i in range(preamble_chirps, n_chirps):
        start = i * chirp_samples
        end = min(start + chirp_samples, n)
        actual = end - start
        if actual < 1:
            break
        t = np.arange(actual) / fs
        symbol_val = np.random.randint(0, n_symbols)
        freq_offset = (symbol_val / n_symbols) * bw
        k = bw / (chirp_samples / fs)
        freq = np.mod(-bw / 2 + freq_offset + k * t, bw) - bw / 2
        phase = 2 * np.pi * np.cumsum(freq) / fs
        sig[start:end] = np.exp(1j * phase)

    return sig


@ensure_length
def synth_vdl2(*, fs=_FS, window_len=_WL):
    """VDL Mode 2 — D8PSK at 10.5 ksym/s with VDL2 framing."""
    baud = 10500.0
    n_frames = np.random.randint(3, 10)
    segments = []

    for _ in range(n_frames):
        # Known preamble: 16 alternating D8PSK symbols for carrier/clock sync
        preamble_sym = np.array([0, 4] * 8, dtype=int)  # alternating 0° and 180°
        # Unique word: 20-symbol pattern for frame sync
        unique_word = np.array([0, 1, 2, 3, 4, 5, 6, 7, 0, 1,
                                2, 3, 4, 5, 6, 7, 0, 1, 2, 3], dtype=int)
        # Frame header + random data payload
        n_data = np.random.randint(50, 450)
        data_sym = np.random.randint(0, 8, n_data)
        symbols = np.concatenate([preamble_sym, unique_word, data_sym])
        seg = psk_mod(symbols, baud, fs=fs, order=8)
        segments.append(seg)
        gap = make_gap(0.005, 0.05, fs)
        segments.append(gap)

    return np.concatenate(segments)


@ensure_length
def synth_drm_wide(*, fs=_FS, window_len=_WL):
    """DRM wide — OFDM digital broadcast at native DRM parameters."""
    n_carriers = np.random.choice([109, 206, 226])
    carrier_spacing = np.random.uniform(40, 50)
    symbol_dur = np.random.uniform(0.02, 0.04)
    n_symbols = max(20, int(0.5 * fs / (symbol_dur * fs)) + 10)
    n_symbols = min(n_symbols, 100)
    return ofdm_carriers(n_carriers, carrier_spacing, symbol_dur, n_symbols,
                         fs=fs)


@ensure_length
def synth_dect(*, fs=_FS, window_len=_WL):
    """DECT — GFSK at 1.152 Mbps, BT=0.5, TDMA 10ms frame."""
    baud = 1152000
    n_slots = np.random.randint(4, 24)  # 24 slots per TDMA frame
    segments = []

    for _ in range(n_slots):
        # Slot: sync + data
        sync_bits = np.array([1, 0, 1, 0] * 8)  # 32-bit sync
        data_bits = np.random.randint(0, 2, np.random.randint(300, 420))
        bits = np.concatenate([sync_bits, data_bits])
        symbols = bits.astype(float)
        seg = gfsk_mod(symbols, 2, baud / 2.0, 1.0 / baud, fs=fs, bt=0.5)
        segments.append(seg)
        # Guard time between slots
        guard = make_gap(10e-6, 50e-6, fs)
        segments.append(guard)

    return np.concatenate(segments)


@ensure_length
def synth_iridium(*, fs=_FS, window_len=_WL):
    """Iridium — DQPSK at 25 ksym/s with simplex burst structure."""
    baud = 25000.0
    n_bursts = np.random.randint(3, 10)
    segments = []

    for _ in range(n_bursts):
        # Burst: preamble + unique word + data
        n_sym = np.random.randint(200, 1000)
        symbols = np.random.randint(0, 4, n_sym)
        seg = psk_mod(symbols, baud, fs=fs, order=4)
        segments.append(seg)
        # Inter-burst gap
        gap = make_gap(0.001, 0.01, fs)
        segments.append(gap)

    return np.concatenate(segments)


# ---------------------------------------------------------------------------
# Synthesizer registry
# ---------------------------------------------------------------------------

MODERATE_SYNTHESIZERS = {
    "BLE": synth_ble, "ZWAVE": synth_zwave, "ADS_B": synth_ads_b,
    "GSM_BURST": synth_gsm_burst,
    "LFM_RADAR": synth_lfm_radar, "FMCW_RADAR": synth_fmcw_radar,
    "PHASE_CODED_RADAR": synth_phase_coded_radar,
    "NOAA_APT": synth_noaa_apt, "COSPAS_SARSAT": synth_cospas_sarsat,
    "LORA_WIDE": synth_lora_wide,
    "VDL2": synth_vdl2, "DRM_WIDE": synth_drm_wide,
    "DECT": synth_dect, "IRIDIUM": synth_iridium,
}

MODERATE_CLASSES = list(MODERATE_SYNTHESIZERS.keys())


class SyntheticModerateGenerator(BaseGenerator):
    name = "synthetic_moderate"
    required_tools = []
    signal_classes = MODERATE_CLASSES
    synthesizers = MODERATE_SYNTHESIZERS
