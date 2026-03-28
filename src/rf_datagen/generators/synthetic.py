"""Pure-Python signal synthesis for 35 signal classes."""

import numpy as np
from scipy.signal import fftconvolve

from ..constants import FS, WINDOW_LEN
from ..dsp import gfsk_mod, fsk_mod, ook_mod, _4fsk_mod, _gmsk_mod, ofdm_carriers
from ..content.typing import CWFistModel, text_to_varicode_bits, text_to_morse_elements
from ..content.ham_text import PSK_TEXTS, CW_PHRASES
from .base import BaseGenerator


def _get_psk_text():
    parts = []
    n_parts = np.random.randint(2, 5)
    for _ in range(n_parts):
        parts.append(np.random.choice(PSK_TEXTS))
    return " ".join(parts)


def _get_cw_text():
    parts = []
    n_parts = np.random.randint(2, 6)
    for _ in range(n_parts):
        parts.append(np.random.choice(CW_PHRASES))
    return " ".join(parts)


# ---------------------------------------------------------------------------
# Mode synthesizers — all accept *, fs=FS, window_len=WINDOW_LEN
# ---------------------------------------------------------------------------

def synth_ft8(*, fs=FS, window_len=WINDOW_LEN):
    costas = np.array([3, 1, 4, 0, 6, 5, 2])
    data1 = np.random.randint(0, 8, 29)
    data2 = np.random.randint(0, 8, 29)
    symbols = np.concatenate([costas, data1, costas, data2, costas])
    return gfsk_mod(symbols, 8, 6.25, 0.160, fs=fs, bt=2.0)


def synth_ft4(*, fs=FS, window_len=WINDOW_LEN):
    n_sym = np.random.randint(50, 106)
    symbols = np.random.randint(0, 4, n_sym)
    return gfsk_mod(symbols, 4, 20.8333, 0.048, fs=fs, bt=1.0)


def synth_wspr(*, fs=FS, window_len=WINDOW_LEN):
    n_sym = np.random.randint(5, 20)
    symbols = np.random.randint(0, 4, n_sym)
    return fsk_mod(symbols, 4, 1.4648, 0.6827, fs=fs)


def synth_jt65(*, fs=FS, window_len=WINDOW_LEN):
    sync_pattern = [
        1,0,0,1,1,0,0,0,1,1,1,1,1,1,0,1,0,1,0,0,0,1,0,1,1,0,0,1,0,0,
        0,1,1,1,0,0,1,1,1,1,0,1,1,0,1,1,1,1,0,0,0,1,1,0,1,0,1,0,1,1,
        0,0,1,1,0,1,0,1,0,1,0,0,1,0,0,0,0,0,0,1,1,0,0,0,0,0,0,0,1,1,
        0,1,0,0,1,0,1,1,0,1,0,1,0,1,0,0,1,1,0,0,1,0,0,1,0,0,0,0,1,1,
        1,1,1,1,0,1,
    ]
    symbols = np.array([
        0 if s else np.random.randint(0, 65) for s in sync_pattern
    ])
    return fsk_mod(symbols, 65, 2.6917, 0.3716, fs=fs)


def synth_jt9(*, fs=FS, window_len=WINDOW_LEN):
    sync_vector = [
        1, 1, 0, 0, 1, 0, 0, 0, 1, 1, 1, 1, 1, 1, 0, 1, 0, 1, 0, 0,
        0, 1, 0, 1, 1, 0, 0, 1, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0,
        1, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 1, 1, 0, 1, 0, 1, 1,
        0, 0, 1, 1, 0, 1, 0, 1, 0, 1, 1, 1, 0, 0, 0, 0, 1, 0, 1,
    ]
    symbols = np.array([
        0 if s else np.random.randint(0, 9) for s in sync_vector
    ])
    return fsk_mod(symbols, 9, 1.7361, 0.576, fs=fs)


def synth_js8(*, fs=FS, window_len=WINDOW_LEN):
    speeds = [0.040, 0.080, 0.160, 0.320]
    weights = [0.25, 0.25, 0.25, 0.25]
    symbol_dur = np.random.choice(speeds, p=weights)
    costas = np.array([3, 1, 4, 0, 6, 5, 2])
    n_data = np.random.randint(15, 40)
    data1 = np.random.randint(0, 8, n_data)
    data2 = np.random.randint(0, 8, n_data)
    symbols = np.concatenate([costas, data1, costas, data2, costas])
    return gfsk_mod(symbols, 8, 6.25, symbol_dur, fs=fs, bt=2.0)


def synth_fm(*, fs=FS, window_len=WINDOW_LEN):
    n = window_len + 500
    t = np.arange(n) / fs
    audio = np.zeros(n)
    n_formants = np.random.randint(2, 5)
    for _ in range(n_formants):
        center = np.random.choice([300, 700, 1100, 1800, 2500])
        bw = np.random.uniform(50, 200)
        for _ in range(np.random.randint(2, 5)):
            f = center + np.random.uniform(-bw, bw)
            a = np.random.uniform(0.2, 1.0)
            audio += a * np.sin(2 * np.pi * f * t + np.random.uniform(0, 2 * np.pi))
    env_freq = np.random.uniform(3, 8)
    envelope = np.clip(np.sin(2 * np.pi * env_freq * t) + 0.3, 0, 1)
    n_pauses = np.random.randint(1, 4)
    for _ in range(n_pauses):
        start = np.random.randint(0, n - 100)
        length = np.random.randint(50, 300)
        envelope[start:min(start + length, n)] *= 0.05
    audio = audio * envelope
    audio /= np.abs(audio).max() + 1e-10
    audio = np.diff(audio, prepend=audio[0]) * 0.5 + audio
    deviation = np.random.uniform(500, 1500)
    phase = 2 * np.pi * deviation * np.cumsum(audio) / fs
    return np.exp(1j * phase)


def synth_sstv(*, fs=FS, window_len=WINDOW_LEN):
    n = window_len + 500
    freq = np.zeros(n, dtype=float)
    pos = 0
    while pos < n:
        sync_len = max(1, int(0.005 * fs))
        end = min(pos + sync_len, n)
        freq[pos:end] = 1200.0
        pos = end
        line_len = max(1, int(np.random.uniform(0.05, 0.15) * fs))
        end = min(pos + line_len, n)
        n_pix = end - pos
        if n_pix > 0:
            num_ctrl = max(2, n_pix // 20)
            ctrl_pts = np.random.uniform(1500, 2300, num_ctrl)
            freq[pos:end] = np.interp(
                np.linspace(0, 1, n_pix),
                np.linspace(0, 1, num_ctrl),
                ctrl_pts,
            )
        pos = end
    phase = 2 * np.pi * np.cumsum(freq) / fs
    return np.exp(1j * phase)


def synth_noise(*, fs=FS, window_len=WINDOW_LEN):
    n = window_len + 100
    variant = np.random.choice(["white", "pink", "bandlimited", "impulsive"])
    if variant == "white":
        return (np.random.randn(n) + 1j * np.random.randn(n)) / np.sqrt(2)
    elif variant == "pink":
        white = np.random.randn(n) + 1j * np.random.randn(n)
        X = np.fft.fft(white)
        freqs = np.fft.fftfreq(n, 1.0 / fs)
        scale = 1.0 / np.sqrt(np.abs(freqs) + 1.0)
        return np.fft.ifft(X * scale)
    elif variant == "bandlimited":
        bw = np.random.uniform(500, 2500)
        white = np.random.randn(n) + 1j * np.random.randn(n)
        X = np.fft.fft(white)
        freqs = np.fft.fftfreq(n, 1.0 / fs)
        X[np.abs(freqs) > bw / 2] = 0
        return np.fft.ifft(X)
    else:
        noise = (np.random.randn(n) + 1j * np.random.randn(n)) / np.sqrt(2)
        for _ in range(np.random.randint(5, 30)):
            pos = np.random.randint(0, n)
            width = np.random.randint(1, 10)
            amp = np.random.uniform(3, 10)
            noise[pos : min(pos + width, n)] *= amp
        return noise


def synth_hellschreiber(*, fs=FS, window_len=WINDOW_LEN):
    n_bits = np.random.randint(100, 400)
    bits = np.random.randint(0, 2, n_bits)
    return ook_mod(bits, 980.0, 1.0 / 122.5, fs=fs)


def synth_mfsk16(*, fs=FS, window_len=WINDOW_LEN):
    n_sym = np.random.randint(15, 60)
    symbols = np.random.randint(0, 16, n_sym)
    return fsk_mod(symbols, 16, 15.625, 0.064, fs=fs)


def synth_mfsk32(*, fs=FS, window_len=WINDOW_LEN):
    n_sym = np.random.randint(30, 100)
    symbols = np.random.randint(0, 32, n_sym)
    return fsk_mod(symbols, 32, 31.25, 0.032, fs=fs)


def synth_contestia(*, fs=FS, window_len=WINDOW_LEN):
    n_sym = np.random.randint(20, 80)
    symbols = np.random.randint(0, 8, n_sym)
    return fsk_mod(symbols, 8, 31.25, 0.032, fs=fs)


def synth_thor(*, fs=FS, window_len=WINDOW_LEN):
    n_sym = np.random.randint(15, 60)
    data = np.random.randint(1, 18, n_sym)
    tones = np.cumsum(data) % 18
    return fsk_mod(tones, 18, 10.766, 0.093, fs=fs)


def synth_packet(*, fs=FS, window_len=WINDOW_LEN):
    n_bits = np.random.randint(100, 500)
    bits = np.random.randint(0, 2, n_bits)
    return fsk_mod(bits, 2, 200.0, 1.0 / 300.0, fs=fs)


def synth_am(*, fs=FS, window_len=WINDOW_LEN):
    n = window_len + 500
    t = np.arange(n) / fs
    audio = np.zeros(n)
    n_tones = np.random.randint(5, 15)
    for _ in range(n_tones):
        f = np.random.uniform(100, 3000)
        a = np.random.uniform(0.1, 1.0)
        audio += a * np.sin(2 * np.pi * f * t + np.random.uniform(0, 2 * np.pi))
    env_freq = np.random.uniform(2, 8)
    envelope = 0.5 + 0.5 * np.sin(2 * np.pi * env_freq * t)
    audio = audio * envelope
    audio /= np.abs(audio).max() + 1e-10
    mod_depth = np.random.uniform(0.3, 0.95)
    carrier_freq = np.random.uniform(-500, 500)
    carrier = np.exp(2j * np.pi * carrier_freq * t)
    return (1.0 + mod_depth * audio) * carrier


def synth_dominoex(*, fs=FS, window_len=WINDOW_LEN):
    variants = {
        4:  (3.906,  0.256),
        8:  (7.813,  0.128),
        11: (10.766, 0.0929),
        16: (15.625, 0.064),
        22: (21.533, 0.0464),
    }
    variant = np.random.choice(
        list(variants.keys()), p=[0.1, 0.2, 0.4, 0.2, 0.1])
    baud, symbol_dur = variants[variant]
    tone_spacing = baud
    n_sym = max(25, int(window_len / (symbol_dur * fs)) + 15)
    increments = np.random.randint(0, 18, n_sym)
    tones = np.cumsum(increments) % 18
    return fsk_mod(tones, 18, tone_spacing, symbol_dur, fs=fs)


def synth_cw(fist=None, *, fs=FS, window_len=WINDOW_LEN, wpm_range=(10, 30)):
    if fist is None:
        fist = CWFistModel()
    wpm = np.random.uniform(wpm_range[0], wpm_range[1])
    tone_freq = np.random.uniform(400, 800)
    text = _get_cw_text()
    words = text.split()
    n_segments = min(np.random.randint(3, 6), max(1, len(words)))
    words_per_seg = max(1, len(words) // n_segments)
    elements = []
    for seg_i in range(n_segments):
        seg_wpm = np.random.uniform(max(5, wpm - 5), wpm + 5)
        seg_unit = 1.2 / seg_wpm
        start_w = seg_i * words_per_seg
        end_w = len(words) if seg_i == n_segments - 1 else (seg_i + 1) * words_per_seg
        seg_text = " ".join(words[start_w:end_w])
        elements.extend(text_to_morse_elements(seg_text, fist, seg_unit))
    total_samples = sum(max(1, int(dur * fs)) for dur, _, _ in elements)
    if total_samples < window_len + 100:
        orig = elements.copy()
        while total_samples < window_len + 500:
            elements.extend(orig)
            total_samples = sum(max(1, int(dur * fs)) for dur, _, _ in elements)
    n = total_samples
    t = np.arange(n) / fs
    envelope = np.zeros(n)
    pos = 0
    for dur, is_on, elem_type in elements:
        samp = max(1, int(dur * fs))
        end = min(pos + samp, n)
        if is_on:
            envelope[pos:end] = 1.0
        pos = end
        if pos >= n:
            break
    rise_fall_s = fist.rise_fall_dit
    edge_len = max(1, int(rise_fall_s * fs))
    if edge_len > 1:
        kernel = np.ones(edge_len) / edge_len
        envelope = fftconvolve(envelope, kernel, mode='same')
    phase0 = np.random.uniform(0, 2 * np.pi)
    return envelope * np.exp(1j * (2 * np.pi * tone_freq * t + phase0))


def synth_rtty(*, fs=FS, window_len=WINDOW_LEN):
    bauds = [45.45, 50.0, 75.0]
    shifts = [170.0, 170.0, 850.0]
    weights = [0.6, 0.2, 0.2]
    idx = np.random.choice(len(bauds), p=weights)
    baud = bauds[idx]
    shift = shifts[idx]
    n_bits = max(30, int((window_len + 500) / (fs / baud)) + 10)
    bits = np.random.randint(0, 2, n_bits)
    return fsk_mod(bits, 2, shift, 1.0 / baud, fs=fs)


def synth_psk31(*, fs=FS, window_len=WINDOW_LEN):
    baud = 31.25
    sps = max(1, int(fs / baud))
    n_sym = max(20, int((window_len + 500) / sps) + 10)
    text = _get_psk_text()
    varicode_bits = text_to_varicode_bits(text)
    while len(varicode_bits) < n_sym:
        varicode_bits.extend(text_to_varicode_bits(_get_psk_text()))
    phase_bits = np.array(varicode_bits[:n_sym])
    n = n_sym * sps
    t = np.arange(n) / fs
    phase = np.zeros(n)
    for i, b in enumerate(phase_bits):
        phase[i * sps:(i + 1) * sps] = b * np.pi
    envelope = np.ones(n)
    ramp_len = sps // 4
    if ramp_len > 1:
        ramp = 0.5 * (1 - np.cos(np.pi * np.arange(ramp_len) / ramp_len))
        for i in range(1, n_sym):
            if phase_bits[i] != phase_bits[i - 1]:
                center = i * sps
                start = max(0, center - ramp_len)
                end = min(n, center + ramp_len)
                half = (end - start) // 2
                envelope[start:start + half] = ramp[:half] if half <= ramp_len else ramp[:half]
                envelope[start + half:end] = ramp[:end - start - half][::-1] if (end - start - half) <= ramp_len else ramp[:end - start - half][::-1]
    carrier_freq = np.random.uniform(-200, 200)
    return envelope * np.exp(1j * (2 * np.pi * carrier_freq * t + phase))


def synth_psk63(*, fs=FS, window_len=WINDOW_LEN):
    baud = 62.5
    sps = max(1, int(fs / baud))
    n_sym = max(30, int((window_len + 500) / sps) + 10)
    text = _get_psk_text()
    varicode_bits = text_to_varicode_bits(text)
    while len(varicode_bits) < n_sym:
        varicode_bits.extend(text_to_varicode_bits(_get_psk_text()))
    phase_bits = np.array(varicode_bits[:n_sym])
    n = n_sym * sps
    t = np.arange(n) / fs
    phase = np.zeros(n)
    for i, b in enumerate(phase_bits):
        phase[i * sps:(i + 1) * sps] = b * np.pi
    envelope = np.ones(n)
    ramp_len = sps // 3
    if ramp_len > 1:
        ramp_down = 0.5 * (1 + np.cos(np.pi * np.arange(ramp_len) / ramp_len))
        for i in range(1, n_sym):
            if phase_bits[i] != phase_bits[i - 1]:
                center = i * sps
                start = max(0, center - ramp_len // 2)
                end = min(n, center + ramp_len // 2)
                seg_len = end - start
                if seg_len > 0:
                    envelope[start:end] = np.linspace(1, 0, seg_len // 2 + 1)[:seg_len // 2].tolist() + \
                                          np.linspace(0, 1, seg_len - seg_len // 2).tolist()
    carrier_freq = np.random.uniform(-200, 200)
    return envelope * np.exp(1j * (2 * np.pi * carrier_freq * t + phase))


def synth_ssb(*, fs=FS, window_len=WINDOW_LEN):
    n = window_len + 500
    t = np.arange(n) / fs
    audio = np.zeros(n)
    n_formants = np.random.randint(3, 6)
    for _ in range(n_formants):
        center = np.random.choice([350, 700, 1200, 1800, 2500])
        for _ in range(np.random.randint(2, 5)):
            f = center + np.random.uniform(-100, 100)
            a = np.random.uniform(0.1, 1.0)
            audio += a * np.sin(2 * np.pi * f * t + np.random.uniform(0, 2 * np.pi))
    env_freq = np.random.uniform(3, 7)
    envelope = np.clip(np.sin(2 * np.pi * env_freq * t) + 0.2, 0, 1)
    n_pauses = np.random.randint(1, 3)
    for _ in range(n_pauses):
        start = np.random.randint(0, n - 100)
        length = np.random.randint(80, 400)
        envelope[start:min(start + length, n)] *= 0.02
    audio = audio * envelope
    audio /= np.abs(audio).max() + 1e-10
    analytic = np.fft.ifft(2 * np.fft.fft(audio) * (np.arange(n) < n // 2))
    is_usb = np.random.random() < 0.5
    if not is_usb:
        analytic = np.conj(analytic)
    carrier_freq = np.random.uniform(-500, 500)
    return analytic * np.exp(2j * np.pi * carrier_freq * t)


def synth_qpsk(*, fs=FS, window_len=WINDOW_LEN):
    baud = np.random.choice([31.25, 62.5, 125.0])
    sps = max(1, int(fs / baud))
    n_sym = max(30, int((window_len + 500) / sps) + 10)
    phases = np.random.randint(0, 4, n_sym) * (np.pi / 2)
    n = n_sym * sps
    t = np.arange(n) / fs
    phase_sig = np.zeros(n)
    for i, p in enumerate(phases):
        phase_sig[i * sps:(i + 1) * sps] = p
    envelope = np.ones(n)
    ramp_len = sps // 4
    if ramp_len > 1:
        ramp = 0.5 * (1 - np.cos(np.pi * np.arange(ramp_len) / ramp_len))
        for i in range(1, n_sym):
            if phases[i] != phases[i - 1]:
                center = i * sps
                start = max(0, center - ramp_len)
                end = min(n, center + ramp_len)
                mid = (start + end) // 2
                envelope[start:mid] = np.minimum(envelope[start:mid],
                    ramp[:mid - start])
                envelope[mid:end] = np.minimum(envelope[mid:end],
                    ramp[:end - mid][::-1])
    carrier_freq = np.random.uniform(-200, 200)
    return envelope * np.exp(1j * (2 * np.pi * carrier_freq * t + phase_sig))


def synth_psk125(*, fs=FS, window_len=WINDOW_LEN):
    baud = np.random.choice([125.0, 250.0, 500.0])
    sps = max(1, int(fs / baud))
    n_sym = max(40, int((window_len + 500) / sps) + 10)
    text = _get_psk_text()
    varicode_bits = text_to_varicode_bits(text)
    while len(varicode_bits) < n_sym:
        varicode_bits.extend(text_to_varicode_bits(_get_psk_text()))
    phase_bits = np.array(varicode_bits[:n_sym])
    n = n_sym * sps
    t = np.arange(n) / fs
    phase = np.zeros(n)
    for i, b in enumerate(phase_bits):
        phase[i * sps:(i + 1) * sps] = b * np.pi
    envelope = np.ones(n)
    ramp_len = sps // 3
    if ramp_len > 1:
        ramp = 0.5 * (1 + np.cos(np.pi * np.arange(ramp_len) / ramp_len))
        for i in range(1, n_sym):
            if phase_bits[i] != phase_bits[i - 1]:
                center = i * sps
                start = max(0, center - ramp_len // 2)
                end = min(n, start + ramp_len)
                envelope[start:end] = ramp[:end - start]
    carrier_freq = np.random.uniform(-300, 300)
    return envelope * np.exp(1j * (2 * np.pi * carrier_freq * t + phase))


def synth_8psk(*, fs=FS, window_len=WINDOW_LEN):
    baud = np.random.choice([125.0, 250.0])
    sps = max(1, int(fs / baud))
    n_sym = max(30, int((window_len + 500) / sps) + 10)
    phases = np.random.randint(0, 8, n_sym) * (np.pi / 4)
    n = n_sym * sps
    t = np.arange(n) / fs
    phase_sig = np.zeros(n)
    for i, p in enumerate(phases):
        phase_sig[i * sps:(i + 1) * sps] = p
    carrier_freq = np.random.uniform(-200, 200)
    return np.exp(1j * (2 * np.pi * carrier_freq * t + phase_sig))


def synth_fsq(*, fs=FS, window_len=WINDOW_LEN):
    n_sym = np.random.randint(20, 80)
    tones = np.random.randint(0, 33, n_sym)
    return fsk_mod(tones, 33, 3.0, 1.0 / 3.0, fs=fs)


def synth_ifkp(*, fs=FS, window_len=WINDOW_LEN):
    n_sym = np.random.randint(15, 50)
    increments = np.random.randint(0, 29, n_sym)
    tones = np.cumsum(increments) % 29
    return fsk_mod(tones, 29, 3.0, 0.5, fs=fs)


def synth_throb(*, fs=FS, window_len=WINDOW_LEN):
    baud = np.random.choice([1.0, 2.0, 4.0])
    symbol_dur = 1.0 / baud
    sps = max(1, int(symbol_dur * fs))
    n_sym = max(10, int((window_len + 500) / sps) + 5)
    tone_freqs = np.array([1468, 1476, 1484, 1492, 1500,
                           1508, 1516, 1524, 1532], dtype=float)
    tone_freqs -= 1500.0
    n = n_sym * sps
    t = np.arange(n) / fs
    sig = np.zeros(n, dtype=np.complex128)
    for i in range(n_sym):
        t1, t2 = np.random.choice(9, 2, replace=False)
        f1, f2 = tone_freqs[t1], tone_freqs[t2]
        seg = t[i * sps:(i + 1) * sps]
        sig[i * sps:(i + 1) * sps] = (
            np.exp(2j * np.pi * f1 * seg) +
            np.exp(2j * np.pi * f2 * seg)
        ) / np.sqrt(2)
    carrier_offset = np.random.uniform(-200, 200)
    return sig * np.exp(2j * np.pi * carrier_offset * t)


def synth_fax(*, fs=FS, window_len=WINDOW_LEN):
    # FAX and SSTV share the same FM line-scan encoding at baseband;
    # the fldigi generator produces real FAX waveforms for training —
    # this synthetic fallback only needs to cover the spectral shape.
    return synth_sstv(fs=fs, window_len=window_len)


def synth_olivia(*, fs=FS, window_len=WINDOW_LEN):
    n_sym = np.random.randint(20, 80)
    symbols = np.random.randint(0, 8, n_sym)
    return fsk_mod(symbols, 8, 31.25, 0.032, fs=fs)


def synth_mt63(*, fs=FS, window_len=WINDOW_LEN):
    return ofdm_carriers(64, 15.625, 0.064, np.random.randint(20, 60), fs=fs)


def synth_rsid(mode_code=None, *, fs=FS, window_len=WINDOW_LEN):
    n_symbols = 15
    symbol_dur = 1.0 / 10.766
    tone_spacing = 10.766
    num_tones = 16
    if mode_code is not None and len(mode_code) == n_symbols:
        symbols = np.array(mode_code) % num_tones
    else:
        symbols = np.random.randint(0, num_tones, n_symbols)
    return fsk_mod(symbols, num_tones, tone_spacing, symbol_dur, fs=fs)


def synth_freedv(*, fs=FS, window_len=WINDOW_LEN):
    n_carriers = np.random.randint(7, 22)
    carrier_spacing = np.random.uniform(60, 90)
    symbol_dur = np.random.uniform(0.015, 0.025)
    sps = max(1, int(symbol_dur * fs))
    n_symbols = max(20, int((window_len + 500) / sps) + 5)
    return ofdm_carriers(n_carriers, carrier_spacing, symbol_dur, n_symbols,
                         fs=fs)


def synth_m17(*, fs=FS, window_len=WINDOW_LEN):
    n_sym = max(20, int((window_len + 500) / (fs / 4800)) + 5)
    dibits = np.random.randint(0, 4, n_sym)
    return _4fsk_mod(dibits, 4800, 2400, 800, fs=fs)


def synth_dmr(*, fs=FS, window_len=WINDOW_LEN):
    n_sym = max(20, int((window_len + 500) / (fs / 4800)) + 5)
    dibits = np.random.randint(0, 4, n_sym)
    sig = _4fsk_mod(dibits, 4800, 1944, 648, fs=fs)
    frame_samples = int(0.060 * fs)
    slot_samples = int(0.0275 * fs)
    gap_samples = frame_samples - 2 * slot_samples
    n = len(sig)
    env = np.zeros(n)
    pos = 0
    while pos < n:
        end1 = min(pos + slot_samples, n)
        env[pos:end1] = 1.0
        pos += slot_samples
        pos += gap_samples // 2
        end2 = min(pos + slot_samples, n)
        env[pos:end2] = 1.0
        pos += slot_samples
        pos += gap_samples // 2
    return sig * env


def synth_dstar(*, fs=FS, window_len=WINDOW_LEN):
    n_bits = max(40, int((window_len + 500) / (fs / 4800)) + 10)
    bits = np.random.randint(0, 2, n_bits)
    return _gmsk_mod(bits, 4800, bt=0.5, fs=fs)


def synth_ysf(*, fs=FS, window_len=WINDOW_LEN):
    n_sym = max(20, int((window_len + 500) / (fs / 4800)) + 5)
    dibits = np.random.randint(0, 4, n_sym)
    return _4fsk_mod(dibits, 4800, 1800, 600, fs=fs)


def synth_p25(*, fs=FS, window_len=WINDOW_LEN):
    n_sym = max(20, int((window_len + 500) / (fs / 4800)) + 5)
    dibits = np.random.randint(0, 4, n_sym)
    p25_fs = np.array([0b01, 0b01, 0b01, 0b00, 0b01, 0b00,
                       0b01, 0b01, 0b01, 0b01, 0b01, 0b00,
                       0b01, 0b01, 0b00, 0b01, 0b01, 0b01,
                       0b00, 0b00, 0b01, 0b01, 0b00, 0b01])
    pos = 0
    while pos + len(p25_fs) < len(dibits):
        dibits[pos:pos + len(p25_fs)] = p25_fs
        pos += 180
    return _4fsk_mod(dibits, 4800, 1800, 600, fs=fs)


def synth_nxdn(*, fs=FS, window_len=WINDOW_LEN):
    if np.random.random() < 0.5:
        n_sym = max(20, int((window_len + 500) / (fs / 4800)) + 5)
        dibits = np.random.randint(0, 4, n_sym)
        return _4fsk_mod(dibits, 4800, 2400, 800, fs=fs)
    else:
        n_sym = max(20, int((window_len + 500) / (fs / 2400)) + 5)
        dibits = np.random.randint(0, 4, n_sym)
        return _4fsk_mod(dibits, 2400, 1050, 350, fs=fs)


def synth_atv(*, fs=FS, window_len=WINDOW_LEN):
    """ATV — horizontal sync pulses + luminance sweep at baseband.

    At 12 kHz sample rate, captures the low-frequency structure of analog TV:
    horizontal sync pulses (~15.625 kHz line rate aliased) and slow luminance
    variations.  Real ATV is wideband, but baseband sync structure is
    distinctive enough for classification.
    """
    n = max(window_len * 10, int(0.5 * fs))
    t = np.arange(n) / fs

    # Horizontal sync pulses — period ~64 us (15625 Hz line rate)
    # At 12 kHz fs this aliases, but we simulate the baseband envelope
    line_period = 1.0 / 15625.0
    line_samples = max(1, int(line_period * fs))
    # Sync pulse ~4.7 us → ~0.074 of line period
    sync_frac = 0.074

    signal = np.zeros(n, dtype=np.float64)
    pos = 0
    line_num = 0
    while pos < n:
        sync_len = max(1, int(sync_frac * line_samples))
        line_end = min(pos + line_samples, n)

        # Sync tip (low level, ~0 IRE)
        se = min(pos + sync_len, n)
        signal[pos:se] = 0.0

        # Active video — luminance sweep (simulate image content)
        active_start = se
        if active_start < line_end:
            active_n = line_end - active_start
            # Vary luminance per line to simulate image
            if line_num % 2 == 0:
                lum = np.random.uniform(0.3, 1.0)
                signal[active_start:line_end] = lum
            else:
                # Gradient
                signal[active_start:line_end] = np.linspace(
                    np.random.uniform(0.2, 0.5),
                    np.random.uniform(0.6, 1.0),
                    active_n)

        pos = line_end
        line_num += 1

    # Add vertical blanking interval every ~262 lines (half-frame)
    vbi_period = int(262 * line_samples)
    vbi_len = int(20 * line_samples)  # ~20 lines of VBI
    if vbi_period > 0:
        vbi_pos = 0
        while vbi_pos + vbi_len < n:
            signal[vbi_pos:min(vbi_pos + vbi_len, n)] *= 0.1
            vbi_pos += vbi_period

    # FM-modulate like real ATV
    deviation = np.random.uniform(500, 2000)
    phase = 2 * np.pi * deviation * np.cumsum(signal) / fs
    return np.exp(1j * phase)


def synth_msk144(*, fs=FS, window_len=WINDOW_LEN):
    """MSK144 — GFSK with Costas array, 72 symbols per frame."""
    costas = np.array([0, 1, 3, 2, 4, 6, 5])
    n_frames = np.random.randint(3, 8)
    segments = []
    for _ in range(n_frames):
        data = np.random.randint(0, 8, 72 - len(costas))
        symbols = np.concatenate([costas, data])
        seg = gfsk_mod(symbols, 8, 200.0, 1.0 / 2000.0, fs=fs, bt=0.5)
        gap = np.zeros(int(np.random.uniform(0.1, 0.5) * fs), dtype=np.complex128)
        segments.extend([seg, gap])
    return np.concatenate(segments)


def synth_eas(*, fs=FS, window_len=WINDOW_LEN):
    """EAS/SAME — dual-tone AFSK at 520.83 Hz (mark) and 1562.5 Hz (space)."""
    mark_freq = 520.83
    space_freq = 1562.5
    baud = 520.83  # ~520.83 baud
    n_bits = np.random.randint(200, 600)
    bits = np.random.randint(0, 2, n_bits)
    sps = max(1, int(fs / baud))
    n = n_bits * sps
    t = np.arange(n) / fs
    freq = np.zeros(n)
    for i, b in enumerate(bits):
        freq[i * sps:(i + 1) * sps] = mark_freq if b else space_freq
    phase = 2 * np.pi * np.cumsum(freq) / fs
    # Three header bursts with pauses
    burst = np.exp(1j * phase)
    pause = np.zeros(int(1.0 * fs), dtype=np.complex128)
    return np.concatenate([burst, pause, burst, pause, burst])


def synth_ardop(*, fs=FS, window_len=WINDOW_LEN):
    """ARDOP — OFDM with ARDOP-like carrier structure (variable carriers)."""
    n_carriers = np.random.choice([1, 2, 4, 8, 16])
    carrier_spacing = np.random.uniform(40, 60)
    symbol_dur = np.random.uniform(0.02, 0.05)
    sps = max(1, int(symbol_dur * fs))
    n_symbols = max(20, int((window_len + 500) / sps) + 5)
    # Leader tone (two-tone)
    leader_len = int(np.random.uniform(0.16, 0.24) * fs)
    t_leader = np.arange(leader_len) / fs
    leader = np.exp(2j * np.pi * 1500 * t_leader) + np.exp(2j * np.pi * 500 * t_leader)
    leader = leader / np.abs(leader).max()
    # Data frames
    data = ofdm_carriers(n_carriers, carrier_spacing, symbol_dur, n_symbols, fs=fs)
    return np.concatenate([leader, data])


def synth_bell103(*, fs=FS, window_len=WINDOW_LEN):
    """Bell 103 modem — 2-FSK at 300 baud, originate: 1270/1070 Hz."""
    mark_freq = 1270.0
    space_freq = 1070.0
    baud = 300.0
    n_bits = max(100, int((window_len + 500) / (fs / baud)) + 20)
    bits = np.random.randint(0, 2, n_bits)
    return fsk_mod(bits, 2, abs(mark_freq - space_freq), 1.0 / baud, fs=fs)


def synth_bell202(*, fs=FS, window_len=WINDOW_LEN):
    """Bell 202 modem — 2-FSK at 1200 baud, 1200/2200 Hz tones."""
    baud = 1200.0
    tone_spacing = 1000.0  # 2200 - 1200
    n_bits = max(100, int((window_len + 500) / (fs / baud)) + 20)
    bits = np.random.randint(0, 2, n_bits)
    return fsk_mod(bits, 2, tone_spacing, 1.0 / baud, fs=fs)


def synth_lora(*, fs=FS, window_len=WINDOW_LEN):
    """LoRa — chirp spread spectrum with cyclic frequency shifts.

    At 12 kHz sample rate we capture narrowband LoRa spectral signatures.
    Spreading factors SF7-SF12, variable bandwidth.
    """
    sf = np.random.randint(7, 13)  # SF7-SF12
    n_symbols = 2 ** sf
    bw = np.random.choice([125, 250, 500])  # kHz (nominal)
    # Scale bandwidth to fit within our sample rate
    effective_bw = min(bw, fs * 0.8)

    n_chirps = np.random.randint(8, 30)
    chirp_samples = max(n_symbols, int(0.01 * (2 ** sf) * fs / 1000))
    total_n = chirp_samples * n_chirps

    t = np.arange(total_n) / fs
    sig = np.zeros(total_n, dtype=np.complex128)

    # 8-symbol preamble (unmodulated up-chirps)
    preamble_chirps = min(8, n_chirps)
    for i in range(preamble_chirps):
        start = i * chirp_samples
        end = start + chirp_samples
        t_chirp = np.arange(chirp_samples) / fs
        # Linear up-chirp: freq sweeps from -bw/2 to +bw/2
        k = effective_bw / (chirp_samples / fs)
        freq = -effective_bw / 2 + k * t_chirp
        phase = 2 * np.pi * np.cumsum(freq) / fs
        sig[start:end] = np.exp(1j * phase)

    # Data symbols (cyclically shifted chirps)
    for i in range(preamble_chirps, n_chirps):
        start = i * chirp_samples
        end = min(start + chirp_samples, total_n)
        actual_len = end - start
        if actual_len < 1:
            break
        t_chirp = np.arange(actual_len) / fs
        symbol_val = np.random.randint(0, n_symbols)
        freq_offset = (symbol_val / n_symbols) * effective_bw
        k = effective_bw / (chirp_samples / fs)
        freq = np.mod(-effective_bw / 2 + freq_offset + k * t_chirp,
                       effective_bw) - effective_bw / 2
        phase = 2 * np.pi * np.cumsum(freq) / fs
        sig[start:end] = np.exp(1j * phase)

    return sig


def synth_pocsag(*, fs=FS, window_len=WINDOW_LEN):
    """POCSAG pager — 2-FSK with 576-bit alternating preamble + sync word."""
    baud = np.random.choice([512, 1200, 2400])
    shift = baud / 2.0  # deviation = baud/2 for POCSAG

    # Preamble: 576 bits alternating 1010...
    preamble = np.array([1, 0] * 288)
    # Sync word: 0x7CD215D8 = 32 bits
    sync_bits = np.array([int(b) for b in f"{0x7CD215D8:032b}"])
    # Data: random codewords (32 bits each, multiple batches)
    n_batches = np.random.randint(1, 5)
    data_bits = np.random.randint(0, 2, n_batches * 16 * 32)  # 16 codewords/batch

    bits = np.concatenate([preamble, sync_bits, data_bits])
    return fsk_mod(bits, 2, shift, 1.0 / baud, fs=fs)


def synth_flex(*, fs=FS, window_len=WINDOW_LEN):
    """FLEX pager — 4-FSK at 1600/3200/6400 baud."""
    baud = np.random.choice([1600, 3200, 6400])
    # FLEX uses 4-FSK with +-800 Hz and +-2400 Hz deviation
    dev_inner = 800.0
    dev_outer = 2400.0

    # Sync: known pattern
    sps = max(1, int(fs / baud))
    sync_dibits = np.array([0b01, 0b00, 0b11, 0b10] * 8)
    n_data = max(200, int((window_len * 5) / sps) + 50)
    data_dibits = np.random.randint(0, 4, n_data)
    dibits = np.concatenate([sync_dibits, data_dibits])

    return _4fsk_mod(dibits, baud, dev_outer, dev_inner, fs=fs)


def synth_hdradio(*, fs=FS, window_len=WINDOW_LEN):
    """HD Radio (NRSC-5) — OFDM multicarrier.

    At 12 kHz FS we capture the narrowband OFDM structure.
    HD Radio uses 1093 subcarriers with ~363 Hz spacing.
    """
    n_carriers = np.random.randint(20, 50)
    carrier_spacing = np.random.uniform(50, 100)
    symbol_dur = np.random.uniform(0.002, 0.005)
    n_symbols = max(40, int((window_len + 500) / (symbol_dur * fs)) + 10)
    return ofdm_carriers(n_carriers, carrier_spacing, symbol_dur, n_symbols,
                         fs=fs)


def synth_dtmf(*, fs=FS, window_len=WINDOW_LEN):
    """DTMF — dual-tone multi-frequency keypad tones."""
    lo_freqs = [697, 770, 852, 941]
    hi_freqs = [1209, 1336, 1477, 1633]

    n_digits = np.random.randint(4, 16)
    tone_dur = np.random.uniform(0.05, 0.15)
    gap_dur = np.random.uniform(0.03, 0.08)

    segments = []
    for _ in range(n_digits):
        lo = np.random.choice(lo_freqs)
        hi = np.random.choice(hi_freqs)
        n_tone = max(1, int(tone_dur * fs))
        t = np.arange(n_tone) / fs
        tone = (np.sin(2 * np.pi * lo * t) +
                np.sin(2 * np.pi * hi * t)) / 2.0
        # Convert to analytic signal
        analytic = np.fft.ifft(
            2 * np.fft.fft(tone) * (np.arange(n_tone) < n_tone // 2))
        segments.append(analytic)
        # Inter-digit gap
        n_gap = max(1, int(gap_dur * fs))
        segments.append(np.zeros(n_gap, dtype=np.complex128))

    return np.concatenate(segments)


def synth_drm(*, fs=FS, window_len=WINDOW_LEN):
    """DRM (Digital Radio Mondiale) — OFDM digital broadcast.

    Uses distinct parameters from HD Radio: narrower carriers,
    different symbol duration, QAM constellation.
    """
    # DRM modes: A=B/W 4.5kHz, B=5kHz, C/D=10kHz/20kHz
    n_carriers = np.random.choice([109, 206, 226, 460])
    # Scale to fit our bandwidth
    n_carriers = min(n_carriers, 60)
    carrier_spacing = np.random.uniform(30, 50)
    symbol_dur = np.random.uniform(0.02, 0.04)  # ~26.67ms for DRM
    n_symbols = max(30, int((window_len + 500) / (symbol_dur * fs)) + 10)
    return ofdm_carriers(n_carriers, carrier_spacing, symbol_dur, n_symbols,
                         fs=fs)


# ---------------------------------------------------------------------------
# Synthesizer registry
# ---------------------------------------------------------------------------

SYNTHESIZERS = {
    "FT8": synth_ft8, "FT4": synth_ft4, "WSPR": synth_wspr,
    "JT65": synth_jt65, "JT9": synth_jt9, "JS8": synth_js8,
    "FM": synth_fm, "SSTV": synth_sstv, "NOISE": synth_noise,
    "HELLSCHREIBER": synth_hellschreiber, "MFSK16": synth_mfsk16,
    "MFSK32": synth_mfsk32, "CONTESTIA": synth_contestia,
    "THOR": synth_thor, "PACKET": synth_packet,
    "AM": synth_am, "DOMINOEX": synth_dominoex,
    "CW": synth_cw, "RTTY": synth_rtty,
    "PSK31": synth_psk31, "PSK63": synth_psk63, "SSB": synth_ssb,
    "QPSK": synth_qpsk, "PSK125": synth_psk125, "8PSK": synth_8psk,
    "FSQ": synth_fsq, "IFKP": synth_ifkp, "THROB": synth_throb,
    "FAX": synth_fax, "OLIVIA": synth_olivia, "MT63": synth_mt63,
    "FREEDV": synth_freedv, "M17": synth_m17,
    "DMR": synth_dmr, "DSTAR": synth_dstar,
    "YSF": synth_ysf, "P25": synth_p25, "NXDN": synth_nxdn,
    "MSK144": synth_msk144, "EAS": synth_eas, "ARDOP": synth_ardop,
    "BELL103": synth_bell103, "BELL202": synth_bell202, "ATV": synth_atv,
    "LORA": synth_lora, "POCSAG": synth_pocsag, "FLEX": synth_flex,
    "HDRADIO": synth_hdradio, "DTMF": synth_dtmf, "DRM": synth_drm,
}

SYNTHETIC_CLASSES = list(SYNTHESIZERS.keys())


class SyntheticGenerator(BaseGenerator):
    name = "synthetic"
    required_tools = []
    signal_classes = SYNTHETIC_CLASSES

    def generate_class(self, class_name, rng=None):
        synth_fn = SYNTHESIZERS[class_name]
        segments = []
        target_samples = max(self.window_len * 10,
                             self.samples_per_class * self.window_len // 4)
        total = 0
        while total < target_samples:
            if class_name == "CW":
                seg = synth_fn(fs=self.fs, window_len=self.window_len,
                               wpm_range=self.config.cw_wpm_range)
            else:
                seg = synth_fn(fs=self.fs, window_len=self.window_len)
            if class_name != "NOISE" and np.random.random() < self.config.rsid_probability:
                rsid = synth_rsid(fs=self.fs, window_len=self.window_len)
                gap = np.zeros(int(np.random.uniform(0.05, 0.2) * self.fs),
                               dtype=np.complex128)
                seg = np.concatenate([rsid, gap, seg])
            segments.append(seg)
            total += len(seg)
        return np.concatenate(segments)
