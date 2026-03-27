"""WSJT-X CLI encoder generator — FT8, FT4, WSPR, JT65, JT9."""

import csv
import os
import shutil
import subprocess
import tempfile
import wave

import numpy as np
from scipy.signal import fftconvolve, resample

from ..constants import FS, WINDOW_LEN
from ..dsp import audio_to_iq
from ..content.ham_text import gen_ft8_message, gen_wspr_message
from ..impairments import extract_windows, apply_impairments
from .base import BaseGenerator

SYNTH_FS = 12000

MODE_PARAMS = {
    "FT8":   (79,   8,  6.25,     6.25,     1000.0),
    "FT4":   (103,  4,  20.8333,  20.8333,  1000.0),
    "JT65":  (126, 65,  2.6917,   2.6917,   1000.0),
    "JT9":   (85,   9,  1.7361,   1.7361,   1000.0),
}

_JT65_SYNC = [
    1,0,0,1,1,0,0,0,1,1,1,1,1,1,0,1,0,1,0,0,0,1,0,1,1,0,0,1,0,0,
    0,1,1,1,0,0,0,0,1,1,0,1,1,0,1,1,1,1,0,0,1,0,0,1,0,0,1,0,1,1,
    0,0,1,1,0,1,0,1,0,1,0,0,1,0,0,0,0,0,0,1,1,0,0,0,0,1,1,0,1,0,
    0,1,0,1,0,1,0,0,1,0,0,0,1,0,1,1,0,0,0,1,1,0,1,0,0,0,1,0,1,0,
    1,1,0,0,0,1,
]

MSG_GENERATORS = {
    "FT8": gen_ft8_message,
    "FT4": gen_ft8_message,
    "JT65": gen_ft8_message,
    "JT9": gen_ft8_message,
    "WSPR": gen_wspr_message,
}


def parse_ft8_symbols(stdout):
    lines = stdout.strip().split('\n')
    symbols = []
    capture = False
    for line in lines:
        if 'Channel symbols' in line:
            capture = True
            continue
        if capture:
            stripped = line.strip()
            if stripped and stripped[0].isdigit():
                symbols.extend(int(c) for c in stripped if c.isdigit())
    if len(symbols) >= 79:
        return symbols[:79]
    return None


def parse_jt65_symbols(stdout):
    lines = stdout.strip().split('\n')
    data_symbols = []
    capture = False
    for line in lines:
        if 'Information-carrying channel symbols' in line:
            capture = True
            continue
        if capture:
            nums = line.strip().split()
            if nums and nums[0].isdigit():
                data_symbols.extend(int(x) for x in nums)
    if len(data_symbols) < 63:
        return None
    symbols = []
    data_idx = 0
    for i in range(126):
        if _JT65_SYNC[i] == 1:
            symbols.append(0)
        else:
            if data_idx < len(data_symbols):
                symbols.append(data_symbols[data_idx] + 2)
                data_idx += 1
            else:
                symbols.append(0)
    return symbols


def parse_jt9_symbols(stdout):
    lines = stdout.strip().split('\n')
    symbols = []
    capture = False
    for line in lines:
        if 'Channel symbols' in line:
            capture = True
            continue
        if capture:
            nums = line.strip().split()
            if nums and nums[0].isdigit():
                symbols.extend(int(x) for x in nums)
    if len(symbols) >= 85:
        return symbols[:85]
    return None


def synthesize_gfsk_tones(symbols, tone_spacing, symbol_rate, base_freq,
                          fs=SYNTH_FS, bt=2.0):
    n_sym = len(symbols)
    samples_per_sym = fs / symbol_rate
    total_samples = int(n_sym * samples_per_sym)
    sym_indices = (np.arange(total_samples) * symbol_rate / fs).astype(int)
    sym_indices = np.clip(sym_indices, 0, n_sym - 1)
    freq = np.array([base_freq + symbols[i] * tone_spacing
                     for i in sym_indices], dtype=np.float64)
    if bt < 50:
        sigma = np.sqrt(np.log(2)) / (np.pi * bt * symbol_rate) * fs
        if sigma > 0.5:
            kernel_len = int(6 * sigma) | 1
            kernel = np.exp(-0.5 * (np.arange(kernel_len) - kernel_len // 2) ** 2
                            / sigma ** 2)
            kernel /= kernel.sum()
            freq = fftconvolve(freq, kernel, mode='same')
    phase = 2 * np.pi * np.cumsum(freq) / fs
    return np.cos(phase)


def encode_ft8(message):
    result = subprocess.run(["ft8code", message],
                            capture_output=True, text=True, timeout=10)
    if result.returncode != 0:
        return None
    symbols = parse_ft8_symbols(result.stdout)
    if symbols is None:
        return None
    _, _, sym_rate, tone_sp, base_f = MODE_PARAMS["FT8"]
    return synthesize_gfsk_tones(symbols, tone_sp, sym_rate, base_f, bt=2.0)


def encode_ft4(message):
    result = subprocess.run(["ft8code", message],
                            capture_output=True, text=True, timeout=10)
    if result.returncode != 0:
        return None
    symbols = parse_ft8_symbols(result.stdout)
    if symbols is None:
        return None
    symbols_4 = [s % 4 for s in symbols]
    _, _, sym_rate, tone_sp, base_f = MODE_PARAMS["FT4"]
    return synthesize_gfsk_tones(symbols_4, tone_sp, sym_rate, base_f, bt=1.0)


def encode_jt65(message):
    result = subprocess.run(["jt65code", message],
                            capture_output=True, text=True, timeout=10)
    if result.returncode != 0:
        return None
    symbols = parse_jt65_symbols(result.stdout)
    if symbols is None:
        return None
    _, _, sym_rate, tone_sp, base_f = MODE_PARAMS["JT65"]
    return synthesize_gfsk_tones(symbols, tone_sp, sym_rate, base_f, bt=100.0)


def encode_jt9(message):
    result = subprocess.run(["jt9code", message],
                            capture_output=True, text=True, timeout=10)
    if result.returncode != 0:
        return None
    symbols = parse_jt9_symbols(result.stdout)
    if symbols is None:
        return None
    _, _, sym_rate, tone_sp, base_f = MODE_PARAMS["JT9"]
    return synthesize_gfsk_tones(symbols, tone_sp, sym_rate, base_f, bt=100.0)


def encode_wspr(message, tmpdir):
    base_freq = 1000.0 + np.random.uniform(-50, 50)
    try:
        result = subprocess.run(
            ["fst4sim", message, "120", f"{base_freq:.1f}",
             "0.0", "0.0", "1.0", "1", "0", "T"],
            capture_output=True, text=True, timeout=30, cwd=tmpdir,
        )
        if result.returncode != 0:
            return None
    except subprocess.TimeoutExpired:
        return None
    wav_path = os.path.join(tmpdir, "000000_0001.wav")
    if not os.path.exists(wav_path):
        return None
    try:
        with wave.open(wav_path, "rb") as wf:
            nframes = wf.getnframes()
            raw = np.frombuffer(wf.readframes(nframes), dtype=np.int16)
        audio = raw.astype(np.float64) / 32768.0
        peak = np.max(np.abs(audio))
        if peak > 0:
            audio = audio * (0.5 / peak)
        return audio
    except Exception:
        return None
    finally:
        try:
            os.remove(wav_path)
        except OSError:
            pass


ENCODERS = {
    "FT8": lambda msg, _: encode_ft8(msg),
    "FT4": lambda msg, _: encode_ft4(msg),
    "JT65": lambda msg, _: encode_jt65(msg),
    "JT9": lambda msg, _: encode_jt9(msg),
    "WSPR": encode_wspr,
}


class WsjtxGenerator(BaseGenerator):
    name = "wsjtx"
    required_tools = ["ft8code", "jt65code", "jt9code", "fst4sim"]
    signal_classes = list(ENCODERS.keys())

    def generate_class(self, class_name, rng=None):
        tmpdir = tempfile.mkdtemp(prefix="wsjtx_gen_")
        try:
            encoder = ENCODERS[class_name]
            msg_gen = MSG_GENERATORS[class_name]
            segments = []
            n_messages = self.config.messages_per_mode
            for _ in range(n_messages):
                msg = msg_gen()
                audio = encoder(msg, tmpdir)
                if audio is not None and len(audio) > 0:
                    iq = audio_to_iq(audio, SYNTH_FS, target_fs=self.fs)
                    if len(iq) >= self.window_len:
                        segments.append(iq)
            if not segments:
                return np.array([], dtype=np.complex128)
            return np.concatenate(segments)
        finally:
            shutil.rmtree(tmpdir, ignore_errors=True)
