"""Optional sdr-library wrappers for Numba-accelerated modulation.

Provides drop-in replacements for modulation.py functions using the `sdr`
PyPI package. Only used when `sdr` is installed and benchmarks show >2x
speedup. Falls back to the NumPy implementations otherwise.

Usage:
    from rf_datagen.dsp.modulation_sdr import get_fast_modulator
    fsk = get_fast_modulator("fsk_mod")  # returns sdr version or None
    if fsk is not None:
        sig = fsk(symbols, num_tones, tone_spacing, symbol_dur, fs)
"""

import time
import numpy as np

from ..constants import FS

_SDR_AVAILABLE = False
try:
    import sdr as _sdr
    _SDR_AVAILABLE = True
except ImportError:
    _sdr = None


def sdr_available():
    """Return True if the sdr library is importable."""
    return _SDR_AVAILABLE


def fsk_mod_sdr(symbols, num_tones, tone_spacing, symbol_dur, fs=FS):
    """FSK modulation using sdr library (Numba-accelerated)."""
    if not _SDR_AVAILABLE:
        return None
    sps = max(1, int(symbol_dur * fs))
    freq_deviation = tone_spacing * (num_tones - 1) / 2.0
    modem = _sdr.FSK(order=num_tones, sps=sps, freq_deviation=freq_deviation)
    symbols_arr = np.asarray(symbols, dtype=int)
    sig = modem.modulate(symbols_arr)
    return sig.astype(np.complex128)


def psk_mod_sdr(phase_bits, baud, fs=FS, order=2):
    """PSK modulation using sdr library (Numba-accelerated)."""
    if not _SDR_AVAILABLE:
        return None
    sps = max(1, int(fs / baud))
    modem = _sdr.PSK(order=order, sps=sps)
    symbols_arr = np.asarray(phase_bits, dtype=int)
    sig = modem.modulate(symbols_arr)
    return sig.astype(np.complex128)


_FAST_MODULATORS = {
    "fsk_mod": fsk_mod_sdr,
    "psk_mod": psk_mod_sdr,
}


def get_fast_modulator(name):
    """Return the sdr-accelerated version of a modulator, or None."""
    if not _SDR_AVAILABLE:
        return None
    return _FAST_MODULATORS.get(name)


def benchmark(n_trials=20, n_symbols=500, fs=FS):
    """Benchmark sdr vs NumPy modulation and return speedup ratios.

    Returns dict of {func_name: {"numpy_ms": float, "sdr_ms": float,
                                  "speedup": float}} or None if sdr unavailable.
    """
    if not _SDR_AVAILABLE:
        return None

    from .modulation import fsk_mod, psk_mod

    results = {}

    # FSK benchmark
    symbols = np.random.randint(0, 4, n_symbols)
    tone_spacing = 6.25
    symbol_dur = 0.16

    # warmup
    fsk_mod(symbols, 4, tone_spacing, symbol_dur, fs)
    fsk_mod_sdr(symbols, 4, tone_spacing, symbol_dur, fs)

    times_np = []
    for _ in range(n_trials):
        t0 = time.perf_counter()
        fsk_mod(symbols, 4, tone_spacing, symbol_dur, fs)
        times_np.append(time.perf_counter() - t0)

    times_sdr = []
    for _ in range(n_trials):
        t0 = time.perf_counter()
        fsk_mod_sdr(symbols, 4, tone_spacing, symbol_dur, fs)
        times_sdr.append(time.perf_counter() - t0)

    np_ms = np.median(times_np) * 1000
    sdr_ms = np.median(times_sdr) * 1000
    results["fsk_mod"] = {
        "numpy_ms": round(np_ms, 3),
        "sdr_ms": round(sdr_ms, 3),
        "speedup": round(np_ms / sdr_ms, 2) if sdr_ms > 0 else 0,
    }

    # PSK benchmark
    bits = np.random.randint(0, 2, n_symbols)
    baud = 31.25

    psk_mod(bits, baud, fs, order=2)
    psk_mod_sdr(bits, baud, fs, order=2)

    times_np = []
    for _ in range(n_trials):
        t0 = time.perf_counter()
        psk_mod(bits, baud, fs, order=2)
        times_np.append(time.perf_counter() - t0)

    times_sdr = []
    for _ in range(n_trials):
        t0 = time.perf_counter()
        psk_mod_sdr(bits, baud, fs, order=2)
        times_sdr.append(time.perf_counter() - t0)

    np_ms = np.median(times_np) * 1000
    sdr_ms = np.median(times_sdr) * 1000
    results["psk_mod"] = {
        "numpy_ms": round(np_ms, 3),
        "sdr_ms": round(sdr_ms, 3),
        "speedup": round(np_ms / sdr_ms, 2) if sdr_ms > 0 else 0,
    }

    return results
