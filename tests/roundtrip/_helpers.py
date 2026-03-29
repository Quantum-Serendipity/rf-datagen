"""Shared helpers for round-trip validation."""

import csv
import json
import os
import shutil
import sys
import tempfile
import time
import wave

import numpy as np
from scipy.signal import resample

from rf_datagen.constants import FS, WINDOW_LEN


# Whisper model cache directory (downloaded on first use)
WHISPER_MODEL_DIR = os.path.join(
    os.environ.get("XDG_CACHE_HOME", os.path.expanduser("~/.cache")),
    "rf-datagen", "whisper-models")


def write_mono_wav(audio, fs, path):
    """Write float64 audio to a mono WAV file."""
    peak = np.max(np.abs(audio))
    if peak > 0:
        audio = audio / peak * 0.95
    samples = (audio * 32767).astype(np.int16)
    with wave.open(path, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(fs)
        wf.writeframes(samples.tobytes())


def add_awgn_audio(audio, snr_db):
    """Add AWGN to real audio at a given SNR (dB)."""
    rms = np.sqrt(np.mean(audio ** 2))
    if rms < 1e-12:
        return audio
    noise_amp = rms * 10 ** (-snr_db / 20)
    noise = np.random.randn(len(audio)) * noise_amp
    return audio + noise


def add_awgn_iq(iq, snr_db):
    """Add complex AWGN to IQ signal."""
    sig_power = np.mean(np.abs(iq) ** 2)
    if sig_power < 1e-20:
        return iq
    noise_power = sig_power * 10 ** (-snr_db / 10)
    noise = np.sqrt(noise_power / 2) * (
        np.random.randn(len(iq)) + 1j * np.random.randn(len(iq)))
    return iq + noise


def generate_test_speech(tmpdir, duration_s=3.0):
    """Generate a test speech raw file (8kHz S16LE) using tones."""
    raw_path = os.path.join(tmpdir, "test_speech.raw")
    fs = 8000
    t = np.arange(int(fs * duration_s)) / fs
    audio = (0.3 * np.sin(2 * np.pi * 440 * t) +
             0.2 * np.sin(2 * np.pi * 880 * t) +
             0.1 * np.sin(2 * np.pi * 1200 * t))
    audio *= (1 + 0.3 * np.sin(2 * np.pi * 2.0 * t))
    peak = np.max(np.abs(audio))
    if peak > 0:
        audio = audio / peak * 0.8
    raw_s16 = (audio * 32767).astype(np.int16)
    raw_s16.tofile(raw_path)
    return raw_path


def ensure_whisper_model(model_name="tiny.en"):
    """Download whisper model if not cached. Returns model path or None."""
    os.makedirs(WHISPER_MODEL_DIR, exist_ok=True)
    model_path = os.path.join(WHISPER_MODEL_DIR, f"ggml-{model_name}.bin")
    if os.path.exists(model_path):
        return model_path

    download_tool = shutil.which("whisper-cpp-download-ggml-model")
    if not download_tool:
        return None

    try:
        result = subprocess.run(
            [download_tool, model_name],
            capture_output=True, text=True, timeout=120,
            cwd=WHISPER_MODEL_DIR)
        if result.returncode == 0 and os.path.exists(model_path):
            return model_path
        for candidate in [f"ggml-{model_name}.bin", f"{model_name}.bin"]:
            p = os.path.join(WHISPER_MODEL_DIR, candidate)
            if os.path.exists(p):
                return p
    except subprocess.TimeoutExpired:
        pass
    return None


import subprocess


def snr_sweep(name, run_one_trial, trials, snr_levels, verbose=False,
              artifact_dir=None, seed=42):
    """Run decode trials across SNR levels.

    Returns list of dicts with per-SNR results.
    """
    np.random.seed(seed + hash(name) % (2**31))

    results = []
    for snr in snr_levels:
        snr_label = "clean" if snr is None else f"{snr}"
        decodes = 0

        for t in range(trials):
            tmpdir = tempfile.mkdtemp(prefix=f"val_{name}_{snr_label}_t{t}_")
            try:
                ok = run_one_trial(snr, t, tmpdir)
                if ok:
                    decodes += 1
                elif verbose and artifact_dir:
                    save_dir = os.path.join(artifact_dir, name.lower(),
                                            f"snr{snr_label}_t{t}")
                    os.makedirs(save_dir, exist_ok=True)
                    for f in os.listdir(tmpdir):
                        src = os.path.join(tmpdir, f)
                        if os.path.isfile(src):
                            shutil.copy2(src, save_dir)
            except Exception as e:
                if verbose:
                    print(f"    {name} snr={snr_label} trial={t}: {e}",
                          file=sys.stderr)
            finally:
                shutil.rmtree(tmpdir, ignore_errors=True)

        rate = decodes / trials if trials > 0 else 0
        results.append({
            "mode": name,
            "snr_db": snr_label,
            "trials": trials,
            "decodes": decodes,
            "decode_rate": round(rate, 3),
        })
        status = "PASS" if rate > 0.5 else ("MARGINAL" if rate > 0 else "FAIL")
        print(f"  {name:>12s}  SNR={snr_label:>5s}  "
              f"{decodes}/{trials} = {rate:.1%}  [{status}]")

    return results
