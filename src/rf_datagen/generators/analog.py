"""Analog voice generator — SSB, AM, FM via TTS."""

import os
import shutil
import tempfile

import numpy as np
from scipy.signal import resample

from ..constants import FS, WINDOW_LEN
from ..dsp import hilbert_analytic, audio_to_iq
from ..dsp.filters import bandpass_filter
from ..content.ham_text import gen_speech_text
from ..content.tts import (TTSEngine, apply_ptt_transients, apply_mic_effects,
                            apply_vox_artifacts, apply_tx_audio_clipping,
                            apply_contest_processing)
from ..impairments import extract_windows, apply_impairments, configure_impairments
from ..logging_config import get_logger
from ..output import atomic_save_npy, atomic_write_csv
from .base import BaseGenerator

log = get_logger("analog")

ANALOG_MODES = {
    "SSB": ["USB", "LSB"],
    "AM":  ["AM"],
    "FM":  ["NBFM"],
}


def modulate_ssb(audio, fs, sideband="USB", *, target_fs=FS):
    filtered = bandpass_filter(audio, fs, 300, 3000)
    target_len = int(len(filtered) * target_fs / fs)
    if target_len < 1:
        return np.array([], dtype=np.complex128)
    resampled = resample(filtered, target_len)
    analytic = hilbert_analytic(resampled)
    if sideband == "LSB":
        return np.conj(analytic)
    return analytic


def modulate_am(audio, fs, mod_index=None, *, target_fs=FS):
    if mod_index is None:
        mod_index = np.random.uniform(0.3, 0.9)
    filtered = bandpass_filter(audio, fs, 100, 3000)
    target_len = int(len(filtered) * target_fs / fs)
    if target_len < 1:
        return np.array([], dtype=np.complex128)
    resampled = resample(filtered, target_len)
    peak = np.max(np.abs(resampled))
    if peak > 0:
        resampled /= peak
    envelope = 1.0 + mod_index * resampled
    return envelope.astype(np.complex128)


def modulate_fm(audio, fs, deviation=None, *, target_fs=FS):
    if deviation is None:
        deviation = np.random.uniform(1500, 2500)
    filtered = bandpass_filter(audio, fs, 50, 3000)
    target_len = int(len(filtered) * target_fs / fs)
    if target_len < 1:
        return np.array([], dtype=np.complex128)
    resampled = resample(filtered, target_len)
    peak = np.max(np.abs(resampled))
    if peak > 0:
        resampled /= peak
    phase = 2 * np.pi * deviation * np.cumsum(resampled) / target_fs
    return np.exp(1j * phase)


class AnalogGenerator(BaseGenerator):
    name = "analog"
    required_tools = ["piper"]
    signal_classes = list(ANALOG_MODES.keys())

    def generate_class(self, class_name, rng=None):
        raise NotImplementedError("Use run() for analog generation")

    def run(self, output_dir, seed=42):
        configure_impairments(self.impairment_config)

        parts_dir = os.path.join(output_dir, "parts")
        os.makedirs(parts_dir, exist_ok=True)

        classes = self._resolve_classes()
        results = {}
        if not classes:
            return results

        voice_cache = self.config.voice_cache
        utterances = self.config.utterances_per_class
        tts = TTSEngine(voice_cache)
        stride = self.impairment_config.effective_stride(self.window_len)
        power_threshold = self.impairment_config.window_power_threshold

        tmpdir = tempfile.mkdtemp(prefix="analog_gen_")

        try:
            for mode_name in classes:
                n_samples = self._boosted_count(mode_name)
                npy_path = os.path.join(parts_dir, f"{mode_name}.npy")
                meta_path = os.path.join(parts_dir, f"{mode_name}_meta.csv")
                hash_path = os.path.join(parts_dir, f"{mode_name}.hash")
                cfg_hash = self._config_hash(mode_name, n_samples)

                if self._check_checkpoint(npy_path, meta_path, hash_path,
                                          n_samples, cfg_hash):
                    log.info("%15s: cached", mode_name)
                    results[mode_name] = {"status": "cached",
                                          "samples": n_samples}
                    continue

                variants = ANALOG_MODES[mode_name]
                mode_iq_segments = []

                for i in range(utterances):
                    text, style = gen_speech_text()
                    audio, wav_fs = tts.synthesize(text, tmpdir)
                    if len(audio) < 1000:
                        continue

                    variant = variants[i % len(variants)]

                    if style == "contest":
                        audio = apply_contest_processing(audio, wav_fs)
                        audio = apply_ptt_transients(audio, wav_fs)
                    else:
                        audio = apply_mic_effects(audio, wav_fs)
                        audio = apply_tx_audio_clipping(audio, wav_fs)
                        audio = apply_ptt_transients(audio, wav_fs)
                    if variant in ("USB", "LSB"):
                        audio = apply_vox_artifacts(audio, wav_fs)

                    if variant in ("USB", "LSB"):
                        iq = modulate_ssb(audio, wav_fs, variant, target_fs=self.fs)
                    elif variant == "AM":
                        iq = modulate_am(audio, wav_fs, target_fs=self.fs)
                    elif variant == "NBFM":
                        iq = modulate_fm(audio, wav_fs, target_fs=self.fs)
                    else:
                        continue

                    if len(iq) >= self.window_len:
                        mode_iq_segments.append(iq)

                if not mode_iq_segments:
                    log.warning("%15s: FAILED (no audio)", mode_name)
                    results[mode_name] = {"status": "failed",
                                          "reason": "no audio"}
                    continue

                combined_iq = np.concatenate(mode_iq_segments)
                raw_windows = extract_windows(
                    combined_iq, window_len=self.window_len,
                    stride=stride, power_threshold=power_threshold)
                if len(raw_windows) == 0:
                    log.warning("%15s: FAILED (no valid windows)", mode_name)
                    results[mode_name] = {"status": "failed",
                                          "reason": "no valid windows"}
                    continue

                samples, meta = apply_impairments(
                    raw_windows, n_samples, fs=self.fs,
                    window_len=self.window_len, return_metadata=True)

                atomic_save_npy(npy_path, samples)
                atomic_write_csv(meta_path, ["scenario"],
                                 [[s] for s in meta["scenarios"]])
                self._write_hash(hash_path, cfg_hash)

                log.info("%15s: %d raw -> %d samples",
                         mode_name, len(raw_windows), len(samples))
                results[mode_name] = {"status": "ok",
                                      "samples": len(samples),
                                      "raw_windows": len(raw_windows)}

        finally:
            shutil.rmtree(tmpdir, ignore_errors=True)

        return results
