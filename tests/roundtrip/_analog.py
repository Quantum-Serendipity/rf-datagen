"""Analog voice validators — correlation and STT."""

import os
import shutil
import subprocess
import wave

import numpy as np
from scipy.signal import resample

from rf_datagen.constants import FS
from rf_datagen.generators.analog import modulate_ssb, modulate_am, modulate_fm

from ._base import BaseRoundtripValidator, register
from ._helpers import (write_mono_wav, add_awgn_audio, add_awgn_iq,
                        ensure_whisper_model)


@register("SSB", "AM", "FM")
class AnalogCorrelationValidator(BaseRoundtripValidator):
    """Validate SSB/AM/FM via modulate -> demodulate -> correlate."""

    required_tools = []
    tier = 2

    def generate_test_audio(self, duration_s=2.0):
        t = np.arange(int(FS * duration_s)) / FS
        audio = (0.4 * np.sin(2 * np.pi * 500 * t) +
                 0.3 * np.sin(2 * np.pi * 1000 * t) +
                 0.2 * np.sin(2 * np.pi * 1500 * t) +
                 0.1 * np.sin(2 * np.pi * 2200 * t))
        audio *= (1 + 0.3 * np.sin(2 * np.pi * 3.0 * t))
        peak = np.max(np.abs(audio))
        if peak > 0:
            audio /= peak
        return audio

    def modulate(self, audio, mode):
        if mode == "SSB":
            return modulate_ssb(audio, FS, sideband="USB")
        elif mode == "AM":
            return modulate_am(audio, FS, mod_index=0.7)
        elif mode == "FM":
            return modulate_fm(audio, FS, deviation=2000)
        return np.array([], dtype=np.complex128)

    def demodulate(self, iq, mode):
        if mode == "SSB":
            return np.real(iq)
        elif mode == "AM":
            envelope = np.abs(iq)
            dc = np.mean(envelope)
            return (envelope - dc) / max(dc, 1e-12)
        elif mode == "FM":
            inst_phase = np.angle(iq[1:] * np.conj(iq[:-1]))
            return inst_phase / np.pi
        return np.array([])

    def correlate(self, original, recovered):
        n = min(len(original), len(recovered))
        if n < 100:
            return 0.0
        a = original[:n]
        b = recovered[:n]
        a = a - np.mean(a)
        b = b - np.mean(b)
        denom = np.sqrt(np.sum(a**2) * np.sum(b**2))
        if denom < 1e-20:
            return 0.0
        return float(np.abs(np.sum(a * b)) / denom)

    def make_trial(self, mode):
        def run(snr_db, trial_idx, tmpdir):
            original = self.generate_test_audio()
            iq = self.modulate(original, mode)
            if len(iq) < 100:
                return False
            if snr_db is not None:
                iq = add_awgn_iq(iq, snr_db)
            recovered = self.demodulate(iq, mode)
            if len(recovered) < 100:
                return False
            corr = self.correlate(original, recovered)
            threshold = 0.7 if snr_db is None else max(0.2, 0.7 - abs(snr_db) * 0.03)
            return corr >= threshold
        return run


@register("SSB_STT", "AM_STT", "FM_STT")
class AnalogSTTValidator(BaseRoundtripValidator):
    """Validate analog voice via TTS -> modulate -> demodulate -> STT."""

    required_tools = ["whisper-cli", "espeak-ng"]
    tier = 1

    KNOWN_PHRASES = [
        "the quick brown fox jumps over the lazy dog",
        "hello radio check how do you copy",
        "the rain in spain falls mainly on the plain",
        "attention please hold for further instructions",
    ]

    def __init__(self):
        self._model_path = None

    def setup(self):
        self._model_path = ensure_whisper_model("tiny.en")
        if self._model_path is None:
            raise RuntimeError("Failed to download whisper model")

    def generate_speech_wav(self, text, tmpdir):
        wav_path = os.path.join(tmpdir, "speech_16k.wav")
        try:
            result = subprocess.run(
                ["espeak-ng", "-v", "en-us", "-s", "140", "-w", wav_path, text],
                capture_output=True, timeout=30)
            if result.returncode != 0 or not os.path.exists(wav_path):
                return None, None
        except (subprocess.TimeoutExpired, FileNotFoundError):
            return None, None
        with wave.open(wav_path, "rb") as wf:
            fs = wf.getframerate()
            raw = np.frombuffer(wf.readframes(wf.getnframes()), dtype=np.int16)
        audio = raw.astype(np.float64) / 32768.0
        return audio, fs

    def modulate(self, audio, fs, mode):
        if mode == "SSB":
            return modulate_ssb(audio, fs, sideband="USB")
        elif mode == "AM":
            return modulate_am(audio, fs, mod_index=0.7)
        elif mode == "FM":
            return modulate_fm(audio, fs, deviation=2000)
        return np.array([], dtype=np.complex128)

    def demodulate_to_wav(self, iq, mode, tmpdir):
        if mode == "SSB":
            audio = np.real(iq)
        elif mode == "AM":
            envelope = np.abs(iq)
            dc = np.mean(envelope)
            audio = (envelope - dc) / max(dc, 1e-12)
        elif mode == "FM":
            audio = np.angle(iq[1:] * np.conj(iq[:-1])) / np.pi
        else:
            return None
        target_len = int(len(audio) * 16000 / FS)
        if target_len < 100:
            return None
        audio_16k = resample(audio, target_len)
        wav_path = os.path.join(tmpdir, "demod_16k.wav")
        write_mono_wav(audio_16k, 16000, wav_path)
        return wav_path

    def transcribe(self, wav_path):
        whisper = shutil.which("whisper-cli")
        if not whisper:
            return None
        try:
            result = subprocess.run(
                [whisper, "-m", self._model_path, "-np", "-nt",
                 "-f", wav_path],
                capture_output=True, text=True, timeout=60)
        except subprocess.TimeoutExpired:
            return None
        return result.stdout.strip()

    def compare(self, transcript, expected_text):
        if not transcript:
            return False
        import re
        clean = lambda s: set(re.sub(r'[^a-z0-9\s]', '', s.lower()).split())
        expected_words = clean(expected_text)
        transcript_words = clean(transcript)
        if not expected_words:
            return False
        matches = 0
        for ew in expected_words:
            if ew in transcript_words:
                matches += 1
            elif len(ew) >= 3 and any(
                    tw.startswith(ew[:3]) or ew.startswith(tw[:3])
                    for tw in transcript_words if len(tw) >= 3):
                matches += 0.5
        return matches >= len(expected_words) * 0.4

    def make_trial(self, mode):
        base_mode = mode.replace("_STT", "")
        def run(snr_db, trial_idx, tmpdir):
            text = self.KNOWN_PHRASES[trial_idx % len(self.KNOWN_PHRASES)]
            audio, fs = self.generate_speech_wav(text, tmpdir)
            if audio is None:
                return False
            iq = self.modulate(audio, fs, base_mode)
            if len(iq) < 100:
                return False
            if snr_db is not None:
                iq = add_awgn_iq(iq, snr_db)
            wav_path = self.demodulate_to_wav(iq, base_mode, tmpdir)
            if wav_path is None:
                return False
            transcript = self.transcribe(wav_path)
            return self.compare(transcript, text)
        return run
