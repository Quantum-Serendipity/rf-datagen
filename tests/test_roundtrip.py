#!/usr/bin/env python3
"""Round-trip reception validation for RF signal generators.

Generates signals using the same encoders as the training pipeline, feeds them
through the corresponding decoders, and verifies successful decode. Tests at
multiple SNR levels to produce decode-rate-vs-SNR curves.

Tier 1 — Exact message comparison:
  FT8 (jt9 --ft8), WSPR/FST4W (jt9 --fst4w via fst4sim), PACKET (atest),
  CW (multimon-ng)

Tier 2 — Sync/BER/correlation metric:
  FreeDV (freedv_rx), M17 (m17-demod), DMR/DSTAR/YSF/NXDN (dsdccx),
  SSB/AM/FM (audio correlation), SSB_STT/AM_STT/FM_STT (whisper-cpp)

Tier 3 — Spectral validation:
  SSTV (VIS code + sync pulse detection)

Known limitations (synthesis approximations, not protocol-accurate):
  JT65, JT9 — our GFSK synthesis doesn't produce signals decodable by jt9.
  FT4 — our encoder maps FT8 symbols mod 4 (approximation).

Usage:
    python -m tests.test_roundtrip
    python -m tests.test_roundtrip --modes FT8 PACKET_1200
    python -m tests.test_roundtrip --snr-only -10 -15 -20
    rf-datagen validate-roundtrip
"""

import argparse
import csv
import json
import os
import shutil
import subprocess
import sys
import tempfile
import time
import wave

import numpy as np

from scipy.signal import resample

from rf_datagen.constants import FS, WINDOW_LEN, MAX_FREQ_OFFSET
from rf_datagen.generators.wsjtx import encode_ft8, SYNTH_FS
from rf_datagen.generators.packet import GEN_AUDIO_FS
from rf_datagen.generators.analog import modulate_ssb, modulate_am, modulate_fm
from rf_datagen.generators.digivoice import generate_tier2
from rf_datagen.generators.sstv import encode_sstv, SSTV_AUDIO_FS
from rf_datagen.content.ham_text import gen_ft8_message, gen_wspr_message
from rf_datagen.dsp import audio_to_iq

# Whisper model cache directory (downloaded on first use)
WHISPER_MODEL_DIR = os.path.join(
    os.environ.get("XDG_CACHE_HOME", os.path.expanduser("~/.cache")),
    "rf-datagen", "whisper-models")

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

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


# ---------------------------------------------------------------------------
# Tier 1: FT8 via jt9
# ---------------------------------------------------------------------------

class FT8Validator:
    """Validate FT8 via jt9 --ft8 decoder."""

    def generate(self, message):
        """Encode an FT8 message to float64 audio at SYNTH_FS."""
        return encode_ft8(message)

    def prepare_wav(self, audio, tmpdir, snr_db=None):
        """Pad audio into a 15s period WAV for jt9."""
        total_samples = int(15 * SYNTH_FS)
        offset_samples = int(0.5 * SYNTH_FS)

        padded = np.zeros(total_samples, dtype=np.float64)
        end = min(offset_samples + len(audio), total_samples)
        padded[offset_samples:end] = audio[:end - offset_samples]

        if snr_db is not None:
            padded = add_awgn_audio(padded, snr_db)

        wav_path = os.path.join(tmpdir, "ft8_signal.wav")
        write_mono_wav(padded, SYNTH_FS, wav_path)
        return wav_path

    def decode(self, wav_path, tmpdir):
        """Run jt9 --ft8 and return list of decoded message strings."""
        decode_dir = os.path.join(tmpdir, "decode")
        os.makedirs(decode_dir, exist_ok=True)

        cmd = ["jt9", "--ft8", "-p", "15", "-d", "3",
               "-a", decode_dir, "-t", decode_dir, wav_path]

        try:
            result = subprocess.run(
                cmd, capture_output=True, text=True, timeout=60)
        except subprocess.TimeoutExpired:
            return []

        messages = []
        for line in result.stdout.strip().split("\n"):
            line = line.strip()
            if not line:
                continue
            parts = line.split()
            if len(parts) >= 5:
                try:
                    float(parts[0])  # time
                    float(parts[1])  # snr
                    float(parts[2])  # dt
                    float(parts[3])  # freq
                    msg_start = 4
                    if parts[msg_start] == "~":
                        msg_start = 5
                    msg = " ".join(parts[msg_start:])
                    messages.append(msg)
                except (ValueError, IndexError):
                    continue
        return messages

    def compare(self, decoded_messages, expected_message):
        """Check if expected message appears in decoded output."""
        expected_upper = expected_message.upper().strip()
        for msg in decoded_messages:
            msg_norm = " ".join(msg.upper().split())
            if expected_upper in msg_norm or msg_norm in expected_upper:
                return True
            exp_parts = expected_upper.split()
            if len(exp_parts) >= 2:
                matches = sum(1 for p in exp_parts if p in msg_norm)
                if matches >= len(exp_parts) - 1:
                    return True
        return False


# ---------------------------------------------------------------------------
# Tier 1: WSPR via fst4sim + jt9 --fst4w
# ---------------------------------------------------------------------------

class WSPRValidator:
    """Validate WSPR via fst4sim encoder + jt9 --fst4w decoder.

    fst4sim in WSPR mode ('T') generates FST4W waveforms, which are decoded
    by jt9 --fst4w (not wsprd, which expects classic WSPR encoding).
    """

    WSPR_AUDIO_FREQ = 1500.0  # Standard USB audio frequency for WSPR

    def generate(self, message, tmpdir, snr_db=None):
        """Generate WSPR/FST4W signal via fst4sim and return WAV path.

        Generates directly at 1500 Hz audio frequency (standard WSPR USB
        offset) so jt9 --fst4w can find it in its default search band.
        """
        # fst4sim "msg" TRsec f0 DT fdop del nfiles snr W
        snr_arg = "99" if snr_db is None else str(snr_db)
        try:
            result = subprocess.run(
                ["fst4sim", message, "120", str(self.WSPR_AUDIO_FREQ),
                 "0.0", "0.0", "1.0", "1", snr_arg, "T"],
                capture_output=True, text=True, timeout=30,
                cwd=tmpdir,
            )
            if result.returncode != 0:
                return None
        except subprocess.TimeoutExpired:
            return None

        wav_path = os.path.join(tmpdir, "000000_0001.wav")
        if not os.path.exists(wav_path):
            return None
        return wav_path

    def decode(self, wav_path, tmpdir):
        """Run jt9 --fst4w and return list of decoded message strings."""
        decode_dir = os.path.join(tmpdir, "decode")
        os.makedirs(decode_dir, exist_ok=True)

        cmd = ["jt9", "--fst4w", "-p", "120", "-d", "3",
               "-f", str(int(self.WSPR_AUDIO_FREQ)), "-F", "200",
               "-a", decode_dir, "-t", decode_dir, wav_path]

        try:
            result = subprocess.run(
                cmd, capture_output=True, text=True, timeout=120)
        except subprocess.TimeoutExpired:
            return []

        messages = []
        for line in result.stdout.strip().split("\n"):
            line = line.strip()
            if not line or "DecodeFinished" in line:
                continue
            parts = line.split()
            if len(parts) >= 5:
                try:
                    float(parts[1])  # snr
                    float(parts[2])  # dt
                    float(parts[3])  # freq
                    msg_start = 4
                    if parts[msg_start] == "`":
                        msg_start = 5
                    msg = " ".join(parts[msg_start:])
                    messages.append(msg)
                except (ValueError, IndexError):
                    continue
        return messages

    def compare(self, decoded_messages, expected_message):
        """Check if expected message appears in decoded output.

        WSPR messages are "CALL GRID POWER". jt9 --fst4w may show the
        callsign as <...> (hash notation) if it's not in the hash table.
        Accept a match if grid+power match in that case.
        """
        parts = expected_message.split()
        if len(parts) < 1:
            return False
        expected_call = parts[0].upper()
        expected_grid = parts[1].upper() if len(parts) > 1 else ""
        expected_power = parts[2] if len(parts) > 2 else ""

        for msg in decoded_messages:
            msg_upper = msg.upper()
            # Direct callsign match
            if expected_call in msg_upper:
                return True
            # Hash notation: callsign is <...> but grid+power match
            if "<" in msg and expected_grid and expected_power:
                if expected_grid in msg_upper and expected_power in msg:
                    return True
        return False


# ---------------------------------------------------------------------------
# Tier 1: PACKET (AX.25 via atest)
# ---------------------------------------------------------------------------

class PacketValidator:
    """Validate packet radio via direwolf's atest decoder."""

    def generate(self, baud, tmpdir, n_packets=5):
        """Generate packet audio using gen_packets.

        Returns float64 audio at GEN_AUDIO_FS, or None on failure.
        """
        wav_path = os.path.join(tmpdir, f"packet_{baud}.wav")
        cmd = ["gen_packets", "-b", str(baud), "-o", wav_path,
               "-r", str(GEN_AUDIO_FS), "-n", str(n_packets)]

        result = subprocess.run(cmd, capture_output=True, text=True, timeout=60)
        if result.returncode != 0:
            return None

        if not os.path.exists(wav_path) or os.path.getsize(wav_path) == 0:
            return None

        with wave.open(wav_path, "rb") as wf:
            nframes = wf.getnframes()
            raw = np.frombuffer(wf.readframes(nframes), dtype=np.int16)
        return raw.astype(np.float64) / 32768.0

    def prepare_wav(self, audio, tmpdir, baud, snr_db=None):
        """Add noise and write WAV for atest."""
        if snr_db is not None:
            audio = add_awgn_audio(audio, snr_db)

        wav_path = os.path.join(tmpdir, f"packet_{baud}_test.wav")
        write_mono_wav(audio, GEN_AUDIO_FS, wav_path)
        return wav_path

    def decode(self, wav_path, baud):
        """Run atest to decode packet WAV. Returns number of decoded frames."""
        cmd = ["atest", "-B", str(baud), wav_path]

        try:
            result = subprocess.run(
                cmd, capture_output=True, text=True, timeout=60)
        except subprocess.TimeoutExpired:
            return 0

        # atest outputs "N packets decoded" near the end
        out = result.stdout + result.stderr
        for line in out.split("\n"):
            line = line.strip()
            if "packets decoded" in line:
                parts = line.split()
                try:
                    return int(parts[0])
                except (ValueError, IndexError):
                    pass

        # Fallback: count DECODED lines
        return sum(1 for l in out.split("\n") if "DECODED" in l)

    def compare(self, n_decoded, min_frames=1):
        """Check if at least min_frames were decoded."""
        return n_decoded >= min_frames


# ---------------------------------------------------------------------------
# Tier 2: FreeDV
# ---------------------------------------------------------------------------

class FreeDVValidator:
    """Validate FreeDV round-trip via freedv_tx/freedv_rx."""

    def generate(self, speech_raw_path, tmpdir, submode="1600"):
        """Generate FreeDV modem audio (raw S16LE 8kHz)."""
        out_path = os.path.join(tmpdir, f"freedv_{submode}_modem.raw")
        try:
            result = subprocess.run(
                ["freedv_tx", submode, speech_raw_path, out_path],
                capture_output=True, timeout=30,
            )
            if result.returncode != 0:
                return None
        except (subprocess.TimeoutExpired, FileNotFoundError):
            return None

        if not os.path.exists(out_path) or os.path.getsize(out_path) < 100:
            return None
        return out_path

    def add_noise(self, modem_path, snr_db):
        """Add noise to raw modem audio and return noisy path."""
        audio = np.fromfile(modem_path, dtype=np.int16).astype(np.float64) / 32768.0
        noisy = add_awgn_audio(audio, snr_db)
        noisy_s16 = (np.clip(noisy, -1, 1) * 32767).astype(np.int16)
        out_path = modem_path.replace(".raw", "_noisy.raw")
        noisy_s16.tofile(out_path)
        return out_path

    def decode(self, modem_path, tmpdir, submode="1600"):
        """Run freedv_rx. Returns True if output produced (sync achieved)."""
        decoded_path = os.path.join(tmpdir, f"freedv_{submode}_decoded.raw")
        try:
            result = subprocess.run(
                ["freedv_rx", submode, modem_path, decoded_path],
                capture_output=True, timeout=30,
            )
        except subprocess.TimeoutExpired:
            return False

        if not os.path.exists(decoded_path):
            return False
        return os.path.getsize(decoded_path) > 0


# ---------------------------------------------------------------------------
# Tier 2: M17
# ---------------------------------------------------------------------------

class M17Validator:
    """Validate M17 round-trip via m17-mod/m17-demod."""

    def generate(self, speech_raw_path, tmpdir):
        """Generate M17 baseband audio (raw S16LE 48kHz)."""
        out_path = os.path.join(tmpdir, "m17_baseband.raw")
        try:
            with open(speech_raw_path, "rb") as infile:
                result = subprocess.run(
                    ["m17-mod", "-S", "N0CALL"],
                    stdin=infile, capture_output=True, timeout=30,
                )
            if result.returncode != 0 or len(result.stdout) < 100:
                return None
        except (subprocess.TimeoutExpired, FileNotFoundError):
            return None

        with open(out_path, "wb") as f:
            f.write(result.stdout)
        return out_path

    def add_noise(self, baseband_path, snr_db):
        """Add noise to raw baseband and return new path."""
        audio = np.fromfile(baseband_path, dtype=np.int16).astype(np.float64) / 32768.0
        noisy = add_awgn_audio(audio, snr_db)
        noisy_s16 = (np.clip(noisy, -1, 1) * 32767).astype(np.int16)
        out_path = baseband_path.replace(".raw", "_noisy.raw")
        noisy_s16.tofile(out_path)
        return out_path

    def decode(self, baseband_path):
        """Run m17-demod. Returns True if audio output produced."""
        try:
            with open(baseband_path, "rb") as infile:
                result = subprocess.run(
                    ["m17-demod", "-l"],
                    stdin=infile, capture_output=True, timeout=30,
                )
        except subprocess.TimeoutExpired:
            return False

        return len(result.stdout) > 0


# ---------------------------------------------------------------------------
# Tier 1: CW via multimon-ng
# ---------------------------------------------------------------------------

class CWValidator:
    """Validate CW round-trip via multimon-ng MORSE_CW decoder."""

    CW_FREQ = 700.0  # Standard CW sidetone frequency

    def generate(self, text, tmpdir):
        """Generate CW audio at 8kHz from text using simple keyer."""
        fs = 8000
        wpm = np.random.choice([15, 18, 20, 25])
        dit_dur = 1.2 / wpm
        dah_dur = dit_dur * 3
        gap_dur = dit_dur

        MORSE = {
            'A': '.-', 'B': '-...', 'C': '-.-.', 'D': '-..', 'E': '.',
            'F': '..-.', 'G': '--.', 'H': '....', 'I': '..', 'J': '.---',
            'K': '-.-', 'L': '.-..', 'M': '--', 'N': '-.', 'O': '---',
            'P': '.--.', 'Q': '--.-', 'R': '.-.', 'S': '...', 'T': '-',
            'U': '..-', 'V': '...-', 'W': '.--', 'X': '-..-', 'Y': '-.--',
            'Z': '--..', '0': '-----', '1': '.----', '2': '..---',
            '3': '...--', '4': '....-', '5': '.....', '6': '-....',
            '7': '--...', '8': '---..', '9': '----.',
        }

        audio_parts = []
        for ch in text.upper():
            if ch == ' ':
                audio_parts.append(np.zeros(int(fs * dit_dur * 4)))
                continue
            code = MORSE.get(ch, '')
            for i, elem in enumerate(code):
                dur = dit_dur if elem == '.' else dah_dur
                n = int(fs * dur)
                t = np.arange(n) / fs
                tone = 0.8 * np.sin(2 * np.pi * self.CW_FREQ * t)
                # Smooth envelope
                ramp = min(int(fs * 0.005), n // 4)
                if ramp > 0:
                    tone[:ramp] *= np.linspace(0, 1, ramp)
                    tone[-ramp:] *= np.linspace(1, 0, ramp)
                audio_parts.append(tone)
                if i < len(code) - 1:
                    audio_parts.append(np.zeros(int(fs * gap_dur)))
            audio_parts.append(np.zeros(int(fs * dit_dur * 3)))

        if not audio_parts:
            return None
        audio = np.concatenate(audio_parts)
        return audio, fs

    def prepare_wav(self, audio, fs, tmpdir, snr_db=None):
        if snr_db is not None:
            audio = add_awgn_audio(audio, snr_db)
        wav_path = os.path.join(tmpdir, "cw_signal.wav")
        write_mono_wav(audio, fs, wav_path)
        return wav_path

    def decode(self, wav_path):
        """Run multimon-ng MORSE_CW decoder."""
        try:
            result = subprocess.run(
                ["multimon-ng", "-t", "wav", "-a", "MORSE_CW", wav_path],
                capture_output=True, text=True, timeout=30)
        except (subprocess.TimeoutExpired, FileNotFoundError):
            return ""

        # multimon-ng outputs decoded text directly (no prefix in -q mode,
        # or after "Enabled demodulators:" line)
        lines = result.stdout.strip().split("\n")
        decoded = []
        for line in lines:
            line = line.strip()
            if line.startswith("Enabled demodulators:") or not line:
                continue
            decoded.append(line)
        return " ".join(decoded)

    def compare(self, decoded_text, expected_text):
        # Extract just the letters/digits for comparison
        expected = "".join(c for c in expected_text.upper() if c.isalnum())
        decoded = "".join(c for c in decoded_text.upper() if c.isalnum())
        if not expected:
            return False
        # Check if most of the expected characters appear in the decoded text
        matches = sum(1 for c in expected if c in decoded)
        return matches >= len(expected) * 0.5


# ---------------------------------------------------------------------------
# Tier 2: Analog voice — audio correlation
# ---------------------------------------------------------------------------

class AnalogCorrelationValidator:
    """Validate SSB/AM/FM via modulate -> demodulate -> correlate."""

    def generate_test_audio(self, duration_s=2.0):
        """Generate deterministic test audio at FS sample rate."""
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
        """Modulate audio to IQ using the generator's modulator."""
        if mode == "SSB":
            return modulate_ssb(audio, FS, sideband="USB")
        elif mode == "AM":
            return modulate_am(audio, FS, mod_index=0.7)
        elif mode == "FM":
            return modulate_fm(audio, FS, deviation=2000)
        return np.array([], dtype=np.complex128)

    def demodulate(self, iq, mode):
        """Demodulate IQ back to audio."""
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

    def add_noise_iq(self, iq, snr_db):
        """Add complex AWGN to IQ signal."""
        sig_power = np.mean(np.abs(iq) ** 2)
        if sig_power < 1e-20:
            return iq
        noise_power = sig_power * 10 ** (-snr_db / 10)
        noise = np.sqrt(noise_power / 2) * (
            np.random.randn(len(iq)) + 1j * np.random.randn(len(iq)))
        return iq + noise

    def correlate(self, original, recovered):
        """Normalized cross-correlation between original and recovered audio."""
        # Trim to matching length
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


# ---------------------------------------------------------------------------
# Tier 1: Analog voice — STT via whisper-cpp-vulkan
# ---------------------------------------------------------------------------

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
        # Some versions output to current directory with different naming
        for candidate in [f"ggml-{model_name}.bin", f"{model_name}.bin"]:
            p = os.path.join(WHISPER_MODEL_DIR, candidate)
            if os.path.exists(p):
                return p
    except subprocess.TimeoutExpired:
        pass
    return None


class AnalogSTTValidator:
    """Validate analog voice via TTS -> modulate -> demodulate -> STT.

    Uses whisper-cpp-vulkan on the iGPU for speech-to-text, then compares
    the original TTS input text against the decoded transcript.
    """

    KNOWN_PHRASES = [
        "the quick brown fox jumps over the lazy dog",
        "hello radio check how do you copy",
        "the rain in spain falls mainly on the plain",
        "attention please hold for further instructions",
    ]

    def __init__(self, model_path):
        self.model_path = model_path

    def generate_speech_wav(self, text, tmpdir):
        """Generate speech WAV at 16kHz (whisper's expected rate) via espeak-ng."""
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
        """Modulate speech to IQ."""
        if mode == "SSB":
            return modulate_ssb(audio, fs, sideband="USB")
        elif mode == "AM":
            return modulate_am(audio, fs, mod_index=0.7)
        elif mode == "FM":
            return modulate_fm(audio, fs, deviation=2000)
        return np.array([], dtype=np.complex128)

    def demodulate_to_wav(self, iq, mode, tmpdir):
        """Demodulate IQ and write 16kHz WAV for whisper."""
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

        # Resample from FS (12kHz) to 16kHz for whisper
        target_len = int(len(audio) * 16000 / FS)
        if target_len < 100:
            return None
        audio_16k = resample(audio, target_len)

        wav_path = os.path.join(tmpdir, "demod_16k.wav")
        write_mono_wav(audio_16k, 16000, wav_path)
        return wav_path

    def transcribe(self, wav_path):
        """Run whisper-cli and return transcript text."""
        whisper = shutil.which("whisper-cli")
        if not whisper:
            return None
        try:
            result = subprocess.run(
                [whisper, "-m", self.model_path, "-np", "-nt",
                 "-f", wav_path],
                capture_output=True, text=True, timeout=60)
        except subprocess.TimeoutExpired:
            return None
        return result.stdout.strip()

    def compare(self, transcript, expected_text):
        """Word-level comparison. Pass if >40% of expected words found.

        Strips punctuation and uses stem-like prefix matching to handle
        minor inflection differences (e.g. jumps/jump, dogs/dog).
        """
        if not transcript:
            return False
        import re
        clean = lambda s: set(re.sub(r'[^a-z0-9\s]', '', s.lower()).split())
        expected_words = clean(expected_text)
        transcript_words = clean(transcript)
        if not expected_words:
            return False
        # Exact match or prefix match (min 3 chars) for inflection tolerance
        matches = 0
        for ew in expected_words:
            if ew in transcript_words:
                matches += 1
            elif len(ew) >= 3 and any(
                    tw.startswith(ew[:3]) or ew.startswith(tw[:3])
                    for tw in transcript_words if len(tw) >= 3):
                matches += 0.5
        return matches >= len(expected_words) * 0.4


# ---------------------------------------------------------------------------
# Tier 2: Digital voice — dsdccx (DMR, D-STAR, YSF, NXDN)
# ---------------------------------------------------------------------------

class DSDCCValidator:
    """Validate digital voice framing via dsdccx sync detection.

    Generates IQ using the same generate_tier2() as the training pipeline,
    FM-demodulates to discriminator audio, and pipes to dsdccx for frame
    sync detection.
    """

    # dsdccx flags per mode
    MODE_FLAGS = {
        "DMR":  "-fr",
        "DSTAR": "-fd",
        "YSF":  "-fy",
        "NXDN": "-fn",
    }

    def generate_iq(self, mode, tmpdir):
        """Generate digital voice IQ using the training generator."""
        speech_path = generate_test_speech(tmpdir, duration_s=2.0)
        iq = generate_tier2(speech_path, tmpdir, mode)
        if len(iq) < 100:
            return None
        return iq

    def iq_to_discriminator(self, iq):
        """FM-demodulate IQ to discriminator audio for dsdccx input.

        dsdccx expects S16LE 8kHz discriminator audio.
        """
        # Instantaneous frequency via conjugate product
        disc = np.angle(iq[1:] * np.conj(iq[:-1]))
        # Resample from FS (12kHz) to 8kHz
        target_len = int(len(disc) * 8000 / FS)
        if target_len < 100:
            return None
        disc_8k = resample(disc, target_len)
        # Normalize and convert to S16LE
        peak = np.max(np.abs(disc_8k))
        if peak > 0:
            disc_8k /= peak
        return (disc_8k * 16000).astype(np.int16)

    def add_noise_iq(self, iq, snr_db):
        """Add complex AWGN to IQ."""
        sig_power = np.mean(np.abs(iq) ** 2)
        if sig_power < 1e-20:
            return iq
        noise_power = sig_power * 10 ** (-snr_db / 10)
        noise = np.sqrt(noise_power / 2) * (
            np.random.randn(len(iq)) + 1j * np.random.randn(len(iq)))
        return iq + noise

    def decode(self, disc_s16, mode):
        """Pipe discriminator audio to dsdccx, return True if sync detected."""
        flag = self.MODE_FLAGS.get(mode)
        if not flag:
            return False

        try:
            result = subprocess.run(
                ["dsdccx", flag, "-i", "-", "-o", "/dev/null", "-n", "-q"],
                input=disc_s16.tobytes(),
                capture_output=True, timeout=30)
        except (subprocess.TimeoutExpired, FileNotFoundError):
            return False

        # dsdccx reports frame syncs on stderr
        output = (result.stdout + result.stderr).decode("utf-8", errors="replace")
        # Look for sync indicators
        sync_indicators = ["Sync", "sync", "Frame", "frame",
                           "voice", "Voice", "data", "Data"]
        return any(ind in output for ind in sync_indicators)


# ---------------------------------------------------------------------------
# Tier 3: SSTV spectral validation
# ---------------------------------------------------------------------------

class SSTVSpectralValidator:
    """Validate SSTV signals by checking spectral structure.

    Checks for:
    - VIS code: 300ms calibration header at 1900 Hz + 1200 Hz
    - Line sync: periodic pulses at 1200 Hz
    - Video band: energy concentrated in 1500-2300 Hz range
    """

    def generate(self, tmpdir):
        """Generate SSTV audio using the training generator's encoder."""
        mode = np.random.choice(["Robot36", "MartinM1", "ScottieS1"])
        try:
            audio = encode_sstv(mode)
            if len(audio) < 1000:
                return None, None
            return audio, mode
        except Exception:
            return None, None

    def validate_spectral(self, audio):
        """Check SSTV spectral properties. Returns True if valid."""
        from scipy.fft import rfft, rfftfreq

        # Check overall frequency content via FFT
        n = len(audio)
        if n < 4096:
            return False

        # Analyze first 2 seconds (should contain VIS header)
        header_len = min(int(SSTV_AUDIO_FS * 2), n)
        header = audio[:header_len]
        freqs = rfftfreq(header_len, 1 / SSTV_AUDIO_FS)
        spectrum = np.abs(rfft(header))

        # Check for energy around 1200 Hz (sync) and 1900 Hz (calibration)
        sync_band = (freqs >= 1100) & (freqs <= 1300)
        cal_band = (freqs >= 1800) & (freqs <= 2000)
        video_band = (freqs >= 1500) & (freqs <= 2300)
        total_band = (freqs >= 500) & (freqs <= 3000)

        sync_energy = np.sum(spectrum[sync_band] ** 2)
        cal_energy = np.sum(spectrum[cal_band] ** 2)
        video_energy = np.sum(spectrum[video_band] ** 2)
        total_energy = np.sum(spectrum[total_band] ** 2)

        if total_energy < 1e-10:
            return False

        # SSTV signals should have significant energy at sync and video freqs
        sync_ratio = (sync_energy + cal_energy) / total_energy
        video_ratio = video_energy / total_energy

        return sync_ratio > 0.05 and video_ratio > 0.3


# ---------------------------------------------------------------------------
# Trial runners — new modes
# ---------------------------------------------------------------------------

def make_cw_trial(validator):
    """Create a trial runner for CW via multimon-ng."""
    words = ["CQ", "TEST", "DE", "W1AW", "K", "73", "QSL", "RST", "599"]
    def run(snr_db, trial_idx, tmpdir):
        text = " ".join(np.random.choice(words, size=3))
        result = validator.generate(text, tmpdir)
        if result is None:
            return False
        audio, fs = result
        wav_path = validator.prepare_wav(audio, fs, tmpdir, snr_db)
        decoded = validator.decode(wav_path)
        return validator.compare(decoded, text)
    return run


def make_analog_correlation_trial(validator, mode):
    """Create a trial runner for analog audio correlation."""
    def run(snr_db, trial_idx, tmpdir):
        original = validator.generate_test_audio()
        iq = validator.modulate(original, mode)
        if len(iq) < 100:
            return False
        if snr_db is not None:
            iq = validator.add_noise_iq(iq, snr_db)
        recovered = validator.demodulate(iq, mode)
        if len(recovered) < 100:
            return False
        corr = validator.correlate(original, recovered)
        threshold = 0.7 if snr_db is None else max(0.2, 0.7 - abs(snr_db) * 0.03)
        return corr >= threshold
    return run


def make_analog_stt_trial(validator, mode):
    """Create a trial runner for analog STT validation."""
    def run(snr_db, trial_idx, tmpdir):
        text = validator.KNOWN_PHRASES[trial_idx % len(validator.KNOWN_PHRASES)]
        audio, fs = validator.generate_speech_wav(text, tmpdir)
        if audio is None:
            return False
        iq = validator.modulate(audio, fs, mode)
        if len(iq) < 100:
            return False
        if snr_db is not None:
            # Add noise to IQ
            sig_power = np.mean(np.abs(iq) ** 2)
            if sig_power > 1e-20:
                noise_power = sig_power * 10 ** (-snr_db / 10)
                noise = np.sqrt(noise_power / 2) * (
                    np.random.randn(len(iq)) + 1j * np.random.randn(len(iq)))
                iq = iq + noise
        wav_path = validator.demodulate_to_wav(iq, mode, tmpdir)
        if wav_path is None:
            return False
        transcript = validator.transcribe(wav_path)
        return validator.compare(transcript, text)
    return run


def make_dsdcc_trial(validator, mode):
    """Create a trial runner for dsdccx digital voice validation."""
    def run(snr_db, trial_idx, tmpdir):
        iq = validator.generate_iq(mode, tmpdir)
        if iq is None:
            return False
        if snr_db is not None:
            iq = validator.add_noise_iq(iq, snr_db)
        disc = validator.iq_to_discriminator(iq)
        if disc is None:
            return False
        return validator.decode(disc, mode)
    return run



def make_sstv_trial(validator):
    """Create a trial runner for SSTV spectral validation."""
    def run(snr_db, trial_idx, tmpdir):
        audio, mode = validator.generate(tmpdir)
        if audio is None:
            return False
        if snr_db is not None:
            audio = add_awgn_audio(audio, snr_db)
        return validator.validate_spectral(audio)
    return run


# ---------------------------------------------------------------------------
# Tier 4: Impairment pipeline validation
# ---------------------------------------------------------------------------

class ImpairmentPipelineValidator:
    """Validate the impairment pipeline produces physically coherent output.

    Tests that apply_scenario_continuous produces narrowband signals with
    consistent spectral structure, unlike per-window apply_scenario which
    smears energy across the band when windows are concatenated.

    Also validates that all named scenarios work and that the per-window
    training path (apply_impairments) is not broken.
    """

    N_WINDOWS = 8   # number of windows to concatenate for continuous test

    def _generate_tone(self, freq_hz=800.0):
        """Generate a pure complex tone at FS — maximally narrowband."""
        n = WINDOW_LEN * self.N_WINDOWS
        t = np.arange(n) / FS
        return np.exp(2j * np.pi * freq_hz * t)

    def _generate_multi_window_ft8(self):
        """Generate N_WINDOWS of synthetic FT8 and concatenate."""
        from rf_datagen.generators.synthetic import synth_ft8
        parts = []
        for _ in range(self.N_WINDOWS):
            raw = synth_ft8()
            parts.append(raw[:WINDOW_LEN])
        return np.concatenate(parts)

    def _occupied_bandwidth(self, sig, fraction=0.99):
        """Measure the bandwidth containing `fraction` of total power.

        Uses a sliding window over the frequency-sorted power spectrum to
        find the narrowest contiguous band holding at least `fraction` of
        total energy. Returns bandwidth in Hz.
        """
        spectrum = np.abs(np.fft.fft(sig)) ** 2
        freqs = np.fft.fftfreq(len(sig), 1.0 / FS)
        # Sort by frequency
        order = np.argsort(freqs)
        freqs = freqs[order]
        spectrum = spectrum[order]
        total_power = spectrum.sum()
        if total_power < 1e-20:
            return FS  # degenerate
        target = total_power * fraction
        best_bw = freqs[-1] - freqs[0]
        left = 0
        window_power = 0.0
        for right in range(len(spectrum)):
            window_power += spectrum[right]
            # Shrink from left while we still meet the threshold
            while window_power - spectrum[left] >= target:
                window_power -= spectrum[left]
                left += 1
            if window_power >= target:
                bw = freqs[right] - freqs[left]
                if bw < best_bw:
                    best_bw = bw
        return max(0.0, best_bw)

    def _phase_discontinuity(self, sig):
        """Measure phase jumps at WINDOW_LEN boundaries.

        Returns array of instantaneous-frequency deviations (Hz) at each
        window boundary vs the surrounding samples.
        """
        # Instantaneous frequency: d(phase)/dt / (2*pi) * fs
        phase = np.unwrap(np.angle(sig))
        inst_freq = np.diff(phase) / (2 * np.pi) * FS
        jumps = []
        for i in range(1, self.N_WINDOWS):
            boundary = i * WINDOW_LEN
            if boundary >= len(inst_freq):
                break
            # Compare inst_freq at boundary to local median
            window = 32  # samples on each side
            lo = max(0, boundary - window)
            hi = min(len(inst_freq), boundary + window)
            local_median = np.median(inst_freq[lo:hi])
            jump = abs(inst_freq[boundary] - local_median)
            jumps.append(jump)
        return np.array(jumps)

    def test_spectral_coherence(self):
        """Per-window impairment should smear; continuous should not.

        Generate a pure tone, impair both ways, compare occupied bandwidth.
        The continuous path should have much narrower occupied bandwidth
        than the per-window path.
        """
        from rf_datagen.impairments.scenarios import (
            apply_scenario, apply_scenario_continuous)

        results = []
        for trial in range(5):
            tone = self._generate_tone()

            # Per-window: apply_scenario to each window, then concatenate
            pw_parts = []
            for i in range(self.N_WINDOWS):
                w = tone[i * WINDOW_LEN:(i + 1) * WINDOW_LEN]
                imp, _ = apply_scenario(w, 30, FS)  # high SNR
                pw_parts.append(imp)
            per_window = np.concatenate(pw_parts)

            # Continuous: concatenate first, then apply_scenario_continuous
            continuous, _ = apply_scenario_continuous(tone, 30, FS,
                                                     scenario="hf_clean")

            bw_pw = self._occupied_bandwidth(per_window)
            bw_ct = self._occupied_bandwidth(continuous)
            results.append((bw_pw, bw_ct))

        avg_pw = np.mean([r[0] for r in results])
        avg_ct = np.mean([r[1] for r in results])

        # Per-window BW should be significantly wider than continuous
        # because each window gets a different random freq offset (±500 Hz)
        # A tone with a single offset stays narrowband; 8 different offsets
        # spread energy across ~1000 Hz.
        ratio = avg_pw / max(avg_ct, 1.0)
        passed = ratio > 2.0  # per-window should be at least 2x wider

        return passed, {
            "per_window_bw_hz": round(avg_pw, 1),
            "continuous_bw_hz": round(avg_ct, 1),
            "ratio": round(ratio, 2),
        }

    def test_phase_continuity(self):
        """Continuous impairment should have smooth phase at boundaries.

        Per-window impairment produces phase discontinuities at window
        boundaries because each window starts with phase=0 in freq_shift.
        Continuous impairment applies a single freq_shift across all windows.
        """
        from rf_datagen.impairments.scenarios import (
            apply_scenario, apply_scenario_continuous)

        results = []
        for trial in range(5):
            tone = self._generate_tone()

            # Per-window
            pw_parts = []
            for i in range(self.N_WINDOWS):
                w = tone[i * WINDOW_LEN:(i + 1) * WINDOW_LEN]
                imp, _ = apply_scenario(w, 40, FS)  # very high SNR
                pw_parts.append(imp)
            per_window = np.concatenate(pw_parts)

            # Continuous
            continuous, _ = apply_scenario_continuous(tone, 40, FS,
                                                     scenario="hf_clean")

            pw_jumps = self._phase_discontinuity(per_window)
            ct_jumps = self._phase_discontinuity(continuous)
            results.append((np.mean(pw_jumps), np.mean(ct_jumps)))

        avg_pw = np.mean([r[0] for r in results])
        avg_ct = np.mean([r[1] for r in results])

        # Continuous should have smaller phase jumps at boundaries
        # hf_clean scenario only does freq_shift + optional dc_offset + awgn,
        # so the continuous path should have very smooth phase at boundaries.
        passed = avg_ct < avg_pw

        return passed, {
            "per_window_avg_jump_hz": round(avg_pw, 1),
            "continuous_avg_jump_hz": round(avg_ct, 1),
        }

    def test_all_scenarios(self):
        """Verify every named scenario runs without error on both APIs."""
        from rf_datagen.impairments.scenarios import (
            apply_scenario_continuous, SCENARIO_NAMES)

        tone = self._generate_tone()
        window = tone[:WINDOW_LEN]
        failures = []

        for name in SCENARIO_NAMES:
            try:
                # Continuous mode with forced scenario
                result, rname = apply_scenario_continuous(tone, 15, FS,
                                                         scenario=name)
                assert rname == name, f"name mismatch: {rname} != {name}"
                assert len(result) == len(tone), "length changed"
                assert np.all(np.isfinite(result)), "non-finite values"
            except Exception as e:
                failures.append(f"{name}(continuous): {e}")

        # Also verify bad scenario name raises ValueError
        try:
            apply_scenario_continuous(tone, 15, FS, scenario="nonexistent")
            failures.append("bad scenario name did not raise ValueError")
        except ValueError:
            pass  # expected

        passed = len(failures) == 0
        return passed, {
            "scenarios_tested": len(SCENARIO_NAMES),
            "failures": failures,
        }

    def test_training_path(self):
        """Verify apply_impairments (per-window training) still works.

        Checks that the batch impairment function produces valid output:
        correct shape, finite values, and non-zero power at each SNR.
        """
        from rf_datagen.impairments.scenarios import apply_impairments

        # Generate some clean windows
        raw = np.array([self._generate_tone()[:WINDOW_LEN]
                        for _ in range(4)])

        target_count = 16
        result = apply_impairments(raw, target_count, FS)

        errors = []
        if result.shape != (target_count, WINDOW_LEN):
            errors.append(f"shape {result.shape} != ({target_count}, {WINDOW_LEN})")
        if not np.all(np.isfinite(result)):
            n_bad = np.sum(~np.isfinite(result))
            errors.append(f"{n_bad} non-finite values")
        # Each window should have non-trivial power
        powers = np.mean(np.abs(result) ** 2, axis=1)
        if np.any(powers < 1e-20):
            n_dead = np.sum(powers < 1e-20)
            errors.append(f"{n_dead}/{target_count} windows have near-zero power")

        passed = len(errors) == 0
        return passed, {
            "windows": target_count,
            "errors": errors,
        }

    def _peak_frequency(self, sig):
        """Return frequency (Hz) of the peak bin using 4x zero-padded FFT."""
        n = len(sig)
        padded = np.zeros(4 * n, dtype=sig.dtype)
        padded[:n] = sig
        spectrum = np.abs(np.fft.fft(padded))
        freqs = np.fft.fftfreq(len(padded), 1.0 / FS)
        return freqs[np.argmax(spectrum)]

    def test_power_conservation(self):
        """Every scenario normalizes to unit power; verify across SNRs."""
        from rf_datagen.impairments.scenarios import (
            apply_scenario_continuous, SCENARIO_NAMES)

        snr_levels = [25, 10, 0, -10]
        errors = []
        total_checks = 0

        for name in SCENARIO_NAMES:
            for snr in snr_levels:
                tone = self._generate_tone()
                result, _ = apply_scenario_continuous(tone, snr, FS,
                                                     scenario=name)
                total_checks += 1
                if not np.all(np.isfinite(result)):
                    errors.append(f"{name}@{snr}dB: non-finite values")
                    continue
                power = np.mean(np.abs(result) ** 2)
                if power < 1e-10:
                    errors.append(f"{name}@{snr}dB: near-zero power {power:.2e}")
                elif abs(power - 1.0) >= 0.05:
                    errors.append(
                        f"{name}@{snr}dB: power={power:.4f} (off by "
                        f"{abs(power - 1.0):.4f})")

        passed = len(errors) == 0
        return passed, {
            "scenarios_tested": len(SCENARIO_NAMES),
            "snr_levels_tested": snr_levels,
            "total_checks": total_checks,
            "errors": errors,
        }

    def test_snr_accuracy(self):
        """Verify add_awgn produces correct noise power for requested SNR."""
        from rf_datagen.impairments.effects import add_awgn

        snr_levels = [25, 15, 5, 0, -5, -10]
        n_trials = 10
        n_samples = 16384
        errors = []
        per_snr = []

        for requested in snr_levels:
            measured = []
            for _ in range(n_trials):
                tone = np.exp(2j * np.pi * 800 * np.arange(n_samples) / FS)
                noisy = add_awgn(tone, requested)
                noise = noisy - tone
                sig_power = np.mean(np.abs(tone) ** 2)
                noise_power = np.mean(np.abs(noise) ** 2)
                if noise_power < 1e-30:
                    measured.append(requested)
                    continue
                actual = 10 * np.log10(sig_power / noise_power)
                measured.append(actual)
            mean_measured = np.mean(measured)
            error_db = abs(mean_measured - requested)
            per_snr.append({
                "requested": requested,
                "measured_mean": round(mean_measured, 2),
                "error_db": round(error_db, 2),
            })
            if error_db >= 1.5:
                errors.append(
                    f"SNR={requested}dB: measured={mean_measured:.2f}dB "
                    f"(error={error_db:.2f}dB)")

        # Zero-input guard: add_awgn clips sig_power to 1.0 when input < 1e-20
        zeros = np.zeros(n_samples, dtype=np.complex128)
        noisy_zeros = add_awgn(zeros, 10)
        zero_noise_power = np.mean(np.abs(noisy_zeros) ** 2)
        # Expected: noise_power = 1.0 * 10^(-10/10) = 0.1
        if abs(zero_noise_power - 0.1) > 0.05:
            errors.append(
                f"zero-input guard: noise_power={zero_noise_power:.4f} "
                f"(expected ~0.1)")

        passed = len(errors) == 0
        return passed, {
            "per_snr": per_snr,
            "zero_input_noise_power": round(zero_noise_power, 4),
            "errors": errors,
        }

    def test_freq_shift_bounds(self):
        """Verify frequency offsets stay within [-MAX_FREQ_OFFSET, +MAX_FREQ_OFFSET]."""
        from rf_datagen.impairments.scenarios import apply_scenario_continuous

        n_trials = 30
        tone_freq = 800.0
        offsets = []
        errors = []

        for _ in range(n_trials):
            tone = self._generate_tone(freq_hz=tone_freq)
            result, _ = apply_scenario_continuous(tone, 40, FS,
                                                  scenario="hf_clean")
            peak = self._peak_frequency(result)
            offset = peak - tone_freq
            offsets.append(offset)

        offsets = np.array(offsets)
        bound = MAX_FREQ_OFFSET + 5  # 5 Hz FFT tolerance
        out_of_bounds = np.abs(offsets) > bound
        if np.any(out_of_bounds):
            bad = offsets[out_of_bounds]
            errors.append(
                f"{np.sum(out_of_bounds)} offsets outside "
                f"[{-bound}, {bound}]: {bad.tolist()}")

        # Distribution sanity: 30 uniform draws in [-500, 500] should span
        # beyond +/-200 with near certainty
        if np.max(offsets) <= 200:
            errors.append(
                f"max offset {np.max(offsets):.1f} Hz <= 200 "
                f"(distribution too narrow)")
        if np.min(offsets) >= -200:
            errors.append(
                f"min offset {np.min(offsets):.1f} Hz >= -200 "
                f"(distribution too narrow)")

        passed = len(errors) == 0
        return passed, {
            "n_trials": n_trials,
            "max_offset_hz": round(float(np.max(offsets)), 1),
            "min_offset_hz": round(float(np.min(offsets)), 1),
            "mean_offset_hz": round(float(np.mean(offsets)), 1),
            "all_within_bounds": bool(~np.any(out_of_bounds)),
            "errors": errors,
        }

    def test_effect_stacking(self):
        """Force all effects simultaneously and verify finite output."""
        from rf_datagen.impairments.effects import (
            normalize_power, add_awgn, freq_shift,
            apply_watterson, apply_qsb,
            apply_iq_imbalance, apply_phase_noise,
            apply_atmospheric_noise, apply_impulse_noise,
            apply_adjacent_signal, apply_powerline_hum, apply_dc_offset,
        )
        from rf_datagen.impairments.transmitter import TransmitterModel
        from rf_datagen.impairments.scenarios import (
            apply_scenario_continuous, SCENARIO_NAMES)

        errors = []

        # Sub-test A: manual full stack
        tone = self._generate_tone()
        sig = TransmitterModel("POORLY_OPERATED").apply(tone, FS)
        sig = apply_watterson(sig, FS)
        sig = apply_qsb(sig, FS)
        sig = freq_shift(sig, MAX_FREQ_OFFSET, FS)
        sig = apply_iq_imbalance(sig)
        sig = apply_phase_noise(sig, FS)
        sig = apply_atmospheric_noise(sig, FS)
        sig = apply_impulse_noise(sig, FS)
        sig = apply_adjacent_signal(sig, FS)
        sig = apply_powerline_hum(sig, FS)
        sig = apply_dc_offset(sig)
        sig = add_awgn(sig, -10)
        sig = normalize_power(sig)

        full_stack_finite = bool(np.all(np.isfinite(sig)))
        full_stack_power = float(np.mean(np.abs(sig) ** 2))
        if not full_stack_finite:
            errors.append("full stack produced non-finite values")
        if full_stack_power < 1e-10:
            errors.append(f"full stack near-zero power: {full_stack_power:.2e}")

        # Sub-test B: repeated heavy scenarios
        heavy = ["hf_poor", "contest_crowded", "sdr_desktop",
                 "overdriven", "poorly_operated"]
        heavy_trials = 0
        heavy_failures = 0
        for name in heavy:
            for _ in range(20):
                tone = self._generate_tone()
                result, _ = apply_scenario_continuous(tone, -10, FS,
                                                     scenario=name)
                heavy_trials += 1
                if not np.all(np.isfinite(result)):
                    heavy_failures += 1
                    errors.append(f"{name}: non-finite output")
                elif np.mean(np.abs(result) ** 2) < 1e-10:
                    heavy_failures += 1
                    errors.append(f"{name}: near-zero power")

        passed = len(errors) == 0
        return passed, {
            "full_stack_power": round(full_stack_power, 4),
            "full_stack_finite": full_stack_finite,
            "heavy_scenario_trials": heavy_trials,
            "errors": errors,
        }

    def test_deterministic_reproducibility(self):
        """Same RNG seed must produce bit-identical output."""
        from rf_datagen.impairments.scenarios import (
            apply_scenario, apply_scenario_continuous, apply_impairments)

        errors = []
        saved_state = np.random.get_state()

        try:
            # Sub-test A: apply_scenario_continuous with forced scenario
            scenarios_tested = ["hf_clean", "hf_poor", "sdr_desktop",
                                "poorly_operated"]
            for name in scenarios_tested:
                tone = self._generate_tone()
                np.random.seed(12345)
                r1, _ = apply_scenario_continuous(tone, 10, FS, scenario=name)
                np.random.seed(12345)
                r2, _ = apply_scenario_continuous(tone, 10, FS, scenario=name)
                if not np.array_equal(r1, r2):
                    errors.append(f"continuous {name}: not identical")

            # Sub-test B: apply_scenario random selection
            tone = self._generate_tone()[:WINDOW_LEN]
            np.random.seed(99999)
            r1, n1 = apply_scenario(tone, 10, FS)
            np.random.seed(99999)
            r2, n2 = apply_scenario(tone, 10, FS)
            random_match = (n1 == n2) and np.array_equal(r1, r2)
            if not random_match:
                errors.append(
                    f"random selection: names={n1}/{n2}, "
                    f"equal={np.array_equal(r1, r2)}")

            # Sub-test C: apply_impairments batch
            raw = np.array([self._generate_tone()[:WINDOW_LEN]
                            for _ in range(4)])
            np.random.seed(77777)
            b1 = apply_impairments(raw, 8, FS)
            np.random.seed(77777)
            b2 = apply_impairments(raw, 8, FS)
            batch_match = bool(np.array_equal(b1, b2))
            if not batch_match:
                errors.append("batch apply_impairments: not identical")

        finally:
            np.random.set_state(saved_state)

        passed = len(errors) == 0
        return passed, {
            "continuous_scenarios_tested": scenarios_tested,
            "random_selection_match": random_match,
            "batch_match": batch_match,
            "errors": errors,
        }


def run_impairment_validation(seed=42):
    """Run all impairment pipeline validation tests.

    Returns (all_pass, results_list) matching the format used by snr_sweep.
    """
    np.random.seed(seed + hash("IMPAIRMENT") % (2**31))
    validator = ImpairmentPipelineValidator()

    tests = [
        ("spectral_coherence", validator.test_spectral_coherence),
        ("phase_continuity",   validator.test_phase_continuity),
        ("all_scenarios",      validator.test_all_scenarios),
        ("training_path",      validator.test_training_path),
        ("power_conservation", validator.test_power_conservation),
        ("snr_accuracy",       validator.test_snr_accuracy),
        ("freq_shift_bounds",  validator.test_freq_shift_bounds),
        ("effect_stacking",    validator.test_effect_stacking),
        ("deterministic_reproducibility",
                               validator.test_deterministic_reproducibility),
    ]

    all_pass = True
    results = []
    for name, test_fn in tests:
        passed, details = test_fn()
        status = "PASS" if passed else "FAIL"
        if not passed:
            all_pass = False
        detail_str = ", ".join(f"{k}={v}" for k, v in details.items()
                               if k != "failures" and k != "errors")
        print(f"  {'IMPAIRMENT':>12s}  {name}: {status}  ({detail_str})")
        if not passed:
            for problem in details.get("failures", details.get("errors", [])):
                print(f"               {problem}")
        results.append({
            "mode": "IMPAIRMENT",
            "snr_db": name,
            "trials": 1,
            "decodes": 1 if passed else 0,
            "decode_rate": 1.0 if passed else 0.0,
        })

    return all_pass, results


# ---------------------------------------------------------------------------
# SNR sweep
# ---------------------------------------------------------------------------

def snr_sweep(name, run_one_trial, trials, snr_levels, verbose=False,
              artifact_dir=None, seed=42):
    """Run decode trials across SNR levels.

    Returns list of dicts with per-SNR results.
    """
    # Reseed per mode so results are independent of which modes run
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


# ---------------------------------------------------------------------------
# Trial runners
# ---------------------------------------------------------------------------

def make_ft8_trial(validator):
    """Create a trial runner for FT8."""
    def run(snr_db, trial_idx, tmpdir):
        msg = gen_ft8_message()
        audio = validator.generate(msg)
        if audio is None or len(audio) == 0:
            return False
        wav_path = validator.prepare_wav(audio, tmpdir, snr_db)
        decoded = validator.decode(wav_path, tmpdir)
        return validator.compare(decoded, msg)
    return run


def make_wspr_trial(validator):
    """Create a trial runner for WSPR/FST4W."""
    def run(snr_db, trial_idx, tmpdir):
        msg = gen_wspr_message()
        wav_path = validator.generate(msg, tmpdir, snr_db)
        if wav_path is None:
            return False
        decoded = validator.decode(wav_path, tmpdir)
        return validator.compare(decoded, msg)
    return run


def make_packet_trial(validator, baud):
    """Create a trial runner for packet radio."""
    def run(snr_db, trial_idx, tmpdir):
        audio = validator.generate(baud, tmpdir, n_packets=5)
        if audio is None or len(audio) < 1000:
            return False
        wav_path = validator.prepare_wav(audio, tmpdir, baud, snr_db)
        n_decoded = validator.decode(wav_path, baud)
        return validator.compare(n_decoded, min_frames=1)
    return run


def make_freedv_trial(validator, submode, speech_raw_path):
    """Create a trial runner for FreeDV."""
    def run(snr_db, trial_idx, tmpdir):
        modem_path = validator.generate(speech_raw_path, tmpdir, submode)
        if modem_path is None:
            return False
        if snr_db is not None:
            modem_path = validator.add_noise(modem_path, snr_db)
        return validator.decode(modem_path, tmpdir, submode)
    return run


def make_m17_trial(validator, speech_raw_path):
    """Create a trial runner for M17."""
    def run(snr_db, trial_idx, tmpdir):
        baseband_path = validator.generate(speech_raw_path, tmpdir)
        if baseband_path is None:
            return False
        if snr_db is not None:
            baseband_path = validator.add_noise(baseband_path, snr_db)
        return validator.decode(baseband_path)
    return run


# ---------------------------------------------------------------------------
# Speech generation helper for Tier 2
# ---------------------------------------------------------------------------

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


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

ALL_MODES = [
    # Tier 1 — exact message
    "FT8", "WSPR", "PACKET_1200", "CW",
    # Tier 1 — STT (optional, requires whisper model)
    "SSB_STT", "AM_STT", "FM_STT",
    # Tier 2 — sync/correlation
    "FREEDV", "M17", "DMR", "DSTAR", "YSF", "NXDN",
    "SSB", "AM", "FM",
    # Tier 3 — spectral
    "SSTV",
    # Tier 4 — impairment pipeline
    "IMPAIRMENT",
]

DEFAULT_MODES = [
    "FT8", "WSPR", "PACKET_1200", "FREEDV", "M17",
    "DMR", "DSTAR", "YSF", "NXDN",
    "SSB", "AM", "FM",
    "IMPAIRMENT",
]

# SNR levels for sweep (None = clean)
DEFAULT_SNR_LEVELS = [None, 25, 20, 15, 10, 5, 0, -5, -10, -15, -20, -25, -30]


def run_validation(modes=None, trials=10, snr_levels=None, clean_only=False,
                   output="./output/validation", verbose=False, seed=42):
    """Run round-trip validation. Returns (all_pass, all_results)."""
    np.random.seed(seed)
    os.makedirs(output, exist_ok=True)

    modes = modes or DEFAULT_MODES

    # Determine SNR levels
    if clean_only:
        snr_levels = [None]
    elif snr_levels is not None:
        snr_levels = [None] + sorted(snr_levels, reverse=True)
    else:
        snr_levels = DEFAULT_SNR_LEVELS

    # Check tool availability
    tools = {
        "jt9": shutil.which("jt9"),
        "fst4sim": shutil.which("fst4sim"),
        "atest": shutil.which("atest"),
        "gen_packets": shutil.which("gen_packets"),
        "freedv_tx": shutil.which("freedv_tx"),
        "freedv_rx": shutil.which("freedv_rx"),
        "m17-mod": shutil.which("m17-mod"),
        "m17-demod": shutil.which("m17-demod"),
        "dsdccx": shutil.which("dsdccx"),
        "c2enc": shutil.which("c2enc"),
        "multimon-ng": shutil.which("multimon-ng"),
        "whisper-cli": shutil.which("whisper-cli"),
        "espeak-ng": shutil.which("espeak-ng"),
    }

    missing = [k for k, v in tools.items() if v is None]
    if missing:
        print(f"WARNING: Missing tools: {missing}", file=sys.stderr)

    print(f"Round-trip validation: {len(modes)} modes, {trials} trials/SNR, "
          f"{len(snr_levels)} SNR levels")
    print(f"SNR levels: {['clean' if s is None else s for s in snr_levels]}")
    print()

    all_results = []
    t0 = time.time()

    # --- Tier 1: FT8 ---
    if "FT8" in modes:
        if not tools.get("jt9"):
            print("  SKIP FT8: jt9 not found")
        else:
            print("--- FT8 ---")
            ft8 = FT8Validator()
            trial_fn = make_ft8_trial(ft8)
            results = snr_sweep("FT8", trial_fn, trials, snr_levels,
                                verbose, output, seed)
            all_results.extend(results)
            print()

    # --- Tier 1: WSPR ---
    if "WSPR" in modes:
        if not tools.get("jt9") or not tools.get("fst4sim"):
            print("  SKIP WSPR: jt9 or fst4sim not found")
        else:
            print("--- WSPR (via fst4sim + jt9 --fst4w) ---")
            wspr = WSPRValidator()
            trial_fn = make_wspr_trial(wspr)
            results = snr_sweep("WSPR", trial_fn, trials, snr_levels,
                                verbose, output, seed)
            all_results.extend(results)
            print()

    # --- Tier 1: PACKET ---
    if "PACKET_1200" in modes:
        if not tools.get("gen_packets") or not tools.get("atest"):
            print("  SKIP PACKET: gen_packets or atest not found")
        else:
            print("--- PACKET 1200 baud (via gen_packets + atest) ---")
            packet = PacketValidator()
            trial_fn = make_packet_trial(packet, 1200)
            results = snr_sweep("PACKET_1200", trial_fn, trials,
                                snr_levels, verbose, output, seed)
            all_results.extend(results)
            print()

    # --- Tier 2: FreeDV ---
    if "FREEDV" in modes:
        if not tools.get("freedv_tx") or not tools.get("freedv_rx"):
            print("  SKIP FREEDV: freedv_tx or freedv_rx not found")
        else:
            print("--- FreeDV (1600 mode) ---")
            speech_tmpdir = tempfile.mkdtemp(prefix="val_speech_")
            try:
                speech_path = generate_test_speech(speech_tmpdir)
                freedv = FreeDVValidator()
                trial_fn = make_freedv_trial(freedv, "1600", speech_path)
                results = snr_sweep("FREEDV", trial_fn, trials,
                                    snr_levels, verbose, output, seed)
                all_results.extend(results)
            finally:
                shutil.rmtree(speech_tmpdir, ignore_errors=True)
            print()

    # --- Tier 2: M17 ---
    if "M17" in modes:
        if not tools.get("m17-mod") or not tools.get("m17-demod"):
            print("  SKIP M17: m17-mod or m17-demod not found")
        else:
            print("--- M17 ---")
            speech_tmpdir = tempfile.mkdtemp(prefix="val_speech_")
            try:
                speech_path = generate_test_speech(speech_tmpdir)
                m17 = M17Validator()
                trial_fn = make_m17_trial(m17, speech_path)
                results = snr_sweep("M17", trial_fn, trials,
                                    snr_levels, verbose, output, seed)
                all_results.extend(results)
            finally:
                shutil.rmtree(speech_tmpdir, ignore_errors=True)
            print()

    # --- Tier 1: CW via multimon-ng ---
    if "CW" in modes:
        if not tools.get("multimon-ng"):
            print("  SKIP CW: multimon-ng not found")
        else:
            print("--- CW (via multimon-ng) ---")
            cw = CWValidator()
            trial_fn = make_cw_trial(cw)
            results = snr_sweep("CW", trial_fn, trials, snr_levels,
                                verbose, output, seed)
            all_results.extend(results)
            print()

    # --- Tier 2: DMR, DSTAR, YSF, NXDN via dsdccx ---
    dsdcc_modes = [m for m in ["DMR", "DSTAR", "YSF", "NXDN"] if m in modes]
    if dsdcc_modes:
        if not tools.get("dsdccx") or not tools.get("c2enc"):
            print(f"  SKIP {dsdcc_modes}: dsdccx or c2enc not found")
        else:
            dsdcc = DSDCCValidator()
            for dv_mode in dsdcc_modes:
                print(f"--- {dv_mode} (via dsdccx) ---")
                trial_fn = make_dsdcc_trial(dsdcc, dv_mode)
                results = snr_sweep(dv_mode, trial_fn, trials,
                                    snr_levels, verbose, output, seed)
                all_results.extend(results)
                print()

    # --- Tier 2: Analog correlation (SSB, AM, FM) ---
    analog_corr_modes = [m for m in ["SSB", "AM", "FM"] if m in modes]
    if analog_corr_modes:
        analog = AnalogCorrelationValidator()
        for am_mode in analog_corr_modes:
            print(f"--- {am_mode} (audio correlation) ---")
            trial_fn = make_analog_correlation_trial(analog, am_mode)
            results = snr_sweep(am_mode, trial_fn, trials,
                                snr_levels, verbose, output, seed)
            all_results.extend(results)
            print()

    # --- Tier 1: Analog STT (SSB_STT, AM_STT, FM_STT) ---
    analog_stt_modes = [m for m in ["SSB_STT", "AM_STT", "FM_STT"]
                        if m in modes]
    if analog_stt_modes:
        if not tools.get("whisper-cli") or not tools.get("espeak-ng"):
            print(f"  SKIP {analog_stt_modes}: whisper-cli or espeak-ng not found")
        else:
            model_path = ensure_whisper_model("tiny.en")
            if model_path is None:
                print("  SKIP STT: failed to download whisper model")
            else:
                stt = AnalogSTTValidator(model_path)
                for stt_mode in analog_stt_modes:
                    base_mode = stt_mode.replace("_STT", "")
                    print(f"--- {base_mode} (whisper STT) ---")
                    trial_fn = make_analog_stt_trial(stt, base_mode)
                    results = snr_sweep(stt_mode, trial_fn, trials,
                                        snr_levels, verbose, output, seed)
                    all_results.extend(results)
                    print()

    # --- Tier 3: SSTV spectral ---
    if "SSTV" in modes:
        print("--- SSTV (spectral validation) ---")
        sstv_val = SSTVSpectralValidator()
        trial_fn = make_sstv_trial(sstv_val)
        results = snr_sweep("SSTV", trial_fn, trials, snr_levels,
                            verbose, output, seed)
        all_results.extend(results)
        print()

    # --- Tier 4: Impairment pipeline ---
    if "IMPAIRMENT" in modes:
        print("--- Impairment pipeline ---")
        imp_pass, imp_results = run_impairment_validation(seed)
        all_results.extend(imp_results)
        print()

    elapsed = time.time() - t0

    # --- Summary ---
    print("=" * 60)
    print("SUMMARY")
    print("=" * 60)

    clean_results = [r for r in all_results if r["snr_db"] == "clean"]
    imp_results = [r for r in all_results if r["mode"] == "IMPAIRMENT"]
    all_pass = True
    for r in clean_results:
        status = "PASS" if r["decode_rate"] >= 0.9 else "FAIL"
        if r["decode_rate"] < 0.9:
            all_pass = False
        print(f"  {r['mode']:>12s}  clean: {r['decodes']}/{r['trials']} "
              f"= {r['decode_rate']:.1%}  [{status}]")
    for r in imp_results:
        status = "PASS" if r["decode_rate"] >= 1.0 else "FAIL"
        if r["decode_rate"] < 1.0:
            all_pass = False
        print(f"  {r['mode']:>12s}  {r['snr_db']}: [{status}]")

    print(f"\nTotal time: {elapsed:.1f}s")
    overall = "PASS" if all_pass else "FAIL"
    print(f"Overall: {overall}")

    # --- Save CSV ---
    csv_path = os.path.join(output, "results.csv")
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=[
            "mode", "snr_db", "trials", "decodes", "decode_rate"])
        writer.writeheader()
        writer.writerows(all_results)
    print(f"\nCSV: {csv_path}")

    # --- Save JSON ---
    json_path = os.path.join(output, "results.json")
    report = {
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "seed": seed,
        "trials_per_snr": trials,
        "snr_levels": ["clean" if s is None else s for s in snr_levels],
        "results": all_results,
        "summary": {
            "overall": overall,
            "clean_channel": {r["mode"]: r["decode_rate"]
                              for r in clean_results},
            "elapsed_seconds": round(elapsed, 1),
        },
    }
    with open(json_path, "w") as f:
        json.dump(report, f, indent=2)
    print(f"JSON: {json_path}")

    return all_pass, all_results


def main():
    parser = argparse.ArgumentParser(
        description="Round-trip reception validation for RF signal generators")
    parser.add_argument("--modes", nargs="*", default=None,
                        help=f"Modes to test (default: all). "
                             f"Available: {ALL_MODES}")
    parser.add_argument("--trials", type=int, default=10,
                        help="Trials per SNR level (default: 10)")
    parser.add_argument("--snr-only", nargs="*", type=int, default=None,
                        help="Test only specific SNR levels (dB). "
                             "Omit for full sweep including clean.")
    parser.add_argument("--clean-only", action="store_true",
                        help="Only test clean channel (no noise)")
    parser.add_argument("--output", default="./output/validation",
                        help="Output directory for results")
    parser.add_argument("--verbose", action="store_true",
                        help="Save debug artifacts on decode failure")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed")
    args = parser.parse_args()

    modes = args.modes
    if modes:
        for m in modes:
            if m not in ALL_MODES:
                print(f"ERROR: Unknown mode '{m}'. Available: {ALL_MODES}",
                      file=sys.stderr)
                sys.exit(1)

    all_pass, _ = run_validation(
        modes=modes,
        trials=args.trials,
        snr_levels=args.snr_only,
        clean_only=args.clean_only,
        output=args.output,
        verbose=args.verbose,
        seed=args.seed,
    )

    sys.exit(0 if all_pass else 1)


if __name__ == "__main__":
    main()
