"""Text-to-speech engine with piper-tts primary and espeak-ng fallback."""

import os
import shutil
import subprocess
import urllib.request
import wave

import numpy as np

from scipy.signal import butter, filtfilt


PIPER_HF_BASE = "https://huggingface.co/rhasspy/piper-voices/resolve/main"

PIPER_VOICES = [
    ("en_US-lessac-medium",
     "Female, American, clear professional",
     "en/en_US/lessac/medium"),
    ("en_US-ryan-medium",
     "Male, American, natural baritone",
     "en/en_US/ryan/medium"),
    ("en_US-amy-medium",
     "Female, American, warm conversational",
     "en/en_US/amy/medium"),
    ("en_US-joe-medium",
     "Male, American, casual",
     "en/en_US/joe/medium"),
    ("en_US-danny-low",
     "Male, American, young adult",
     "en/en_US/danny/low"),
    ("en_US-kathleen-low",
     "Female, American, mature",
     "en/en_US/kathleen/low"),
    ("en_GB-alba-medium",
     "Female, British RP",
     "en/en_GB/alba/medium"),
    ("en_GB-alan-medium",
     "Male, British RP",
     "en/en_GB/alan/medium"),
    ("en_GB-northern_english_male-medium",
     "Male, Northern English accent",
     "en/en_GB/northern_english_male/medium"),
    ("en_GB-cori-medium",
     "Female, British, younger",
     "en/en_GB/cori/medium"),
    ("de_DE-thorsten-medium",
     "Male, German",
     "de/de_DE/thorsten/medium"),
    ("de_DE-kerstin-low",
     "Female, German",
     "de/de_DE/kerstin/low"),
    ("fr_FR-siwis-medium",
     "Female, French",
     "fr/fr_FR/siwis/medium"),
    ("fr_FR-gilles-low",
     "Male, French",
     "fr/fr_FR/gilles/low"),
    ("es_ES-sharvard-medium",
     "Female, Castilian Spanish",
     "es/es_ES/sharvard/medium"),
    ("it_IT-riccardo-x_low",
     "Male, Italian",
     "it/it_IT/riccardo/x_low"),
]

ESPEAK_VOICES = [
    "en-us", "en-us+f3", "en-gb", "en-gb+f4",
    "en-au", "de", "fr", "es", "it", "pt",
]

SPEED_RANGE = (120, 200)


def _download_voice(model_name, hf_subpath, cache_dir):
    """Download a piper voice model (.onnx + .onnx.json) from HuggingFace."""
    os.makedirs(cache_dir, exist_ok=True)
    onnx_path = os.path.join(cache_dir, f"{model_name}.onnx")
    json_path = os.path.join(cache_dir, f"{model_name}.onnx.json")

    if os.path.exists(onnx_path) and os.path.exists(json_path):
        return onnx_path

    for suffix, local_path in [(".onnx", onnx_path), (".onnx.json", json_path)]:
        url = f"{PIPER_HF_BASE}/{hf_subpath}/{model_name}{suffix}"
        try:
            print(f"    Downloading {model_name}{suffix} ...", flush=True)
            urllib.request.urlretrieve(url, local_path)
        except Exception as e:
            print(f"    FAILED: {e}")
            for p in (onnx_path, json_path):
                try:
                    os.remove(p)
                except OSError:
                    pass
            return None

    return onnx_path


def _load_piper_voices(cache_dir):
    """Download all piper voice models and verify the piper binary works."""
    if shutil.which("piper") is None:
        return []

    voices = []
    for model_name, description, hf_subpath in PIPER_VOICES:
        onnx_path = _download_voice(model_name, hf_subpath, cache_dir)
        if onnx_path is None:
            continue
        voices.append((onnx_path, description, model_name))

    return voices


class TTSEngine:
    """Text-to-speech engine with piper CLI primary and espeak-ng fallback."""

    def __init__(self, voice_cache_dir):
        self.piper_voices = []
        self.use_piper = False

        print("  Loading piper-tts voices...")
        self.piper_voices = _load_piper_voices(voice_cache_dir)

        if self.piper_voices:
            self.use_piper = True
            print(f"  Loaded {len(self.piper_voices)} piper voices:")
            for _, desc, name in self.piper_voices:
                print(f"    {name}: {desc}")
        else:
            print("  Piper-TTS unavailable, using espeak-ng fallback")
            if shutil.which("espeak-ng") is None:
                raise RuntimeError(
                    "Neither piper-tts nor espeak-ng available. "
                    "Install via: nix develop")

    def synthesize(self, text, tmpdir="/tmp"):
        """Generate speech audio. Returns (samples_float64, sample_rate)."""
        if self.use_piper:
            return self._synth_piper(text, tmpdir)
        return self._synth_espeak(text, tmpdir)

    def _synth_piper(self, text, tmpdir="/tmp"):
        """Synthesize with a random piper voice via piper CLI binary."""
        onnx_path, _, _ = self.piper_voices[
            np.random.randint(len(self.piper_voices))]

        wav_path = os.path.join(tmpdir, "piper_out.wav")

        try:
            result = subprocess.run(
                ["piper", "-m", onnx_path, "-f", wav_path],
                input=text, capture_output=True, text=True, timeout=30,
            )
            if result.returncode != 0:
                return np.array([]), 0
        except (subprocess.TimeoutExpired, FileNotFoundError):
            return np.array([]), 0

        if not os.path.exists(wav_path) or os.path.getsize(wav_path) == 0:
            return np.array([]), 0

        try:
            with wave.open(wav_path, "rb") as wf:
                nframes = wf.getnframes()
                wav_fs = wf.getframerate()
                raw = np.frombuffer(wf.readframes(nframes), dtype=np.int16)
        except Exception:
            return np.array([]), 0
        finally:
            try:
                os.remove(wav_path)
            except OSError:
                pass

        audio = raw.astype(np.float64) / 32768.0
        return audio, wav_fs

    def _synth_espeak(self, text, tmpdir="/tmp"):
        """Synthesize with espeak-ng (fallback)."""
        voice = np.random.choice(ESPEAK_VOICES)
        speed = np.random.randint(*SPEED_RANGE)
        wav_path = os.path.join(tmpdir, "speech.wav")

        try:
            result = subprocess.run(
                ["espeak-ng", "-v", voice, "-s", str(speed),
                 "-w", wav_path, text],
                capture_output=True, text=True, timeout=30,
            )
        except (subprocess.TimeoutExpired, OSError):
            return np.array([]), 0
        if result.returncode != 0:
            return np.array([]), 0

        if not os.path.exists(wav_path) or os.path.getsize(wav_path) == 0:
            return np.array([]), 0

        try:
            with wave.open(wav_path, "rb") as wf:
                nframes = wf.getnframes()
                wav_fs = wf.getframerate()
                raw = np.frombuffer(wf.readframes(nframes), dtype=np.int16)
        except Exception:
            return np.array([]), 0
        finally:
            try:
                os.remove(wav_path)
            except OSError:
                pass

        audio = raw.astype(np.float64) / 32768.0
        return audio, wav_fs


def apply_ptt_transients(audio, fs):
    """Add PTT click transients to beginning/end of audio (70% of samples)."""
    if np.random.random() > 0.70:
        return audio

    result = audio.copy()

    press_dur = int(np.random.uniform(0.0005, 0.005) * fs)
    press_db = np.random.uniform(-20, -6)
    press_amp = np.max(np.abs(audio) + 1e-10) * 10 ** (press_db / 20)
    press_noise = press_amp * np.random.randn(press_dur)
    result = np.concatenate([press_noise, result])

    release_click_dur = int(np.random.uniform(0.001, 0.005) * fs)
    tail_dur = int(np.random.uniform(0.05, 0.2) * fs)
    release_db = np.random.uniform(-25, -10)
    release_amp = np.max(np.abs(audio) + 1e-10) * 10 ** (release_db / 20)
    click = release_amp * np.random.randn(release_click_dur)
    tail = release_amp * np.random.randn(tail_dur)
    decay = np.exp(-np.arange(tail_dur) / (tail_dur * 0.3))
    tail *= decay
    result = np.concatenate([result, click, tail])

    return result


def apply_mic_effects(audio, fs):
    """Apply realistic microphone effects to speech audio."""
    result = audio.copy()

    gain = np.random.uniform(0.3, 1.5)
    result *= gain

    noise_db = np.random.uniform(-30, -15)
    noise_amp = np.max(np.abs(audio) + 1e-10) * 10 ** (noise_db / 20)
    threshold = 0.05 * np.max(np.abs(audio) + 1e-10)
    active = np.abs(audio) > threshold
    noise = noise_amp * np.random.randn(len(result))
    result[active] += noise[active]

    duration_s = len(audio) / fs
    if duration_s > 3.0:
        n_breaths = np.random.randint(1, 3)
        for _ in range(n_breaths):
            breath_start = np.random.randint(int(fs * 0.5), max(int(fs * 0.5) + 1, len(result) - int(fs * 0.5)))
            breath_dur = int(np.random.uniform(0.2, 0.5) * fs)
            breath_end = min(breath_start + breath_dur, len(result))
            dip_db = np.random.uniform(-12, -6)
            dip_linear = 10 ** (dip_db / 20)
            dip_len = breath_end - breath_start
            if dip_len > 2:
                dip_env = 1.0 - (1.0 - dip_linear) * np.sin(
                    np.pi * np.arange(dip_len) / dip_len)
                result[breath_start:breath_end] *= dip_env

    return result


def apply_vox_artifacts(audio, fs):
    """Apply VOX (voice-operated transmit) artifacts (20% of SSB samples)."""
    if np.random.random() > 0.20:
        return audio

    result = audio.copy()

    threshold = np.max(np.abs(audio) + 1e-10) * np.random.uniform(0.1, 0.3)
    hang_time = int(np.random.uniform(0.3, 0.8) * fs)

    gate = np.zeros(len(result), dtype=bool)
    hang_counter = 0
    for i in range(len(result)):
        if np.abs(result[i]) > threshold:
            gate[i] = True
            hang_counter = hang_time
        elif hang_counter > 0:
            gate[i] = True
            hang_counter -= 1

    result[~gate] = 0.0

    gate_diff = np.diff(gate.astype(int))
    trigger_points = np.where(np.abs(gate_diff) > 0)[0]

    for tp in trigger_points:
        burst_dur = int(np.random.uniform(0.005, 0.020) * fs)
        burst_amp = np.max(np.abs(audio) + 1e-10) * np.random.uniform(0.05, 0.2)
        burst = burst_amp * np.random.randn(burst_dur)
        end = min(tp + burst_dur, len(result))
        result[tp:end] += burst[:end - tp]

    return result


def apply_tx_audio_clipping(audio, fs, force=False, drive_range=(1.5, 4.0)):
    """Apply transmitter audio chain soft clipping."""
    if not force and np.random.random() > 0.15:
        return audio

    result = audio.copy()
    drive = np.random.uniform(*drive_range)

    peak = np.max(np.abs(result)) + 1e-10
    result = result / peak
    result = np.tanh(drive * result) / np.tanh(drive)
    result = result * peak

    return result


def apply_contest_processing(audio, fs):
    """Apply contest-style speech processing chain."""
    result = audio.copy()

    nyq = fs / 2
    hp_norm = min(400 / nyq, 0.99)
    if hp_norm < 0.99:
        b, a = butter(2, hp_norm, btype='high')
        result = filtfilt(b, a, result)

    peak = np.max(np.abs(result)) + 1e-10
    result = result / peak

    ratio = np.random.uniform(6.0, 12.0)
    threshold = np.random.uniform(-12, -6)
    thresh_lin = 10 ** (threshold / 20)

    env = np.abs(result)
    alpha_attack = 1.0 - np.exp(-1.0 / (0.001 * fs))
    alpha_release = 1.0 - np.exp(-1.0 / (0.050 * fs))
    smooth_env = np.zeros_like(env)
    smooth_env[0] = env[0]
    for i in range(1, len(env)):
        if env[i] > smooth_env[i - 1]:
            smooth_env[i] = alpha_attack * env[i] + (1 - alpha_attack) * smooth_env[i - 1]
        else:
            smooth_env[i] = alpha_release * env[i] + (1 - alpha_release) * smooth_env[i - 1]

    gain = np.ones_like(smooth_env)
    above = smooth_env > thresh_lin
    gain[above] = thresh_lin * (smooth_env[above] / thresh_lin) ** (1.0 / ratio) / (smooth_env[above] + 1e-10)
    result = result * gain

    clip_level = 10 ** (-3.0 / 20)
    result = np.clip(result, -clip_level, clip_level)

    bp_low = 300 / nyq
    bp_high = min(2700 / nyq, 0.99)
    if bp_low < bp_high:
        b, a = butter(4, [max(bp_low, 0.001), bp_high], btype='band')
        result = filtfilt(b, a, result)

    result = np.tanh(2.0 * result)
    result = result * peak

    return result
