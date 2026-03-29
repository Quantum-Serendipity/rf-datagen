"""Multimon-ng validators — DTMF, POCSAG, EAS, FLEX, ACARS."""

import os
import subprocess

import numpy as np

from ._base import BaseRoundtripValidator, register
from ._helpers import write_mono_wav, add_awgn_audio


# ---------------------------------------------------------------------------
# DTMF
# ---------------------------------------------------------------------------

_DTMF_FREQS = {
    '1': (697, 1209), '2': (697, 1336), '3': (697, 1477),
    '4': (770, 1209), '5': (770, 1336), '6': (770, 1477),
    '7': (852, 1209), '8': (852, 1336), '9': (852, 1477),
    '*': (941, 1209), '0': (941, 1336), '#': (941, 1477),
    'A': (697, 1633), 'B': (770, 1633), 'C': (852, 1633), 'D': (941, 1633),
}


def _generate_dtmf_audio(digits, fs=8000, tone_dur=0.1, gap_dur=0.05):
    """Generate DTMF tone pairs for a digit sequence."""
    parts = []
    for d in digits:
        if d not in _DTMF_FREQS:
            continue
        f1, f2 = _DTMF_FREQS[d]
        n = int(fs * tone_dur)
        t = np.arange(n) / fs
        tone = 0.5 * (np.sin(2 * np.pi * f1 * t) +
                       np.sin(2 * np.pi * f2 * t))
        parts.append(tone)
        parts.append(np.zeros(int(fs * gap_dur)))
    if not parts:
        return np.array([])
    return np.concatenate(parts)


@register("DTMF")
class DTMFValidator(BaseRoundtripValidator):
    """Validate DTMF via tone generation + multimon-ng decoder."""

    required_tools = ["multimon-ng"]
    tier = 1

    def make_trial(self, mode):
        def run(snr_db, trial_idx, tmpdir):
            # Generate random digit sequence
            digits = "".join(np.random.choice(list("0123456789"), size=5))
            audio = _generate_dtmf_audio(digits, fs=8000)
            if len(audio) == 0:
                return False
            if snr_db is not None:
                audio = add_awgn_audio(audio, snr_db)
            wav_path = os.path.join(tmpdir, "dtmf.wav")
            write_mono_wav(audio, 8000, wav_path)

            # Decode
            try:
                result = subprocess.run(
                    ["multimon-ng", "-t", "wav", "-a", "DTMF", wav_path],
                    capture_output=True, text=True, timeout=15)
            except (subprocess.TimeoutExpired, FileNotFoundError):
                return False

            # Parse decoded digits from multimon-ng output
            decoded_digits = []
            for line in result.stdout.split("\n"):
                line = line.strip()
                if "DTMF:" in line:
                    # Format: "DTMF: X"
                    parts = line.split("DTMF:")
                    if len(parts) > 1:
                        d = parts[1].strip()
                        if d and d[0] in _DTMF_FREQS:
                            decoded_digits.append(d[0])

            decoded = "".join(decoded_digits)
            # Accept if >=60% of digits decoded correctly in sequence
            if not decoded:
                return False
            matches = 0
            di = 0
            for expected_d in digits:
                while di < len(decoded):
                    if decoded[di] == expected_d:
                        matches += 1
                        di += 1
                        break
                    di += 1
            return matches >= len(digits) * 0.6

        return run


# ---------------------------------------------------------------------------
# POCSAG
# ---------------------------------------------------------------------------

@register("POCSAG")
class POCSAGValidator(BaseRoundtripValidator):
    """Validate POCSAG via multimon-ng sync detection (tier 2)."""

    required_tools = ["multimon-ng"]
    tier = 2

    def make_trial(self, mode):
        def run(snr_db, trial_idx, tmpdir):
            # Use the POCSAG synthesizer from the training pipeline
            from rf_datagen.generators.synthetic import synth_pocsag
            from rf_datagen.constants import FS
            from rf_datagen.dsp import iq_to_audio

            iq = synth_pocsag()
            if len(iq) < 100:
                return False

            # Demodulate IQ to audio (FM discriminator)
            disc = np.angle(iq[1:] * np.conj(iq[:-1]))
            # Resample to 22050 for multimon-ng
            from scipy.signal import resample
            target_len = int(len(disc) * 22050 / FS)
            if target_len < 100:
                return False
            audio = resample(disc, target_len)
            peak = np.max(np.abs(audio))
            if peak > 0:
                audio = audio / peak

            if snr_db is not None:
                audio = add_awgn_audio(audio, snr_db)

            wav_path = os.path.join(tmpdir, "pocsag.wav")
            write_mono_wav(audio, 22050, wav_path)

            try:
                result = subprocess.run(
                    ["multimon-ng", "-t", "wav", "-a", "POCSAG1200", wav_path],
                    capture_output=True, text=True, timeout=15)
            except (subprocess.TimeoutExpired, FileNotFoundError):
                return False

            # Tier 2: accept if any POCSAG sync/frame detected
            output = result.stdout + result.stderr
            return ("POCSAG" in output and
                    any(k in output for k in ["Alpha:", "Numeric:",
                                              "Address:", "Function:"]))

        return run


# ---------------------------------------------------------------------------
# EAS
# ---------------------------------------------------------------------------

@register("EAS")
class EASValidator(BaseRoundtripValidator):
    """Validate EAS/SAME via sameeas encoder + multimon-ng decoder."""

    required_tools = ["multimon-ng"]
    tier = 1

    def make_trial(self, mode):
        def run(snr_db, trial_idx, tmpdir):
            from rf_datagen.generators.sameeas import generate_same_signal
            from rf_datagen.content.ham_text import gen_same_message

            message = gen_same_message()
            audio, sr = generate_same_signal(message)
            if len(audio) == 0:
                return False

            if snr_db is not None:
                audio = add_awgn_audio(audio, snr_db)

            wav_path = os.path.join(tmpdir, "eas.wav")
            write_mono_wav(audio, sr, wav_path)

            try:
                result = subprocess.run(
                    ["multimon-ng", "-t", "wav", "-a", "EAS", wav_path],
                    capture_output=True, text=True, timeout=15)
            except (subprocess.TimeoutExpired, FileNotFoundError):
                return False

            # Check if EAS header decoded (ZCZC prefix in SAME messages)
            output = result.stdout
            return "EAS:" in output or "ZCZC" in output

        return run


# ---------------------------------------------------------------------------
# FLEX (tier 2 — sync detection)
# ---------------------------------------------------------------------------

@register("FLEX")
class FLEXValidator(BaseRoundtripValidator):
    """Validate FLEX sync pattern detection via multimon-ng."""

    required_tools = ["multimon-ng"]
    tier = 2

    def make_trial(self, mode):
        def run(snr_db, trial_idx, tmpdir):
            from rf_datagen.generators.synthetic import synth_flex
            from rf_datagen.constants import FS
            from scipy.signal import resample

            iq = synth_flex()
            if len(iq) < 100:
                return False

            disc = np.angle(iq[1:] * np.conj(iq[:-1]))
            target_len = int(len(disc) * 22050 / FS)
            if target_len < 100:
                return False
            audio = resample(disc, target_len)
            peak = np.max(np.abs(audio))
            if peak > 0:
                audio = audio / peak

            if snr_db is not None:
                audio = add_awgn_audio(audio, snr_db)

            wav_path = os.path.join(tmpdir, "flex.wav")
            write_mono_wav(audio, 22050, wav_path)

            try:
                result = subprocess.run(
                    ["multimon-ng", "-t", "wav", "-a", "FLEX", wav_path],
                    capture_output=True, text=True, timeout=15)
            except (subprocess.TimeoutExpired, FileNotFoundError):
                return False

            output = result.stdout + result.stderr
            return "FLEX" in output

        return run


# ---------------------------------------------------------------------------
# ACARS (tier 2 — preamble detection)
# ---------------------------------------------------------------------------

@register("ACARS")
class ACARSValidator(BaseRoundtripValidator):
    """Validate ACARS preamble detection via multimon-ng."""

    required_tools = ["multimon-ng"]
    tier = 2

    def make_trial(self, mode):
        def run(snr_db, trial_idx, tmpdir):
            from rf_datagen.generators.synthetic import synth_acars
            from rf_datagen.constants import FS
            from scipy.signal import resample

            iq = synth_acars()
            if len(iq) < 100:
                return False

            disc = np.angle(iq[1:] * np.conj(iq[:-1]))
            target_len = int(len(disc) * 22050 / FS)
            if target_len < 100:
                return False
            audio = resample(disc, target_len)
            peak = np.max(np.abs(audio))
            if peak > 0:
                audio = audio / peak

            if snr_db is not None:
                audio = add_awgn_audio(audio, snr_db)

            wav_path = os.path.join(tmpdir, "acars.wav")
            write_mono_wav(audio, 22050, wav_path)

            try:
                result = subprocess.run(
                    ["multimon-ng", "-t", "wav", "-a", "ACARS", wav_path],
                    capture_output=True, text=True, timeout=15)
            except (subprocess.TimeoutExpired, FileNotFoundError):
                return False

            output = result.stdout + result.stderr
            return "ACARS" in output

        return run
