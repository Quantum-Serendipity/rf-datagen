"""Extended WSJT-X validators — FT4, JT65, JT9.

These modes use synthesis approximations that are not fully protocol-accurate,
so expected_fail=True. Decode rates are tracked for regression monitoring.
"""

import os
import subprocess

import numpy as np

from rf_datagen.generators.wsjtx import SYNTH_FS
from rf_datagen.content.ham_text import gen_ft8_message

from ._base import BaseRoundtripValidator, register
from ._helpers import write_mono_wav, add_awgn_audio


@register("FT4")
class FT4Validator(BaseRoundtripValidator):
    """Validate FT4 via jt9 --ft4 decoder.

    Known limitation: our encoder maps FT8 symbols mod 4 (approximation),
    so decode rates will be low.
    """

    required_tools = ["jt9"]
    tier = 1
    expected_fail = True

    def make_trial(self, mode):
        def run(snr_db, trial_idx, tmpdir):
            from rf_datagen.generators.synthetic import synth_ft4
            from rf_datagen.constants import FS
            from rf_datagen.dsp import iq_to_audio

            msg = gen_ft8_message()
            iq = synth_ft4()
            if len(iq) < 100:
                return False

            # Convert IQ to audio for jt9
            audio = np.real(iq)
            # Resample to SYNTH_FS (12000 Hz)
            from scipy.signal import resample
            if FS != SYNTH_FS:
                target_len = int(len(audio) * SYNTH_FS / FS)
                audio = resample(audio, target_len)

            # Pad into a 7.5s period for FT4
            total_samples = int(7.5 * SYNTH_FS)
            offset_samples = int(0.2 * SYNTH_FS)
            padded = np.zeros(total_samples, dtype=np.float64)
            end = min(offset_samples + len(audio), total_samples)
            padded[offset_samples:end] = audio[:end - offset_samples]

            if snr_db is not None:
                padded = add_awgn_audio(padded, snr_db)

            wav_path = os.path.join(tmpdir, "ft4_signal.wav")
            write_mono_wav(padded, SYNTH_FS, wav_path)

            decode_dir = os.path.join(tmpdir, "decode")
            os.makedirs(decode_dir, exist_ok=True)
            cmd = ["jt9", "--ft4", "-p", "7.5", "-d", "3",
                   "-a", decode_dir, "-t", decode_dir, wav_path]
            try:
                result = subprocess.run(
                    cmd, capture_output=True, text=True, timeout=60)
            except subprocess.TimeoutExpired:
                return False

            # Any decoded message counts as success
            for line in result.stdout.strip().split("\n"):
                if line.strip() and len(line.split()) >= 5:
                    return True
            return False

        return run


@register("JT65")
class JT65Validator(BaseRoundtripValidator):
    """Validate JT65 via jt9 --jt65 decoder.

    Known limitation: our GFSK synthesis doesn't produce signals decodable
    by jt9.
    """

    required_tools = ["jt9"]
    tier = 1
    expected_fail = True

    def make_trial(self, mode):
        def run(snr_db, trial_idx, tmpdir):
            from rf_datagen.generators.synthetic import synth_jt65
            from rf_datagen.constants import FS
            from scipy.signal import resample

            iq = synth_jt65()
            if len(iq) < 100:
                return False

            audio = np.real(iq)
            if FS != SYNTH_FS:
                target_len = int(len(audio) * SYNTH_FS / FS)
                audio = resample(audio, target_len)

            # JT65 uses 60s period
            total_samples = int(60 * SYNTH_FS)
            offset_samples = int(1.0 * SYNTH_FS)
            padded = np.zeros(total_samples, dtype=np.float64)
            end = min(offset_samples + len(audio), total_samples)
            padded[offset_samples:end] = audio[:end - offset_samples]

            if snr_db is not None:
                padded = add_awgn_audio(padded, snr_db)

            wav_path = os.path.join(tmpdir, "jt65_signal.wav")
            write_mono_wav(padded, SYNTH_FS, wav_path)

            decode_dir = os.path.join(tmpdir, "decode")
            os.makedirs(decode_dir, exist_ok=True)
            cmd = ["jt9", "--jt65", "-p", "60", "-d", "3",
                   "-a", decode_dir, "-t", decode_dir, wav_path]
            try:
                result = subprocess.run(
                    cmd, capture_output=True, text=True, timeout=120)
            except subprocess.TimeoutExpired:
                return False

            for line in result.stdout.strip().split("\n"):
                if line.strip() and len(line.split()) >= 5:
                    return True
            return False

        return run


@register("JT9")
class JT9Validator(BaseRoundtripValidator):
    """Validate JT9 via jt9 decoder.

    Known limitation: our GFSK synthesis doesn't produce protocol-accurate
    JT9 signals.
    """

    required_tools = ["jt9"]
    tier = 1
    expected_fail = True

    def make_trial(self, mode):
        def run(snr_db, trial_idx, tmpdir):
            from rf_datagen.generators.synthetic import synth_jt9
            from rf_datagen.constants import FS
            from scipy.signal import resample

            iq = synth_jt9()
            if len(iq) < 100:
                return False

            audio = np.real(iq)
            if FS != SYNTH_FS:
                target_len = int(len(audio) * SYNTH_FS / FS)
                audio = resample(audio, target_len)

            # JT9 uses 60s period
            total_samples = int(60 * SYNTH_FS)
            offset_samples = int(1.0 * SYNTH_FS)
            padded = np.zeros(total_samples, dtype=np.float64)
            end = min(offset_samples + len(audio), total_samples)
            padded[offset_samples:end] = audio[:end - offset_samples]

            if snr_db is not None:
                padded = add_awgn_audio(padded, snr_db)

            wav_path = os.path.join(tmpdir, "jt9_signal.wav")
            write_mono_wav(padded, SYNTH_FS, wav_path)

            decode_dir = os.path.join(tmpdir, "decode")
            os.makedirs(decode_dir, exist_ok=True)
            cmd = ["jt9", "--jt9", "-p", "60", "-d", "3",
                   "-a", decode_dir, "-t", decode_dir, wav_path]
            try:
                result = subprocess.run(
                    cmd, capture_output=True, text=True, timeout=120)
            except subprocess.TimeoutExpired:
                return False

            for line in result.stdout.strip().split("\n"):
                if line.strip() and len(line.split()) >= 5:
                    return True
            return False

        return run
