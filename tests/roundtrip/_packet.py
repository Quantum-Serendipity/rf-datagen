"""PACKET validator via direwolf atest."""

import os
import subprocess
import wave

import numpy as np

from rf_datagen.generators.packet import GEN_AUDIO_FS

from ._base import BaseRoundtripValidator, register
from ._helpers import write_mono_wav, add_awgn_audio


@register("PACKET_1200")
class PacketValidator(BaseRoundtripValidator):
    """Validate packet radio via direwolf's atest decoder."""

    required_tools = ["gen_packets", "atest"]
    tier = 1

    def generate(self, baud, tmpdir, n_packets=5):
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
        if snr_db is not None:
            audio = add_awgn_audio(audio, snr_db)
        wav_path = os.path.join(tmpdir, f"packet_{baud}_test.wav")
        write_mono_wav(audio, GEN_AUDIO_FS, wav_path)
        return wav_path

    def decode(self, wav_path, baud):
        cmd = ["atest", "-B", str(baud), wav_path]
        try:
            result = subprocess.run(
                cmd, capture_output=True, text=True, timeout=60)
        except subprocess.TimeoutExpired:
            return 0
        out = result.stdout + result.stderr
        for line in out.split("\n"):
            line = line.strip()
            if "packets decoded" in line:
                parts = line.split()
                try:
                    return int(parts[0])
                except (ValueError, IndexError):
                    pass
        return sum(1 for l in out.split("\n") if "DECODED" in l)

    def make_trial(self, mode):
        baud = 1200
        def run(snr_db, trial_idx, tmpdir):
            audio = self.generate(baud, tmpdir, n_packets=5)
            if audio is None or len(audio) < 1000:
                return False
            wav_path = self.prepare_wav(audio, tmpdir, baud, snr_db)
            n_decoded = self.decode(wav_path, baud)
            return n_decoded >= 1
        return run
