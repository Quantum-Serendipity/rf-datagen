"""Packet radio generator via direwolf gen_packets."""

import os
import shutil
import subprocess
import tempfile
import wave

import numpy as np
from scipy.signal import resample

from ..constants import FS, WINDOW_LEN
from ..dsp import audio_to_iq
from ..content.ham_text import gen_packet_content
from .base import BaseGenerator

GEN_AUDIO_FS = 44100
PACKET_BAUDS = [300, 1200, 9600]


def generate_packets(baud, n_packets=50, tmpdir="/tmp"):
    wav_path = os.path.join(tmpdir, f"packet_{baud}.wav")
    cmd = ["gen_packets", "-b", str(baud), "-o", wav_path,
           "-r", str(GEN_AUDIO_FS), "-n", str(n_packets)]
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=60)
    if result.returncode != 0:
        return np.array([])
    if not os.path.exists(wav_path) or os.path.getsize(wav_path) == 0:
        return np.array([])
    try:
        with wave.open(wav_path, "rb") as wf:
            nframes = wf.getnframes()
            raw = np.frombuffer(wf.readframes(nframes), dtype=np.int16)
        audio = raw.astype(np.float64) / 32768.0
    except Exception:
        return np.array([])
    finally:
        try:
            os.remove(wav_path)
        except OSError:
            pass
    return audio


class PacketGenerator(BaseGenerator):
    name = "packet"
    required_tools = ["gen_packets"]
    signal_classes = ["PACKET"]

    def generate_class(self, class_name, rng=None):
        tmpdir = tempfile.mkdtemp(prefix="packet_gen_")
        try:
            all_iq_segments = []
            packets_per_baud = self.config.packets_per_baud
            for baud in PACKET_BAUDS:
                audio = generate_packets(baud, packets_per_baud, tmpdir)
                if len(audio) < 1000:
                    continue
                iq = audio_to_iq(audio, GEN_AUDIO_FS, target_fs=self.fs)
                if len(iq) >= self.window_len:
                    all_iq_segments.append(iq)
            if not all_iq_segments:
                return np.array([], dtype=np.complex128)
            return np.concatenate(all_iq_segments)
        finally:
            shutil.rmtree(tmpdir, ignore_errors=True)
