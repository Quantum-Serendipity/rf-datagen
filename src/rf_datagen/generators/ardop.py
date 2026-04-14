"""ARDOP generator — uses ardopcf TNC with audio pipe I/O."""

import os
import shutil
import socket
import subprocess
import tempfile
import time
import wave

import numpy as np

from ..constants import FS
from ..dsp import audio_to_iq
from ..logging_config import get_logger
from .base import BaseGenerator

log = get_logger("ardop")

ARDOPCF_AUDIO_FS = 12000
CTRL_PORT = 8515
DATA_PORT = 8516


def _find_free_port(start=8515):
    """Find a free TCP port starting from the given port."""
    for port in range(start, start + 100):
        try:
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                s.bind(("127.0.0.1", port))
                return port
        except OSError:
            continue
    return None


def _send_cmd(sock, cmd):
    """Send a command to ardopcf control port."""
    sock.sendall((cmd + "\r").encode())
    time.sleep(0.05)


def _recv_response(sock, timeout=2.0):
    """Read available data from control socket."""
    sock.settimeout(timeout)
    try:
        return sock.recv(4096).decode(errors="replace")
    except socket.timeout:
        return ""


class ArdopGenerator(BaseGenerator):
    name = "ardop"
    required_tools = ["ardopcf"]
    signal_classes = ["ARDOP"]

    def generate_class(self, class_name, rng=None):
        """Generate ARDOP signal by running ardopcf TNC and sending test frames.

        ardopcf can produce audio output to a WAV file via its -w flag.
        We start the TNC, send data frames at various speeds, and capture
        the generated audio.
        """
        tmpdir = tempfile.mkdtemp(prefix="ardop_gen_")
        segments = []

        try:
            speeds = self.config.ardop_speeds
            for speed in speeds:
                audio = self._generate_speed(speed, tmpdir)
                if len(audio) < 1000:
                    continue
                iq = audio_to_iq(audio, ARDOPCF_AUDIO_FS, target_fs=self.fs)
                if len(iq) >= self.window_len:
                    segments.append(iq)
        finally:
            shutil.rmtree(tmpdir, ignore_errors=True)

        if not segments:
            return np.array([], dtype=np.complex128)
        return np.concatenate(segments)

    def _generate_speed(self, speed, tmpdir):
        """Generate ARDOP frames at a specific speed using ardopcf."""
        wav_path = os.path.join(tmpdir, f"ardop_{speed}.wav")
        ctrl_port = _find_free_port(8515)
        if ctrl_port is None:
            return np.array([])
        data_port = ctrl_port + 1

        # Start ardopcf with WAV output
        proc = None
        try:
            proc = subprocess.Popen(
                ["ardopcf", str(ctrl_port),
                 "-w", wav_path,
                 "--ptt", "0",  # no PTT control
                 ],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
            time.sleep(0.5)  # let TNC start

            if proc.poll() is not None:
                return np.array([])

            # Connect to control port
            ctrl = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            ctrl.settimeout(3.0)
            try:
                ctrl.connect(("127.0.0.1", ctrl_port))
            except (ConnectionRefusedError, socket.timeout):
                return np.array([])

            try:
                _recv_response(ctrl, timeout=1.0)

                # Configure TNC
                _send_cmd(ctrl, "INITIALIZE")
                _recv_response(ctrl)
                _send_cmd(ctrl, "PROTOCOLMODE ARQ")
                _recv_response(ctrl)

                # Send test frames at the requested speed
                n_frames = max(5, self.config.messages_per_mode // 10)
                for _ in range(n_frames):
                    # Generate random data payload
                    payload = os.urandom(np.random.randint(16, 128))
                    frame_cmd = f"SENDID"
                    _send_cmd(ctrl, frame_cmd)
                    _recv_response(ctrl, timeout=1.0)

                # Allow time for audio generation
                time.sleep(1.0)
                _send_cmd(ctrl, "CLOSE")
                _recv_response(ctrl)
            finally:
                ctrl.close()

        except Exception as e:
            log.debug("ardopcf error at speed %d: %s", speed, e)
            return np.array([])
        finally:
            if proc is not None:
                proc.terminate()
                try:
                    proc.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    proc.kill()
                    proc.wait()

        # Read output WAV
        if not os.path.exists(wav_path) or os.path.getsize(wav_path) == 0:
            return np.array([])

        try:
            with wave.open(wav_path, "rb") as wf:
                nframes = wf.getnframes()
                raw = np.frombuffer(wf.readframes(nframes), dtype=np.int16)
            return raw.astype(np.float64) / 32768.0
        except Exception as e:
            log.warning("ARDOP WAV read failed: %s", e)
            return np.array([])
        finally:
            try:
                os.remove(wav_path)
            except OSError:
                pass
