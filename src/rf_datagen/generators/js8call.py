"""JS8Call generator — headless JS8Call with TCP JSON API control."""

import json
import os
import shutil
import signal
import socket
import subprocess
import tempfile
import time

import numpy as np

from ..constants import FS
from ..dsp import audio_to_iq
from ..content.ham_text import gen_js8_message
from ..isolation import IsolatedPulseServer
from ..logging_config import get_logger
from .base import BaseGenerator

log = get_logger("js8call")

CAPTURE_FS = 48000
JS8_TCP_PORT = 2442

# JS8Call sub-mode speed offsets (Hz offset in waterfall)
JS8_SUBMODES = {
    "normal": 0,
    "fast":   1,
    "turbo":  2,
    "slow":   4,
}


def _find_free_port(start=2442):
    """Find a free TCP port starting from the given port."""
    for port in range(start, start + 100):
        try:
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                s.bind(("127.0.0.1", port))
                return port
        except OSError:
            continue
    return None


class JS8CallInstance:
    """Manages a single headless JS8Call instance."""

    def __init__(self, work_dir, pa_env, tcp_port=None):
        self.work_dir = work_dir
        self.config_dir = os.path.join(work_dir, "js8call_config")
        self.pa_env = pa_env
        self.tcp_port = tcp_port or _find_free_port(JS8_TCP_PORT)
        self.sink_name = "capture"
        self.proc = None
        self.sock = None

    def _make_env(self):
        env = self.pa_env.copy()
        env["HOME"] = self.config_dir
        return env

    def _write_config(self):
        """Write JS8Call configuration for headless operation."""
        os.makedirs(self.config_dir, exist_ok=True)
        config_dir = os.path.join(self.config_dir, ".config", "JS8Call")
        os.makedirs(config_dir, exist_ok=True)

        # JS8Call uses QSettings INI format
        config_path = os.path.join(config_dir, "JS8Call.ini")
        with open(config_path, "w") as f:
            f.write("[General]\n")
            f.write(f"TCPEnabled=true\n")
            f.write(f"TCPPort={self.tcp_port}\n")
            f.write(f"TCPAddress=127.0.0.1\n")
            f.write(f"AudioInput=default\n")
            f.write(f"AudioOutput=default\n")
            f.write(f"MyCall=W1AW\n")
            f.write(f"MyGrid=FN31\n")

    def start(self, timeout=60):
        self._write_config()
        env = self._make_env()

        self.proc = subprocess.Popen(
            ["xvfb-run", "-a", "js8call", "--config",
             os.path.join(self.config_dir, ".config", "JS8Call")],
            env=env, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
            start_new_session=True,
        )

        # Wait for TCP API to become available
        deadline = time.time() + timeout
        while time.time() < deadline:
            if self.proc.poll() is not None:
                raise RuntimeError(
                    f"js8call exited with code {self.proc.returncode}")
            try:
                self.sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                self.sock.settimeout(3.0)
                self.sock.connect(("127.0.0.1", self.tcp_port))
                return
            except (ConnectionRefusedError, socket.timeout, OSError):
                if self.sock:
                    self.sock.close()
                    self.sock = None
                time.sleep(1.0)

        raise RuntimeError(f"JS8Call TCP API not ready after {timeout}s")

    def send_command(self, cmd_type, value="", params=None):
        """Send a JSON command to JS8Call TCP API."""
        msg = {
            "type": cmd_type,
            "value": value,
        }
        if params:
            msg["params"] = params
        data = json.dumps(msg) + "\n"
        try:
            self.sock.sendall(data.encode())
        except (OSError, BrokenPipeError):
            log.debug("JS8Call socket send failed")

    def recv_response(self, timeout=3.0):
        """Read a response from JS8Call TCP API."""
        self.sock.settimeout(timeout)
        try:
            data = self.sock.recv(8192)
            return data.decode(errors="replace")
        except socket.timeout:
            return ""
        except OSError:
            return ""

    def tx_message(self, message, submode="normal"):
        """Queue a message for transmission."""
        # Set sub-mode speed
        speed_val = JS8_SUBMODES.get(submode, 0)
        self.send_command("STATION.SET_SPEED", str(speed_val))
        time.sleep(0.2)

        # Send the message
        self.send_command("TX.SEND_MESSAGE", message)

    def capture_tx(self, message, submode="normal", max_wait=30.0):
        """Transmit a message and capture the audio output."""
        audio_path = os.path.join(self.work_dir, "js8_capture.raw")
        env = self._make_env()

        rec_proc = None
        audio_fd = open(audio_path, "wb")
        try:
            rec_proc = subprocess.Popen(
                ["parec", f"--device={self.sink_name}.monitor",
                 f"--rate={CAPTURE_FS}", "--channels=1", "--format=s16le"],
                stdout=audio_fd, env=env,
            )
            time.sleep(0.3)

            self.tx_message(message, submode)

            # Wait for TX to complete
            deadline = time.time() + max_wait
            while time.time() < deadline:
                resp = self.recv_response(timeout=1.0)
                if "TX.COMPLETED" in resp or "RX.DIRECTED" in resp:
                    break
                time.sleep(0.5)

            time.sleep(0.5)
        finally:
            if rec_proc is not None:
                rec_proc.terminate()
                try:
                    rec_proc.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    rec_proc.kill()
                    rec_proc.wait()
            audio_fd.close()

        if not os.path.exists(audio_path) or os.path.getsize(audio_path) == 0:
            return np.array([], dtype=np.int16)

        raw_data = np.fromfile(audio_path, dtype=np.int16)
        try:
            os.remove(audio_path)
        except OSError:
            pass

        if len(raw_data) < CAPTURE_FS:
            return np.array([], dtype=np.int16)

        # Trim silence
        threshold = max(1, int(np.max(np.abs(raw_data)) * 0.01))
        nonsilent = np.where(np.abs(raw_data) > threshold)[0]
        if len(nonsilent) == 0:
            return np.array([], dtype=np.int16)
        margin = int(0.1 * CAPTURE_FS)
        start = max(0, nonsilent[0] - margin)
        end = min(len(raw_data), nonsilent[-1] + margin)
        return raw_data[start:end]

    def stop(self):
        if self.sock:
            try:
                self.sock.close()
            except OSError:
                pass
        if self.proc and self.proc.poll() is None:
            try:
                pgid = os.getpgid(self.proc.pid)
                os.killpg(pgid, signal.SIGTERM)
            except (ProcessLookupError, PermissionError):
                pass
            try:
                self.proc.wait(timeout=10)
            except subprocess.TimeoutExpired:
                try:
                    pgid = os.getpgid(self.proc.pid)
                    os.killpg(pgid, signal.SIGKILL)
                except (ProcessLookupError, PermissionError):
                    pass
                self.proc.wait()


class Js8callGenerator(BaseGenerator):
    name = "js8call"
    required_tools = ["js8call", "parec", "xvfb-run"]
    signal_classes = ["JS8"]

    def generate_class(self, class_name, rng=None):
        tmpdir = tempfile.mkdtemp(prefix="js8call_gen_")
        instance = None

        try:
            with IsolatedPulseServer() as pa:
                pa_env = pa.clean_env()
                instance = JS8CallInstance(tmpdir, pa_env)
                instance.start()

                segments = []
                n_messages = self.config.messages_per_mode
                submodes = list(JS8_SUBMODES.keys())

                for i in range(n_messages):
                    msg = gen_js8_message()
                    submode = submodes[i % len(submodes)]

                    raw_audio = instance.capture_tx(msg, submode=submode)
                    if len(raw_audio) < 1000:
                        continue

                    iq = audio_to_iq(raw_audio, CAPTURE_FS, target_fs=self.fs)
                    if len(iq) >= self.window_len:
                        segments.append(iq)

                if not segments:
                    return np.array([], dtype=np.complex128)
                return np.concatenate(segments)

        except Exception as e:
            log.warning("JS8Call generation failed: %s", e)
            return np.array([], dtype=np.complex128)
        finally:
            if instance:
                instance.stop()
            shutil.rmtree(tmpdir, ignore_errors=True)
