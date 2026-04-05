"""Isolated PulseAudio server for headless audio capture."""

import os
import shutil
import subprocess
import tempfile
import time

from . import pid_registry


class IsolatedPulseServer:
    """Context manager that runs a completely isolated PulseAudio daemon.

    Provides a Unix socket for PULSE_SERVER with a null sink for capture.
    """

    def __init__(self, sink_name="capture"):
        self.sink_name = sink_name
        self.monitor_source = f"{sink_name}.monitor"
        self._tmpdir = None
        self._socket_path = None
        self._proc = None

    def __enter__(self):
        self._tmpdir = tempfile.mkdtemp(prefix="pa_isolated_")
        self._socket_path = os.path.join(self._tmpdir, "native")

        config_path = os.path.join(self._tmpdir, "default.pa")
        with open(config_path, "w") as f:
            f.write(
                f"load-module module-native-protocol-unix "
                f"auth-anonymous=1 socket={self._socket_path}\n"
                f"load-module module-null-sink "
                f"sink_name={self.sink_name} "
                f"sink_properties=device.description=isolated_capture "
                f"rate=48000 channels=1\n"
                f"load-module module-always-sink\n"
                f"set-default-sink {self.sink_name}\n"
            )

        pa_env = os.environ.copy()
        pa_env["XDG_RUNTIME_DIR"] = self._tmpdir
        # Fully isolate from session D-Bus to avoid name collisions
        # when running multiple PA instances.
        pa_env.pop("DBUS_SESSION_BUS_ADDRESS", None)
        pa_env["DBUS_SESSION_BUS_ADDRESS"] = "disabled:"
        self._proc = subprocess.Popen(
            ["pulseaudio",
             "--daemonize=false",
             "-n",
             f"--file={config_path}",
             "--exit-idle-time=-1",
             "--use-pid-file=false",
             "--log-level=error"],
            env=pa_env,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.PIPE,
        )

        deadline = time.time() + 10
        while time.time() < deadline:
            if os.path.exists(self._socket_path):
                break
            if self._proc.poll() is not None:
                stderr = self._proc.stderr.read().decode()
                raise RuntimeError(
                    f"Isolated PulseAudio exited: {stderr}")
            time.sleep(0.1)
        else:
            self._proc.kill()
            raise RuntimeError("Isolated PulseAudio socket not created")

        try:
            pid_registry.register_child(
                self._proc.pid, "pulseaudio", tmpdir=self._tmpdir)
        except Exception:
            pass

        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self._proc and self._proc.poll() is None:
            try:
                pid_registry.unregister_child(self._proc.pid)
            except Exception:
                pass
            self._proc.terminate()
            try:
                self._proc.wait(timeout=5)
            except subprocess.TimeoutExpired:
                self._proc.kill()
                self._proc.wait()
        if self._tmpdir:
            shutil.rmtree(self._tmpdir, ignore_errors=True)

    def env(self):
        """Return environment variables that isolate audio to this server."""
        return {
            "PULSE_SERVER": f"unix:{self._socket_path}",
        }

    def clean_env(self, base_env=None):
        """Return a full environment dict with audio/display isolation."""
        env = (base_env or os.environ).copy()
        env.update(self.env())
        env.pop("DISPLAY", None)
        env.pop("WAYLAND_DISPLAY", None)
        return env
