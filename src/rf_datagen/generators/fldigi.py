"""fldigi XML-RPC generator — 18+ signal classes via headless fldigi."""

import os
import signal
import shutil
import subprocess
import time
import xmlrpc.client

import numpy as np
from scipy.signal import resample

from ..constants import FS, WINDOW_LEN
from ..dsp import hilbert_analytic, audio_to_iq
from ..content.ham_text import get_text_for_mode
from ..content.typing import TypingCadenceModel
from ..impairments import extract_windows, apply_impairments, configure_impairments
from ..isolation import IsolatedPulseServer
from ..logging_config import get_logger
from ..output import atomic_save_npy, atomic_write_csv
from .. import pid_registry
from .base import BaseGenerator

log = get_logger("fldigi")

CAPTURE_FS = 48000
MODE_TIMEOUT = 900  # 15 minutes max per mode before giving up
MAX_CHUNK_TIMEOUT = 120  # 2 minutes max waiting for a single TX chunk

MODE_CHARS_PER_SEC = {
    "PSK31": 5, "PSK63": 10, "RTTY": 8, "OLIVIA": 3,
    "CW": 5, "DOMINOEX": 3, "THOR": 3, "MT63": 10,
    "CONTESTIA": 3, "HELLSCHREIBER": 5, "MFSK16": 5, "MFSK32": 10,
    "NAVTEX": 8, "PACKET": 10, "FAX": 2,
    "QPSK": 10, "PSK125": 20, "8PSK": 20,
    "FSQ": 1, "IFKP": 1, "THROB": 1,
}

FLDIGI_MODES = {
    "PSK31":         ["BPSK31"],
    "PSK63":         ["BPSK63"],
    "RTTY":          ["RTTY"],
    "OLIVIA":        ["OLIVIA-8/250", "OLIVIA-8/500", "OLIVIA-8/1000",
                      "OLIVIA-16/500", "OLIVIA-16/1000", "OLIVIA-32/1000"],
    "CW":            ["CW"],
    "DOMINOEX":      ["DomEX-4", "DomEX-5", "DomEX-8", "DomEX-11",
                      "DomEX-16", "DomEX-22"],
    "THOR":          ["THOR4", "THOR5", "THOR8", "THOR11", "THOR16", "THOR22"],
    "MT63":          ["MT63-500", "MT63-1K", "MT63-2K"],
    "CONTESTIA":     ["Contestia/4-250", "Contestia/4-500", "Contestia/8-250",
                      "Contestia/8-500", "Contestia/8-1000", "Contestia/16-500",
                      "Contestia/16-1000"],
    "HELLSCHREIBER": ["Feld-Hell", "Slow-Hell", "Hell-x5", "Hell-x9",
                      "FSKHell", "FSKHell-105", "Hell-80"],
    "MFSK16":        ["MFSK-16"],
    "MFSK32":        ["MFSK-32"],
    "NAVTEX":        ["NAVTEX"],
    "PACKET":        ["PKT300", "PKT1200"],
    "FAX":           ["WEFAX-IOC576", "WEFAX-IOC288"],
    "QPSK":          ["QPSK31", "QPSK63", "QPSK125", "QPSK250"],
    "PSK125":        ["BPSK125", "BPSK250", "BPSK500"],
    "8PSK":          ["8PSK125", "8PSK250"],
    "FSQ":           ["FSQ-3"],
    "IFKP":          ["IFKP"],
    "THROB":         ["Throb-1", "Throb-2", "Throb-4",
                      "ThrobX-1", "ThrobX-2", "ThrobX-4"],
}


def feed_characters_with_cadence(server, text, cadence):
    if cadence.profile == 'copy_paste':
        pos = 0
        while pos < len(text):
            remaining = len(text) - pos
            if remaining <= 10:
                burst_len = remaining
            else:
                burst_len = np.random.randint(10, min(51, remaining + 1))
            chunk = text[pos:pos + burst_len]
            server.text.add_tx(chunk)
            pos += burst_len
            if pos < len(text):
                time.sleep(cadence.pause_duration())
    else:
        for char in text:
            if cadence.should_typo() and char.isalpha():
                wrong = chr(np.random.randint(ord('a'), ord('z') + 1))
                server.text.add_tx(wrong)
                time.sleep(cadence.char_delay() * 0.5)
                server.text.add_tx('\b')
                time.sleep(cadence.char_delay())
            server.text.add_tx(char)
            if cadence.should_pause(char):
                time.sleep(cadence.pause_duration())
            else:
                time.sleep(cadence.char_delay())


class FLDigiInstance:
    def __init__(self, instance_id, xmlrpc_port, work_dir, pa_env):
        self.id = instance_id
        self.port = xmlrpc_port
        self.work_dir = work_dir
        self.config_dir = os.path.join(work_dir, f"fldigi_{instance_id}")
        self.audio_dir = os.path.join(work_dir, f"audio_{instance_id}")
        self.sink_name = "capture"
        self.pa_env = pa_env
        self.fldigi_proc = None
        self.server = None

    def _make_env(self):
        env = self.pa_env.copy()
        env["HOME"] = self.config_dir
        return env

    def _write_config(self):
        config_path = os.path.join(self.config_dir, "fldigi_def.xml")
        with open(config_path, "w") as f:
            f.write('<?xml version="1.0" encoding="UTF-8"?>\n'
                    '<FLDIGI_DEFS>\n'
                    '  <CONFIGURATION>\n'
                    '    <AUDIOIO>2</AUDIOIO>\n'
                    '  </CONFIGURATION>\n'
                    '  <ID>\n'
                    '    <TXRSID>1</TXRSID>\n'
                    '  </ID>\n'
                    '</FLDIGI_DEFS>\n')

    def start(self):
        os.makedirs(self.config_dir, exist_ok=True)
        os.makedirs(self.audio_dir, exist_ok=True)
        self._write_config()
        env = self._make_env()
        self.fldigi_proc = subprocess.Popen(
            ["xvfb-run", "-a", "fldigi",
             "--config-dir", self.config_dir,
             "--xmlrpc-server-port", str(self.port)],
            env=env, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
            start_new_session=True,
        )
        self.server = xmlrpc.client.ServerProxy(
            f"http://127.0.0.1:{self.port}")
        self._wait_ready()

    def _wait_ready(self, timeout=60):
        deadline = time.time() + timeout
        while time.time() < deadline:
            try:
                self.server.fldigi.name()
                return
            except Exception:
                if self.fldigi_proc.poll() is not None:
                    raise RuntimeError(
                        f"fldigi exited with code {self.fldigi_proc.returncode}")
                time.sleep(0.5)
        raise RuntimeError(f"fldigi XML-RPC not ready after {timeout}s")

    def _is_alive(self):
        if self.fldigi_proc is None or self.fldigi_proc.poll() is not None:
            return False
        try:
            self.server.fldigi.name()
            return True
        except Exception:
            return False

    def _restart(self):
        self.stop()
        self._write_config()
        env = self._make_env()
        self.fldigi_proc = subprocess.Popen(
            ["xvfb-run", "-a", "fldigi",
             "--config-dir", self.config_dir,
             "--xmlrpc-server-port", str(self.port)],
            env=env, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
            start_new_session=True,
        )
        self.server = xmlrpc.client.ServerProxy(
            f"http://127.0.0.1:{self.port}")
        self._wait_ready()

    def _tx_chunk(self, mode_name, chunk_text, cadence=None):
        audio_path = os.path.join(self.audio_dir, f"{mode_name}_chunk.raw")
        rec_proc = None
        audio_fd = open(audio_path, "wb")
        try:
            rec_proc = subprocess.Popen(
                ["parec", f"--device={self.sink_name}.monitor",
                 f"--rate={CAPTURE_FS}", "--channels=1", "--format=s16le"],
                stdout=audio_fd, env=self._make_env(),
            )
            time.sleep(0.3)
            self.server.text.clear_tx()
            if cadence is not None:
                feed_characters_with_cadence(self.server, chunk_text, cadence)
            else:
                self.server.text.add_tx(chunk_text)
            self.server.main.tx()
            chars_per_sec = MODE_CHARS_PER_SEC.get(mode_name, 5)
            est_secs = min(len(chunk_text) / chars_per_sec + 10.0,
                           MAX_CHUNK_TIMEOUT)
            deadline = time.time() + est_secs
            tx_started = False
            while time.time() < deadline:
                try:
                    state = self.server.main.get_trx_state()
                    if state == "TX":
                        tx_started = True
                    elif tx_started and state == "RX":
                        break
                except Exception:
                    break
                time.sleep(0.5)
            try:
                self.server.main.rx()
            except Exception:
                pass
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
        threshold = max(1, int(np.max(np.abs(raw_data)) * 0.01))
        nonsilent = np.where(np.abs(raw_data) > threshold)[0]
        if len(nonsilent) == 0:
            return np.array([], dtype=np.int16)
        margin = int(0.1 * CAPTURE_FS)
        start = max(0, nonsilent[0] - margin)
        end = min(len(raw_data), nonsilent[-1] + margin)
        return raw_data[start:end]

    def generate_mode(self, mode_name, text, target_fs=FS, deadline=None):
        CHUNK_CHARS = 500
        MAX_RETRIES = 3
        modem_variants = FLDIGI_MODES[mode_name]
        cadence = TypingCadenceModel()
        chunks = [text[i:i + CHUNK_CHARS] for i in range(0, len(text), CHUNK_CHARS)]
        all_audio = []
        for ci, chunk in enumerate(chunks):
            if deadline and time.time() > deadline:
                log.warning("%s: mode timeout at chunk %d/%d, "
                            "using %d chunks captured so far",
                            mode_name, ci, len(chunks), len(all_audio))
                break
            modem_name = modem_variants[ci % len(modem_variants)]
            for attempt in range(MAX_RETRIES):
                if not self._is_alive():
                    log.debug("fldigi instance %d restarting", self.id)
                    self._restart()
                try:
                    self.server.modem.set_by_name(modem_name)
                    time.sleep(0.5)
                    try:
                        self.server.modem.set_carrier(1500)
                    except Exception:
                        pass
                except Exception:
                    continue
                raw = self._tx_chunk(mode_name, chunk, cadence=cadence)
                if len(raw) > 0:
                    all_audio.append(raw)
                    break
                elif not self._is_alive():
                    continue
                else:
                    break
        if not all_audio:
            return np.array([], dtype=np.complex128)
        combined = np.concatenate(all_audio)
        return audio_to_iq(combined, CAPTURE_FS, target_fs=target_fs)

    def stop(self):
        if self.fldigi_proc and self.fldigi_proc.poll() is None:
            try:
                pgid = os.getpgid(self.fldigi_proc.pid)
                os.killpg(pgid, signal.SIGTERM)
            except (ProcessLookupError, PermissionError):
                pass
            try:
                self.fldigi_proc.wait(timeout=10)
            except subprocess.TimeoutExpired:
                try:
                    pgid = os.getpgid(self.fldigi_proc.pid)
                    os.killpg(pgid, signal.SIGKILL)
                except (ProcessLookupError, PermissionError):
                    pass
                self.fldigi_proc.wait()


class FldigiGenerator(BaseGenerator):
    name = "fldigi"
    required_tools = ["fldigi", "parec", "xvfb-run"]
    signal_classes = list(FLDIGI_MODES.keys())

    def generate_class(self, class_name, rng=None):
        # This generator overrides run() for multi-instance orchestration
        raise NotImplementedError("Use run() for fldigi generation")

    def run(self, output_dir, seed=42, port=7362):
        """Sequential mode generation: one mode at a time, checkpoint after each.

        Uses a single fldigi instance to avoid multi-thread crash risk.
        A crash loses at most the current in-flight mode; all completed
        modes are already saved and will be skipped on restart.
        """
        import tempfile
        from .._state import shutdown_requested

        # Pre-flight: skip entirely if all modes cached
        # (avoids starting PulseAudio + fldigi instances for nothing)
        cached = self._check_all_cached(output_dir)
        if cached is not None:
            log.info("fldigi: all %d modes cached — skipping", len(cached))
            return cached

        configure_impairments(self.impairment_config)

        parts_dir = os.path.join(output_dir, "parts", self.name)
        os.makedirs(parts_dir, exist_ok=True)

        classes = self._resolve_classes()
        all_results = {}
        if not classes:
            return all_results

        stride = self.impairment_config.effective_stride(self.window_len)
        power_threshold = self.impairment_config.window_power_threshold

        log.info("fldigi: %d modes, 1 instance (sequential)", len(classes))

        tmpdir = tempfile.mkdtemp(prefix="fldigi_gen_")
        inst = None

        try:
            with IsolatedPulseServer() as pa:
                pa_env = pa.clean_env()
                inst = FLDigiInstance(0, port, tmpdir, pa_env)
                inst.start()
                try:
                    if inst.fldigi_proc:
                        pid_registry.register_child(
                            inst.fldigi_proc.pid, "fldigi",
                            port=port, config_dir=inst.config_dir)
                except Exception:
                    pass

                for mode_name in classes:
                    if shutdown_requested():
                        log.warning("Shutdown requested — stopping fldigi "
                                    "after %d/%d modes",
                                    len(all_results), len(classes))
                        break

                    npy_path = os.path.join(parts_dir, f"{mode_name}.npy")
                    meta_path = os.path.join(parts_dir,
                                             f"{mode_name}_meta.csv")
                    hash_path = os.path.join(parts_dir, f"{mode_name}.hash")
                    n_samples = self._boosted_count(mode_name)
                    cfg_hash = self._config_hash(mode_name, n_samples)

                    if self._check_checkpoint(npy_path, meta_path,
                                              hash_path, n_samples,
                                              cfg_hash):
                        log.info("%15s: cached", mode_name)
                        all_results[mode_name] = {"status": "cached",
                                                  "samples": n_samples}
                        continue

                    cps = MODE_CHARS_PER_SEC.get(mode_name, 5)
                    est_chars = int(
                        n_samples * (self.window_len / self.fs)
                        / 2 * cps * 1.5)
                    text = get_text_for_mode(mode_name, max(500, est_chars))
                    mode_deadline = time.time() + MODE_TIMEOUT

                    # Restart instance if it died between modes
                    if not inst._is_alive():
                        log.warning("fldigi instance died, restarting")
                        inst._restart()

                    iq = inst.generate_mode(mode_name, text,
                                            target_fs=self.fs,
                                            deadline=mode_deadline)
                    if len(iq) < self.window_len:
                        log.warning("%15s: FAILED (signal too short)",
                                    mode_name)
                        all_results[mode_name] = {
                            "status": "failed",
                            "reason": "signal too short"}
                        continue

                    raw_windows = extract_windows(
                        iq, window_len=self.window_len,
                        stride=stride,
                        power_threshold=power_threshold)
                    if len(raw_windows) == 0:
                        all_results[mode_name] = {
                            "status": "failed",
                            "reason": "no valid windows"}
                        continue

                    samples, meta = apply_impairments(
                        raw_windows, n_samples, fs=self.fs,
                        window_len=self.window_len,
                        return_metadata=True)
                    atomic_save_npy(npy_path, samples)
                    atomic_write_csv(meta_path, ["scenario"],
                                     [[s] for s in meta["scenarios"]])
                    self._write_hash(hash_path, cfg_hash)
                    log.info("%15s: %d raw -> %d samples",
                             mode_name, len(raw_windows), len(samples))
                    all_results[mode_name] = {
                        "status": "ok",
                        "samples": len(samples),
                        "raw_windows": len(raw_windows)}

        finally:
            if inst is not None:
                try:
                    if inst.fldigi_proc:
                        pid_registry.unregister_child(inst.fldigi_proc.pid)
                except Exception:
                    pass
                inst.stop()
            shutil.rmtree(tmpdir, ignore_errors=True)

        return all_results
