"""fldigi XML-RPC generator — 18+ signal classes via headless fldigi."""

import os
import signal
import shutil
import subprocess
import threading
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
MODE_TIMEOUT_MIN = 900  # floor: at least 15 minutes per mode
MAX_CHUNK_TIMEOUT = 120  # 2 minutes max waiting for a single TX chunk
CHUNK_CHARS = 500  # characters per TX chunk

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
    def __init__(self, instance_id, xmlrpc_port, work_dir, pa_env=None):
        self.id = instance_id
        self.port = xmlrpc_port
        self.work_dir = work_dir
        self.config_dir = os.path.join(work_dir, f"fldigi_{instance_id}")
        self.audio_dir = os.path.join(work_dir, f"audio_{instance_id}")
        self.sink_name = "capture"
        self.pa_env = pa_env
        self._own_pa = None  # per-instance PulseAudio server
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
        # Each instance gets its own PulseAudio server so we don't
        # overload a single PA daemon with many concurrent parec clients.
        if self.pa_env is None:
            self._own_pa = IsolatedPulseServer()
            self._own_pa.__enter__()
            self.pa_env = self._own_pa.clean_env()
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
            if rec_proc.poll() is not None:
                log.error("parec exited immediately — PulseAudio may be dead")
                return np.array([], dtype=np.int16)
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
                if rec_proc.poll() is not None:
                    log.error("parec died during TX — PulseAudio may be dead")
                    break
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
        MAX_RETRIES = 3
        modem_variants = FLDIGI_MODES[mode_name]
        cadence = TypingCadenceModel()
        chunks = [text[i:i + CHUNK_CHARS] for i in range(0, len(text), CHUNK_CHARS)]
        all_audio = []
        consecutive_failures = 0
        for ci, chunk in enumerate(chunks):
            if deadline and time.time() > deadline:
                log.warning("%s: mode timeout at chunk %d/%d, "
                            "using %d chunks captured so far",
                            mode_name, ci, len(chunks), len(all_audio))
                break
            if consecutive_failures >= 5:
                log.error("%s: %d consecutive chunk failures, aborting mode "
                          "(PulseAudio may be dead)",
                          mode_name, consecutive_failures)
                break
            modem_name = modem_variants[ci % len(modem_variants)]
            chunk_ok = False
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
                    chunk_ok = True
                    break
                elif not self._is_alive():
                    continue
                else:
                    break
            if chunk_ok:
                consecutive_failures = 0
            else:
                consecutive_failures += 1
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
        if self._own_pa is not None:
            self._own_pa.__exit__(None, None, None)
            self._own_pa = None


class FldigiGenerator(BaseGenerator):
    name = "fldigi"
    required_tools = ["fldigi", "parec", "xvfb-run"]
    signal_classes = list(FLDIGI_MODES.keys())

    def generate_class(self, class_name, rng=None):
        # This generator overrides run() for multi-instance orchestration
        raise NotImplementedError("Use run() for fldigi generation")

    def _postprocess_mode(self, mode_name, raw_chunks, n_total_chunks,
                          parts_dir, stride, power_threshold):
        """Concatenate audio chunks, convert to IQ, window, impair, save.

        Called when all TX chunks for a mode have been collected.
        Returns (mode_name, result_dict).
        """
        valid = [a for a in raw_chunks if a is not None and len(a) > 0]
        del raw_chunks
        n_valid = len(valid)

        if not valid:
            log.warning("%15s: FAILED (%d/%d chunks empty)",
                        mode_name, n_total_chunks, n_total_chunks)
            return mode_name, {"status": "failed", "reason": "no valid chunks"}

        log.info("%15s: %d/%d chunks captured, post-processing...",
                 mode_name, n_valid, n_total_chunks)

        combined = np.concatenate(valid)
        del valid
        iq = audio_to_iq(combined, CAPTURE_FS, target_fs=self.fs)
        del combined

        if len(iq) < self.window_len:
            log.warning("%15s: FAILED (signal too short)", mode_name)
            return mode_name, {"status": "failed",
                               "reason": "signal too short"}

        raw_windows = extract_windows(
            iq, window_len=self.window_len, stride=stride,
            power_threshold=power_threshold)
        del iq

        if len(raw_windows) == 0:
            log.warning("%15s: FAILED (no valid windows)", mode_name)
            return mode_name, {"status": "failed",
                               "reason": "no valid windows"}

        n_samples = self._boosted_count(mode_name)
        samples, meta = apply_impairments(
            raw_windows, n_samples, fs=self.fs,
            window_len=self.window_len, return_metadata=True)
        n_raw = len(raw_windows)
        del raw_windows
        n_out = len(samples)

        npy_path = os.path.join(parts_dir, f"{mode_name}.npy")
        meta_path = os.path.join(parts_dir, f"{mode_name}_meta.csv")
        hash_path = os.path.join(parts_dir, f"{mode_name}.hash")

        atomic_save_npy(npy_path, samples)
        snrs = meta.get("snrs", [])
        meta_rows = []
        for i, s in enumerate(meta["scenarios"]):
            snr = snrs[i] if i < len(snrs) else ""
            meta_rows.append([s, snr])
        atomic_write_csv(meta_path, ["scenario", "snr"], meta_rows)
        cfg_hash = self._config_hash(mode_name, n_samples)
        self._write_hash(hash_path, cfg_hash)
        del samples, meta

        log.info("%15s: %d raw -> %d samples (from %d/%d chunks)",
                 mode_name, n_raw, n_out, n_valid, n_total_chunks)
        return mode_name, {"status": "ok", "samples": n_out,
                           "raw_windows": n_raw}

    def run(self, output_dir, seed=42, port=7362):
        """Chunk-level parallel generation across all fldigi instances.

        Instead of assigning whole modes to instances 1:1, pre-splits every
        uncached mode into TX chunks and feeds them through a shared work
        queue.  All N instances pull chunks regardless of which mode they
        belong to, so 2 remaining modes still saturate 21 instances.

        Post-processing (windowing + impairments) runs in a small pool
        (max 3 concurrent) to cap memory, and fires as soon as all chunks
        for a mode are collected — overlapping with ongoing TX work.
        """
        import queue
        import tempfile
        from concurrent.futures import ThreadPoolExecutor, as_completed
        from .._state import shutdown_requested

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

        # Pre-check cache so we only launch instances for uncached modes
        uncached = []
        for mode_name in classes:
            n_samples = self._boosted_count(mode_name)
            cfg_hash = self._config_hash(mode_name, n_samples)
            npy_path = os.path.join(parts_dir, f"{mode_name}.npy")
            meta_path = os.path.join(parts_dir, f"{mode_name}_meta.csv")
            hash_path = os.path.join(parts_dir, f"{mode_name}.hash")
            if self._check_checkpoint(npy_path, meta_path, hash_path,
                                      n_samples, cfg_hash):
                log.info("%15s: cached", mode_name)
                all_results[mode_name] = {"status": "cached",
                                          "samples": n_samples}
            else:
                uncached.append(mode_name)

        if not uncached:
            log.info("fldigi: all %d modes cached — skipping", len(classes))
            return all_results

        max_workers = self.config.workers or 14

        stride = self.impairment_config.effective_stride(self.window_len)
        power_threshold = self.impairment_config.window_power_threshold

        # --- Pre-split all uncached modes into chunk work items ---
        mode_info = {}       # mode -> {n_chunks, n_samples}
        per_mode_chunks = [] # list of lists, one per mode

        for mode_name in uncached:
            n_samples = self._boosted_count(mode_name)
            cps = MODE_CHARS_PER_SEC.get(mode_name, 5)
            est_chars = int(
                n_samples * (self.window_len / self.fs) / 2 * cps * 1.5)
            text = get_text_for_mode(mode_name, max(500, est_chars))
            chunks = [text[i:i + CHUNK_CHARS]
                      for i in range(0, len(text), CHUNK_CHARS)]
            modem_variants = FLDIGI_MODES[mode_name]

            mode_info[mode_name] = {
                "n_chunks": len(chunks),
                "n_samples": n_samples,
            }

            items = []
            for ci, chunk_text in enumerate(chunks):
                modem_name = modem_variants[ci % len(modem_variants)]
                items.append((mode_name, ci, chunk_text, modem_name))
            per_mode_chunks.append(items)

        # Interleave chunks across modes so all modes make progress
        # together: chunk 0 of each mode, then chunk 1 of each, etc.
        work_queue = queue.Queue()
        max_chunk_idx = max(len(mc) for mc in per_mode_chunks)
        for ci in range(max_chunk_idx):
            for mode_chunks in per_mode_chunks:
                if ci < len(mode_chunks):
                    work_queue.put(mode_chunks[ci])
        del per_mode_chunks

        total_chunks = work_queue.qsize()
        n_workers = max(1, min(max_workers, total_chunks))

        log.info("fldigi: %d uncached modes, %d chunks across %d instances",
                 len(uncached), total_chunks, n_workers)
        for mn in uncached:
            log.info("  %15s: %d chunks, target %d samples",
                     mn, mode_info[mn]["n_chunks"],
                     mode_info[mn]["n_samples"])

        # Per-mode audio collectors (thread-safe via audio_lock)
        audio_lock = threading.Lock()
        audio_slots = {m: [None] * mode_info[m]["n_chunks"]
                       for m in uncached}
        chunks_done = {m: 0 for m in uncached}
        postproc_submitted = set()

        # Thread-safe results collection
        results_lock = threading.Lock()

        tmpdir = tempfile.mkdtemp(prefix="fldigi_gen_")
        instances = []

        try:
            for i in range(n_workers):
                inst = FLDigiInstance(i, port + i, tmpdir)
                inst.start()
                try:
                    if inst.fldigi_proc:
                        pid_registry.register_child(
                            inst.fldigi_proc.pid, "fldigi",
                            port=port + i,
                            config_dir=inst.config_dir)
                except Exception:
                    pass
                instances.append(inst)

            # Post-processing pool: max 3 concurrent to cap memory.
            # Runs alongside TX — as soon as all chunks for a mode
            # are collected, post-processing starts while other
            # instances keep pulling TX chunks.
            postproc_pool = ThreadPoolExecutor(max_workers=3)
            postproc_futures = []

            def _submit_postproc(mode_name):
                """Submit post-processing if all chunks for mode are done."""
                with audio_lock:
                    if chunks_done[mode_name] < mode_info[mode_name]["n_chunks"]:
                        return
                    if mode_name in postproc_submitted:
                        return
                    postproc_submitted.add(mode_name)
                    raw_chunks = audio_slots[mode_name]
                    audio_slots[mode_name] = None  # free ref

                n_total = mode_info[mode_name]["n_chunks"]

                def _do_postproc():
                    name, result = self._postprocess_mode(
                        mode_name, raw_chunks, n_total,
                        parts_dir, stride, power_threshold)
                    with results_lock:
                        all_results[name] = result

                f = postproc_pool.submit(_do_postproc)
                postproc_futures.append(f)

            def _tx_worker(worker_id, inst):
                """Pull chunks from shared queue and TX them."""
                consecutive_failures = 0

                while not shutdown_requested():
                    try:
                        item = work_queue.get(timeout=2)
                    except queue.Empty:
                        break

                    mode_name, chunk_idx, chunk_text, modem_name = item

                    # Ensure fldigi is alive and set modem
                    try:
                        if not inst._is_alive():
                            inst._restart()
                        inst.server.modem.set_by_name(modem_name)
                        time.sleep(0.3)
                        try:
                            inst.server.modem.set_carrier(1500)
                        except Exception:
                            pass
                    except Exception:
                        consecutive_failures += 1
                        if consecutive_failures >= 10:
                            log.error("Instance %d: %d consecutive "
                                      "failures, stopping",
                                      worker_id, consecutive_failures)
                            # Mark chunk done (empty) so mode can finish
                            with audio_lock:
                                chunks_done[mode_name] += 1
                            _submit_postproc(mode_name)
                            break
                        # Mark chunk done (empty) and continue
                        with audio_lock:
                            chunks_done[mode_name] += 1
                        _submit_postproc(mode_name)
                        time.sleep(1)
                        continue

                    cadence = TypingCadenceModel()
                    raw = inst._tx_chunk(mode_name, chunk_text,
                                         cadence=cadence)

                    with audio_lock:
                        if len(raw) > 0:
                            audio_slots[mode_name][chunk_idx] = raw
                            consecutive_failures = 0
                        else:
                            consecutive_failures += 1
                        chunks_done[mode_name] += 1

                    _submit_postproc(mode_name)

                    if consecutive_failures >= 10:
                        log.error("Instance %d: %d consecutive "
                                  "failures, stopping",
                                  worker_id, consecutive_failures)
                        break

            with ThreadPoolExecutor(max_workers=n_workers) as tx_pool:
                tx_futures = [
                    tx_pool.submit(_tx_worker, i, inst)
                    for i, inst in enumerate(instances)
                ]
                for f in as_completed(tx_futures):
                    try:
                        f.result()
                    except Exception as e:
                        log.error("TX worker failed: %s", e)

            # Wait for any in-flight post-processing to finish
            for f in as_completed(postproc_futures):
                try:
                    f.result()
                except Exception as e:
                    log.error("Post-processing failed: %s", e)

            postproc_pool.shutdown(wait=True)

            # Safety: post-process any modes that weren't submitted
            # (e.g. if all workers died before finishing a mode's chunks)
            for mode_name in uncached:
                if mode_name not in postproc_submitted \
                        and mode_name not in all_results:
                    log.warning("%15s: post-processing partial "
                                "(%d/%d chunks)",
                                mode_name, chunks_done.get(mode_name, 0),
                                mode_info[mode_name]["n_chunks"])
                    with audio_lock:
                        postproc_submitted.add(mode_name)
                        raw_chunks = audio_slots.get(mode_name) or []
                        audio_slots[mode_name] = None
                    n_total = mode_info[mode_name]["n_chunks"]
                    name, result = self._postprocess_mode(
                        mode_name, raw_chunks, n_total,
                        parts_dir, stride, power_threshold)
                    all_results[name] = result

        finally:
            for inst in instances:
                try:
                    if inst.fldigi_proc:
                        pid_registry.unregister_child(inst.fldigi_proc.pid)
                except Exception:
                    pass
                inst.stop()
            shutil.rmtree(tmpdir, ignore_errors=True)

        return all_results
