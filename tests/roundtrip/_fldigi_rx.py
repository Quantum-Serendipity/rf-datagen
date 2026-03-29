"""fldigi RX pipeline validator — 18 modes decoded via headless fldigi.

Pipeline: TX fldigi → capture audio → add AWGN → play to RX fldigi → compare.
"""

import os
import shutil
import subprocess
import tempfile
import time

import numpy as np

from rf_datagen.generators.fldigi import (
    FLDigiInstance, FLDIGI_MODES, MODE_CHARS_PER_SEC, CAPTURE_FS)
from rf_datagen.isolation import IsolatedPulseServer

from ._base import BaseRoundtripValidator, register
from ._helpers import write_mono_wav, add_awgn_audio


# All fldigi-decodable modes (18 narrowband modes excluding CW, PACKET, FAX)
_FLDIGI_RX_MODES = [
    "PSK31", "PSK63", "QPSK", "PSK125", "8PSK",
    "RTTY", "OLIVIA", "DOMINOEX", "MT63", "HELLSCHREIBER",
    "MFSK16", "MFSK32", "CONTESTIA", "THOR",
    "FSQ", "IFKP", "THROB", "NAVTEX",
]

# Short test texts per mode — kept short for fast TX/RX cycle
_TEST_TEXTS = {
    "PSK31":         "CQ CQ DE W1AW W1AW K",
    "PSK63":         "CQ CQ DE W1AW W1AW K",
    "QPSK":          "CQ CQ DE W1AW W1AW K",
    "PSK125":        "CQ CQ DE W1AW W1AW K",
    "8PSK":          "CQ CQ DE W1AW W1AW K",
    "RTTY":          "RYRYRY DE W1AW W1AW K",
    "OLIVIA":        "CQ CQ DE W1AW K",
    "DOMINOEX":      "CQ CQ DE W1AW K",
    "MT63":          "CQ CQ DE W1AW K",
    "HELLSCHREIBER": "CQ DE W1AW",
    "MFSK16":        "CQ CQ DE W1AW K",
    "MFSK32":        "CQ CQ DE W1AW K",
    "CONTESTIA":     "CQ CQ DE W1AW K",
    "THOR":          "CQ CQ DE W1AW K",
    "FSQ":           "CQ DE W1AW",
    "IFKP":          "CQ DE W1AW",
    "THROB":         "CQ W1AW",
    "NAVTEX":        "ZCZC FA00 WEATHER FORECAST",
}


def _levenshtein_ratio(a, b):
    """Simple Levenshtein distance ratio (0.0 = no match, 1.0 = exact)."""
    if not a or not b:
        return 0.0
    if a == b:
        return 1.0
    n, m = len(a), len(b)
    if n > m:
        a, b = b, a
        n, m = m, n
    prev = list(range(n + 1))
    for j in range(1, m + 1):
        curr = [j] + [0] * n
        for i in range(1, n + 1):
            cost = 0 if a[i - 1] == b[j - 1] else 1
            curr[i] = min(prev[i] + 1, curr[i - 1] + 1, prev[i - 1] + cost)
        prev = curr
    dist = prev[n]
    return 1.0 - dist / max(n, m)


def _compare_fldigi_text(expected, decoded, mode):
    """Compare TX text to RX decoded text.

    Case-insensitive. Accepts if decoded contains >=50% of expected as a
    substring or Levenshtein ratio >= 0.4.
    """
    if not decoded or not expected:
        return False
    exp = expected.upper().strip()
    dec = decoded.upper().strip()
    # Direct substring
    if exp in dec:
        return True
    # Check if most expected words appear
    exp_words = exp.split()
    if exp_words:
        matches = sum(1 for w in exp_words if w in dec)
        if matches >= len(exp_words) * 0.5:
            return True
    # Levenshtein fallback
    return _levenshtein_ratio(exp, dec) >= 0.4


@register(*_FLDIGI_RX_MODES)
class FldigiRXValidator(BaseRoundtripValidator):
    """Validate fldigi modes via TX→capture→noise→RX pipeline."""

    required_tools = ["fldigi", "parec", "paplay", "xvfb-run"]
    tier = 1

    def __init__(self):
        self._pa = None
        self._tx = None
        self._rx = None
        self._tmpdir = None

    def setup(self):
        self._tmpdir = tempfile.mkdtemp(prefix="fldigi_rx_")
        self._pa = IsolatedPulseServer()
        self._pa.__enter__()
        pa_env = self._pa.clean_env()
        # TX instance on port 7380
        self._tx = FLDigiInstance(0, 7380, self._tmpdir, pa_env)
        self._tx.start()
        # RX instance on port 7381
        self._rx = FLDigiInstance(1, 7381, self._tmpdir, pa_env)
        self._rx.start()

    def teardown(self):
        if self._tx:
            self._tx.stop()
        if self._rx:
            self._rx.stop()
        if self._pa:
            self._pa.__exit__(None, None, None)
        if self._tmpdir:
            shutil.rmtree(self._tmpdir, ignore_errors=True)

    def _tx_generate(self, mode, text):
        """Generate audio via TX fldigi, return int16 raw audio."""
        fldigi_modem = FLDIGI_MODES[mode][0]
        try:
            self._tx.server.modem.set_by_name(fldigi_modem)
            time.sleep(0.3)
            try:
                self._tx.server.modem.set_carrier(1500)
            except Exception:
                pass
        except Exception:
            return np.array([], dtype=np.int16)
        return self._tx._tx_chunk(mode, text)

    def _rx_decode(self, wav_path, mode, timeout_s=None):
        """Play WAV to PA, let RX fldigi decode, return decoded text."""
        fldigi_modem = FLDIGI_MODES[mode][0]
        try:
            self._rx.server.modem.set_by_name(fldigi_modem)
            time.sleep(0.3)
            try:
                self._rx.server.modem.set_carrier(1500)
            except Exception:
                pass
            self._rx.server.text.clear_rx()
        except Exception:
            return ""

        # Play noisy WAV into PulseAudio (RX reads from monitor source)
        pa_env = self._rx.pa_env
        try:
            subprocess.run(
                ["paplay", "--file-format=wav", wav_path],
                env=pa_env, timeout=timeout_s or 60,
                stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        except (subprocess.TimeoutExpired, FileNotFoundError):
            return ""

        # Wait for decoding to settle (no new chars for 2s)
        prev_text = ""
        stable_since = time.time()
        deadline = time.time() + (timeout_s or 15)
        while time.time() < deadline:
            try:
                text = self._rx.server.text.get_rx()
            except Exception:
                break
            if text != prev_text:
                prev_text = text
                stable_since = time.time()
            elif time.time() - stable_since >= 2.0:
                break
            time.sleep(0.3)

        try:
            return self._rx.server.text.get_rx()
        except Exception:
            return prev_text

    def make_trial(self, mode):
        text = _TEST_TEXTS.get(mode, "CQ CQ DE W1AW K")
        cps = MODE_CHARS_PER_SEC.get(mode, 5)
        est_duration = len(text) / cps + 10

        def run(snr_db, trial_idx, tmpdir):
            # TX: generate audio
            raw_audio = self._tx_generate(mode, text)
            if len(raw_audio) == 0:
                return False

            # Convert to float, add noise, save WAV
            audio = raw_audio.astype(np.float64) / 32768.0
            if snr_db is not None:
                audio = add_awgn_audio(audio, snr_db)
            wav_path = os.path.join(tmpdir, "fldigi_test.wav")
            write_mono_wav(audio, CAPTURE_FS, wav_path)

            # RX: decode
            decoded = self._rx_decode(wav_path, mode,
                                      timeout_s=est_duration + 10)
            return _compare_fldigi_text(text, decoded, mode)

        return run
