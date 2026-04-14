"""SSTV generator via PySSTV."""

import importlib
import os

import numpy as np
from scipy.signal import resample

from ..constants import FS, WINDOW_LEN
from ..dsp import hilbert_analytic, audio_to_iq
from ..content.images import random_image
from ..impairments import extract_windows, apply_impairments
from ..logging_config import get_logger
from .base import BaseGenerator

log = get_logger("sstv")

SSTV_AUDIO_FS = 48000

SSTV_MODES = {
    "Robot36":   ("pysstv.color", "Robot36",   320, 240),
    "Robot72":   ("pysstv.color", "Robot72",   320, 240),
    "MartinM1":  ("pysstv.color", "MartinM1",  320, 256),
    "MartinM2":  ("pysstv.color", "MartinM2",  320, 256),
    "ScottieS1": ("pysstv.color", "ScottieS1", 320, 256),
    "ScottieS2": ("pysstv.color", "ScottieS2", 320, 256),
    "PD90":      ("pysstv.color", "PD90",      320, 256),
    "PD120":     ("pysstv.color", "PD120",     640, 496),
}


def _get_sstv_class(module_name, class_name):
    mod = importlib.import_module(module_name)
    return getattr(mod, class_name)


def encode_sstv(mode_name):
    module_name, class_name, width, height = SSTV_MODES[mode_name]
    sstv_class = _get_sstv_class(module_name, class_name)
    img = random_image(width, height)
    sstv = sstv_class(img, SSTV_AUDIO_FS, 16)
    sstv.vox_enabled = False
    samples = np.array(list(sstv.gen_samples()), dtype=np.float64)
    peak = np.max(np.abs(samples))
    if peak > 0:
        samples /= peak
    return samples


class SstvGenerator(BaseGenerator):
    name = "sstv"
    required_tools = []
    signal_classes = ["SSTV"]

    def generate_class(self, class_name, rng=None):
        all_iq_segments = []
        images_per_mode = self.config.images_per_mode
        for mode_name in SSTV_MODES:
            for _ in range(images_per_mode):
                try:
                    audio = encode_sstv(mode_name)
                    if len(audio) < 1000:
                        continue
                    iq = audio_to_iq(audio, SSTV_AUDIO_FS, target_fs=self.fs)
                    if len(iq) >= self.window_len:
                        all_iq_segments.append(iq)
                except Exception as e:
                    log.warning("SSTV encode failed for mode %s: %s", mode_name, e)
                    continue
        if not all_iq_segments:
            return np.array([], dtype=np.complex128)
        return np.concatenate(all_iq_segments)
