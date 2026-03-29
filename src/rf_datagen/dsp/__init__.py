"""Core DSP primitives — single source of truth for modulation and filtering."""

from .analytic import hilbert_analytic, audio_to_iq
from .modulation import (gfsk_mod, fsk_mod, ook_mod, psk_mod,
                          _4fsk_mod, _gmsk_mod, _pi4dqpsk_mod,
                          ofdm_carriers, chirp_mod, dsss_mod,
                          oqpsk_mod, ppm_mod, ofdm_full)
from .filters import bandpass_filter, rrc_filter, gaussian_filter, gaussian_filter_sigma
from .modulation_sdr import sdr_available, get_fast_modulator, benchmark as sdr_benchmark

__all__ = [
    "hilbert_analytic", "audio_to_iq",
    "gfsk_mod", "fsk_mod", "ook_mod", "psk_mod",
    "_4fsk_mod", "_gmsk_mod", "_pi4dqpsk_mod", "ofdm_carriers",
    "chirp_mod", "dsss_mod", "oqpsk_mod", "ppm_mod", "ofdm_full",
    "bandpass_filter", "rrc_filter", "gaussian_filter",
    "sdr_available", "get_fast_modulator", "sdr_benchmark",
]
