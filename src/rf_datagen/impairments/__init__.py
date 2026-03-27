"""Channel models and augmentation — realistic RF impairments."""

from .effects import (
    normalize_power, add_awgn, freq_shift,
    apply_watterson, apply_rayleigh, apply_rician,
    apply_qsb, apply_atmospheric_noise, apply_clock_drift,
    apply_iq_imbalance, apply_phase_noise, apply_dc_offset,
    apply_adc_quantization, apply_clock_jitter,
    apply_nonlinear_distortion, apply_image_rejection,
    apply_impulse_noise, apply_adjacent_signal,
    apply_powerline_hum, apply_narrowband_birdie,
    apply_time_mask, apply_freq_mask,
    extract_windows,
)
from .transmitter import TransmitterModel
from .scenarios import (
    configure as configure_impairments,
    apply_scenario, apply_scenario_continuous,
    apply_impairments, SCENARIO_NAMES,
)

__all__ = [
    "normalize_power", "add_awgn", "freq_shift",
    "apply_watterson", "apply_rayleigh", "apply_rician",
    "apply_qsb", "apply_atmospheric_noise",
    "apply_scenario", "apply_scenario_continuous", "apply_impairments",
    "configure_impairments",
    "extract_windows", "TransmitterModel",
    "SCENARIO_NAMES",
]
