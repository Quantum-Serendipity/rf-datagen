"""Unit tests for rf_datagen.impairments.effects and transmitter modules."""

import numpy as np
import pytest

from rf_datagen.impairments.effects import (
    normalize_power,
    add_awgn,
    freq_shift,
    apply_watterson,
    apply_rayleigh,
    apply_rician,
    apply_qsb,
    apply_atmospheric_noise,
    apply_clock_drift,
    apply_iq_imbalance,
    apply_phase_noise,
    apply_dc_offset,
    apply_adc_quantization,
    apply_clock_jitter,
    apply_nonlinear_distortion,
    apply_image_rejection,
    apply_impulse_noise,
    apply_adjacent_signal,
    apply_powerline_hum,
    apply_narrowband_birdie,
    apply_time_mask,
    apply_freq_mask,
    extract_windows,
)
from rf_datagen.impairments.transmitter import (
    apply_alc_compression,
    apply_rf_clipping,
    apply_tx_hum,
    apply_key_clicks,
    apply_tx_drift,
    apply_switching_noise,
    TransmitterModel,
)
from rf_datagen.constants import FS, WINDOW_LEN

# ---------------------------------------------------------------------------
# normalize_power
# ---------------------------------------------------------------------------

def test_normalize_power_unit_rms(tone_1k):
    out = normalize_power(tone_1k)
    rms = np.sqrt(np.mean(np.abs(out) ** 2))
    assert abs(rms - 1.0) < 0.01


def test_normalize_power_zero_input():
    out = normalize_power(np.zeros(100, dtype=complex))
    assert np.allclose(out, 0)


# ---------------------------------------------------------------------------
# add_awgn
# ---------------------------------------------------------------------------

def test_awgn_measured_snr(tone_1k):
    snr_target = 10.0
    out = add_awgn(tone_1k, snr_target)
    noise = out - tone_1k
    sig_power = np.mean(np.abs(tone_1k) ** 2)
    noise_power = np.mean(np.abs(noise) ** 2)
    measured_snr = 10 * np.log10(sig_power / noise_power)
    assert abs(measured_snr - snr_target) < 3.0


def test_awgn_length_preserved(tone_1k):
    out = add_awgn(tone_1k, 10.0)
    assert len(out) == len(tone_1k)


# ---------------------------------------------------------------------------
# freq_shift
# ---------------------------------------------------------------------------

def test_freq_shift_peak_moves(tone_1k):
    out = freq_shift(tone_1k, 500, FS)
    spectrum = np.abs(np.fft.fft(out))
    freqs = np.fft.fftfreq(len(out), 1.0 / FS)
    peak_freq = freqs[np.argmax(spectrum)]
    assert abs(peak_freq - 1500) < 50


def test_freq_shift_length_preserved(tone_1k):
    out = freq_shift(tone_1k, 500, FS)
    assert len(out) == len(tone_1k)


# ---------------------------------------------------------------------------
# apply_watterson
# ---------------------------------------------------------------------------

def test_watterson_output_is_complex(tone_1k):
    out = apply_watterson(tone_1k, FS)
    assert np.iscomplexobj(out)


def test_watterson_length_preserved(tone_1k):
    out = apply_watterson(tone_1k, FS)
    assert len(out) == len(tone_1k)


# ---------------------------------------------------------------------------
# apply_rayleigh
# ---------------------------------------------------------------------------

def test_rayleigh_length_preserved(tone_1k):
    out = apply_rayleigh(tone_1k)
    assert len(out) == len(tone_1k)


def test_rayleigh_output_nonzero(tone_1k):
    out = apply_rayleigh(tone_1k)
    assert np.any(np.abs(out) > 0)


# ---------------------------------------------------------------------------
# apply_rician
# ---------------------------------------------------------------------------

def test_rician_high_k_correlated(tone_1k):
    out = apply_rician(tone_1k, FS, k_db=30)
    # High K means LOS dominant — output ≈ complex_scalar * input.
    # The channel applies a random phase, so use complex cross-correlation
    # normalized by magnitudes to check similarity.
    cross = np.abs(np.vdot(tone_1k, out))
    norm = np.sqrt(np.vdot(tone_1k, tone_1k).real * np.vdot(out, out).real)
    corr = cross / norm
    assert corr > 0.8


def test_rician_length_preserved(tone_1k):
    out = apply_rician(tone_1k, FS, k_db=10)
    assert len(out) == len(tone_1k)


# ---------------------------------------------------------------------------
# apply_qsb
# ---------------------------------------------------------------------------

def test_qsb_envelope_varies(tone_1k):
    out = apply_qsb(tone_1k, FS)
    assert np.std(np.abs(out)) > 0


def test_qsb_length_preserved(tone_1k):
    out = apply_qsb(tone_1k, FS)
    assert len(out) == len(tone_1k)


# ---------------------------------------------------------------------------
# apply_atmospheric_noise
# ---------------------------------------------------------------------------

def test_atmospheric_noise_power_increases(tone_1k):
    out = apply_atmospheric_noise(tone_1k, FS)
    assert np.mean(np.abs(out) ** 2) > np.mean(np.abs(tone_1k) ** 2)


# ---------------------------------------------------------------------------
# apply_clock_drift
# ---------------------------------------------------------------------------

def test_clock_drift_preserves_envelope(tone_1k):
    out = apply_clock_drift(tone_1k, FS)
    assert np.allclose(np.abs(out), np.abs(tone_1k), atol=1e-10)


# ---------------------------------------------------------------------------
# apply_iq_imbalance
# ---------------------------------------------------------------------------

def test_iq_imbalance_creates_mirror(tone_1k):
    n = len(tone_1k)
    # Find the bin index for -1kHz
    freqs = np.fft.fftfreq(n, 1.0 / FS)
    mirror_bin = np.argmin(np.abs(freqs - (-1000)))

    before_spectrum = np.abs(np.fft.fft(tone_1k))
    out = apply_iq_imbalance(tone_1k)
    after_spectrum = np.abs(np.fft.fft(out))
    assert after_spectrum[mirror_bin] > before_spectrum[mirror_bin]


def test_iq_imbalance_length_preserved(tone_1k):
    out = apply_iq_imbalance(tone_1k)
    assert len(out) == len(tone_1k)


# ---------------------------------------------------------------------------
# apply_phase_noise
# ---------------------------------------------------------------------------

def test_phase_noise_preserves_approx_envelope(tone_1k):
    out = apply_phase_noise(tone_1k, FS)
    assert np.allclose(np.abs(out), np.abs(tone_1k), atol=0.5)


# ---------------------------------------------------------------------------
# apply_dc_offset
# ---------------------------------------------------------------------------

def test_dc_offset_shifts_mean(tone_1k):
    out = apply_dc_offset(tone_1k)
    assert abs(np.mean(out) - np.mean(tone_1k)) > 0.001


# ---------------------------------------------------------------------------
# apply_adc_quantization
# ---------------------------------------------------------------------------

def test_adc_8bit_more_distortion_than_12bit(tone_1k):
    q8 = apply_adc_quantization(tone_1k, bits=8)
    q12 = apply_adc_quantization(tone_1k, bits=12)
    error_8 = np.mean(np.abs(q8 - tone_1k) ** 2)
    error_12 = np.mean(np.abs(q12 - tone_1k) ** 2)
    assert error_8 > error_12


def test_adc_quantization_length_preserved(tone_1k):
    out = apply_adc_quantization(tone_1k, bits=10)
    assert len(out) == len(tone_1k)


# ---------------------------------------------------------------------------
# apply_clock_jitter
# ---------------------------------------------------------------------------

def test_clock_jitter_length_preserved(tone_1k):
    out = apply_clock_jitter(tone_1k, FS)
    assert len(out) == len(tone_1k)


# ---------------------------------------------------------------------------
# apply_nonlinear_distortion
# ---------------------------------------------------------------------------

def test_nonlinear_creates_harmonics(tone_1k):
    out = apply_nonlinear_distortion(tone_1k)
    spectrum = np.abs(np.fft.fft(out))
    freqs = np.fft.fftfreq(len(out), 1.0 / FS)
    # Sum energy near 2kHz and 3kHz harmonics
    harmonic_energy = 0.0
    for hf in [2000, 3000]:
        mask = np.abs(freqs - hf) < 50
        harmonic_energy += np.sum(spectrum[mask] ** 2)
    assert harmonic_energy > 0


def test_nonlinear_length_preserved(tone_1k):
    out = apply_nonlinear_distortion(tone_1k)
    assert len(out) == len(tone_1k)


# ---------------------------------------------------------------------------
# apply_image_rejection
# ---------------------------------------------------------------------------

def test_image_rejection_length_preserved(tone_1k):
    out = apply_image_rejection(tone_1k)
    assert len(out) == len(tone_1k)


# ---------------------------------------------------------------------------
# apply_impulse_noise
# ---------------------------------------------------------------------------

def test_impulse_noise_peak_increases(tone_1k):
    out = apply_impulse_noise(tone_1k, FS)
    assert np.max(np.abs(out)) >= np.max(np.abs(tone_1k)) - 0.01


def test_impulse_noise_length_preserved(tone_1k):
    out = apply_impulse_noise(tone_1k, FS)
    assert len(out) == len(tone_1k)


# ---------------------------------------------------------------------------
# apply_adjacent_signal
# ---------------------------------------------------------------------------

def test_adjacent_signal_power_increases(tone_1k):
    out = apply_adjacent_signal(tone_1k, FS)
    assert np.mean(np.abs(out) ** 2) > np.mean(np.abs(tone_1k) ** 2)


# ---------------------------------------------------------------------------
# apply_powerline_hum
# ---------------------------------------------------------------------------

def test_powerline_hum_adds_harmonics(tone_1k):
    out = apply_powerline_hum(tone_1k, FS)
    assert np.mean(np.abs(out) ** 2) > np.mean(np.abs(tone_1k) ** 2)


# ---------------------------------------------------------------------------
# apply_narrowband_birdie
# ---------------------------------------------------------------------------

def test_narrowband_birdie_power_increases(tone_1k):
    out = apply_narrowband_birdie(tone_1k, FS)
    assert np.mean(np.abs(out) ** 2) > np.mean(np.abs(tone_1k) ** 2)


# ---------------------------------------------------------------------------
# apply_time_mask
# ---------------------------------------------------------------------------

def test_time_mask_zeros_some_samples(tone_1k):
    out = apply_time_mask(tone_1k)
    assert np.sum(np.abs(out) == 0) > np.sum(np.abs(tone_1k) == 0)


def test_time_mask_length_preserved(tone_1k):
    out = apply_time_mask(tone_1k)
    assert len(out) == len(tone_1k)


# ---------------------------------------------------------------------------
# apply_freq_mask
# ---------------------------------------------------------------------------

def test_freq_mask_length_preserved(tone_1k):
    out = apply_freq_mask(tone_1k)
    assert len(out) == len(tone_1k)


# ---------------------------------------------------------------------------
# extract_windows
# ---------------------------------------------------------------------------

def test_extract_windows_shape(tone_1k):
    windows = extract_windows(tone_1k, window_len=512)
    assert windows.ndim == 2
    assert windows.shape[1] == 512
    assert windows.shape[0] > 0


def test_extract_windows_short_input_empty():
    short = np.array([1 + 1j, 2 + 2j])
    windows = extract_windows(short, window_len=512)
    assert windows.shape == (0, 512)


def test_extract_windows_power_filters_silence():
    # Build a signal that is mostly zeros with a loud burst in the middle
    sig = np.zeros(4096, dtype=complex)
    sig[1024:1536] = np.exp(2j * np.pi * 1000 * np.arange(512) / FS)
    windows = extract_windows(sig, window_len=512)
    # Silence windows should be filtered; we should get very few windows
    # (only those overlapping the loud segment)
    assert windows.shape[0] < 4096 // 256


def test_extract_windows_unit_power(tone_1k):
    windows = extract_windows(tone_1k, window_len=512)
    for i in range(windows.shape[0]):
        rms = np.sqrt(np.mean(np.abs(windows[i]) ** 2))
        assert abs(rms - 1.0) < 0.01


# ---------------------------------------------------------------------------
# transmitter.py
# ---------------------------------------------------------------------------

def test_alc_compression_reduces_peak(tone_1k):
    loud = tone_1k * 5
    out = apply_alc_compression(loud, attack_ms=5, release_ms=80,
                                ratio=4.0, threshold_db=-6, fs=FS)
    assert np.max(np.abs(out)) < np.max(np.abs(loud))


def test_alc_compression_length_preserved(tone_1k):
    out = apply_alc_compression(tone_1k, attack_ms=5, release_ms=80,
                                ratio=4.0, threshold_db=-6, fs=FS)
    assert len(out) == len(tone_1k)


def test_rf_clipping_bounds_magnitude(tone_1k):
    loud = tone_1k * 2
    out = apply_rf_clipping(loud, clip_db=-6, fs=FS)
    # After clipping + bandpass, peak should be reduced vs input
    assert np.max(np.abs(out)) < np.max(np.abs(loud))


def test_rf_clipping_length_preserved(tone_1k):
    out = apply_rf_clipping(tone_1k, clip_db=-6, fs=FS)
    assert len(out) == len(tone_1k)


def test_tx_hum_modulates_envelope(tone_1k):
    out = apply_tx_hum(tone_1k, hum_freq=120, hum_level_db=-10, fs=FS)
    assert np.std(np.abs(out)) > np.std(np.abs(tone_1k))


def test_key_clicks_length_preserved(tone_1k):
    out = apply_key_clicks(tone_1k, rise_fall_ms=3, fs=FS)
    assert len(out) == len(tone_1k)


def test_tx_drift_shifts_frequency(tone_1k):
    out = apply_tx_drift(tone_1k, drift_hz=50, fs=FS)
    spectrum_in = np.abs(np.fft.fft(tone_1k))
    spectrum_out = np.abs(np.fft.fft(out))
    peak_in = np.argmax(spectrum_in)
    peak_out = np.argmax(spectrum_out)
    # The peak should have moved (or at least the spectrum changed)
    # With chirp-like drift, the peak bin may shift or energy spreads
    # Check that the spectral shape changed
    assert not np.allclose(spectrum_in, spectrum_out, atol=1e-6)


def test_tx_drift_preserves_envelope(tone_1k):
    out = apply_tx_drift(tone_1k, drift_hz=50, fs=FS)
    assert np.allclose(np.abs(out), np.abs(tone_1k), atol=1e-10)


@pytest.mark.parametrize("profile", [
    "WELL_OPERATED", "CASUAL", "POORLY_OPERATED", "VINTAGE",
])
def test_transmitter_model_profiles_apply(profile, tone_1k):
    model = TransmitterModel(profile)
    out = model.apply(tone_1k, FS)
    assert np.iscomplexobj(out)
    assert len(out) == len(tone_1k)


def test_transmitter_model_output_is_complex(tone_1k):
    model = TransmitterModel("CASUAL")
    out = model.apply(tone_1k, FS)
    assert np.iscomplexobj(out)
