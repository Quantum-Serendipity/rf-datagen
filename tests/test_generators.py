"""Unit tests for rf-datagen generator modules."""

import numpy as np
import pytest

from rf_datagen.generators.synthetic import SYNTHESIZERS
from rf_datagen.generators.analog import modulate_ssb, modulate_am, modulate_fm
from rf_datagen.generators.digivoice import modulate_4fsk, modulate_gmsk
from rf_datagen.generators.wsjtx import (
    parse_ft8_symbols,
    parse_jt65_symbols,
    parse_jt9_symbols,
    synthesize_gfsk_tones,
)
from rf_datagen.constants import FS, WINDOW_LEN


# ---------------------------------------------------------------------------
# Parametrized tests across all 37 synthesizer modes
# ---------------------------------------------------------------------------

ALL_MODES = list(SYNTHESIZERS.keys())


@pytest.mark.parametrize("mode", ALL_MODES)
def test_synth_output_is_complex(mode):
    output = SYNTHESIZERS[mode](fs=FS, window_len=WINDOW_LEN)
    assert np.iscomplexobj(output), f"{mode} output should be complex"


@pytest.mark.parametrize("mode", ALL_MODES)
def test_synth_output_long_enough(mode):
    output = SYNTHESIZERS[mode](fs=FS, window_len=WINDOW_LEN)
    assert len(output) >= WINDOW_LEN, (
        f"{mode} output length {len(output)} < WINDOW_LEN {WINDOW_LEN}"
    )


@pytest.mark.parametrize("mode", ALL_MODES)
def test_synth_output_finite(mode):
    output = SYNTHESIZERS[mode](fs=FS, window_len=WINDOW_LEN)
    assert np.all(np.isfinite(output)), f"{mode} output contains non-finite values"


# ---------------------------------------------------------------------------
# Mode-specific structural tests
# ---------------------------------------------------------------------------

def test_synth_cw_has_on_off_keying():
    output = SYNTHESIZERS["CW"](fs=FS, window_len=WINDOW_LEN)
    envelope = np.abs(output)
    peak = np.max(envelope)
    assert peak > 0, "CW signal should have non-zero peak"
    # There should be samples below 10% of peak (key-up) and above 50% (key-down)
    assert np.any(envelope < 0.1 * peak), "CW should have near-zero (key-up) regions"
    assert np.any(envelope > 0.5 * peak), "CW should have near-max (key-down) regions"


def test_synth_fm_constant_envelope():
    output = SYNTHESIZERS["FM"](fs=FS, window_len=WINDOW_LEN)
    envelope = np.abs(output)
    ratio = np.std(envelope) / (np.mean(envelope) + 1e-12)
    assert ratio < 0.15, f"FM envelope variation {ratio:.3f} exceeds 0.15 threshold"


def test_synth_am_has_carrier():
    output = SYNTHESIZERS["AM"](fs=FS, window_len=WINDOW_LEN)
    envelope = np.abs(output)
    assert np.mean(envelope) > 0.5 * np.max(envelope), (
        "AM signal should have a strong carrier component"
    )


def test_synth_ssb_single_sideband():
    output = SYNTHESIZERS["SSB"](fs=FS, window_len=WINDOW_LEN)
    spectrum = np.fft.fft(output)
    n = len(spectrum)
    positive_energy = np.sum(np.abs(spectrum[1:n // 2]) ** 2)
    negative_energy = np.sum(np.abs(spectrum[n // 2 + 1:]) ** 2)
    total = positive_energy + negative_energy
    # One half should have >70% of the energy
    dominant = max(positive_energy, negative_energy)
    assert dominant / total > 0.70, (
        f"SSB should concentrate >70% energy in one sideband, "
        f"got {dominant / total:.2f}"
    )


def test_synth_dmr_has_tdma_gaps():
    output = SYNTHESIZERS["DMR"](fs=FS, window_len=WINDOW_LEN)
    envelope = np.abs(output)
    # DMR TDMA gaps are ~30 samples each; check for windows of 10 samples
    # with envelope essentially zero
    window_size = 10
    gap_count = 0
    for i in range(0, len(envelope) - window_size, window_size):
        chunk = envelope[i:i + window_size]
        if np.max(chunk) < 0.01:
            gap_count += 1
    assert gap_count >= 3, (
        f"DMR signal should have multiple TDMA gap windows, found {gap_count}"
    )


def test_synth_noise_broadband():
    output = SYNTHESIZERS["NOISE"](fs=FS, window_len=WINDOW_LEN)
    spectrum = np.abs(np.fft.fft(output)) ** 2
    n = len(spectrum)
    half = spectrum[1:n // 2]
    # Only consider bins with significant energy (handles bandlimited variant
    # which zeroes out-of-band frequencies)
    threshold = np.max(half) * 1e-4  # -40 dB from peak
    active_bins = half[half > threshold]
    assert len(active_bins) > 10, "Noise should occupy multiple spectral bins"
    # Within the active passband, PSD should be relatively flat
    kernel_size = max(1, len(active_bins) // 20)
    smoothed = np.convolve(active_bins, np.ones(kernel_size) / kernel_size,
                           mode="valid")
    ratio_db = 10 * np.log10(np.max(smoothed) / (np.min(smoothed) + 1e-30))
    assert ratio_db < 20, f"Noise PSD max/min ratio {ratio_db:.1f} dB exceeds 20 dB"


def test_synth_mt63_is_ofdm():
    output = SYNTHESIZERS["MT63"](fs=FS, window_len=WINDOW_LEN)
    spectrum = np.abs(np.fft.fft(output))
    peak = np.max(spectrum)
    threshold = peak * 10 ** (-20 / 20)  # -20 dB from max
    n_peaks_above = np.sum(spectrum > threshold)
    assert n_peaks_above > 10, (
        f"MT63 should show >10 spectral peaks above -20 dB, found {n_peaks_above}"
    )


def test_synth_ft8_79_symbols():
    output = SYNTHESIZERS["FT8"](fs=FS, window_len=WINDOW_LEN)
    # 79 symbols * 0.160s/symbol * 12000 Hz = 151680 samples
    assert len(output) > 100000, (
        f"FT8 output length {len(output)} is too short for 79 symbols"
    )


# ---------------------------------------------------------------------------
# analog.py pure modulation tests
# ---------------------------------------------------------------------------

@pytest.fixture
def test_audio():
    """1 second of 440 Hz sine at 48000 Hz."""
    return np.sin(2 * np.pi * 440 * np.arange(48000) / 48000)


def test_ssb_output_is_complex(test_audio):
    output = modulate_ssb(test_audio, 48000)
    assert np.iscomplexobj(output)


def test_ssb_usb_positive_spectrum(test_audio):
    output = modulate_ssb(test_audio, 48000, sideband="USB")
    spectrum = np.fft.fft(output)
    n = len(spectrum)
    positive_energy = np.sum(np.abs(spectrum[1:n // 2]) ** 2)
    negative_energy = np.sum(np.abs(spectrum[n // 2 + 1:]) ** 2)
    total = positive_energy + negative_energy
    assert positive_energy / total > 0.70, (
        f"USB should have >70% positive freq energy, got {positive_energy / total:.2f}"
    )


def test_ssb_lsb_negative_spectrum(test_audio):
    output = modulate_ssb(test_audio, 48000, sideband="LSB")
    spectrum = np.fft.fft(output)
    n = len(spectrum)
    positive_energy = np.sum(np.abs(spectrum[1:n // 2]) ** 2)
    negative_energy = np.sum(np.abs(spectrum[n // 2 + 1:]) ** 2)
    total = positive_energy + negative_energy
    assert negative_energy / total > 0.70, (
        f"LSB should have >70% negative freq energy, got {negative_energy / total:.2f}"
    )


def test_am_output_is_complex(test_audio):
    output = modulate_am(test_audio, 48000)
    assert np.iscomplexobj(output)


def test_am_envelope_correlates_with_audio(test_audio):
    output = modulate_am(test_audio, 48000, mod_index=0.8)
    # Resample audio to 12000 for comparison
    from scipy.signal import resample
    target_len = int(len(test_audio) * FS / 48000)
    resampled_audio = resample(test_audio, target_len)
    envelope = np.abs(output)
    # Trim to matching lengths
    min_len = min(len(envelope), len(resampled_audio))
    corr = np.abs(np.corrcoef(envelope[:min_len], resampled_audio[:min_len])[0, 1])
    assert corr > 0.3, f"AM envelope should correlate with audio, got r={corr:.3f}"


def test_fm_output_is_complex(test_audio):
    output = modulate_fm(test_audio, 48000)
    assert np.iscomplexobj(output)


def test_fm_constant_envelope(test_audio):
    output = modulate_fm(test_audio, 48000)
    envelope = np.abs(output)
    ratio = np.std(envelope) / (np.mean(envelope) + 1e-12)
    assert ratio < 0.05, f"FM envelope variation {ratio:.4f} exceeds 0.05 threshold"


# ---------------------------------------------------------------------------
# digivoice.py modulation tests
# ---------------------------------------------------------------------------

def test_4fsk_unit_envelope():
    dibits = np.array([0, 1, 2, 3] * 50, dtype=np.uint8)
    output = modulate_4fsk(dibits, sym_rate=4800, dev_outer=1944, dev_inner=648)
    envelope = np.abs(output)
    within_tolerance = np.sum(np.abs(envelope - 1.0) < 0.15) / len(envelope)
    assert within_tolerance > 0.80, (
        f"4FSK envelope should be ~1.0 for >80% of samples, got {within_tolerance:.2f}"
    )


def test_gmsk_unit_envelope():
    bits = np.random.randint(0, 2, 500).astype(np.uint8)
    output = modulate_gmsk(bits, bit_rate=4800)
    envelope = np.abs(output)
    within_tolerance = np.sum(np.abs(envelope - 1.0) < 0.15) / len(envelope)
    assert within_tolerance > 0.80, (
        f"GMSK envelope should be ~1.0 for >80% of samples, got {within_tolerance:.2f}"
    )


# ---------------------------------------------------------------------------
# wsjtx.py parser and synthesizer tests
# ---------------------------------------------------------------------------

def test_parse_ft8_valid():
    stdout = "Message:  CQ W1AW FN31\nChannel symbols:\n" + "0" * 79 + "\n"
    result = parse_ft8_symbols(stdout)
    assert result is not None
    assert len(result) == 79
    assert all(isinstance(s, int) for s in result)


def test_parse_ft8_garbage_returns_none():
    assert parse_ft8_symbols("garbage input with no channel symbols") is None


def test_parse_jt65_valid():
    header = "Source-encoded message:\nInformation-carrying channel symbols:\n"
    data_line = "  " + " ".join(["0"] * 63) + "\n"
    stdout = header + data_line
    result = parse_jt65_symbols(stdout)
    assert result is not None
    assert len(result) == 126
    assert all(isinstance(s, int) for s in result)


def test_parse_jt9_valid():
    stdout = "Message\nChannel symbols:\n " + " ".join(["0"] * 85) + "\n"
    result = parse_jt9_symbols(stdout)
    assert result is not None
    assert len(result) == 85
    assert all(isinstance(s, int) for s in result)


def test_synth_gfsk_tones_output_length():
    symbols = [0] * 79
    output = synthesize_gfsk_tones(
        symbols, tone_spacing=6.25, symbol_rate=6.25, base_freq=1500, fs=12000,
    )
    expected = int(79 * 12000 / 6.25)  # 151680
    assert len(output) == expected, (
        f"Expected {expected} samples, got {len(output)}"
    )


def test_synth_gfsk_tones_output_is_real():
    symbols = [0, 1, 2, 3, 4, 5, 6, 7] * 10
    output = synthesize_gfsk_tones(
        symbols, tone_spacing=6.25, symbol_rate=6.25, base_freq=1500, fs=12000,
    )
    assert not np.iscomplexobj(output), "GFSK tones output should be real-valued"
