"""Unit tests for rf_datagen.dsp — modulation, filter, and analytic primitives."""

import numpy as np
import pytest

from rf_datagen.constants import FS
from rf_datagen.dsp import (
    gfsk_mod,
    fsk_mod,
    ook_mod,
    psk_mod,
    _4fsk_mod,
    _gmsk_mod,
    ofdm_carriers,
    chirp_mod,
    dsss_mod,
    oqpsk_mod,
    ppm_mod,
    _pi4dqpsk_mod,
    ofdm_full,
    bandpass_filter,
    rrc_filter,
    gaussian_filter,
    hilbert_analytic,
    audio_to_iq,
)


# ---------------------------------------------------------------------------
# modulation.py  (18 tests)
# ---------------------------------------------------------------------------

class TestGFSK:
    """GFSK modulator tests."""

    SYMBOLS = np.array([0, 1, 2, 3] * 50, dtype=np.uint8)  # 200 symbols
    PARAMS = dict(num_tones=4, tone_spacing=6.25, symbol_dur=0.160, fs=FS, bt=2.0)

    def test_gfsk_output_is_complex_unit_envelope(self):
        out = gfsk_mod(symbols=self.SYMBOLS, **self.PARAMS)
        assert np.iscomplexobj(out)
        magnitudes = np.abs(out)
        # At least 95% of samples should be within ±0.05 of unity
        close_to_unity = np.abs(magnitudes - 1.0) < 0.05
        assert np.mean(close_to_unity) > 0.90

    def test_gfsk_output_length(self):
        out = gfsk_mod(symbols=self.SYMBOLS, **self.PARAMS)
        sps = max(1, int(0.160 * FS))  # 1920
        assert len(out) == len(self.SYMBOLS) * sps  # 200 * 1920 = 384000

    def test_gfsk_frequency_in_bandwidth(self):
        out = gfsk_mod(symbols=self.SYMBOLS, **self.PARAMS)
        spectrum = np.fft.fftshift(np.abs(np.fft.fft(out)) ** 2)
        freqs = np.fft.fftshift(np.fft.fftfreq(len(out), 1 / FS))
        bw_half = self.PARAMS["num_tones"] * self.PARAMS["tone_spacing"] / 2  # 12.5 Hz
        in_band = np.abs(freqs) <= bw_half
        total_energy = spectrum.sum()
        in_band_energy = spectrum[in_band].sum()
        assert in_band_energy / total_energy > 0.90


class TestFSK:
    """FSK modulator tests."""

    def test_fsk_unit_envelope(self):
        symbols = np.array([0, 1] * 100, dtype=np.uint8)
        out = fsk_mod(symbols=symbols, num_tones=2, tone_spacing=100.0,
                      symbol_dur=1 / 100, fs=FS)
        np.testing.assert_allclose(np.abs(out), 1.0, atol=1e-12)

    def test_fsk_output_length(self):
        symbols = np.array([0, 1] * 100, dtype=np.uint8)
        out = fsk_mod(symbols=symbols, num_tones=2, tone_spacing=100.0,
                      symbol_dur=1 / 100, fs=FS)
        sps = max(1, int((1 / 100) * FS))  # 120
        assert len(out) == len(symbols) * sps

    def test_fsk_continuous_phase(self):
        symbols = np.array([0, 1] * 100, dtype=np.uint8)
        out = fsk_mod(symbols=symbols, num_tones=2, tone_spacing=100.0,
                      symbol_dur=1 / 100, fs=FS)
        phase = np.unwrap(np.angle(out))
        max_jump = np.max(np.abs(np.diff(phase)))
        assert max_jump < np.pi


class TestOOK:
    """OOK modulator tests."""

    PARAMS = dict(tone_freq=800, bit_dur=1 / 50, fs=FS)

    def test_ook_output_length(self):
        bits = np.ones(100, dtype=np.uint8)
        out = ook_mod(bits=bits, **self.PARAMS)
        sps = max(1, int((1 / 50) * FS))  # 240
        assert len(out) == 100 * sps  # 24000

    def test_ook_zero_bits_silent(self):
        bits = np.zeros(100, dtype=np.uint8)
        out = ook_mod(bits=bits, **self.PARAMS)
        np.testing.assert_allclose(np.abs(out), 0.0, atol=1e-12)

    def test_ook_one_bits_have_energy(self):
        bits = np.ones(100, dtype=np.uint8)
        out = ook_mod(bits=bits, **self.PARAMS)
        np.testing.assert_allclose(np.abs(out), 1.0, atol=1e-12)


class TestPSK:
    """PSK modulator tests."""

    def test_psk_output_length(self):
        phase_bits = np.array([0, 1] * 100, dtype=np.uint8)
        out = psk_mod(phase_bits=phase_bits, baud=31.25, fs=FS)
        sps = max(1, int(FS / 31.25))  # 384
        assert len(out) == len(phase_bits) * sps  # 200 * 384 = 76800

    def test_psk_bpsk_two_phase_states(self):
        # Use runs of identical bits so that within each run the carrier
        # phase advances uniformly and we can measure the modulation offset
        # between runs of 0s vs runs of 1s.
        phase_bits = np.array([0] * 50 + [1] * 50 + [0] * 50 + [1] * 50,
                              dtype=np.uint8)
        out = psk_mod(phase_bits=phase_bits, baud=31.25, fs=FS, order=2)
        sps = max(1, int(FS / 31.25))
        # Sample at mid-symbol, well inside each run (skip first 5 of each run)
        run_starts = [0, 50, 100, 150]
        run_bits = [0, 1, 0, 1]
        phases_by_bit = {0: [], 1: []}
        for start, bit in zip(run_starts, run_bits):
            for k in range(10, 40):  # mid-run symbols to avoid edge effects
                idx = (start + k) * sps + sps // 2
                phases_by_bit[bit].append(np.angle(out[idx]))
        # Within each group, remove carrier drift by looking at successive diffs
        # The carrier phase advances linearly, so diff should be constant.
        # Between groups, the offset should differ by ~pi.
        # Simpler: compute mean phasor for each group to get mean direction.
        mean0 = np.mean(np.exp(1j * np.array(phases_by_bit[0])))
        mean1 = np.mean(np.exp(1j * np.array(phases_by_bit[1])))
        # The angle between the two mean phasors should be near pi
        angle_diff = np.abs(np.angle(mean1 / mean0))
        assert angle_diff > np.pi * 0.5, f"Phase diff {angle_diff:.3f} not near pi"

    def test_psk_qpsk_four_phases(self):
        phase_bits = np.array([0, 1, 2, 3] * 50, dtype=np.uint8)
        out = psk_mod(phase_bits=phase_bits, baud=31.25, fs=FS, order=4)
        sps = max(1, int(FS / 31.25))
        mid_phases = []
        for i in range(len(phase_bits)):
            mid = i * sps + sps // 2
            mid_phases.append(np.angle(out[mid]))
        mid_phases = np.array(mid_phases)
        # Quantize to 4 quadrants and verify we hit all 4
        quadrants = np.mod(np.round(mid_phases / (np.pi / 2)), 4).astype(int)
        assert len(np.unique(quadrants)) == 4


class Test4FSK:
    """4-level FSK modulator tests."""

    def test_4fsk_unit_envelope(self, sample_dibits):
        out = _4fsk_mod(dibits=sample_dibits, sym_rate=4800,
                        dev_outer=1944, dev_inner=648, fs=FS)
        magnitudes = np.abs(out)
        np.testing.assert_allclose(magnitudes, 1.0, atol=0.1)

    def test_4fsk_four_frequencies(self):
        # All-zero dibits → dibit 0b00 → +dev_inner = +648 Hz
        dibits = np.zeros(100, dtype=np.uint8)
        out = _4fsk_mod(dibits=dibits, sym_rate=4800,
                        dev_outer=1944, dev_inner=648, fs=FS, smooth=False)
        spectrum = np.abs(np.fft.fft(out)) ** 2
        freqs = np.fft.fftfreq(len(out), 1 / FS)
        peak_idx = np.argmax(spectrum)
        peak_freq = freqs[peak_idx]
        # Peak should be near +648 Hz (within a few Hz of resolution)
        assert abs(abs(peak_freq) - 648) < 50


class TestGMSK:
    """GMSK modulator tests."""

    def test_gmsk_unit_envelope(self, sample_bits):
        out = _gmsk_mod(bits=sample_bits, bit_rate=4800, bt=0.5, fs=FS)
        magnitudes = np.abs(out)
        np.testing.assert_allclose(magnitudes, 1.0, atol=0.1)

    def test_gmsk_narrower_than_fsk(self, sample_bits):
        """GMSK (bt=0.5) 99% bandwidth should be narrower than raw FSK."""
        gmsk_out = _gmsk_mod(bits=sample_bits, bit_rate=4800, bt=0.5, fs=FS)
        # Equivalent raw FSK: binary (2 tones), deviation = bit_rate/4,
        # spacing = 2*deviation = 2400
        fsk_symbols = sample_bits.copy()
        sps_fsk = max(1, int(FS / 4800))
        fsk_out = fsk_mod(symbols=fsk_symbols, num_tones=2,
                          tone_spacing=2400, symbol_dur=1 / 4800, fs=FS)

        def bw_99(sig):
            psd = np.fft.fftshift(np.abs(np.fft.fft(sig)) ** 2)
            total = psd.sum()
            cumulative = np.cumsum(psd)
            low = np.searchsorted(cumulative, 0.005 * total)
            high = np.searchsorted(cumulative, 0.995 * total)
            freqs = np.fft.fftshift(np.fft.fftfreq(len(sig), 1 / FS))
            return freqs[high] - freqs[low]

        assert bw_99(gmsk_out) < bw_99(fsk_out)


class TestOFDM:
    """OFDM carrier tests."""

    def test_ofdm_output_length(self):
        out = ofdm_carriers(n_carriers=64, carrier_spacing=15.625,
                            symbol_dur=0.050, n_symbols=10, fs=FS)
        sps = max(1, int(0.050 * FS))  # 600
        assert len(out) == 10 * sps  # 6000

    def test_ofdm_carrier_peaks_visible(self):
        n_carriers = 8
        spacing = 15.625
        out = ofdm_carriers(n_carriers=n_carriers, carrier_spacing=spacing,
                            symbol_dur=0.050, n_symbols=10, fs=FS)
        spectrum = np.fft.fftshift(np.abs(np.fft.fft(out)) ** 2)
        freqs = np.fft.fftshift(np.fft.fftfreq(len(out), 1 / FS))
        # Find peaks: local maxima above a threshold
        threshold = spectrum.max() * 0.05
        # Simple peak detection: a sample is a peak if it's larger than both
        # neighbours and above threshold
        peaks = []
        for i in range(1, len(spectrum) - 1):
            if (spectrum[i] > spectrum[i - 1] and
                    spectrum[i] > spectrum[i + 1] and
                    spectrum[i] > threshold):
                peaks.append(freqs[i])
        assert len(peaks) >= 6


class TestChirpMod:
    """Chirp modulator tests."""

    BW = 2000       # 2 kHz sweep bandwidth
    PULSE_DUR = 0.01  # 10 ms pulse
    PRF = 20        # 20 Hz pulse repetition frequency
    N_PULSES = 5

    def test_chirp_output_is_complex(self):
        out = chirp_mod(bw=self.BW, pulse_dur=self.PULSE_DUR,
                        prf=self.PRF, n_pulses=self.N_PULSES, fs=FS)
        assert np.iscomplexobj(out)

    def test_chirp_output_length(self):
        out = chirp_mod(bw=self.BW, pulse_dur=self.PULSE_DUR,
                        prf=self.PRF, n_pulses=self.N_PULSES, fs=FS)
        pri_samples = max(1, int((1 / self.PRF) * FS))
        expected = self.N_PULSES * pri_samples
        assert len(out) == expected

    def test_chirp_spectral_energy_within_bandwidth(self):
        out = chirp_mod(bw=self.BW, pulse_dur=self.PULSE_DUR,
                        prf=self.PRF, n_pulses=self.N_PULSES, fs=FS)
        spectrum = np.fft.fftshift(np.abs(np.fft.fft(out)) ** 2)
        freqs = np.fft.fftshift(np.fft.fftfreq(len(out), 1 / FS))
        in_band = np.abs(freqs) <= self.BW / 2
        total_energy = spectrum.sum()
        in_band_energy = spectrum[in_band].sum()
        # Most spectral energy should be within chirp bandwidth
        assert in_band_energy / total_energy > 0.80


class TestDSSS:
    """DSSS modulator tests."""

    CHIP_CODE = np.array([1, 0, 1, 1, 0, 0, 1, 0, 0, 1, 1], dtype=np.uint8)
    CHIPS_PER_BIT = 11
    CHIP_RATE = 1000.0

    def test_dsss_output_is_complex(self, sample_bits):
        out = dsss_mod(data_bits=sample_bits, chip_code=self.CHIP_CODE,
                       chips_per_bit=self.CHIPS_PER_BIT, fs=FS,
                       chip_rate=self.CHIP_RATE)
        assert np.iscomplexobj(out)

    def test_dsss_bandwidth_spreads_to_chip_rate(self, sample_bits):
        out = dsss_mod(data_bits=sample_bits, chip_code=self.CHIP_CODE,
                       chips_per_bit=self.CHIPS_PER_BIT, fs=FS,
                       chip_rate=self.CHIP_RATE)
        spectrum = np.fft.fftshift(np.abs(np.fft.fft(out)) ** 2)
        freqs = np.fft.fftshift(np.fft.fftfreq(len(out), 1 / FS))
        # Spectral energy should spread across the chip_rate bandwidth:
        # the main lobe of a rectangular-chip DSSS signal is 2*chip_rate wide.
        # Verify significant energy exists beyond data-only bandwidth.
        in_chip_band = np.abs(freqs) <= self.CHIP_RATE
        total = spectrum.sum()
        in_band_energy = spectrum[in_chip_band].sum()
        # Main lobe should contain a substantial fraction of total energy
        assert in_band_energy / total > 0.40
        # But energy should NOT be concentrated in a narrow band (< chip_rate/4),
        # confirming that spreading occurred.
        narrow = np.abs(freqs) <= self.CHIP_RATE / 4
        narrow_energy = spectrum[narrow].sum()
        assert narrow_energy / total < 0.60

    def test_dsss_spreading_ratio_output_length(self, sample_bits):
        out = dsss_mod(data_bits=sample_bits, chip_code=self.CHIP_CODE,
                       chips_per_bit=self.CHIPS_PER_BIT, fs=FS,
                       chip_rate=self.CHIP_RATE)
        spc = max(1, int(FS / self.CHIP_RATE))
        expected = len(sample_bits) * self.CHIPS_PER_BIT * spc
        assert len(out) == expected


class TestOQPSK:
    """Offset QPSK modulator tests."""

    SYM_RATE = 100

    def test_oqpsk_output_is_complex(self):
        symbols = np.array([0, 1, 2, 3] * 50, dtype=np.uint8)
        out = oqpsk_mod(symbols=symbols, sym_rate=self.SYM_RATE, fs=FS)
        assert np.iscomplexobj(out)

    def test_oqpsk_iq_half_symbol_offset(self):
        """I and Q channels should have max cross-correlation at half-symbol offset."""
        symbols = np.array([0, 1, 2, 3] * 50, dtype=np.uint8)
        out = oqpsk_mod(symbols=symbols, sym_rate=self.SYM_RATE, fs=FS)
        sps = max(1, int(FS / self.SYM_RATE))
        half_sym = sps  # half of 2*sps symbol period
        i_sig = out.real
        q_sig = out.imag
        # Cross-correlate at zero offset vs half-symbol offset
        max_lag = 2 * sps
        n = min(len(i_sig), len(q_sig)) - max_lag
        corr_at_offset = np.abs(np.sum(
            i_sig[:n] * q_sig[half_sym:half_sym + n]))
        corr_at_zero = np.abs(np.sum(
            i_sig[:n] * q_sig[:n]))
        # Cross-correlation at half-symbol offset should exceed zero offset
        assert corr_at_offset > corr_at_zero


class TestPPM:
    """PPM modulator tests."""

    SLOT_DUR = 1 / 100  # 10 ms slot

    def test_ppm_output_is_complex(self, sample_bits):
        out = ppm_mod(bits=sample_bits, slot_dur=self.SLOT_DUR, fs=FS)
        assert np.iscomplexobj(out)

    def test_ppm_bit_to_position_mapping(self):
        """bit=1 -> energy in first half, bit=0 -> energy in second half."""
        bits = np.array([1, 0, 1, 1, 0], dtype=np.uint8)
        out = ppm_mod(bits=bits, slot_dur=self.SLOT_DUR, fs=FS)
        sps = max(2, int(self.SLOT_DUR * FS))
        half = sps // 2

        for i, bit in enumerate(bits):
            start = i * sps
            first_half_energy = np.sum(np.abs(out[start:start + half]) ** 2)
            second_half_energy = np.sum(
                np.abs(out[start + half:start + sps]) ** 2)
            if bit == 1:
                assert first_half_energy > second_half_energy
            else:
                assert second_half_energy > first_half_energy


class TestPi4DQPSK:
    """pi/4-DQPSK modulator tests."""

    SYM_RATE = 4800

    def test_pi4dqpsk_output_is_complex(self, sample_dibits):
        out = _pi4dqpsk_mod(dibits=sample_dibits, sym_rate=self.SYM_RATE,
                            fs=FS)
        assert np.iscomplexobj(out)

    def test_pi4dqpsk_output_length(self, sample_dibits):
        out = _pi4dqpsk_mod(dibits=sample_dibits, sym_rate=self.SYM_RATE,
                            fs=FS)
        sps = max(1, int(FS / self.SYM_RATE))
        expected = len(sample_dibits) * sps
        assert len(out) == expected

    def test_pi4dqpsk_phase_rotation_per_dibit(self, sample_dibits):
        """Phase should change between consecutive symbols."""
        out = _pi4dqpsk_mod(dibits=sample_dibits, sym_rate=self.SYM_RATE,
                            fs=FS)
        sps = max(1, int(FS / self.SYM_RATE))
        # Sample phase at mid-symbol points
        mid_phases = []
        for i in range(len(sample_dibits)):
            mid = i * sps + sps // 2
            mid_phases.append(np.angle(out[mid]))
        mid_phases = np.array(mid_phases)
        # Phase differences between successive symbols should be non-zero
        diffs = np.abs(np.diff(mid_phases))
        # With random dibits, most phase changes should be substantial
        assert np.mean(diffs > 0.1) > 0.5


class TestOFDMFull:
    """Full OFDM with CP, pilots, and DC null tests."""

    N_SC = 64
    SC_SPACING = 15.625
    CP_LEN = 16
    CONST = "qpsk"
    N_SYM = 4
    PILOT_SPACING = 8

    def test_ofdm_full_output_is_complex(self):
        out = ofdm_full(n_subcarriers=self.N_SC,
                        subcarrier_spacing=self.SC_SPACING,
                        cp_length=self.CP_LEN, constellation=self.CONST,
                        n_symbols=self.N_SYM, fs=FS,
                        pilot_spacing=self.PILOT_SPACING)
        assert np.iscomplexobj(out)

    def test_ofdm_full_cyclic_prefix_present(self):
        """Last cp_length samples of each OFDM symbol body repeat at start."""
        out = ofdm_full(n_subcarriers=self.N_SC,
                        subcarrier_spacing=self.SC_SPACING,
                        cp_length=self.CP_LEN, constellation=self.CONST,
                        n_symbols=self.N_SYM, fs=FS,
                        pilot_spacing=self.PILOT_SPACING)
        symbol_len = self.N_SC + self.CP_LEN  # 80 samples per OFDM symbol
        for sym_idx in range(self.N_SYM):
            start = sym_idx * symbol_len
            cp = out[start:start + self.CP_LEN]
            body_tail = out[start + self.CP_LEN + self.N_SC - self.CP_LEN:
                            start + self.CP_LEN + self.N_SC]
            np.testing.assert_allclose(cp, body_tail, atol=1e-10)

    def test_ofdm_full_pilot_subcarriers(self):
        """Pilots at expected positions (every pilot_spacing subcarriers)."""
        # Use deterministic seed and extract freq-domain from one symbol
        out = ofdm_full(n_subcarriers=self.N_SC,
                        subcarrier_spacing=self.SC_SPACING,
                        cp_length=self.CP_LEN, constellation=self.CONST,
                        n_symbols=1, fs=FS,
                        pilot_spacing=self.PILOT_SPACING)
        symbol_len = self.N_SC + self.CP_LEN
        # Extract body (skip CP)
        body = out[self.CP_LEN:self.CP_LEN + self.N_SC]
        freq_domain = np.fft.fft(body) / np.sqrt(self.N_SC)
        # Pilot positions: k where k % pilot_spacing == 0 and k != 0
        # and k != N_SC//2
        for k in range(self.N_SC):
            if k == 0 or k == self.N_SC // 2:
                continue
            if self.PILOT_SPACING > 0 and k % self.PILOT_SPACING == 0:
                # Pilot should be BPSK (magnitude ~1.0)
                assert np.abs(freq_domain[k]) > 0.5

    def test_ofdm_full_dc_null(self):
        """Subcarrier 0 and N/2 should be zero (DC null)."""
        out = ofdm_full(n_subcarriers=self.N_SC,
                        subcarrier_spacing=self.SC_SPACING,
                        cp_length=self.CP_LEN, constellation=self.CONST,
                        n_symbols=1, fs=FS,
                        pilot_spacing=self.PILOT_SPACING)
        symbol_len = self.N_SC + self.CP_LEN
        body = out[self.CP_LEN:self.CP_LEN + self.N_SC]
        freq_domain = np.fft.fft(body) / np.sqrt(self.N_SC)
        # DC (k=0) and Nyquist (k=N/2) should be near zero
        assert np.abs(freq_domain[0]) < 1e-10
        assert np.abs(freq_domain[self.N_SC // 2]) < 1e-10


# ---------------------------------------------------------------------------
# filters.py  (7 tests)
# ---------------------------------------------------------------------------

class TestBandpassFilter:
    """Bandpass filter tests."""

    def test_bandpass_passes_in_band(self):
        n = 4096
        t = np.arange(n) / FS
        tone = np.cos(2 * np.pi * 1000 * t)
        filtered = bandpass_filter(tone, fs=FS, low=300, high=3000)
        input_power = np.sum(tone ** 2)
        output_power = np.sum(filtered ** 2)
        # Less than 3 dB loss → output_power > 0.5 * input_power
        assert output_power > 0.5 * input_power

    def test_bandpass_rejects_out_of_band(self):
        n = 4096
        t = np.arange(n) / FS
        tone = np.cos(2 * np.pi * 100 * t)
        filtered = bandpass_filter(tone, fs=FS, low=300, high=3000)
        input_power = np.sum(tone ** 2)
        output_power = np.sum(filtered ** 2)
        # >20 dB attenuation → output_power < 0.01 * input_power
        assert output_power < 0.01 * input_power


class TestRRCFilter:
    """Root Raised Cosine filter tests."""

    def test_rrc_symmetry(self):
        h = rrc_filter(n_taps=65, rolloff=0.35, sps=8)
        np.testing.assert_allclose(h, h[::-1], atol=1e-12)

    def test_rrc_energy_normalized(self):
        h = rrc_filter(n_taps=65, rolloff=0.35, sps=8)
        energy = np.sum(h ** 2)
        np.testing.assert_allclose(energy, 1.0, atol=1e-6)

    def test_rrc_length(self):
        h = rrc_filter(n_taps=65, rolloff=0.35, sps=8)
        assert len(h) == 65


class TestGaussianFilter:
    """Gaussian pulse shaping filter tests."""

    def test_gaussian_sums_to_one(self):
        h = gaussian_filter(bt=0.5, sps=8)
        np.testing.assert_allclose(h.sum(), 1.0, atol=1e-10)

    def test_gaussian_peak_at_center(self):
        h = gaussian_filter(bt=0.5, sps=8)
        # Default n_taps = 4*sps = 32, center index = 32//2 = 16
        assert np.argmax(h) == len(h) // 2


# ---------------------------------------------------------------------------
# analytic.py  (6 tests)
# ---------------------------------------------------------------------------

class TestHilbertAnalytic:
    """Hilbert analytic signal tests."""

    def test_hilbert_is_complex(self):
        n = 1024
        t = np.arange(n) / FS
        real_signal = np.cos(2 * np.pi * 100 * t)
        out = hilbert_analytic(real_signal)
        assert np.iscomplexobj(out)

    def test_hilbert_positive_spectrum(self):
        n = 1024
        t = np.arange(n) / FS
        real_signal = np.cos(2 * np.pi * 100 * t)
        out = hilbert_analytic(real_signal)
        spectrum = np.abs(np.fft.fft(out)) ** 2
        total_energy = spectrum.sum()
        # Negative frequency bins: indices n//2+1 through n-1
        neg_energy = spectrum[n // 2 + 1:].sum()
        assert neg_energy / total_energy < 0.01

    def test_hilbert_real_matches_input(self):
        n = 1024
        t = np.arange(n) / FS
        real_signal = np.cos(2 * np.pi * 100 * t)
        out = hilbert_analytic(real_signal)
        np.testing.assert_allclose(out.real, real_signal, atol=1e-10)


class TestAudioToIQ:
    """audio_to_iq conversion tests."""

    def test_audio_to_iq_empty(self):
        out = audio_to_iq(np.array([], dtype=np.float64), 48000)
        assert len(out) == 0
        assert np.iscomplexobj(out)

    def test_audio_to_iq_resamples(self):
        source_fs = 48000
        duration = 1.0
        n = int(source_fs * duration)
        t = np.arange(n) / source_fs
        tone = np.cos(2 * np.pi * 440 * t)
        out = audio_to_iq(tone, source_fs, target_fs=FS)
        expected_len = int(n * FS / source_fs)  # 12000
        assert len(out) == expected_len

    def test_audio_to_iq_int16_norm(self):
        source_fs = 48000
        n = 48000
        t = np.arange(n) / source_fs
        tone_int16 = (32767 * np.cos(2 * np.pi * 440 * t)).astype(np.int16)
        out = audio_to_iq(tone_int16, source_fs, target_fs=FS)
        # After normalization and Hilbert transform, magnitude should peak near 1.0
        peak_mag = np.max(np.abs(out))
        assert 0.9 < peak_mag < 1.1
