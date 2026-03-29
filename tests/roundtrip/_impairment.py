"""Impairment pipeline validator."""

import numpy as np

from rf_datagen.constants import FS, WINDOW_LEN, MAX_FREQ_OFFSET

from ._base import BaseRoundtripValidator, register


@register("IMPAIRMENT")
class ImpairmentPipelineValidator(BaseRoundtripValidator):
    """Validate the impairment pipeline produces physically coherent output."""

    required_tools = []
    tier = 2

    N_WINDOWS = 8

    def _generate_tone(self, freq_hz=800.0):
        n = WINDOW_LEN * self.N_WINDOWS
        t = np.arange(n) / FS
        return np.exp(2j * np.pi * freq_hz * t)

    def _occupied_bandwidth(self, sig, fraction=0.99):
        spectrum = np.abs(np.fft.fft(sig)) ** 2
        freqs = np.fft.fftfreq(len(sig), 1.0 / FS)
        order = np.argsort(freqs)
        freqs = freqs[order]
        spectrum = spectrum[order]
        total_power = spectrum.sum()
        if total_power < 1e-20:
            return FS
        target = total_power * fraction
        best_bw = freqs[-1] - freqs[0]
        left = 0
        window_power = 0.0
        for right in range(len(spectrum)):
            window_power += spectrum[right]
            while window_power - spectrum[left] >= target:
                window_power -= spectrum[left]
                left += 1
            if window_power >= target:
                bw = freqs[right] - freqs[left]
                if bw < best_bw:
                    best_bw = bw
        return max(0.0, best_bw)

    def _phase_discontinuity(self, sig):
        phase = np.unwrap(np.angle(sig))
        inst_freq = np.diff(phase) / (2 * np.pi) * FS
        jumps = []
        for i in range(1, self.N_WINDOWS):
            boundary = i * WINDOW_LEN
            if boundary >= len(inst_freq):
                break
            window = 32
            lo = max(0, boundary - window)
            hi = min(len(inst_freq), boundary + window)
            local_median = np.median(inst_freq[lo:hi])
            jump = abs(inst_freq[boundary] - local_median)
            jumps.append(jump)
        return np.array(jumps)

    def _peak_frequency(self, sig):
        n = len(sig)
        padded = np.zeros(4 * n, dtype=sig.dtype)
        padded[:n] = sig
        spectrum = np.abs(np.fft.fft(padded))
        freqs = np.fft.fftfreq(len(padded), 1.0 / FS)
        return freqs[np.argmax(spectrum)]

    def test_spectral_coherence(self):
        from rf_datagen.impairments.scenarios import (
            apply_scenario, apply_scenario_continuous)
        results = []
        for trial in range(5):
            tone = self._generate_tone()
            pw_parts = []
            for i in range(self.N_WINDOWS):
                w = tone[i * WINDOW_LEN:(i + 1) * WINDOW_LEN]
                imp, _ = apply_scenario(w, 30, FS)
                pw_parts.append(imp)
            per_window = np.concatenate(pw_parts)
            continuous, _ = apply_scenario_continuous(tone, 30, FS,
                                                     scenario="hf_clean")
            bw_pw = self._occupied_bandwidth(per_window)
            bw_ct = self._occupied_bandwidth(continuous)
            results.append((bw_pw, bw_ct))
        avg_pw = np.mean([r[0] for r in results])
        avg_ct = np.mean([r[1] for r in results])
        ratio = avg_pw / max(avg_ct, 1.0)
        passed = ratio > 2.0
        return passed, {
            "per_window_bw_hz": round(avg_pw, 1),
            "continuous_bw_hz": round(avg_ct, 1),
            "ratio": round(ratio, 2),
        }

    def test_phase_continuity(self):
        from rf_datagen.impairments.scenarios import (
            apply_scenario, apply_scenario_continuous)
        results = []
        for trial in range(5):
            tone = self._generate_tone()
            pw_parts = []
            for i in range(self.N_WINDOWS):
                w = tone[i * WINDOW_LEN:(i + 1) * WINDOW_LEN]
                imp, _ = apply_scenario(w, 40, FS)
                pw_parts.append(imp)
            per_window = np.concatenate(pw_parts)
            continuous, _ = apply_scenario_continuous(tone, 40, FS,
                                                     scenario="hf_clean")
            pw_jumps = self._phase_discontinuity(per_window)
            ct_jumps = self._phase_discontinuity(continuous)
            results.append((np.mean(pw_jumps), np.mean(ct_jumps)))
        avg_pw = np.mean([r[0] for r in results])
        avg_ct = np.mean([r[1] for r in results])
        passed = avg_ct < avg_pw
        return passed, {
            "per_window_avg_jump_hz": round(avg_pw, 1),
            "continuous_avg_jump_hz": round(avg_ct, 1),
        }

    def test_all_scenarios(self):
        from rf_datagen.impairments.scenarios import (
            apply_scenario_continuous, SCENARIO_NAMES)
        tone = self._generate_tone()
        failures = []
        for name in SCENARIO_NAMES:
            try:
                result, rname = apply_scenario_continuous(tone, 15, FS,
                                                         scenario=name)
                assert rname == name, f"name mismatch: {rname} != {name}"
                assert len(result) == len(tone), "length changed"
                assert np.all(np.isfinite(result)), "non-finite values"
            except Exception as e:
                failures.append(f"{name}(continuous): {e}")
        try:
            apply_scenario_continuous(tone, 15, FS, scenario="nonexistent")
            failures.append("bad scenario name did not raise ValueError")
        except ValueError:
            pass
        passed = len(failures) == 0
        return passed, {
            "scenarios_tested": len(SCENARIO_NAMES),
            "failures": failures,
        }

    def test_training_path(self):
        from rf_datagen.impairments.scenarios import apply_impairments
        raw = np.array([self._generate_tone()[:WINDOW_LEN] for _ in range(4)])
        target_count = 16
        result = apply_impairments(raw, target_count, FS)
        errors = []
        if result.shape != (target_count, WINDOW_LEN):
            errors.append(f"shape {result.shape} != ({target_count}, {WINDOW_LEN})")
        if not np.all(np.isfinite(result)):
            n_bad = np.sum(~np.isfinite(result))
            errors.append(f"{n_bad} non-finite values")
        powers = np.mean(np.abs(result) ** 2, axis=1)
        if np.any(powers < 1e-20):
            n_dead = np.sum(powers < 1e-20)
            errors.append(f"{n_dead}/{target_count} windows have near-zero power")
        passed = len(errors) == 0
        return passed, {"windows": target_count, "errors": errors}

    def test_power_conservation(self):
        from rf_datagen.impairments.scenarios import (
            apply_scenario_continuous, SCENARIO_NAMES)
        snr_levels = [25, 10, 0, -10]
        errors = []
        total_checks = 0
        for name in SCENARIO_NAMES:
            for snr in snr_levels:
                tone = self._generate_tone()
                result, _ = apply_scenario_continuous(tone, snr, FS,
                                                     scenario=name)
                total_checks += 1
                if not np.all(np.isfinite(result)):
                    errors.append(f"{name}@{snr}dB: non-finite values")
                    continue
                power = np.mean(np.abs(result) ** 2)
                if power < 1e-10:
                    errors.append(f"{name}@{snr}dB: near-zero power {power:.2e}")
                elif abs(power - 1.0) >= 0.05:
                    errors.append(
                        f"{name}@{snr}dB: power={power:.4f} (off by "
                        f"{abs(power - 1.0):.4f})")
        passed = len(errors) == 0
        return passed, {
            "scenarios_tested": len(SCENARIO_NAMES),
            "snr_levels_tested": snr_levels,
            "total_checks": total_checks,
            "errors": errors,
        }

    def test_snr_accuracy(self):
        from rf_datagen.impairments.effects import add_awgn
        snr_levels = [25, 15, 5, 0, -5, -10]
        n_trials = 10
        n_samples = 16384
        errors = []
        per_snr = []
        for requested in snr_levels:
            measured = []
            for _ in range(n_trials):
                tone = np.exp(2j * np.pi * 800 * np.arange(n_samples) / FS)
                noisy = add_awgn(tone, requested)
                noise = noisy - tone
                sig_power = np.mean(np.abs(tone) ** 2)
                noise_power = np.mean(np.abs(noise) ** 2)
                if noise_power < 1e-30:
                    measured.append(requested)
                    continue
                actual = 10 * np.log10(sig_power / noise_power)
                measured.append(actual)
            mean_measured = np.mean(measured)
            error_db = abs(mean_measured - requested)
            per_snr.append({
                "requested": requested,
                "measured_mean": round(mean_measured, 2),
                "error_db": round(error_db, 2),
            })
            if error_db >= 1.5:
                errors.append(
                    f"SNR={requested}dB: measured={mean_measured:.2f}dB "
                    f"(error={error_db:.2f}dB)")
        zeros = np.zeros(n_samples, dtype=np.complex128)
        noisy_zeros = add_awgn(zeros, 10)
        zero_noise_power = np.mean(np.abs(noisy_zeros) ** 2)
        if abs(zero_noise_power - 0.1) > 0.05:
            errors.append(
                f"zero-input guard: noise_power={zero_noise_power:.4f} "
                f"(expected ~0.1)")
        passed = len(errors) == 0
        return passed, {
            "per_snr": per_snr,
            "zero_input_noise_power": round(zero_noise_power, 4),
            "errors": errors,
        }

    def test_freq_shift_bounds(self):
        from rf_datagen.impairments.scenarios import apply_scenario_continuous
        n_trials = 30
        tone_freq = 800.0
        offsets = []
        errors = []
        for _ in range(n_trials):
            tone = self._generate_tone(freq_hz=tone_freq)
            result, _ = apply_scenario_continuous(tone, 40, FS,
                                                  scenario="hf_clean")
            peak = self._peak_frequency(result)
            offset = peak - tone_freq
            offsets.append(offset)
        offsets = np.array(offsets)
        bound = MAX_FREQ_OFFSET + 5
        out_of_bounds = np.abs(offsets) > bound
        if np.any(out_of_bounds):
            bad = offsets[out_of_bounds]
            errors.append(
                f"{np.sum(out_of_bounds)} offsets outside "
                f"[{-bound}, {bound}]: {bad.tolist()}")
        if np.max(offsets) <= 200:
            errors.append(
                f"max offset {np.max(offsets):.1f} Hz <= 200 "
                f"(distribution too narrow)")
        if np.min(offsets) >= -200:
            errors.append(
                f"min offset {np.min(offsets):.1f} Hz >= -200 "
                f"(distribution too narrow)")
        passed = len(errors) == 0
        return passed, {
            "n_trials": n_trials,
            "max_offset_hz": round(float(np.max(offsets)), 1),
            "min_offset_hz": round(float(np.min(offsets)), 1),
            "mean_offset_hz": round(float(np.mean(offsets)), 1),
            "all_within_bounds": bool(~np.any(out_of_bounds)),
            "errors": errors,
        }

    def test_effect_stacking(self):
        from rf_datagen.impairments.effects import (
            normalize_power, add_awgn, freq_shift,
            apply_watterson, apply_qsb,
            apply_iq_imbalance, apply_phase_noise,
            apply_atmospheric_noise, apply_impulse_noise,
            apply_adjacent_signal, apply_powerline_hum, apply_dc_offset,
        )
        from rf_datagen.impairments.transmitter import TransmitterModel
        from rf_datagen.impairments.scenarios import (
            apply_scenario_continuous, SCENARIO_NAMES)

        errors = []
        tone = self._generate_tone()
        sig = TransmitterModel("POORLY_OPERATED").apply(tone, FS)
        sig = apply_watterson(sig, FS)
        sig = apply_qsb(sig, FS)
        sig = freq_shift(sig, MAX_FREQ_OFFSET, FS)
        sig = apply_iq_imbalance(sig)
        sig = apply_phase_noise(sig, FS)
        sig = apply_atmospheric_noise(sig, FS)
        sig = apply_impulse_noise(sig, FS)
        sig = apply_adjacent_signal(sig, FS)
        sig = apply_powerline_hum(sig, FS)
        sig = apply_dc_offset(sig)
        sig = add_awgn(sig, -10)
        sig = normalize_power(sig)
        full_stack_finite = bool(np.all(np.isfinite(sig)))
        full_stack_power = float(np.mean(np.abs(sig) ** 2))
        if not full_stack_finite:
            errors.append("full stack produced non-finite values")
        if full_stack_power < 1e-10:
            errors.append(f"full stack near-zero power: {full_stack_power:.2e}")
        heavy = ["hf_poor", "contest_crowded", "sdr_desktop",
                 "overdriven", "poorly_operated"]
        heavy_trials = 0
        for name in heavy:
            for _ in range(20):
                tone = self._generate_tone()
                result, _ = apply_scenario_continuous(tone, -10, FS,
                                                     scenario=name)
                heavy_trials += 1
                if not np.all(np.isfinite(result)):
                    errors.append(f"{name}: non-finite output")
                elif np.mean(np.abs(result) ** 2) < 1e-10:
                    errors.append(f"{name}: near-zero power")
        passed = len(errors) == 0
        return passed, {
            "full_stack_power": round(full_stack_power, 4),
            "full_stack_finite": full_stack_finite,
            "heavy_scenario_trials": heavy_trials,
            "errors": errors,
        }

    def test_deterministic_reproducibility(self):
        from rf_datagen.impairments.scenarios import (
            apply_scenario, apply_scenario_continuous, apply_impairments)
        errors = []
        saved_state = np.random.get_state()
        random_match = False
        batch_match = False
        scenarios_tested = []
        try:
            scenarios_tested = ["hf_clean", "hf_poor", "sdr_desktop",
                                "poorly_operated"]
            for name in scenarios_tested:
                tone = self._generate_tone()
                np.random.seed(12345)
                r1, _ = apply_scenario_continuous(tone, 10, FS, scenario=name)
                np.random.seed(12345)
                r2, _ = apply_scenario_continuous(tone, 10, FS, scenario=name)
                if not np.array_equal(r1, r2):
                    errors.append(f"continuous {name}: not identical")
            tone = self._generate_tone()[:WINDOW_LEN]
            np.random.seed(99999)
            r1, n1 = apply_scenario(tone, 10, FS)
            np.random.seed(99999)
            r2, n2 = apply_scenario(tone, 10, FS)
            random_match = (n1 == n2) and np.array_equal(r1, r2)
            if not random_match:
                errors.append(
                    f"random selection: names={n1}/{n2}, "
                    f"equal={np.array_equal(r1, r2)}")
            raw = np.array([self._generate_tone()[:WINDOW_LEN]
                            for _ in range(4)])
            np.random.seed(77777)
            b1 = apply_impairments(raw, 8, FS)
            np.random.seed(77777)
            b2 = apply_impairments(raw, 8, FS)
            batch_match = bool(np.array_equal(b1, b2))
            if not batch_match:
                errors.append("batch apply_impairments: not identical")
        finally:
            np.random.set_state(saved_state)
        passed = len(errors) == 0
        return passed, {
            "continuous_scenarios_tested": scenarios_tested,
            "random_selection_match": random_match,
            "batch_match": batch_match,
            "errors": errors,
        }

    def run_custom(self, mode, seed=42):
        """Run all impairment pipeline validation tests."""
        np.random.seed(seed + hash("IMPAIRMENT") % (2**31))

        tests = [
            ("spectral_coherence", self.test_spectral_coherence),
            ("phase_continuity",   self.test_phase_continuity),
            ("all_scenarios",      self.test_all_scenarios),
            ("training_path",      self.test_training_path),
            ("power_conservation", self.test_power_conservation),
            ("snr_accuracy",       self.test_snr_accuracy),
            ("freq_shift_bounds",  self.test_freq_shift_bounds),
            ("effect_stacking",    self.test_effect_stacking),
            ("deterministic_reproducibility",
                                   self.test_deterministic_reproducibility),
        ]

        all_pass = True
        results = []
        for name, test_fn in tests:
            passed, details = test_fn()
            status = "PASS" if passed else "FAIL"
            if not passed:
                all_pass = False
            detail_str = ", ".join(f"{k}={v}" for k, v in details.items()
                                   if k != "failures" and k != "errors")
            print(f"  {'IMPAIRMENT':>12s}  {name}: {status}  ({detail_str})")
            if not passed:
                for problem in details.get("failures",
                                           details.get("errors", [])):
                    print(f"               {problem}")
            results.append({
                "mode": "IMPAIRMENT",
                "snr_db": name,
                "trials": 1,
                "decodes": 1 if passed else 0,
                "decode_rate": 1.0 if passed else 0.0,
            })
        return all_pass, results

    def make_trial(self, mode):
        # Not used — IMPAIRMENT uses run_custom instead
        raise NotImplementedError("Use run_custom for IMPAIRMENT mode")
