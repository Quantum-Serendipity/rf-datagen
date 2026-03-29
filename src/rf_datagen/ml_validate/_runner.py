"""ML validation runner — orchestrates inference across backends."""

import os
import sys
import time

import numpy as np

from ._mapping import load_mapping, mappable_signals, get_expected_class
from ._adapters import adapt_torchsig, adapt_radioml_2016, adapt_batch
from ._report import save_results


# Backend registry
_BACKENDS = {}


def register_backend(name):
    """Decorator to register a model backend."""
    def decorator(cls):
        _BACKENDS[name] = cls
        return cls
    return decorator


def available_backends():
    """Return list of available backend names (with importable deps)."""
    avail = []
    for name, cls in _BACKENDS.items():
        try:
            cls()  # Attempt instantiation
            avail.append(name)
        except Exception:
            pass
    return avail


class MLValidationRunner:
    """Orchestrate ML validation across selected backends and modes."""

    def __init__(self, model="torchsig", modes=None, samples=50,
                 snr_sweep=False, snr_levels=None,
                 output="./output/ml_validation",
                 device="cpu", threshold=0.5):
        self.model_name = model
        self.modes = modes
        self.samples = samples
        self.snr_sweep = snr_sweep
        self.snr_levels = snr_levels or [-10, 0, 10, 20]
        self.output = output
        self.device = device
        self.threshold = threshold

    def _get_backends(self):
        """Resolve requested backends."""
        # Lazy import backends
        try:
            from . import torchsig_backend  # noqa: F401
        except ImportError:
            pass

        if self.model_name == "all":
            return {n: _BACKENDS[n] for n in _BACKENDS}
        if self.model_name not in _BACKENDS:
            print(f"ERROR: Unknown model backend '{self.model_name}'. "
                  f"Available: {list(_BACKENDS.keys())}", file=sys.stderr)
            return {}
        return {self.model_name: _BACKENDS[self.model_name]}

    def _generate_samples(self, signal_name, n_samples, snr_db=None):
        """Generate IQ samples for a signal class using the synth pipeline."""
        from ..qc import _all_synthesizers

        synths = _all_synthesizers()
        key = signal_name
        if key not in synths:
            return []

        synth_fn = synths[key]
        samples = []
        for _ in range(n_samples):
            try:
                iq = synth_fn()
                if snr_db is not None:
                    from ..impairments.effects import add_awgn
                    iq = add_awgn(iq, snr_db)
                samples.append(iq)
            except Exception:
                continue
        return samples

    def run(self):
        """Execute ML validation. Returns results dict."""
        os.makedirs(self.output, exist_ok=True)

        backends = self._get_backends()
        if not backends:
            return {"error": "No backends available"}

        # Determine modes to validate
        all_results = []
        t0 = time.time()

        for backend_name, backend_cls in backends.items():
            print(f"\n--- {backend_name} ---")

            try:
                backend = backend_cls()
                backend.load()
            except Exception as e:
                print(f"  ERROR: Failed to load {backend_name}: {e}",
                      file=sys.stderr)
                continue

            modes = self.modes or mappable_signals(backend_name)
            if not modes:
                print(f"  No mappable signals for {backend_name}")
                continue

            snr_levels = self.snr_levels if self.snr_sweep else [None]

            for snr in snr_levels:
                snr_label = "clean" if snr is None else f"{snr}"

                for signal_name in modes:
                    expected = get_expected_class(signal_name, backend_name)
                    if expected is None:
                        continue

                    # Generate IQ samples
                    iq_list = self._generate_samples(
                        signal_name, self.samples, snr_db=snr)
                    if not iq_list:
                        print(f"  {signal_name}: no samples generated")
                        continue

                    # Run inference
                    correct = 0
                    total = len(iq_list)
                    confidences = []

                    for iq in iq_list:
                        try:
                            predictions = backend.predict(
                                adapt_torchsig(iq)[np.newaxis, ...])
                            if predictions:
                                pred_class, conf = predictions[0]
                                confidences.append(conf)
                                if pred_class == expected:
                                    correct += 1
                        except Exception:
                            total -= 1
                            continue

                    accuracy = correct / total if total > 0 else 0
                    mean_conf = np.mean(confidences) if confidences else 0

                    status = ("PASS" if accuracy >= self.threshold
                              else "FAIL")
                    print(f"  {signal_name:>15s}  SNR={snr_label:>5s}  "
                          f"{correct}/{total} = {accuracy:.1%}  "
                          f"conf={mean_conf:.2f}  [{status}]")

                    all_results.append({
                        "signal": signal_name,
                        "model": backend_name,
                        "snr_db": snr_label,
                        "expected": expected,
                        "total": total,
                        "correct": correct,
                        "accuracy": round(accuracy, 3),
                        "mean_confidence": round(float(mean_conf), 3),
                    })

        elapsed = time.time() - t0

        # Save results
        results = {
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
            "elapsed_seconds": round(elapsed, 1),
            "threshold": self.threshold,
            "results": all_results,
        }

        save_results(results, self.output)

        # Summary
        print(f"\n{'=' * 60}")
        print(f"ML Validation complete in {elapsed:.1f}s")
        if all_results:
            overall_acc = np.mean([r["accuracy"] for r in all_results])
            print(f"Overall accuracy: {overall_acc:.1%}")
            n_pass = sum(1 for r in all_results
                         if r["accuracy"] >= self.threshold)
            print(f"Pass rate: {n_pass}/{len(all_results)}")

        return results
