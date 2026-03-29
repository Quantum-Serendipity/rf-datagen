"""ML validation subpackage — external model inference for signal verification.

Usage:
    from rf_datagen.ml_validate import run_ml_validation
    results = run_ml_validation(model="torchsig", modes=["PSK31", "FT8"])

Requires optional dependencies:
    pip install rf-datagen[ml-validate]
"""

import sys


def _require_torch():
    """Lazy-import guard for torch."""
    try:
        import torch
        return torch
    except ImportError:
        sys.exit(
            "ERROR: torch not found. Install with:\n"
            "  pip install rf-datagen[ml-validate]")


def _require_onnxruntime():
    """Lazy-import guard for onnxruntime."""
    try:
        import onnxruntime
        return onnxruntime
    except ImportError:
        sys.exit(
            "ERROR: onnxruntime not found. Install with:\n"
            "  pip install rf-datagen[ml-validate]")


def run_ml_validation(model="torchsig", modes=None, samples=50,
                      snr_sweep=False, snr_levels=None,
                      output="./output/ml_validation",
                      device="cpu", threshold=0.5):
    """Run ML-based classification validation.

    Parameters
    ----------
    model : str
        Model backend: "torchsig", "rfml", "cgdnn", or "all"
    modes : list[str] or None
        Signal modes to validate (default: all mapped)
    samples : int
        Number of IQ samples per mode
    snr_sweep : bool
        Run across multiple SNR levels
    snr_levels : list[int] or None
        SNR levels for sweep (default: [-10, 0, 10, 20])
    output : str
        Output directory for results
    device : str
        "cpu" or "openvino"
    threshold : float
        Minimum confidence for correct classification

    Returns
    -------
    dict : Results with per-mode accuracy and confusion data
    """
    from ._runner import MLValidationRunner
    runner = MLValidationRunner(
        model=model, modes=modes, samples=samples,
        snr_sweep=snr_sweep, snr_levels=snr_levels,
        output=output, device=device, threshold=threshold)
    return runner.run()


__all__ = ["run_ml_validation"]
