"""Runner — orchestrates validation across all registered validators."""

import argparse
import csv
import json
import os
import sys
import time

import numpy as np

from ._base import VALIDATORS
from ._helpers import snr_sweep

# Import all validator modules to trigger @register decorators
from . import _wsjtx          # noqa: F401 — FT8, WSPR
from . import _packet          # noqa: F401 — PACKET_1200
from . import _cw              # noqa: F401 — CW
from . import _analog          # noqa: F401 — SSB, AM, FM, SSB_STT, AM_STT, FM_STT
from . import _digivoice       # noqa: F401 — FREEDV, M17
from . import _dsdcc           # noqa: F401 — DMR, DSTAR, YSF, NXDN, P25
from . import _sstv            # noqa: F401 — SSTV
from . import _impairment      # noqa: F401 — IMPAIRMENT
from . import _fldigi_rx       # noqa: F401 — 18 fldigi modes
from . import _minimodem       # noqa: F401 — BELL103, BELL202
from . import _multimon        # noqa: F401 — DTMF, POCSAG, EAS, FLEX, ACARS
from . import _wsjtx_ext       # noqa: F401 — FT4, JT65, JT9
from . import _js8call         # noqa: F401 — JS8


# ---------------------------------------------------------------------------
# Mode lists
# ---------------------------------------------------------------------------

ALL_MODES = sorted(VALIDATORS.keys())

# Default modes: original set (backward compatible)
DEFAULT_MODES = [
    "FT8", "WSPR", "PACKET_1200", "FREEDV", "M17",
    "DMR", "DSTAR", "YSF", "NXDN",
    "SSB", "AM", "FM",
    "IMPAIRMENT",
]

# Extended defaults: original + new Phase 1-2 modes (excludes STT, fldigi)
EXTENDED_MODES = DEFAULT_MODES + [
    "P25", "CW", "SSTV",
    "BELL103", "BELL202",
    "DTMF", "POCSAG", "EAS",
]

# fldigi modes (separate because they require fldigi + PulseAudio)
FLDIGI_MODES_LIST = [
    "PSK31", "PSK63", "QPSK", "PSK125", "8PSK",
    "RTTY", "OLIVIA", "DOMINOEX", "MT63", "HELLSCHREIBER",
    "MFSK16", "MFSK32", "CONTESTIA", "THOR",
    "FSQ", "IFKP", "THROB", "NAVTEX",
]

# Quick fldigi subset for fast smoke tests
FLDIGI_QUICK = ["PSK31", "RTTY", "OLIVIA"]

# Expected-fail modes (tracked for regression, not counted as failures)
EXPECTED_FAIL_MODES = [
    m for m, cls in VALIDATORS.items() if cls.expected_fail
]

# SNR levels for sweep (None = clean)
DEFAULT_SNR_LEVELS = [None, 25, 20, 15, 10, 5, 0, -5, -10, -15, -20, -25, -30]


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------

def run_validation(modes=None, trials=10, snr_levels=None, clean_only=False,
                   output="./output/validation", verbose=False, seed=42):
    """Run round-trip validation. Returns (all_pass, all_results)."""
    np.random.seed(seed)
    os.makedirs(output, exist_ok=True)

    modes = modes or DEFAULT_MODES

    if clean_only:
        snr_levels = [None]
    elif snr_levels is not None:
        snr_levels = [None] + sorted(snr_levels, reverse=True)
    else:
        snr_levels = DEFAULT_SNR_LEVELS

    # Report missing tools
    missing_any = False
    for m in modes:
        if m not in VALIDATORS:
            print(f"  ERROR: Unknown mode '{m}'. "
                  f"Available: {sorted(VALIDATORS.keys())}", file=sys.stderr)
            return False, []
        cls = VALIDATORS[m]
        missing = cls.missing_tools()
        if missing:
            missing_any = True
            print(f"  WARNING: {m} missing tools: {missing}", file=sys.stderr)

    print(f"Round-trip validation: {len(modes)} modes, {trials} trials/SNR, "
          f"{len(snr_levels)} SNR levels")
    print(f"SNR levels: {['clean' if s is None else s for s in snr_levels]}")
    print()

    all_results = []
    t0 = time.time()

    # Group modes by validator class to share setup/teardown
    # Order: process modes in the order requested, but batch same-class modes
    seen_classes = {}  # cls -> (instance, [modes])
    mode_order = []    # (cls_instance, mode) in original order
    for m in modes:
        cls = VALIDATORS[m]
        if not cls.available():
            print(f"  SKIP {m}: missing tools {cls.missing_tools()}")
            continue
        if cls not in seen_classes:
            instance = cls()
            seen_classes[cls] = instance
        mode_order.append((seen_classes[cls], m))

    # Setup all unique validator instances
    setup_instances = set()
    for inst, m in mode_order:
        if inst not in setup_instances:
            try:
                inst.setup()
            except Exception as e:
                print(f"  SKIP {type(inst).__name__}: setup failed: {e}",
                      file=sys.stderr)
                # Remove all modes for this instance
                mode_order = [(i, mm) for i, mm in mode_order if i is not inst]
                continue
            setup_instances.add(inst)

    try:
        for inst, m in mode_order:
            if inst not in setup_instances:
                continue

            cls = VALIDATORS[m]
            expected_fail = cls.expected_fail
            label = f"{m} (expected-fail)" if expected_fail else m
            print(f"--- {label} ---")

            # Special handler (e.g., IMPAIRMENT)
            if hasattr(inst, 'run_custom'):
                try:
                    _, results = inst.run_custom(m, seed)
                    all_results.extend(results)
                except Exception as e:
                    print(f"  ERROR: {m}: {e}", file=sys.stderr)
            else:
                try:
                    trial_fn = inst.make_trial(m)
                    results = snr_sweep(m, trial_fn, trials, snr_levels,
                                        verbose, output, seed)
                    all_results.extend(results)
                except Exception as e:
                    print(f"  ERROR: {m}: {e}", file=sys.stderr)

            print()
    finally:
        # Teardown all instances
        for inst in setup_instances:
            try:
                inst.teardown()
            except Exception:
                pass

    elapsed = time.time() - t0

    # --- Summary ---
    print("=" * 60)
    print("SUMMARY")
    print("=" * 60)

    clean_results = [r for r in all_results if r["snr_db"] == "clean"]
    imp_results = [r for r in all_results if r["mode"] == "IMPAIRMENT"]
    all_pass = True

    for r in clean_results:
        m = r["mode"]
        cls = VALIDATORS.get(m)
        expected_fail = cls.expected_fail if cls else False
        status = "PASS" if r["decode_rate"] >= 0.9 else "FAIL"
        if r["decode_rate"] < 0.9 and not expected_fail:
            all_pass = False
        suffix = " (expected-fail)" if expected_fail else ""
        print(f"  {m:>12s}  clean: {r['decodes']}/{r['trials']} "
              f"= {r['decode_rate']:.1%}  [{status}]{suffix}")

    for r in imp_results:
        status = "PASS" if r["decode_rate"] >= 1.0 else "FAIL"
        if r["decode_rate"] < 1.0:
            all_pass = False
        print(f"  {r['mode']:>12s}  {r['snr_db']}: [{status}]")

    print(f"\nTotal time: {elapsed:.1f}s")
    overall = "PASS" if all_pass else "FAIL"
    print(f"Overall: {overall}")

    # --- Save CSV ---
    csv_path = os.path.join(output, "results.csv")
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=[
            "mode", "snr_db", "trials", "decodes", "decode_rate"])
        writer.writeheader()
        writer.writerows(all_results)
    print(f"\nCSV: {csv_path}")

    # --- Save JSON ---
    json_path = os.path.join(output, "results.json")
    report = {
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "seed": seed,
        "trials_per_snr": trials,
        "snr_levels": ["clean" if s is None else s for s in snr_levels],
        "results": all_results,
        "summary": {
            "overall": overall,
            "clean_channel": {r["mode"]: r["decode_rate"]
                              for r in clean_results},
            "elapsed_seconds": round(elapsed, 1),
        },
    }
    with open(json_path, "w") as f:
        json.dump(report, f, indent=2)
    print(f"JSON: {json_path}")

    return all_pass, all_results


def main():
    parser = argparse.ArgumentParser(
        description="Round-trip reception validation for RF signal generators")
    parser.add_argument("--modes", nargs="*", default=None,
                        help=f"Modes to test (default: core set). "
                             f"Available: {ALL_MODES}")
    parser.add_argument("--trials", type=int, default=10,
                        help="Trials per SNR level (default: 10)")
    parser.add_argument("--snr-only", nargs="*", type=int, default=None,
                        help="Test only specific SNR levels (dB)")
    parser.add_argument("--clean-only", action="store_true",
                        help="Only test clean channel (no noise)")
    parser.add_argument("--output", default="./output/validation",
                        help="Output directory for results")
    parser.add_argument("--verbose", action="store_true",
                        help="Save debug artifacts on decode failure")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed")
    args = parser.parse_args()

    modes = args.modes
    if modes:
        for m in modes:
            if m not in ALL_MODES:
                print(f"ERROR: Unknown mode '{m}'. Available: {ALL_MODES}",
                      file=sys.stderr)
                sys.exit(1)

    all_pass, _ = run_validation(
        modes=modes,
        trials=args.trials,
        snr_levels=args.snr_only,
        clean_only=args.clean_only,
        output=args.output,
        verbose=args.verbose,
        seed=args.seed,
    )

    sys.exit(0 if all_pass else 1)
