"""CLI entry point for rf-datagen."""

import argparse
import os
import signal
import shutil
import sys
import time
from pathlib import Path

import numpy as np

from .config import load_config, GeneratorConfig
from .constants import SIGNAL_LABELS
from .domains import DOMAINS, labels_for_domain, SIGNAL_DOMAIN_MAP
from .generators import GENERATORS
from . import _state
from .logging_config import setup_logging, get_logger
from .output import assemble_parts, save_dataset, atomic_write_json
from . import pid_registry

log = get_logger("cli")


def _diversity_check(iq_data, tags, n_pairs=200, threshold=0.85):
    """Spot-check signal diversity per class via magnitude spectrum similarity.

    Samples random pairs of windows within each class, computes cosine
    similarity on magnitude spectra (invariant to freq_shift phase).
    High similarity means many windows came from the same underlying signal.

    Returns list of warning strings for low-diversity classes.
    """
    from collections import Counter
    warnings = []
    counts = Counter(tags)

    for label in sorted(counts):
        count = counts[label]
        if count < 50:
            continue
        indices = [i for i, t in enumerate(tags) if t == label]
        n = min(n_pairs, len(indices))
        pairs_a = np.random.choice(indices, n)
        pairs_b = np.random.choice(indices, n)
        # Magnitude spectra (removes freq_shift phase rotation)
        spec_a = np.abs(np.fft.fft(iq_data[pairs_a], axis=1))
        spec_b = np.abs(np.fft.fft(iq_data[pairs_b], axis=1))
        dots = np.sum(spec_a * spec_b, axis=1)
        norms = np.sqrt(np.sum(spec_a ** 2, axis=1)
                        * np.sum(spec_b ** 2, axis=1))
        sims = dots / (norms + 1e-10)
        mean_sim = float(np.mean(sims))
        if mean_sim > threshold:
            warnings.append(
                f"{label}: mean spectral similarity {mean_sim:.3f} "
                f"(threshold {threshold}) — possible low signal diversity")

    return warnings


def cmd_generate(args):
    """Generate IQ dataset."""
    cfg = load_config(args.config)

    # CLI overrides
    if args.output:
        cfg.dataset.output_dir = args.output
    if args.seed is not None:
        cfg.dataset.seed = args.seed
    if args.domains:
        cfg.dataset.domains = [d.strip() for d in args.domains.split(",")]

    output_dir = cfg.dataset.output_dir
    os.makedirs(output_dir, exist_ok=True)

    setup_logging(output_dir=output_dir,
                  verbose=args.verbose, quiet=args.quiet)

    # Copy config alongside output for reproducibility
    if args.config and os.path.exists(args.config):
        shutil.copy2(args.config, os.path.join(output_dir, "config.toml"))

    # Filter generators — config is authoritative, CLI --generators narrows further
    if args.generators:
        requested = set(args.generators.split(","))
        gen_names = [g for g in requested if g in GENERATORS]
        unknown = requested - set(GENERATORS.keys())
        if unknown:
            log.warning("Unknown generators: %s", ", ".join(sorted(unknown)))
    else:
        gen_names = [g for g in cfg.generators if g in GENERATORS]

    # Check prerequisites and build run plan
    plan = []
    for name in gen_names:
        gen_cfg = cfg.generators[name]
        if not gen_cfg.enabled:
            continue
        # Apply dataset.workers as fallback for generators without own workers
        if gen_cfg.workers == 0 and cfg.dataset.workers != 0:
            gen_cfg.workers = cfg.dataset.workers
        gen = GENERATORS[name](gen_cfg, cfg.impairments,
                               fs=cfg.dataset.sample_rate,
                               window_len=cfg.dataset.window_length)
        missing = gen.check_prerequisites()
        if missing:
            log.warning("%s: skipping — missing tools: %s", name,
                        ", ".join(missing))
            continue
        plan.append((name, gen))

    if not plan:
        log.error("No generators to run")
        return 1

    # Clean up orphans from any previous crashed run
    try:
        if not pid_registry.cleanup_stale():
            log.error("Aborting — another generation is running")
            return 1
        pid_registry.init_registry()
    except Exception as e:
        log.debug("PID registry init failed (non-fatal): %s", e)

    total_classes = sum(len(g.signal_classes) for _, g in plan)
    gen_list = ", ".join(n for n, _ in plan)
    log.info("Generating across %d classes using: %s", total_classes, gen_list)
    log.info("Output: %s", output_dir)

    # Install graceful shutdown handler
    _state.reset_shutdown()
    prev_sigint = signal.getsignal(signal.SIGINT)
    prev_sigterm = signal.getsignal(signal.SIGTERM)

    def _handle_shutdown(signum, frame):
        if _state.shutdown_requested():
            # Second signal — force exit
            sys.exit(1)
        _state.request_shutdown()
        sig_name = signal.Signals(signum).name
        log.warning("%s received — finishing current class, "
                    "then stopping. Press again to force quit.", sig_name)

    signal.signal(signal.SIGINT, _handle_shutdown)
    signal.signal(signal.SIGTERM, _handle_shutdown)

    t_start = time.time()
    all_results = {}
    enabled_domains = cfg.dataset.domains

    try:
        for domain_name in enabled_domains:
            domain = DOMAINS[domain_name]
            domain_labels = set(labels_for_domain(domain_name))

            # Use domain-specific output subdirectory when multiple domains
            if len(enabled_domains) > 1:
                domain_dir = os.path.join(output_dir, domain_name)
            else:
                domain_dir = output_dir

            for name, gen in plan:
                if _state.shutdown_requested():
                    log.warning("Shutdown requested — skipping %s", name)
                    break

                # Filter generator classes to this domain
                domain_classes = [c for c in gen.signal_classes
                                  if c in domain_labels]
                if not domain_classes:
                    continue

                # Create a domain-specific generator instance
                gen_cfg = cfg.generators[name]
                domain_gen = GENERATORS[name](
                    gen_cfg, cfg.impairments,
                    fs=domain.sample_rate,
                    window_len=domain.window_length)
                # Override signal_classes to domain subset
                domain_gen.signal_classes = domain_classes

                log.info("[%s] %s: %d classes",
                         domain_name, name, len(domain_classes))
                gen_results = domain_gen.run(domain_dir,
                                             seed=cfg.dataset.seed)
                if gen_results:
                    all_results.update(gen_results)
    finally:
        signal.signal(signal.SIGINT, prev_sigint)
        signal.signal(signal.SIGTERM, prev_sigterm)
        try:
            pid_registry.remove_registry()
        except Exception:
            pass

    # Assemble — one dataset per domain
    total_windows = 0
    total_size_mb = 0.0
    all_tags = []
    all_diversity_warnings = []

    for domain_name in enabled_domains:
        domain = DOMAINS[domain_name]
        domain_labels = labels_for_domain(domain_name)

        if len(enabled_domains) > 1:
            domain_dir = os.path.join(output_dir, domain_name)
            prefix = f"rf_datagen_{domain_name}"
        else:
            domain_dir = output_dir
            prefix = "rf_datagen"

        log.info("Assembling %s dataset...", domain_name)
        iq_data, tags, scenarios, snrs = assemble_parts(
            domain_dir,
            window_len=domain.window_length,
            labels=domain_labels)

        if len(iq_data) == 0:
            log.warning("No data generated for domain %s", domain_name)
            continue

        # Diversity spot-check before saving
        dw = _diversity_check(iq_data, tags)
        all_diversity_warnings.extend(dw)

        save_dataset(iq_data, tags, scenarios, domain_dir, prefix=prefix,
                     snrs=snrs)

        iq_path = os.path.join(domain_dir, f"{prefix}_iq.npy")
        if os.path.exists(iq_path):
            total_size_mb += os.path.getsize(iq_path) / (1024 * 1024)
        total_windows += len(iq_data)
        all_tags.extend(tags)

    elapsed = time.time() - t_start

    if total_windows == 0:
        log.error("No data generated")
        return 1

    log.info("Done in %.0fs — %d windows, %.0f MB", elapsed, total_windows,
             total_size_mb)

    # Per-class summary
    from collections import Counter
    counts = Counter(all_tags)
    log.info("Per-class counts (%d classes):", len(counts))
    for label in SIGNAL_LABELS:
        if label in counts:
            log.info("  %15s: %d", label, counts[label])

    # Post-assembly diversity spot-check
    for w in all_diversity_warnings:
        log.warning("DIVERSITY: %s", w)

    # Generation report: failures
    failed = {k: v for k, v in all_results.items() if v["status"] == "failed"}
    ok_count = sum(1 for v in all_results.values() if v["status"] in ("ok", "cached"))

    if failed:
        log.warning("FAILED (%d classes):", len(failed))
        for cls, info in sorted(failed.items()):
            log.warning("  %15s: %s", cls, info.get("reason", "unknown"))

    # Write generation report
    report = {
        "domains": enabled_domains,
        "total_classes": total_classes,
        "generated": ok_count,
        "failed": len(failed),
        "total_windows": total_windows,
        "elapsed_s": round(elapsed, 1),
        "size_mb": round(total_size_mb, 1),
        "per_class": all_results,
        "diversity_warnings": all_diversity_warnings,
    }
    report_path = os.path.join(output_dir, "generation_report.json")
    atomic_write_json(report_path, report)

    if all_diversity_warnings:
        log.error("DIVERSITY CHECK FAILED — %d classes have low signal "
                  "diversity. Training on this data will produce a bad model.",
                  len(all_diversity_warnings))
        return 3

    if failed:
        log.warning("Generated %d/%d classes (%d failed)",
                    ok_count, total_classes, len(failed))
        return 2  # partial success
    return 0


def cmd_list(args):
    """List all signal classes and their generators."""
    # Build class -> generators mapping
    class_gens = {label: [] for label in SIGNAL_LABELS}
    for name, gen_cls in GENERATORS.items():
        gen = gen_cls(GeneratorConfig())
        for cls in gen.signal_classes:
            if cls in class_gens:
                class_gens[cls].append(name)

    print(f"Signal Classes ({len(SIGNAL_LABELS)}):")
    for label in SIGNAL_LABELS:
        gens = ", ".join(class_gens[label]) if class_gens[label] else "(none)"
        print(f"  {label:<15s} {gens}")

    return 0


def cmd_validate(args):
    """Validate dataset integrity."""
    data_dir = args.data_dir
    iq_path = os.path.join(data_dir, "rf_datagen_iq.npy")
    csv_path = os.path.join(data_dir, "rf_datagen_tags.csv")

    errors = []

    if not os.path.exists(iq_path):
        errors.append(f"Missing: {iq_path}")
    if not os.path.exists(csv_path):
        errors.append(f"Missing: {csv_path}")

    if errors:
        for e in errors:
            print(f"  ERROR: {e}")
        return 1

    iq = np.load(iq_path)
    print(f"IQ data: {iq.shape}, dtype={iq.dtype}")

    if iq.ndim != 2:
        errors.append(f"Expected 2D array, got {iq.ndim}D")
    elif iq.shape[1] != WINDOW_LEN:
        errors.append(f"Window length {iq.shape[1]} != expected {WINDOW_LEN}")

    if not np.iscomplexobj(iq):
        errors.append(f"Expected complex dtype, got {iq.dtype}")

    nan_count = np.sum(np.isnan(iq))
    if nan_count > 0:
        errors.append(f"{nan_count} NaN values found")

    inf_count = np.sum(np.isinf(iq))
    if inf_count > 0:
        errors.append(f"{inf_count} Inf values found")

    # Check CSV
    import csv
    with open(csv_path) as f:
        reader = csv.reader(f)
        header = next(reader)
        rows = list(reader)

    print(f"Metadata: {len(rows)} rows, columns: {header}")

    if len(rows) != iq.shape[0]:
        errors.append(f"Row count mismatch: CSV={len(rows)}, IQ={iq.shape[0]}")

    # Check labels
    label_col = header.index("mode") if "mode" in header else None
    if label_col is not None:
        labels = set(row[label_col] for row in rows)
        unknown = labels - set(SIGNAL_LABELS)
        if unknown:
            errors.append(f"Unknown labels: {unknown}")
        print(f"Classes present: {len(labels)}/{len(SIGNAL_LABELS)}")
        missing = set(SIGNAL_LABELS) - labels
        if missing:
            print(f"  Missing classes: {', '.join(sorted(missing))}")

    if errors:
        print("\nValidation FAILED:")
        for e in errors:
            print(f"  ERROR: {e}")
        return 1

    print("\nValidation PASSED")
    return 0


def cmd_validate_roundtrip(args):
    """Round-trip encode/decode validation."""
    from tests.test_roundtrip import run_validation, ALL_MODES

    modes = args.modes
    if modes:
        for m in modes:
            if m not in ALL_MODES:
                print(f"ERROR: Unknown mode '{m}'. Available: {ALL_MODES}",
                      file=sys.stderr)
                return 1

    snr_levels = None
    if args.snr_only is not None:
        snr_levels = args.snr_only

    all_pass, _ = run_validation(
        modes=modes,
        trials=args.trials,
        snr_levels=snr_levels,
        clean_only=args.clean_only,
        output=args.output or "./output/validation",
        verbose=args.verbose,
        seed=args.seed if args.seed is not None else 42,
    )
    return 0 if all_pass else 1


def cmd_validate_ml(args):
    """ML-based classification validation."""
    from .ml_validate import run_ml_validation

    results = run_ml_validation(
        model=args.model,
        modes=args.modes,
        samples=args.samples,
        snr_sweep=args.snr_sweep,
        snr_levels=args.snr_levels,
        output=args.output or "./output/ml_validation",
        device=args.device,
        threshold=args.threshold,
    )
    if "error" in results:
        return 1
    return 0


def _cmd_e2e(args):
    """Lazy import wrapper for e2e pipeline."""
    from .e2e import cmd_e2e
    return cmd_e2e(args)


def cmd_qc(args):
    """Dispatch to QC inspection subcommands."""
    from . import qc

    if args.qc_command is None:
        print("Usage: rf-datagen qc {text,audio,modulated,impaired,dataset,report,probe,benchmark}")
        return 1

    dispatch = {
        "text": qc.cmd_text,
        "audio": qc.cmd_audio,
        "modulated": qc.cmd_modulated,
        "impaired": qc.cmd_impaired,
        "dataset": qc.cmd_dataset,
        "report": qc.cmd_report,
        "probe": qc.cmd_probe,
        "benchmark": qc.cmd_benchmark,
    }
    return dispatch[args.qc_command](args)


def cmd_cleanup(args):
    """Clean up orphaned processes from a crashed generation run."""
    setup_logging(verbose=args.verbose, quiet=False)
    cleaned = pid_registry.cleanup_stale(force=args.force)
    if cleaned:
        log.info("Cleanup complete")
        return 0
    return 1


def cmd_inspect(args):
    """Show spectrogram/stats for a class (requires matplotlib)."""
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib is required for inspect. Install with: pip install matplotlib")
        return 1

    data_dir = args.data_dir
    iq_path = os.path.join(data_dir, "rf_datagen_iq.npy")
    csv_path = os.path.join(data_dir, "rf_datagen_tags.csv")

    if not os.path.exists(iq_path) or not os.path.exists(csv_path):
        print(f"Dataset not found in {data_dir}")
        return 1

    iq = np.load(iq_path)

    import csv
    with open(csv_path) as f:
        reader = csv.reader(f)
        header = next(reader)
        rows = list(reader)

    label_col = header.index("mode") if "mode" in header else 1
    labels = [row[label_col] for row in rows]

    target = args.signal_class.upper()
    indices = [i for i, l in enumerate(labels) if l == target]

    if not indices:
        print(f"No samples found for class '{target}'")
        available = sorted(set(labels))
        print(f"Available: {', '.join(available)}")
        return 1

    print(f"{target}: {len(indices)} samples")

    # Show 4 random samples
    n_show = min(4, len(indices))
    show_idx = np.random.choice(indices, n_show, replace=False)

    fig, axes = plt.subplots(2, n_show, figsize=(4 * n_show, 6))
    if n_show == 1:
        axes = axes.reshape(2, 1)

    from .constants import FS

    for col, idx in enumerate(show_idx):
        window = iq[idx]

        # Time domain (magnitude)
        axes[0, col].plot(np.abs(window), linewidth=0.5)
        axes[0, col].set_title(f"{target} #{idx}")
        axes[0, col].set_xlabel("Sample")
        axes[0, col].set_ylabel("|IQ|")

        # Spectrogram
        axes[1, col].specgram(window, NFFT=256, Fs=FS, noverlap=128,
                              cmap="viridis")
        axes[1, col].set_xlabel("Time (s)")
        axes[1, col].set_ylabel("Freq (Hz)")

    plt.tight_layout()
    plt.suptitle(f"{target} — {len(indices)} total samples", y=1.02)

    out_png = os.path.join(data_dir, f"inspect_{target}.png")
    plt.savefig(out_png, dpi=150, bbox_inches="tight")
    print(f"Saved: {out_png}")
    plt.show()

    # Stats
    class_data = iq[indices]
    power = np.mean(np.abs(class_data) ** 2, axis=1)
    print(f"  Mean power: {np.mean(power):.4f}")
    print(f"  Power std:  {np.std(power):.4f}")
    print(f"  Min power:  {np.min(power):.6f}")
    print(f"  Max power:  {np.max(power):.4f}")

    return 0


def main():
    parser = argparse.ArgumentParser(
        prog="rf-datagen",
        description="RF signal IQ dataset generator for ML training")

    sub = parser.add_subparsers(dest="command")

    # generate
    p_gen = sub.add_parser("generate", help="Generate IQ dataset")
    p_gen.add_argument("--config", "-c", default=None,
                       help="Path to config.toml (default: built-in defaults)")
    p_gen.add_argument("--generators", "-g", default=None,
                       help="Comma-separated generator names (default: all enabled)")
    p_gen.add_argument("--output", "-o", default=None,
                       help="Output directory (overrides config)")
    p_gen.add_argument("--seed", "-s", type=int, default=None,
                       help="Random seed (overrides config)")
    p_gen.add_argument("--domains", "-d", default=None,
                       help="Comma-separated domain names: narrowband,moderate,wideband")
    p_gen.add_argument("--verbose", "-v", action="store_true",
                       help="Enable verbose (DEBUG) console output")
    p_gen.add_argument("--quiet", "-q", action="store_true",
                       help="Suppress INFO messages (WARNING only)")

    # list
    sub.add_parser("list", help="List all signal classes and their generators")

    # cleanup
    p_clean = sub.add_parser("cleanup",
                              help="Kill orphaned processes from a crashed run")
    p_clean.add_argument("--force", action="store_true",
                          help="Force cleanup even if parent appears alive")
    p_clean.add_argument("--verbose", "-v", action="store_true",
                          help="Enable verbose output")

    # validate
    p_val = sub.add_parser("validate", help="Validate dataset integrity")
    p_val.add_argument("data_dir", help="Path to dataset directory")

    # validate-roundtrip
    p_vrt = sub.add_parser("validate-roundtrip",
                           help="Round-trip encode/decode validation")
    p_vrt.add_argument("--modes", nargs="*", default=None,
                       help="Modes to test (default: all)")
    p_vrt.add_argument("--trials", type=int, default=10,
                       help="Trials per SNR level (default: 10)")
    p_vrt.add_argument("--snr-only", nargs="*", type=int, default=None,
                       help="Test only specific SNR levels (dB)")
    p_vrt.add_argument("--clean-only", action="store_true",
                       help="Only test clean channel (no noise)")
    p_vrt.add_argument("--output", "-o", default=None,
                       help="Output directory for results")
    p_vrt.add_argument("--verbose", action="store_true",
                       help="Save debug artifacts on decode failure")
    p_vrt.add_argument("--seed", "-s", type=int, default=None,
                       help="Random seed")

    # validate-ml
    p_vml = sub.add_parser("validate-ml",
                           help="ML-based classification validation")
    p_vml.add_argument("--model", default="torchsig",
                       choices=["torchsig", "rfml", "cgdnn", "all"],
                       help="Model backend (default: torchsig)")
    p_vml.add_argument("--modes", nargs="*", default=None,
                       help="Signal modes to validate (default: all mapped)")
    p_vml.add_argument("--samples", type=int, default=50,
                       help="IQ samples per mode (default: 50)")
    p_vml.add_argument("--snr-sweep", action="store_true",
                       help="Run across multiple SNR levels")
    p_vml.add_argument("--snr-levels", nargs="*", type=int, default=None,
                       help="SNR levels for sweep (default: -10 0 10 20)")
    p_vml.add_argument("--output", "-o", default=None,
                       help="Output directory for results")
    p_vml.add_argument("--device", default="cpu",
                       choices=["cpu", "openvino"],
                       help="Inference device (default: cpu)")
    p_vml.add_argument("--threshold", type=float, default=0.5,
                       help="Min accuracy to pass (default: 0.5)")

    # e2e — end-to-end validation pipeline
    p_e2e = sub.add_parser("e2e",
                           help="End-to-end generate + validate pipeline")
    p_e2e.add_argument("--config", "-c", default=None,
                       help="Path to config.toml")
    p_e2e.add_argument("--output", "-o", default=None,
                       help="Output directory")
    p_e2e.add_argument("--domains", "-d", default=None,
                       help="Comma-separated domains: narrowband,moderate,wideband")
    p_e2e.add_argument("--skip-generate", action="store_true",
                       help="Use existing output (skip generation)")
    p_e2e.add_argument("--skip-roundtrip", action="store_true",
                       help="Skip round-trip decode tests")
    p_e2e.add_argument("--skip-ml", action="store_true",
                       help="Skip ML classification (needs torch)")
    p_e2e.add_argument("--spectral-samples", type=int, default=20,
                       help="Windows per class for spectral checks (default: 20)")
    p_e2e.add_argument("--ml-samples", type=int, default=30,
                       help="Windows per class for ML (default: 30)")
    p_e2e.add_argument("--strict", action="store_true",
                       help="Fail on advisory gate failures too")
    p_e2e.add_argument("--json-report", default=None,
                       help="Custom path for JSON report")
    p_e2e.add_argument("--html-report", default=None,
                       help="Generate HTML report at this path")
    p_e2e.add_argument("--seed", "-s", type=int, default=42,
                       help="Random seed")
    p_e2e.add_argument("--verbose", "-v", action="store_true")
    p_e2e.add_argument("--quiet", "-q", action="store_true")

    # inspect
    p_ins = sub.add_parser("inspect",
                           help="Plot spectrogram/stats for a class from dataset")
    p_ins.add_argument("data_dir", help="Path to dataset directory")
    p_ins.add_argument("--class", dest="signal_class", required=True,
                       help="Signal class name (e.g. FT8)")

    # qc — quality-control inspection tool
    p_qc = sub.add_parser("qc", help="QC inspection and reporting tool")
    qc_sub = p_qc.add_subparsers(dest="qc_command")

    p_qc_text = qc_sub.add_parser("text", help="Show generated text content")
    p_qc_text.add_argument("--generator", default="analog",
                           choices=["analog", "fldigi", "packet", "digivoice",
                                    "all"],
                           help="Which text generator (default: analog)")
    p_qc_text.add_argument("--mode", default=None,
                           help="Mode for fldigi generator (default: PSK31)")
    p_qc_text.add_argument("--count", type=int, default=10)
    p_qc_text.add_argument("--seed", type=int, default=42)

    p_qc_audio = qc_sub.add_parser("audio", help="Export TTS speech as WAV")
    p_qc_audio.add_argument("--count", type=int, default=5)
    p_qc_audio.add_argument("--output", default="/tmp/qc_inspect/audio")
    p_qc_audio.add_argument("--voice-cache",
                            default="artifacts/data/piper-voices")
    p_qc_audio.add_argument("--play", action="store_true",
                            help="Play audio via aplay")
    p_qc_audio.add_argument("--seed", type=int, default=42)

    p_qc_mod = qc_sub.add_parser("modulated",
                                  help="Visualize clean modulated IQ signals")
    p_qc_mod.add_argument("--mode", nargs="+", default=["FT8"],
                          help="Signal mode(s) (default: FT8)")
    p_qc_mod.add_argument("--all-modes", action="store_true",
                          help="Generate for all available modes")
    p_qc_mod.add_argument("--count", type=int, default=3)
    p_qc_mod.add_argument("--snr-grid", action="store_true",
                          help="Also generate SNR comparison grid")
    p_qc_mod.add_argument("--output", default="/tmp/qc_inspect/modulated")
    p_qc_mod.add_argument("--seed", type=int, default=42)

    p_qc_imp = qc_sub.add_parser("impaired",
                                  help="Visualize signals after impairments")
    p_qc_imp.add_argument("--mode", required=True, help="Signal mode")
    p_qc_imp.add_argument("--snr", type=int, default=10,
                          help="SNR in dB (default: 10)")
    p_qc_imp.add_argument("--all-snr", action="store_true",
                          help="Generate at all SNR levels")
    p_qc_imp.add_argument("--scenario", default=None,
                          help="Force specific impairment scenario")
    p_qc_imp.add_argument("--count", type=int, default=3)
    p_qc_imp.add_argument("--output", default="/tmp/qc_inspect/impaired")
    p_qc_imp.add_argument("--seed", type=int, default=42)

    p_qc_ds = qc_sub.add_parser("dataset",
                                 help="Inspect existing .npy training data")
    p_qc_ds.add_argument("--path", required=True,
                         help="Path to dataset directory")
    p_qc_ds.add_argument("--mode", default=None,
                         help="Filter to specific mode")
    p_qc_ds.add_argument("--count", type=int, default=10)
    p_qc_ds.add_argument("--output", default=None,
                         help="Output dir for visualizations")
    p_qc_ds.add_argument("--seed", type=int, default=42)

    p_qc_rpt = qc_sub.add_parser("report",
                                   help="Full HTML report for one mode")
    p_qc_rpt.add_argument("--mode", required=True, help="Signal mode")
    p_qc_rpt.add_argument("--count", type=int, default=3,
                          help="Samples per section")
    p_qc_rpt.add_argument("--output", default="/tmp/qc_inspect/reports")
    p_qc_rpt.add_argument("--voice-cache",
                           default="artifacts/data/piper-voices")
    p_qc_rpt.add_argument("--seed", type=int, default=42)

    p_qc_probe = qc_sub.add_parser("probe",
                                     help="GNU Radio probe — MITM debug utility")
    p_qc_probe.add_argument("--mode", required=True, help="Signal mode")
    p_qc_probe.add_argument("--action", default="analyze",
                            choices=["analyze", "decode", "transform"],
                            help="Probe action (default: analyze)")
    p_qc_probe.add_argument("--point", default="after-generation",
                            choices=["after-generation", "after-impairments",
                                     "after-windowing", "custom"],
                            help="Pipeline tap point")
    p_qc_probe.add_argument("--snr", type=int, default=None,
                            help="Apply AWGN at this SNR before probing")
    p_qc_probe.add_argument("--decoder", default=None,
                            help="Decoder for decode action")
    p_qc_probe.add_argument("--flowgraph", default=None,
                            help="Flowgraph for transform action")
    p_qc_probe.add_argument("--transform-params", nargs="*",
                            help="Transform params as key=value pairs")
    p_qc_probe.add_argument("--input", default=None,
                            help="Input IQ file (complex64) instead of generating")
    p_qc_probe.add_argument("--output", default=None,
                            help="Output IQ file for transform results")
    p_qc_probe.add_argument("--output-dir", default=None,
                            help="Output dir for visualizations")
    p_qc_probe.add_argument("--seed", type=int, default=42)

    p_qc_bench = qc_sub.add_parser("benchmark",
                                      help="Benchmark sdr vs NumPy modulation")
    p_qc_bench.add_argument("--trials", type=int, default=20,
                            help="Number of trials per function (default: 20)")
    p_qc_bench.add_argument("--symbols", type=int, default=500,
                            help="Number of symbols per trial (default: 500)")

    args = parser.parse_args()

    if args.command is None:
        parser.print_help()
        return 1

    commands = {
        "generate": cmd_generate,
        "list": cmd_list,
        "cleanup": cmd_cleanup,
        "validate": cmd_validate,
        "validate-roundtrip": cmd_validate_roundtrip,
        "validate-ml": cmd_validate_ml,
        "e2e": _cmd_e2e,
        "inspect": cmd_inspect,
        "qc": cmd_qc,
    }

    sys.exit(commands[args.command](args))


if __name__ == "__main__":
    main()
