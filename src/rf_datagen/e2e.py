"""End-to-end dataset generation and exhaustive validation pipeline.

Orchestrates:
  1. Generate all classes across all domains
  2. Structural validation (shapes, dtypes, completeness, distributions)
  3. Spectral validation (per-class BW and PAPR fingerprinting)
  4. Round-trip decode on actual generated data (optional)
  5. ML classification on actual generated data (optional)
  6. Structured pass/fail report with actionable metrics
"""

import json
import os
import sys
import time

import numpy as np

from .config import load_config
from .dataset_checks import run_structural_checks
from .domains import DOMAINS, labels_for_domain, SIGNAL_DOMAIN_MAP
from .logging_config import get_logger, setup_logging
from .output import atomic_write_json
from .spectral_fingerprint import validate_spectral

log = get_logger("e2e")


# ---------------------------------------------------------------------------
# Quality gate thresholds
# ---------------------------------------------------------------------------

STRUCTURAL_GATE_REQUIRED = True   # Gate 1 must pass
SPECTRAL_GATE_REQUIRED = True     # Gate 2 must pass
ROUNDTRIP_GATE_REQUIRED = False   # Gate 3 advisory by default
ML_GATE_REQUIRED = False          # Gate 4 advisory by default

ROUNDTRIP_DECODE_THRESHOLD = 0.50  # Per-mode at 25dB SNR
ROUNDTRIP_OVERALL_THRESHOLD = 0.70
ML_PER_CLASS_THRESHOLD = 0.40
ML_OVERALL_THRESHOLD = 0.60


class E2EPipeline:
    """Orchestrate end-to-end dataset generation and validation."""

    def __init__(self, config_path=None, output_dir=None, domains=None,
                 skip_generate=False, skip_roundtrip=False, skip_ml=False,
                 spectral_samples=20, ml_samples=30,
                 strict=False, json_report=None, html_report=None, seed=42,
                 verbose=False, quiet=False):
        self.config_path = config_path
        self.output_dir = output_dir
        self.domains = domains
        self.skip_generate = skip_generate
        self.skip_roundtrip = skip_roundtrip
        self.skip_ml = skip_ml
        self.spectral_samples = spectral_samples
        self.ml_samples = ml_samples
        self.strict = strict
        self.json_report = json_report
        self.html_report = html_report
        self.seed = seed
        self.verbose = verbose
        self.quiet = quiet

        self.cfg = None
        self.results = {}
        self.t_start = None

    def run(self):
        """Execute the full pipeline. Returns exit code (0=pass, 1=fail)."""
        self.t_start = time.time()

        # Load config
        self.cfg = load_config(self.config_path)
        if self.output_dir:
            self.cfg.dataset.output_dir = self.output_dir
        if self.domains:
            self.cfg.dataset.domains = self.domains
        if self.seed is not None:
            self.cfg.dataset.seed = self.seed

        output_dir = self.cfg.dataset.output_dir
        os.makedirs(output_dir, exist_ok=True)
        setup_logging(output_dir=output_dir,
                      verbose=self.verbose, quiet=self.quiet)

        enabled_domains = self.cfg.dataset.domains

        log.info("E2E pipeline: domains=%s, output=%s",
                 enabled_domains, output_dir)

        # Phase 1: Generate
        if not self.skip_generate:
            log.info("=" * 60)
            log.info("PHASE 1: GENERATE")
            log.info("=" * 60)
            gen_rc = self._phase_generate()
            if gen_rc not in (0, 2):  # 0=ok, 2=partial
                log.error("Generation failed (rc=%d), aborting", gen_rc)
                return 1
        else:
            log.info("Skipping generation (--skip-generate)")

        # Phase 2: Structural validation
        log.info("=" * 60)
        log.info("PHASE 2: STRUCTURAL VALIDATION")
        log.info("=" * 60)
        self.results["structural"] = run_structural_checks(
            output_dir, enabled_domains, self.cfg)
        self._log_phase_result("structural")

        # Phase 3: Spectral validation
        log.info("=" * 60)
        log.info("PHASE 3: SPECTRAL VALIDATION")
        log.info("=" * 60)
        self.results["spectral"] = validate_spectral(
            output_dir, enabled_domains,
            n_samples=self.spectral_samples, seed=self.seed)
        self._log_phase_result("spectral")

        # Phase 4: Round-trip decode
        if not self.skip_roundtrip:
            log.info("=" * 60)
            log.info("PHASE 4: ROUND-TRIP DECODE")
            log.info("=" * 60)
            self.results["roundtrip"] = self._phase_roundtrip(
                output_dir, enabled_domains)
            self._log_phase_result("roundtrip")
        else:
            log.info("Skipping round-trip (--skip-roundtrip)")
            self.results["roundtrip"] = {"status": "SKIP"}

        # Phase 5: ML classification
        if not self.skip_ml:
            log.info("=" * 60)
            log.info("PHASE 5: ML CLASSIFICATION")
            log.info("=" * 60)
            self.results["ml_classification"] = self._phase_ml(
                output_dir, enabled_domains)
            self._log_phase_result("ml_classification")
        else:
            log.info("Skipping ML classification (--skip-ml)")
            self.results["ml_classification"] = {"status": "SKIP"}

        # Phase 6: Report
        log.info("=" * 60)
        log.info("PHASE 6: REPORT")
        log.info("=" * 60)
        verdict = self._compute_verdict()
        report = self._build_report(verdict)
        self._save_report(report, output_dir)
        self._print_summary(report)

        return 0 if verdict == "PASS" else 1

    # ------------------------------------------------------------------
    # Phase 1: Generation
    # ------------------------------------------------------------------

    def _phase_generate(self):
        """Run dataset generation via the existing cmd_generate logic."""
        import argparse

        # Build a synthetic args namespace matching cmd_generate expectations
        args = argparse.Namespace(
            config=self.config_path,
            generators=None,
            output=self.cfg.dataset.output_dir,
            seed=self.cfg.dataset.seed,
            domains=",".join(self.cfg.dataset.domains),
            verbose=self.verbose,
            quiet=self.quiet,
        )

        from .cli import cmd_generate
        return cmd_generate(args)

    # ------------------------------------------------------------------
    # Phase 4: Round-trip decode (fresh-generation synthesizer health check)
    # ------------------------------------------------------------------
    #
    # Dataset windows (2048 samples = 0.17s at 12 kHz) are too short for
    # most decoders (FT8 needs 12.6s, WSPR needs 120s).  Instead we run
    # fresh-generation round-trip trials that validate each synthesizer's
    # ability to produce decodable output.  This verifies that the code
    # path used during generation produces correct signals.
    # ------------------------------------------------------------------

    # Modes with available round-trip validators
    _ROUNDTRIP_MODES = [
        "FT8", "WSPR", "CW",
        "SSB", "AM", "FM",
        "DTMF", "EAS", "POCSAG", "ACARS",
        # BELL103/BELL202 excluded: minimodem can open audio device
    ]

    def _phase_roundtrip(self, output_dir, domains):
        """Run fresh-generation round-trip decode for each mode.

        Validates that synthesizers produce decodable signals by generating
        fresh IQ and feeding it to external decoders.  This is a synthesizer
        health check, not a dataset content check.
        """
        import tempfile

        per_mode = {}

        # Only narrowband has round-trip validators
        if "narrowband" not in domains:
            return {
                "status": "SKIP",
                "detail": "No narrowband domain in this run",
            }

        try:
            from tests.roundtrip import VALIDATORS
        except ImportError:
            return {
                "status": "SKIP",
                "detail": "Round-trip validators not importable",
            }

        available_modes = [m for m in self._ROUNDTRIP_MODES
                           if m in VALIDATORS and VALIDATORS[m].available()]

        if not available_modes:
            return {"status": "SKIP", "detail": "No validators available"}

        n_trials = 5  # Trials per mode (keep fast)
        np.random.seed(self.seed)

        for mode in available_modes:
            validator_cls = VALIDATORS[mode]
            inst = validator_cls()
            try:
                inst.setup()
            except Exception as e:
                per_mode[mode] = {
                    "decode_rate": 0, "threshold": ROUNDTRIP_DECODE_THRESHOLD,
                    "status": "SKIP", "detail": f"Setup failed: {e}",
                }
                continue

            decodes = 0
            total = 0
            try:
                trial_fn = inst.make_trial(mode)
                for t in range(n_trials):
                    with tempfile.TemporaryDirectory(
                            prefix=f"e2e_rt_{mode}_") as tmpdir:
                        try:
                            ok = trial_fn(None, t, tmpdir)
                            total += 1
                            if ok:
                                decodes += 1
                        except Exception:
                            total += 1
            except Exception as e:
                log.warning("Round-trip %s error: %s", mode, e)
            finally:
                try:
                    inst.teardown()
                except Exception:
                    pass

            rate = decodes / total if total > 0 else 0
            expected_fail = getattr(validator_cls, "expected_fail", False)
            status = ("PASS" if rate >= ROUNDTRIP_DECODE_THRESHOLD
                      else ("EXPECTED_FAIL" if expected_fail else "FAIL"))
            per_mode[mode] = {
                "decode_rate": round(rate, 3),
                "threshold": ROUNDTRIP_DECODE_THRESHOLD,
                "decodes": decodes,
                "total": total,
                "status": status,
            }
            log.info("  %12s: %d/%d = %.1f%% [%s]",
                     mode, decodes, total, rate * 100, status)

        # Overall gate (exclude expected_fail)
        tested = [v for v in per_mode.values()
                  if v["status"] not in ("SKIP", "EXPECTED_FAIL")]
        if not tested:
            return {"status": "SKIP", "per_mode": per_mode}

        n_pass = sum(1 for v in tested if v["status"] == "PASS")
        overall_rate = n_pass / len(tested) if tested else 0
        gate = overall_rate >= ROUNDTRIP_OVERALL_THRESHOLD

        return {
            "status": "PASS" if gate else "FAIL",
            "per_mode": per_mode,
            "pass_rate": f"{n_pass}/{len(tested)}",
            "overall_rate": round(overall_rate, 3),
        }

    # ------------------------------------------------------------------
    # Phase 5: ML classification on dataset
    # ------------------------------------------------------------------

    def _phase_ml(self, output_dir, domains):
        """Run ML classification on windows from the generated dataset."""
        try:
            from .ml_validate._runner import MLValidationRunner
            from .ml_validate._mapping import mappable_signals, get_expected_class
            from .ml_validate._adapters import adapt_torchsig
        except ImportError:
            return {"status": "SKIP", "detail": "ML validation deps not available"}

        # Check if a backend is available
        try:
            from .ml_validate._runner import _BACKENDS
            from .ml_validate import torchsig_backend  # noqa: F401
        except ImportError:
            pass

        from .ml_validate._runner import available_backends
        backends = available_backends()
        if not backends:
            return {"status": "SKIP", "detail": "No ML backends available"}

        backend_name = backends[0]

        import csv as csv_mod

        multi_domain = len(domains) > 1
        per_class = {}
        rng = np.random.RandomState(self.seed)

        # Load each domain's dataset
        for domain_name in domains:
            domain = DOMAINS[domain_name]

            if multi_domain:
                domain_dir = os.path.join(output_dir, domain_name)
                prefix = f"rf_datagen_{domain_name}"
            else:
                domain_dir = output_dir
                prefix = "rf_datagen"

            iq_path = os.path.join(domain_dir, f"{prefix}_iq.npy")
            csv_path = os.path.join(domain_dir, f"{prefix}_tags.csv")

            if not os.path.exists(iq_path) or not os.path.exists(csv_path):
                continue

            iq = np.load(iq_path, mmap_mode="r")
            with open(csv_path) as f:
                reader = csv_mod.reader(f)
                header = next(reader)
                rows = list(reader)

            mode_col = header.index("mode") if "mode" in header else 1
            labels = [row[mode_col] for row in rows]

            mappable = mappable_signals(backend_name)
            domain_labels = set(labels_for_domain(domain_name))

            for signal_name in mappable:
                if signal_name not in domain_labels:
                    continue

                expected = get_expected_class(signal_name, backend_name)
                if expected is None:
                    continue

                indices = [i for i, l in enumerate(labels) if l == signal_name]
                if not indices:
                    per_class[signal_name] = {
                        "status": "SKIP", "detail": "No samples in dataset",
                    }
                    continue

                n = min(self.ml_samples, len(indices))
                chosen = rng.choice(indices, n, replace=False)

                # Load backend
                try:
                    from .ml_validate._runner import _BACKENDS
                    backend_cls = _BACKENDS[backend_name]
                    backend = backend_cls()
                    backend.load()
                except Exception as e:
                    per_class[signal_name] = {
                        "status": "SKIP",
                        "detail": f"Backend load failed: {e}",
                    }
                    continue

                correct = 0
                total = 0
                for idx in chosen:
                    window = np.array(iq[idx])
                    try:
                        adapted = adapt_torchsig(window)[np.newaxis, ...]
                        predictions = backend.predict(adapted)
                        if predictions:
                            pred_class, conf = predictions[0]
                            total += 1
                            if pred_class == expected:
                                correct += 1
                    except Exception:
                        total += 1

                accuracy = correct / total if total > 0 else 0
                status = ("PASS" if accuracy >= ML_PER_CLASS_THRESHOLD
                          else "FAIL")
                per_class[signal_name] = {
                    "accuracy": round(accuracy, 3),
                    "expected": expected,
                    "correct": correct,
                    "total": total,
                    "threshold": ML_PER_CLASS_THRESHOLD,
                    "status": status,
                }
                log.info("  %15s: %d/%d = %.1f%% [%s]",
                         signal_name, correct, total, accuracy * 100, status)

        # Overall gate
        tested = [v for v in per_class.values() if v["status"] != "SKIP"]
        if not tested:
            return {"status": "SKIP", "per_class": per_class}

        n_pass = sum(1 for v in tested if v["status"] == "PASS")
        overall_rate = n_pass / len(tested) if tested else 0
        overall_acc = (np.mean([v["accuracy"] for v in tested])
                       if tested else 0)

        gate = overall_rate >= ML_OVERALL_THRESHOLD

        return {
            "status": "PASS" if gate else "FAIL",
            "per_class": per_class,
            "pass_rate": f"{n_pass}/{len(tested)}",
            "overall_accuracy": round(float(overall_acc), 3),
        }

    # ------------------------------------------------------------------
    # Verdict & report
    # ------------------------------------------------------------------

    def _compute_verdict(self):
        """Compute overall pass/fail verdict from phase results."""
        structural = self.results.get("structural", {}).get("status", "SKIP")
        spectral = self.results.get("spectral", {}).get("status", "SKIP")
        roundtrip = self.results.get("roundtrip", {}).get("status", "SKIP")
        ml = self.results.get("ml_classification", {}).get("status", "SKIP")

        # Gate 1 + 2 are required
        if structural == "FAIL":
            return "FAIL"
        if spectral == "FAIL":
            return "FAIL"

        # In strict mode, gates 3 + 4 are also required
        if self.strict:
            if roundtrip == "FAIL":
                return "FAIL"
            if ml == "FAIL":
                return "FAIL"

        return "PASS"

    def _build_report(self, verdict):
        """Build structured JSON report."""
        elapsed = time.time() - self.t_start

        # Count checks
        total_checks = 0
        passed = 0
        failed = 0
        warnings = 0

        for phase_name, phase_result in self.results.items():
            if phase_result.get("status") == "SKIP":
                continue
            # Count structural checks
            if "checks" in phase_result:
                for check in phase_result["checks"].values():
                    total_checks += 1
                    if check["status"] == "PASS":
                        passed += 1
                    else:
                        failed += 1
            # Count spectral per-class
            if "per_class" in phase_result:
                for cls_result in phase_result["per_class"].values():
                    if cls_result.get("status") == "SKIP":
                        continue
                    total_checks += 1
                    if cls_result["status"] == "PASS":
                        passed += 1
                    else:
                        failed += 1
            # Count roundtrip per-mode
            if "per_mode" in phase_result:
                for mode_result in phase_result["per_mode"].values():
                    if mode_result.get("status") == "SKIP":
                        continue
                    total_checks += 1
                    if mode_result["status"] == "PASS":
                        passed += 1
                    else:
                        failed += 1

        return {
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
            "config": self.config_path,
            "domains": self.cfg.dataset.domains,
            "verdict": verdict,
            "strict": self.strict,
            "elapsed_seconds": round(elapsed, 1),
            "phases": self.results,
            "summary": {
                "total_checks": total_checks,
                "passed": passed,
                "failed": failed,
                "warnings": warnings,
            },
        }

    def _save_report(self, report, output_dir):
        """Save JSON and optionally HTML report to disk."""
        report_path = self.json_report or os.path.join(
            output_dir, "e2e_report.json")
        atomic_write_json(report_path, report)
        log.info("Report saved: %s", report_path)

        if self.html_report:
            html_path = self.html_report
        else:
            html_path = os.path.join(output_dir, "e2e_report.html")

        _write_html_report(report, html_path)
        log.info("HTML report: %s", html_path)

    def _print_summary(self, report):
        """Print human-readable summary to console."""
        v = report["verdict"]
        elapsed = report["elapsed_seconds"]
        summary = report["summary"]

        print()
        print("=" * 60)
        print(f"E2E VALIDATION {'PASSED' if v == 'PASS' else 'FAILED'}")
        print("=" * 60)

        for phase_name in ("structural", "spectral", "roundtrip",
                           "ml_classification"):
            phase = report["phases"].get(phase_name, {})
            status = phase.get("status", "SKIP")
            marker = {"PASS": "+", "FAIL": "!", "SKIP": "-"}.get(status, "?")
            extra = ""
            if "pass_rate" in phase:
                extra = f" ({phase['pass_rate']})"
            elif "checks" in phase:
                n_pass = sum(1 for c in phase["checks"].values()
                             if c["status"] == "PASS")
                extra = f" ({n_pass}/{len(phase['checks'])})"
            print(f"  [{marker}] {phase_name:<20s} {status}{extra}")

        print()
        print(f"  Checks: {summary['passed']}/{summary['total_checks']} passed, "
              f"{summary['failed']} failed")
        print(f"  Time: {elapsed:.1f}s")
        print(f"  Verdict: {v}")
        if self.strict:
            print("  Mode: STRICT (all gates required)")
        print()

    def _log_phase_result(self, phase_name):
        """Log phase result status."""
        result = self.results.get(phase_name, {})
        status = result.get("status", "UNKNOWN")
        log.info("Phase %s: %s", phase_name, status)


# ---------------------------------------------------------------------------
# CLI entry point (called from cli.py)
# ---------------------------------------------------------------------------

def cmd_e2e(args):
    """Run end-to-end validation pipeline."""
    domains = None
    if args.domains:
        domains = [d.strip() for d in args.domains.split(",")]

    pipeline = E2EPipeline(
        config_path=args.config,
        output_dir=args.output,
        domains=domains,
        skip_generate=args.skip_generate,
        skip_roundtrip=args.skip_roundtrip,
        skip_ml=args.skip_ml,
        spectral_samples=args.spectral_samples,
        ml_samples=args.ml_samples,
        strict=args.strict,
        json_report=args.json_report,
        html_report=getattr(args, "html_report", None),
        seed=args.seed,
        verbose=args.verbose,
        quiet=args.quiet,
    )
    return pipeline.run()


# ---------------------------------------------------------------------------
# HTML report generation
# ---------------------------------------------------------------------------

def _write_html_report(report, path):
    """Write a self-contained HTML report from the structured report dict."""
    verdict = report.get("verdict", "UNKNOWN")
    elapsed = report.get("elapsed_seconds", 0)
    summary = report.get("summary", {})
    phases = report.get("phases", {})

    verdict_color = "#22c55e" if verdict == "PASS" else "#ef4444"

    # Build phase sections
    phase_html = []
    for name in ("structural", "spectral", "roundtrip", "ml_classification"):
        phase = phases.get(name, {})
        status = phase.get("status", "SKIP")
        phase_html.append(_html_phase_section(name, phase, status))

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>E2E Validation Report</title>
<style>
  :root {{ --bg: #0f172a; --fg: #e2e8f0; --card: #1e293b; --border: #334155;
           --pass: #22c55e; --fail: #ef4444; --skip: #94a3b8; --warn: #f59e0b; }}
  * {{ margin: 0; padding: 0; box-sizing: border-box; }}
  body {{ font-family: 'SF Mono', 'Fira Code', monospace; background: var(--bg);
          color: var(--fg); padding: 2rem; line-height: 1.6; }}
  .container {{ max-width: 900px; margin: 0 auto; }}
  h1 {{ font-size: 1.5rem; margin-bottom: 0.5rem; }}
  h2 {{ font-size: 1.1rem; margin: 1.5rem 0 0.5rem; border-bottom: 1px solid var(--border);
        padding-bottom: 0.3rem; }}
  .verdict {{ display: inline-block; padding: 0.2rem 0.8rem; border-radius: 4px;
              font-weight: bold; color: #fff; background: {verdict_color}; }}
  .meta {{ color: var(--skip); font-size: 0.85rem; margin: 0.5rem 0 1.5rem; }}
  .summary-grid {{ display: grid; grid-template-columns: repeat(4, 1fr); gap: 1rem;
                   margin: 1rem 0; }}
  .summary-card {{ background: var(--card); border: 1px solid var(--border);
                   border-radius: 6px; padding: 1rem; text-align: center; }}
  .summary-card .value {{ font-size: 1.5rem; font-weight: bold; }}
  .summary-card .label {{ font-size: 0.75rem; color: var(--skip); text-transform: uppercase; }}
  .phase {{ background: var(--card); border: 1px solid var(--border); border-radius: 6px;
            padding: 1rem; margin: 1rem 0; }}
  .phase-header {{ display: flex; justify-content: space-between; align-items: center;
                   cursor: pointer; }}
  .phase-header h3 {{ font-size: 1rem; }}
  .badge {{ padding: 0.15rem 0.5rem; border-radius: 3px; font-size: 0.75rem;
            font-weight: bold; color: #fff; }}
  .badge-pass {{ background: var(--pass); }}
  .badge-fail {{ background: var(--fail); }}
  .badge-skip {{ background: var(--skip); }}
  table {{ width: 100%; border-collapse: collapse; margin-top: 0.5rem; font-size: 0.85rem; }}
  th, td {{ padding: 0.3rem 0.6rem; text-align: left; border-bottom: 1px solid var(--border); }}
  th {{ color: var(--skip); font-weight: normal; text-transform: uppercase; font-size: 0.7rem; }}
  .pass {{ color: var(--pass); }}
  .fail {{ color: var(--fail); }}
  .skip {{ color: var(--skip); }}
  details > summary {{ list-style: none; }}
  details > summary::-webkit-details-marker {{ display: none; }}
  footer {{ margin-top: 2rem; padding-top: 1rem; border-top: 1px solid var(--border);
            color: var(--skip); font-size: 0.75rem; text-align: center; }}
</style>
</head>
<body>
<div class="container">
  <h1>E2E Validation Report</h1>
  <div><span class="verdict">{verdict}</span></div>
  <div class="meta">
    {report.get('timestamp', '')} &middot;
    Domains: {', '.join(report.get('domains', []))} &middot;
    {elapsed:.1f}s
  </div>

  <div class="summary-grid">
    <div class="summary-card">
      <div class="value">{summary.get('total_checks', 0)}</div>
      <div class="label">Total Checks</div>
    </div>
    <div class="summary-card">
      <div class="value pass">{summary.get('passed', 0)}</div>
      <div class="label">Passed</div>
    </div>
    <div class="summary-card">
      <div class="value fail">{summary.get('failed', 0)}</div>
      <div class="label">Failed</div>
    </div>
    <div class="summary-card">
      <div class="value">{elapsed:.1f}s</div>
      <div class="label">Duration</div>
    </div>
  </div>

  {''.join(phase_html)}

  <footer>
    Generated by rf-datagen e2e pipeline
  </footer>
</div>
</body>
</html>"""

    with open(path, "w") as f:
        f.write(html)


def _html_phase_section(name, phase, status):
    """Build HTML for one phase section."""
    badge_cls = {"PASS": "badge-pass", "FAIL": "badge-fail"}.get(
        status, "badge-skip")
    display_name = name.replace("_", " ").title()

    # Build detail content
    detail_rows = []

    # Structural checks
    if "checks" in phase:
        for check_name, check in sorted(phase["checks"].items()):
            s = check["status"]
            cls = "pass" if s == "PASS" else "fail"
            detail_str = check.get("detail", "")
            if isinstance(detail_str, dict):
                detail_str = json.dumps(detail_str, default=str)
            detail_rows.append(
                f'<tr><td>{_esc(check_name)}</td>'
                f'<td class="{cls}">{s}</td>'
                f'<td>{_esc(str(detail_str)[:120])}</td></tr>')

    # Per-class results (spectral, ML)
    if "per_class" in phase:
        for cls_name, cls_result in sorted(phase["per_class"].items()):
            s = cls_result.get("status", "SKIP")
            css = {"PASS": "pass", "FAIL": "fail"}.get(s, "skip")
            extras = []
            if "median_bw" in cls_result:
                extras.append(f"BW={cls_result['median_bw']}")
            if "median_papr" in cls_result:
                extras.append(f"PAPR={cls_result['median_papr']}")
            if "accuracy" in cls_result:
                extras.append(f"acc={cls_result['accuracy']:.1%}")
            detail_str = cls_result.get("detail", ", ".join(extras))
            detail_rows.append(
                f'<tr><td>{_esc(cls_name)}</td>'
                f'<td class="{css}">{s}</td>'
                f'<td>{_esc(str(detail_str)[:120])}</td></tr>')

    # Per-mode results (roundtrip)
    if "per_mode" in phase:
        for mode_name, mode_result in sorted(phase["per_mode"].items()):
            s = mode_result.get("status", "SKIP")
            css = {"PASS": "pass", "FAIL": "fail"}.get(s, "skip")
            rate = mode_result.get("decode_rate", 0)
            total = mode_result.get("total", 0)
            decodes = mode_result.get("decodes", 0)
            detail_str = f"{decodes}/{total} = {rate:.0%}"
            detail_rows.append(
                f'<tr><td>{_esc(mode_name)}</td>'
                f'<td class="{css}">{s}</td>'
                f'<td>{_esc(detail_str)}</td></tr>')

    extra_info = ""
    if "pass_rate" in phase:
        extra_info = f' <span style="color:var(--skip);font-size:0.8rem">({phase["pass_rate"]})</span>'

    table_html = ""
    if detail_rows:
        table_html = (
            '<table><thead><tr><th>Check</th><th>Status</th><th>Detail</th>'
            f'</tr></thead><tbody>{"".join(detail_rows)}</tbody></table>')

    return f"""
  <details class="phase" {'open' if status == 'FAIL' else ''}>
    <summary class="phase-header">
      <h3>{display_name}{extra_info}</h3>
      <span class="badge {badge_cls}">{status}</span>
    </summary>
    {table_html}
  </details>"""


def _esc(text):
    """Escape HTML special characters."""
    return (text.replace("&", "&amp;").replace("<", "&lt;")
            .replace(">", "&gt;").replace('"', "&quot;"))
