"""Structural and statistical validation of generated datasets.

Loads assembled .npy + tags CSV from disk and runs completeness, integrity,
and distribution checks.  Returns a structured results dict consumed by
the e2e pipeline.
"""

import csv
import os

import numpy as np

from .domains import DOMAINS, labels_for_domain, SIGNAL_DOMAIN_MAP
from .logging_config import get_logger

log = get_logger("dataset_checks")


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def run_structural_checks(output_dir, domains, config=None):
    """Run all structural checks across requested domains.

    Parameters
    ----------
    output_dir : str
        Root output directory (may contain per-domain subdirs).
    domains : list[str]
        Domain names to check (e.g. ["narrowband", "moderate"]).
    config : Config or None
        If provided, used for expected sample counts and scenario weights.

    Returns
    -------
    dict : {status, checks: {name: {status, detail}}}
    """
    checks = {}
    multi_domain = len(domains) > 1

    all_expected_labels = []
    for d in domains:
        all_expected_labels.extend(labels_for_domain(d))

    for domain_name in domains:
        domain = DOMAINS[domain_name]
        expected_labels = labels_for_domain(domain_name)

        if multi_domain:
            domain_dir = os.path.join(output_dir, domain_name)
            prefix = f"rf_datagen_{domain_name}"
        else:
            domain_dir = output_dir
            prefix = "rf_datagen"

        iq_path = os.path.join(domain_dir, f"{prefix}_iq.npy")
        csv_path = os.path.join(domain_dir, f"{prefix}_tags.csv")

        # --- File existence ---
        if not os.path.exists(iq_path):
            checks[f"{domain_name}_file_exists"] = {
                "status": "FAIL",
                "detail": f"Missing {iq_path}",
            }
            continue
        if not os.path.exists(csv_path):
            checks[f"{domain_name}_file_exists"] = {
                "status": "FAIL",
                "detail": f"Missing {csv_path}",
            }
            continue

        checks[f"{domain_name}_file_exists"] = {
            "status": "PASS", "detail": "IQ and CSV present",
        }

        # Load data
        iq = np.load(iq_path, mmap_mode="r")
        tags_rows, tags_header = _load_tags_csv(csv_path)

        domain_checks = _check_domain(
            iq, tags_rows, tags_header,
            domain_name, domain, expected_labels, config,
        )
        checks.update(domain_checks)

    overall = "PASS" if all(
        c["status"] == "PASS" for c in checks.values()
    ) else "FAIL"

    return {"status": overall, "checks": checks}


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _load_tags_csv(csv_path):
    """Load tags CSV, return (rows, header)."""
    with open(csv_path) as f:
        reader = csv.reader(f)
        header = next(reader)
        rows = list(reader)
    return rows, header


def _check_domain(iq, tags_rows, tags_header, domain_name, domain,
                  expected_labels, config):
    """Run all checks for a single domain. Returns dict of check results."""
    pfx = f"{domain_name}_"
    checks = {}

    # Column indices
    mode_col = tags_header.index("mode") if "mode" in tags_header else 1
    scenario_col = (tags_header.index("scenario")
                    if "scenario" in tags_header else None)
    labels_in_csv = [row[mode_col] for row in tags_rows]

    # --- Shape consistency ---
    if iq.ndim != 2:
        checks[pfx + "shape"] = {
            "status": "FAIL",
            "detail": f"Expected 2D, got {iq.ndim}D",
        }
        return checks  # Can't proceed

    if iq.shape[1] != domain.window_length:
        checks[pfx + "shape"] = {
            "status": "FAIL",
            "detail": (f"Window length {iq.shape[1]} != "
                       f"expected {domain.window_length}"),
        }
    else:
        checks[pfx + "shape"] = {"status": "PASS", "detail": str(iq.shape)}

    # --- Dtype ---
    if not np.iscomplexobj(iq):
        checks[pfx + "dtype"] = {
            "status": "FAIL",
            "detail": f"Expected complex, got {iq.dtype}",
        }
    else:
        checks[pfx + "dtype"] = {"status": "PASS", "detail": str(iq.dtype)}

    # --- Metadata alignment ---
    if len(tags_rows) != iq.shape[0]:
        checks[pfx + "metadata_alignment"] = {
            "status": "FAIL",
            "detail": f"CSV rows={len(tags_rows)}, IQ rows={iq.shape[0]}",
        }
    else:
        checks[pfx + "metadata_alignment"] = {
            "status": "PASS",
            "detail": f"{len(tags_rows)} rows",
        }

    # --- Class completeness ---
    present = set(labels_in_csv)
    missing = [l for l in expected_labels if l not in present]
    if missing:
        checks[pfx + "class_completeness"] = {
            "status": "FAIL",
            "detail": f"Missing {len(missing)}/{len(expected_labels)}: "
                      f"{', '.join(missing[:10])}"
                      f"{'...' if len(missing) > 10 else ''}",
        }
    else:
        checks[pfx + "class_completeness"] = {
            "status": "PASS",
            "detail": f"{len(present)}/{len(expected_labels)} classes",
        }

    # --- Sample counts ---
    from collections import Counter
    counts = Counter(labels_in_csv)

    if config is not None:
        # Derive target from max generator samples_per_class in config
        # (different generators may contribute different counts)
        gen_targets = [g.samples_per_class for g in config.generators.values()
                       if g.enabled]
        target = max(gen_targets) if gen_targets else domain.default_samples_per_class
        tolerance = 0.10  # ±10%
        bad_counts = {}
        for label in expected_labels:
            c = counts.get(label, 0)
            if c == 0:
                continue
            if abs(c - target) / target > tolerance:
                bad_counts[label] = c

        if bad_counts:
            checks[pfx + "sample_counts"] = {
                "status": "FAIL",
                "detail": {
                    "target": target,
                    "tolerance": f"±{int(tolerance*100)}%",
                    "out_of_range": bad_counts,
                },
            }
        else:
            checks[pfx + "sample_counts"] = {
                "status": "PASS",
                "detail": {
                    "target": target,
                    "min": min(counts.values()) if counts else 0,
                    "max": max(counts.values()) if counts else 0,
                },
            }
    else:
        # Without config, just report counts
        checks[pfx + "sample_counts"] = {
            "status": "PASS",
            "detail": {
                "total": len(labels_in_csv),
                "classes": len(counts),
                "min": min(counts.values()) if counts else 0,
                "max": max(counts.values()) if counts else 0,
            },
        }

    # --- NaN/Inf check (sample up to 500 rows for speed on large datasets) ---
    n_check = min(500, iq.shape[0])
    check_indices = np.linspace(0, iq.shape[0] - 1, n_check, dtype=int)
    sample = np.array(iq[check_indices])
    nan_count = int(np.sum(np.isnan(sample)))
    inf_count = int(np.sum(np.isinf(sample)))

    if nan_count > 0 or inf_count > 0:
        checks[pfx + "no_nan_inf"] = {
            "status": "FAIL",
            "detail": f"NaN={nan_count}, Inf={inf_count} in {n_check} sampled rows",
        }
    else:
        checks[pfx + "no_nan_inf"] = {
            "status": "PASS",
            "detail": f"Clean ({n_check} rows sampled)",
        }

    # --- Zero-power check ---
    powers = np.mean(np.abs(sample) ** 2, axis=1)
    zero_count = int(np.sum(powers < 1e-30))
    if zero_count > 0:
        checks[pfx + "no_zero_power"] = {
            "status": "FAIL",
            "detail": f"{zero_count}/{n_check} sampled rows have ~zero power",
        }
    else:
        checks[pfx + "no_zero_power"] = {
            "status": "PASS",
            "detail": f"All {n_check} sampled rows have nonzero power",
        }

    # --- Scenario distribution ---
    if scenario_col is not None and config is not None:
        scenarios = [row[scenario_col] for row in tags_rows if row[scenario_col]]
        if scenarios:
            checks[pfx + "scenario_distribution"] = _check_scenario_distribution(
                scenarios, config.impairments.scenario_weights)
        else:
            checks[pfx + "scenario_distribution"] = {
                "status": "PASS",
                "detail": "No scenario data (pre-impairment dataset?)",
            }

    # --- SNR distribution ---
    snr_col = None
    if "snr_db" in tags_header:
        snr_col = tags_header.index("snr_db")
    # SNR info may be embedded in scenario metadata; skip if not in CSV
    if snr_col is not None:
        snr_values = []
        for row in tags_rows:
            try:
                snr_values.append(float(row[snr_col]))
            except (ValueError, IndexError):
                pass
        if snr_values:
            checks[pfx + "snr_distribution"] = _check_snr_distribution(
                snr_values, config)

    return checks


def _check_scenario_distribution(scenarios, expected_weights):
    """Chi-squared-like check on scenario distribution vs config weights.

    Uses a relaxed ±50% tolerance per scenario rather than a formal
    chi-squared test, since sample sizes vary widely.
    """
    from collections import Counter
    observed = Counter(scenarios)
    total = sum(observed.values())

    if total < 100:
        return {
            "status": "PASS",
            "detail": f"Too few samples ({total}) for distribution check",
        }

    bad = {}
    for scenario, weight in expected_weights.items():
        expected_count = weight * total
        actual_count = observed.get(scenario, 0)
        if expected_count < 5:
            continue  # Too few expected to judge
        ratio = actual_count / expected_count if expected_count > 0 else 0
        if ratio < 0.5 or ratio > 2.0:
            bad[scenario] = {
                "expected": round(expected_count, 1),
                "actual": actual_count,
                "ratio": round(ratio, 2),
            }

    if bad:
        return {
            "status": "FAIL",
            "detail": {
                "total_samples": total,
                "out_of_range": bad,
            },
        }
    return {
        "status": "PASS",
        "detail": f"All scenarios within ±50% of expected ({total} samples)",
    }


def _check_snr_distribution(snr_values, config):
    """Check that SNR levels are roughly evenly distributed."""
    from collections import Counter
    snr_counts = Counter(round(s) for s in snr_values)
    total = len(snr_values)

    if total < 100:
        return {
            "status": "PASS",
            "detail": f"Too few samples ({total}) for SNR distribution check",
        }

    expected_per_level = total / len(snr_counts) if snr_counts else 0
    if expected_per_level < 5:
        return {
            "status": "PASS",
            "detail": "Too few per-level samples for distribution check",
        }

    bad = {}
    for level, count in sorted(snr_counts.items()):
        ratio = count / expected_per_level
        if ratio < 0.5 or ratio > 2.0:
            bad[level] = {"count": count, "expected": round(expected_per_level, 1)}

    if bad:
        return {
            "status": "FAIL",
            "detail": {"out_of_range": bad},
        }
    return {
        "status": "PASS",
        "detail": f"{len(snr_counts)} SNR levels, ~{int(expected_per_level)} each",
    }
