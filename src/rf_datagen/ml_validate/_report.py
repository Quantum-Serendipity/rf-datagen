"""Report generation — CSV, JSON, and optional confusion matrix."""

import csv
import json
import os


def save_results(results, output_dir):
    """Save validation results to CSV and JSON."""
    os.makedirs(output_dir, exist_ok=True)

    # JSON
    json_path = os.path.join(output_dir, "results.json")
    with open(json_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"JSON: {json_path}")

    # CSV
    csv_path = os.path.join(output_dir, "results.csv")
    rows = results.get("results", [])
    if rows:
        fieldnames = list(rows[0].keys())
        with open(csv_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)
        print(f"CSV: {csv_path}")

    return json_path, csv_path


def save_confusion_matrix(results, output_dir):
    """Save confusion matrix as PNG (requires matplotlib)."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return None

    rows = results.get("results", [])
    if not rows:
        return None

    # Build confusion data
    signals = sorted(set(r["signal"] for r in rows))
    expected_classes = sorted(set(r["expected"] for r in rows if r["expected"]))

    if not expected_classes:
        return None

    # For each signal, accumulate predictions
    # This is a simplified version — full confusion matrix needs per-sample data
    fig, ax = plt.subplots(figsize=(max(8, len(signals) * 0.5),
                                    max(6, len(signals) * 0.4)))
    accuracies = []
    labels = []
    for s in signals:
        s_rows = [r for r in rows if r["signal"] == s and r["snr_db"] == "clean"]
        if s_rows:
            acc = s_rows[0]["accuracy"]
            accuracies.append(acc)
            labels.append(s)

    if accuracies:
        colors = ["green" if a >= 0.5 else "red" for a in accuracies]
        ax.barh(range(len(labels)), accuracies, color=colors, alpha=0.7)
        ax.set_yticks(range(len(labels)))
        ax.set_yticklabels(labels, fontsize=8)
        ax.set_xlabel("Accuracy")
        ax.set_title("ML Classification Accuracy by Signal")
        ax.set_xlim(0, 1)
        ax.axvline(0.5, color="gray", linestyle="--", alpha=0.5)

    plt.tight_layout()
    png_path = os.path.join(output_dir, "accuracy_chart.png")
    plt.savefig(png_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Chart: {png_path}")
    return png_path
