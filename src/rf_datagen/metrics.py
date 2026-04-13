"""Lightweight in-process metrics for generation runs.

Collects per-class timing, status, and throughput data.  Thread-safe.
Writes a ``generation_metrics.json`` summary at the end of a run.
"""

import json
import os
import threading
import time

from .logging_config import get_logger

log = get_logger("metrics")


class RunMetrics:
    """Accumulate generation metrics for a single run."""

    def __init__(self):
        self._lock = threading.Lock()
        self._start = time.monotonic()
        self._classes = {}  # class_name -> dict
        self._exceptions = []

    def class_started(self, generator, class_name, domain=None):
        with self._lock:
            self._classes[class_name] = {
                "generator": generator,
                "domain": domain,
                "status": "in_progress",
                "started_at": time.monotonic(),
            }

    def class_completed(self, generator, class_name, samples, duration_s,
                        status="ok"):
        with self._lock:
            self._classes[class_name] = {
                "generator": generator,
                "status": status,
                "samples": samples,
                "duration_s": round(duration_s, 2),
            }

    def class_failed(self, generator, class_name, reason):
        with self._lock:
            self._classes[class_name] = {
                "generator": generator,
                "status": "failed",
                "reason": reason,
            }

    def record_exception(self, generator, class_name, exc):
        with self._lock:
            self._exceptions.append({
                "generator": generator,
                "class": class_name,
                "type": type(exc).__name__,
                "message": str(exc),
            })

    def update_from_results(self, generator_name, results):
        """Bulk-update from a generator's result dict."""
        with self._lock:
            for class_name, info in results.items():
                entry = {
                    "generator": generator_name,
                    "status": info.get("status", "unknown"),
                }
                if "samples" in info:
                    entry["samples"] = info["samples"]
                if "time_s" in info:
                    entry["duration_s"] = info["time_s"]
                if "reason" in info:
                    entry["reason"] = info["reason"]
                self._classes[class_name] = entry

    def summary(self):
        """Return a summary dict of the run."""
        with self._lock:
            elapsed = time.monotonic() - self._start
            ok = sum(1 for c in self._classes.values()
                     if c.get("status") == "ok")
            cached = sum(1 for c in self._classes.values()
                         if c.get("status") == "cached")
            failed = sum(1 for c in self._classes.values()
                         if c.get("status") == "failed")
            total_samples = sum(c.get("samples", 0)
                                for c in self._classes.values())
            failures = {name: info.get("reason", "unknown")
                        for name, info in self._classes.items()
                        if info.get("status") == "failed"}
            return {
                "duration_s": round(elapsed, 1),
                "classes_ok": ok,
                "classes_cached": cached,
                "classes_failed": failed,
                "classes_total": len(self._classes),
                "total_samples": total_samples,
                "samples_per_second": (round(total_samples / elapsed)
                                       if elapsed > 0 else 0),
                "failures": failures,
                "exceptions": list(self._exceptions),
            }

    def save(self, output_dir):
        """Write metrics to ``generation_metrics.json``."""
        path = os.path.join(output_dir, "generation_metrics.json")
        try:
            with open(path, "w") as f:
                json.dump(self.summary(), f, indent=2)
        except OSError as e:
            log.warning("Failed to write metrics: %s", e)
