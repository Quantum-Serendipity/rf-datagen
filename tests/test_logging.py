"""Tests for structured logging setup."""

import json
import logging
import os

import numpy as np
import pytest


def test_get_logger_returns_usable_logger():
    """get_logger returns a logger that supports positional args."""
    from rf_datagen.logging_config import get_logger
    log = get_logger("test_module")
    # Should not raise — positional formatting must work
    log.info("count: %d, name: %s", 42, "test")


def test_setup_logging_creates_log_file(tmp_path):
    """setup_logging with output_dir creates generation.log."""
    # Use a fresh logger namespace to avoid state leaks
    import rf_datagen.logging_config as lc
    # Reset state for this test
    old_configured = lc._configured
    lc._configured = False
    try:
        lc.setup_logging(output_dir=str(tmp_path), verbose=False, quiet=False)
        log = lc.get_logger("test_file")
        log.info("test message for file")

        log_path = tmp_path / "generation.log"
        assert log_path.exists(), "generation.log should be created"
        content = log_path.read_text()
        assert "test message for file" in content
    finally:
        lc.shutdown_logging()
        lc._configured = old_configured


def test_bind_and_clear_context():
    """Context binding and clearing work without errors."""
    from rf_datagen.logging_config import bind_context, clear_context
    bind_context(generator="synthetic", class_name="CW")
    clear_context()


def test_metrics_class_tracking():
    """RunMetrics tracks class completions correctly."""
    from rf_datagen.metrics import RunMetrics
    m = RunMetrics()
    m.class_completed("synthetic", "CW", samples=100, duration_s=1.5)
    m.class_completed("synthetic", "FM", samples=200, duration_s=2.0)
    m.class_failed("fldigi", "PSK31", reason="signal too short")

    s = m.summary()
    assert s["classes_ok"] == 2
    assert s["classes_failed"] == 1
    assert s["total_samples"] == 300
    assert "PSK31" in s["failures"]


def test_metrics_save(tmp_path):
    """RunMetrics.save writes valid JSON."""
    from rf_datagen.metrics import RunMetrics
    m = RunMetrics()
    m.class_completed("synthetic", "CW", samples=50, duration_s=0.5)
    m.save(str(tmp_path))

    path = tmp_path / "generation_metrics.json"
    assert path.exists()
    data = json.loads(path.read_text())
    assert data["classes_ok"] == 1
    assert data["total_samples"] == 50


def test_metrics_update_from_results():
    """RunMetrics.update_from_results bulk-updates correctly."""
    from rf_datagen.metrics import RunMetrics
    m = RunMetrics()
    results = {
        "CW": {"status": "ok", "samples": 100, "time_s": 1.0},
        "FM": {"status": "cached", "samples": 200},
        "PSK31": {"status": "failed", "reason": "no tools"},
    }
    m.update_from_results("synthetic", results)
    s = m.summary()
    assert s["classes_ok"] == 1
    assert s["classes_cached"] == 1
    assert s["classes_failed"] == 1


def test_progress_tracker_log_mode():
    """LogProgressTracker emits progress without errors."""
    from rf_datagen.progress import LogProgressTracker
    p = LogProgressTracker(10, interval_secs=0, interval_classes=1)
    for i in range(10):
        p.update(f"CLASS_{i}")
    p.finish()
    assert p.completed == 10


def test_progress_tracker_factory():
    """create_progress_tracker returns a tracker without crashing."""
    from rf_datagen.progress import create_progress_tracker, ProgressTracker
    p = create_progress_tracker(5)
    assert isinstance(p, ProgressTracker)
    p.update("test")
    p.finish()
