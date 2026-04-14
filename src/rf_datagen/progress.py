"""Progress tracking for generation runs.

RichProgressTracker for interactive TTY sessions (colored bars + ETA).
LogProgressTracker for non-TTY / CI (periodic log lines).
"""

import sys
import threading
import time

from .logging_config import get_logger

log = get_logger("progress")


class ProgressTracker:
    """Base progress tracker interface."""

    def __init__(self, total_classes):
        self._total = total_classes
        self._completed = 0
        self._lock = threading.Lock()

    def update(self, class_name=None, status="ok"):
        """Mark one class as completed."""
        with self._lock:
            self._completed += 1

    def finish(self):
        """Clean up progress display."""

    @property
    def completed(self):
        with self._lock:
            return self._completed


class LogProgressTracker(ProgressTracker):
    """Emits periodic log lines for non-TTY environments."""

    def __init__(self, total_classes, interval_secs=30, interval_classes=5):
        super().__init__(total_classes)
        self._interval_secs = interval_secs
        self._interval_classes = interval_classes
        self._last_log_time = time.monotonic()
        self._last_log_count = 0
        self._start = time.monotonic()

    def update(self, class_name=None, status="ok"):
        with self._lock:
            self._completed += 1
            now = time.monotonic()
            elapsed_since_log = now - self._last_log_time
            classes_since_log = self._completed - self._last_log_count

            if (elapsed_since_log >= self._interval_secs or
                    classes_since_log >= self._interval_classes):
                pct = 100 * self._completed / max(self._total, 1)
                elapsed = now - self._start
                if self._completed > 0:
                    eta = elapsed * (self._total - self._completed) / self._completed
                    eta_str = f"ETA ~{eta / 60:.0f}m" if eta > 60 else f"ETA ~{eta:.0f}s"
                else:
                    eta_str = "ETA unknown"
                log.info("Progress: %d/%d classes (%.0f%%), %s",
                         self._completed, self._total, pct, eta_str)
                self._last_log_time = now
                self._last_log_count = self._completed

    def finish(self):
        elapsed = time.monotonic() - self._start
        log.info("Generation complete: %d/%d classes in %.0fs",
                 self._completed, self._total, elapsed)


class RichProgressTracker(ProgressTracker):
    """Interactive multi-bar progress display using rich."""

    def __init__(self, total_classes):
        super().__init__(total_classes)
        from rich.progress import (Progress, BarColumn, TextColumn,
                                   TimeElapsedColumn, TimeRemainingColumn,
                                   MofNCompleteColumn)
        self._progress = Progress(
            TextColumn("[bold blue]{task.description}"),
            BarColumn(bar_width=30),
            MofNCompleteColumn(),
            TimeElapsedColumn(),
            TimeRemainingColumn(),
        )
        self._task_id = self._progress.add_task(
            "Generating", total=total_classes)
        self._progress.start()

    def update(self, class_name=None, status="ok"):
        with self._lock:
            self._completed += 1
            desc = f"Generating ({class_name})" if class_name else "Generating"
            self._progress.update(self._task_id, advance=1, description=desc)

    def finish(self):
        self._progress.update(self._task_id, description="Complete")
        self._progress.stop()


def create_progress_tracker(total_classes):
    """Create the appropriate progress tracker for the current environment."""
    if total_classes <= 0:
        return ProgressTracker(0)
    try:
        if sys.stderr.isatty():
            return RichProgressTracker(total_classes)
    except Exception as e:
        log.warning("Rich progress tracker unavailable: %s", e)
    return LogProgressTracker(total_classes)
