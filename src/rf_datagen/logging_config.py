"""Structured logging setup for rf-datagen."""

import logging
import os
import sys


LOG_FORMAT = "%(asctime)s %(levelname)-5s [%(name)s] %(message)s"
LOG_DATE_FORMAT = "%H:%M:%S"

_configured = False


def setup_logging(output_dir=None, verbose=False, quiet=False):
    """Configure logging for a generation run.

    - Console handler: human-readable, INFO by default (DEBUG with
      --verbose, WARNING with --quiet).
    - File handler (if output_dir set): writes to generation.log with
      DEBUG level for full audit trail.
    """
    global _configured
    if _configured:
        return
    _configured = True

    root = logging.getLogger("rf_datagen")
    root.setLevel(logging.DEBUG)

    # Console handler
    console = logging.StreamHandler(sys.stderr)
    if quiet:
        console.setLevel(logging.WARNING)
    elif verbose:
        console.setLevel(logging.DEBUG)
    else:
        console.setLevel(logging.INFO)
    console.setFormatter(logging.Formatter(LOG_FORMAT, datefmt=LOG_DATE_FORMAT))
    root.addHandler(console)

    # File handler
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        log_path = os.path.join(output_dir, "generation.log")
        fh = logging.FileHandler(log_path, mode="a")
        fh.setLevel(logging.DEBUG)
        fh.setFormatter(logging.Formatter(
            "%(asctime)s %(levelname)-5s [%(name)s] %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S"))
        root.addHandler(fh)


def get_logger(name):
    """Get a logger under the rf_datagen namespace."""
    return logging.getLogger(f"rf_datagen.{name}")
