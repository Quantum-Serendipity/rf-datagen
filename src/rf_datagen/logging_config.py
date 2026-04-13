"""Structured logging setup for rf-datagen.

Uses structlog with contextvars for automatic context propagation.
Console output is human-readable (colored when TTY); file output is
JSON-lines for machine parsing.  Multiprocess workers use a
QueueHandler so all log records serialize through a single listener.
"""

import logging
import logging.handlers
import multiprocessing
import os
import sys

import structlog

_configured = False
_log_queue = None
_queue_listener = None


# ---------------------------------------------------------------------------
# structlog processors
# ---------------------------------------------------------------------------

def _add_process_info(logger, method_name, event_dict):
    """Add pid to every log record."""
    event_dict["pid"] = os.getpid()
    return event_dict


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def setup_logging(output_dir=None, verbose=False, quiet=False):
    """Configure structured logging for a generation run.

    - Console: human-readable via structlog ConsoleRenderer (stderr).
    - File (if output_dir): JSON-lines to ``generation.jsonl`` at DEBUG.
    - Legacy file: plain text to ``generation.log`` at DEBUG.

    Safe to call multiple times; only the first call takes effect.
    """
    global _configured, _log_queue, _queue_listener
    if _configured:
        return
    _configured = True

    # --- stdlib root logger (structlog wraps this) -------------------------
    root = logging.getLogger("rf_datagen")
    root.setLevel(logging.DEBUG)
    # Remove any handlers inherited from prior setup
    root.handlers.clear()

    # --- Console handler ---------------------------------------------------
    if quiet:
        console_level = logging.WARNING
    elif verbose:
        console_level = logging.DEBUG
    else:
        console_level = logging.INFO

    console = logging.StreamHandler(sys.stderr)
    console.setLevel(console_level)
    root.addHandler(console)

    # --- File handlers (if output_dir) -------------------------------------
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

        # Plain-text log (human-readable, backward-compat)
        log_path = os.path.join(output_dir, "generation.log")
        fh = logging.FileHandler(log_path, mode="a")
        fh.setLevel(logging.DEBUG)
        root.addHandler(fh)

        # JSON-lines log (machine-parseable structured output)
        jsonl_path = os.path.join(output_dir, "generation.jsonl")
        jh = logging.FileHandler(jsonl_path, mode="a")
        jh.setLevel(logging.DEBUG)

    # --- structlog configuration -------------------------------------------
    # Use render_to_log_kwargs so structlog renders the event string and
    # passes it to stdlib's %-formatting.  This works with plain
    # logging.Formatter on all handlers (console, file, QueueHandler).
    shared_processors = [
        structlog.contextvars.merge_contextvars,
        structlog.stdlib.add_log_level,
        structlog.stdlib.add_logger_name,
        _add_process_info,
        structlog.processors.StackInfoRenderer(),
        structlog.processors.format_exc_info,
        structlog.stdlib.ProcessorFormatter.wrap_for_formatter,
    ]

    # ProcessorFormatter renders the final output for each handler
    console_formatter = structlog.stdlib.ProcessorFormatter(
        processors=[
            structlog.stdlib.ProcessorFormatter.remove_processors_meta,
            structlog.dev.ConsoleRenderer(colors=sys.stderr.isatty()),
        ],
        foreign_pre_chain=shared_processors[:-1],
    )
    console.setFormatter(console_formatter)

    if output_dir:
        plain_formatter = structlog.stdlib.ProcessorFormatter(
            processors=[
                structlog.stdlib.ProcessorFormatter.remove_processors_meta,
                structlog.dev.ConsoleRenderer(colors=False),
            ],
            foreign_pre_chain=shared_processors[:-1],
        )
        fh.setFormatter(plain_formatter)

        json_formatter = structlog.stdlib.ProcessorFormatter(
            processors=[
                structlog.stdlib.ProcessorFormatter.remove_processors_meta,
                structlog.processors.JSONRenderer(),
            ],
            foreign_pre_chain=shared_processors[:-1],
        )
        jh.setFormatter(json_formatter)
        root.addHandler(jh)

    structlog.configure(
        processors=shared_processors,
        logger_factory=structlog.stdlib.LoggerFactory(),
        wrapper_class=structlog.stdlib.BoundLogger,
        cache_logger_on_first_use=True,
    )

    # --- Multiprocess log queue --------------------------------------------
    _log_queue = multiprocessing.Queue(-1)
    # The queue listener drains records from forked workers into the
    # parent's handlers.  Workers call setup_worker_logging() to
    # install a QueueHandler that feeds this queue.
    _queue_listener = logging.handlers.QueueListener(
        _log_queue, *root.handlers, respect_handler_level=True)
    _queue_listener.start()


def shutdown_logging():
    """Stop the queue listener.  Call at process exit."""
    global _queue_listener
    if _queue_listener is not None:
        _queue_listener.stop()
        _queue_listener = None


def setup_worker_logging(log_queue):
    """Configure logging in a forked worker process.

    Replaces all handlers on the rf_datagen root logger with a single
    QueueHandler that sends records to the parent's listener.

    Call this at the top of any function executed via
    ProcessPoolExecutor with fork context.
    """
    root = logging.getLogger("rf_datagen")
    root.handlers.clear()
    root.addHandler(logging.handlers.QueueHandler(log_queue))
    root.setLevel(logging.DEBUG)


def get_log_queue():
    """Return the multiprocessing Queue for passing to workers."""
    return _log_queue


def get_logger(name):
    """Get a structured logger under the rf_datagen namespace.

    Returns a structlog BoundLogger that wraps
    ``logging.getLogger("rf_datagen.<name>")``.

    Backward-compatible: ``log.info("format %s", arg)`` works because
    structlog's PositionalArgumentsFormatter handles it, and we fall
    through to stdlib formatting.
    """
    return structlog.get_logger(f"rf_datagen.{name}")


# ---------------------------------------------------------------------------
# Context helpers — thin wrappers so callers don't import structlog
# ---------------------------------------------------------------------------

def bind_context(**kw):
    """Bind key-value pairs to the current structlog context.

    Call at worker entry points to tag all subsequent log lines with
    generator name, class name, domain, worker_id, etc.
    """
    structlog.contextvars.bind_contextvars(**kw)


def clear_context():
    """Clear all bound context variables."""
    structlog.contextvars.clear_contextvars()
