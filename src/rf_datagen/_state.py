"""Shared runtime state — no rf_datagen imports allowed here."""


_shutdown_requested = False


def shutdown_requested():
    """Return True if a graceful shutdown has been requested."""
    return _shutdown_requested


def request_shutdown():
    """Set the shutdown flag (called from signal handlers)."""
    global _shutdown_requested
    _shutdown_requested = True


def reset_shutdown():
    """Clear the shutdown flag (called at generation start)."""
    global _shutdown_requested
    _shutdown_requested = False
