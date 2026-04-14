"""PID registry for tracking child processes and cleaning up orphans."""

import atexit
import fcntl
import json
import os
import signal
import threading
import time

from .logging_config import get_logger

log = get_logger("pid_registry")

_REGISTRY_DIR = os.path.join(os.path.expanduser("~"), ".cache", "rf-datagen")
_REGISTRY_PATH = os.path.join(_REGISTRY_DIR, "pids.json")
_lock = threading.Lock()


def _read_registry():
    """Read registry file, returning None if missing or corrupt."""
    try:
        with open(_REGISTRY_PATH) as f:
            return json.load(f)
    except (OSError, json.JSONDecodeError):
        return None


def _write_registry(data):
    """Atomically write registry file with file-level locking."""
    os.makedirs(_REGISTRY_DIR, exist_ok=True)
    tmp_path = _REGISTRY_PATH + ".tmp"
    with open(tmp_path, "w") as f:
        fcntl.flock(f.fileno(), fcntl.LOCK_EX)
        json.dump(data, f, indent=2)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp_path, _REGISTRY_PATH)


def _pid_alive(pid):
    """Check if a process is alive."""
    try:
        os.kill(pid, 0)
        return True
    except PermissionError:
        return True  # process exists but we can't signal it
    except ProcessLookupError:
        return False


def init_registry():
    """Initialize a fresh registry for this generation run."""
    data = {
        "parent_pid": os.getpid(),
        "started_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "children": [],
    }
    with _lock:
        _write_registry(data)
    log.debug("PID registry initialized (parent=%d)", os.getpid())


def register_child(pid, proc_type, **metadata):
    """Register a child process in the registry."""
    entry = {"pid": pid, "type": proc_type}
    entry.update(metadata)
    with _lock:
        data = _read_registry()
        if data is None:
            return
        data["children"].append(entry)
        _write_registry(data)


def unregister_child(pid):
    """Remove a child process from the registry."""
    with _lock:
        data = _read_registry()
        if data is None:
            return
        data["children"] = [c for c in data["children"] if c["pid"] != pid]
        _write_registry(data)


def cleanup_stale(force=False):
    """Clean up orphaned processes from a previous crashed run.

    Returns True if cleanup succeeded or wasn't needed.
    Returns False if another generation is currently running.
    """
    data = _read_registry()
    if data is None:
        return True  # no registry, nothing to clean

    parent_pid = data.get("parent_pid")
    if parent_pid is None:
        remove_registry()
        return True

    # If the parent is alive and it's not us, another generation is running
    if _pid_alive(parent_pid) and parent_pid != os.getpid():
        if not force:
            log.warning("Another generation is running (PID %d, started %s). "
                        "Use --force or wait for it to finish.",
                        parent_pid, data.get("started_at", "unknown"))
            return False
        log.warning("Force-cleaning registry from PID %d", parent_pid)

    children = data.get("children", [])
    if not children:
        remove_registry()
        return True

    # Parent is dead (or force) — kill orphaned children
    killed = 0
    for child in children:
        pid = child.get("pid")
        if pid is None:
            continue
        if not _pid_alive(pid):
            continue

        proc_type = child.get("type", "unknown")
        log.info("Killing orphaned %s process (PID %d)", proc_type, pid)
        try:
            os.kill(pid, signal.SIGTERM)
        except (ProcessLookupError, PermissionError):
            continue
        killed += 1

    # Wait briefly, then SIGKILL any survivors
    if killed > 0:
        time.sleep(3)
        for child in children:
            pid = child.get("pid")
            if pid and _pid_alive(pid):
                try:
                    os.kill(pid, signal.SIGKILL)
                    log.debug("SIGKILL sent to PID %d", pid)
                except (ProcessLookupError, PermissionError):
                    pass

    # Clean up tmpdirs referenced in metadata
    for child in children:
        tmpdir = child.get("tmpdir")
        if tmpdir and os.path.isdir(tmpdir):
            import shutil
            try:
                shutil.rmtree(tmpdir)
                log.debug("Removed stale tmpdir: %s", tmpdir)
            except OSError as e:
                log.debug("Failed to remove stale tmpdir %s: %s", tmpdir, e)
        config_dir = child.get("config_dir")
        if config_dir and os.path.isdir(config_dir):
            import shutil
            try:
                shutil.rmtree(config_dir)
                log.debug("Removed stale config_dir: %s", config_dir)
            except OSError as e:
                log.debug("Failed to remove stale config_dir %s: %s",
                          config_dir, e)

    if killed > 0:
        log.info("Cleaned up %d orphaned processes", killed)
    else:
        log.debug("No live orphan processes found")

    remove_registry()
    return True


def remove_registry():
    """Remove the registry file on clean exit."""
    try:
        os.remove(_REGISTRY_PATH)
    except OSError:
        pass


def _atexit_cleanup():
    """Best-effort cleanup on process exit."""
    data = _read_registry()
    if data is None:
        return
    parent_pid = data.get("parent_pid")
    if parent_pid == os.getpid():
        # We're the owner — clean up our children
        cleanup_stale(force=True)


atexit.register(_atexit_cleanup)
