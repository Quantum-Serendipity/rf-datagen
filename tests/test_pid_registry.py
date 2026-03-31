"""Tests for rf_datagen.pid_registry — orphan process tracking."""

import json
import os
import subprocess
import time

import pytest

from rf_datagen import pid_registry


@pytest.fixture(autouse=True)
def _isolate_registry(tmp_path, monkeypatch):
    """Redirect registry to tmp_path so tests don't touch real state."""
    reg_dir = str(tmp_path / "cache")
    reg_path = os.path.join(reg_dir, "pids.json")
    monkeypatch.setattr(pid_registry, "_REGISTRY_DIR", reg_dir)
    monkeypatch.setattr(pid_registry, "_REGISTRY_PATH", reg_path)
    yield


def test_init_creates_registry(tmp_path):
    pid_registry.init_registry()
    data = json.loads(open(pid_registry._REGISTRY_PATH).read())
    assert data["parent_pid"] == os.getpid()
    assert data["children"] == []
    assert "started_at" in data


def test_register_and_unregister(tmp_path):
    pid_registry.init_registry()

    pid_registry.register_child(12345, "fldigi", port=7362)
    data = json.loads(open(pid_registry._REGISTRY_PATH).read())
    assert len(data["children"]) == 1
    assert data["children"][0]["pid"] == 12345
    assert data["children"][0]["type"] == "fldigi"
    assert data["children"][0]["port"] == 7362

    pid_registry.unregister_child(12345)
    data = json.loads(open(pid_registry._REGISTRY_PATH).read())
    assert len(data["children"]) == 0


def test_cleanup_stale_no_registry():
    """cleanup_stale returns True when no registry exists."""
    assert pid_registry.cleanup_stale() is True


def test_cleanup_stale_dead_parent(tmp_path):
    """Cleanup kills children when parent is dead."""
    # Start a real subprocess we can track
    proc = subprocess.Popen(["sleep", "300"])
    child_pid = proc.pid

    # Create registry with a fake dead parent
    data = {
        "parent_pid": 99999999,  # almost certainly dead
        "started_at": "2026-03-30T12:00:00",
        "children": [{"pid": child_pid, "type": "test"}],
    }
    os.makedirs(pid_registry._REGISTRY_DIR, exist_ok=True)
    with open(pid_registry._REGISTRY_PATH, "w") as f:
        json.dump(data, f)

    result = pid_registry.cleanup_stale()
    assert result is True

    # The sleep process should be dead
    try:
        proc.wait(timeout=5)
    except subprocess.TimeoutExpired:
        proc.kill()
        proc.wait()
        pytest.fail("Child process was not killed by cleanup")

    # Registry file should be removed
    assert not os.path.exists(pid_registry._REGISTRY_PATH)


def test_cleanup_stale_live_parent_blocks():
    """cleanup_stale returns False when parent is still alive (us)."""
    pid_registry.init_registry()

    # Modify parent_pid to a different live PID (init=1)
    data = json.loads(open(pid_registry._REGISTRY_PATH).read())
    data["parent_pid"] = 1  # init process, always alive
    with open(pid_registry._REGISTRY_PATH, "w") as f:
        json.dump(data, f)

    result = pid_registry.cleanup_stale()
    assert result is False


def test_remove_registry():
    pid_registry.init_registry()
    assert os.path.exists(pid_registry._REGISTRY_PATH)
    pid_registry.remove_registry()
    assert not os.path.exists(pid_registry._REGISTRY_PATH)


def test_remove_nonexistent_is_noop():
    """remove_registry on missing file doesn't raise."""
    pid_registry.remove_registry()
