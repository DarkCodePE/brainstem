"""
Tests for the auto-fetch flock — single-holder semantics + cleanup on exit.
"""

from __future__ import annotations

import os
import threading

import pytest

from wiki_autofetch.lockfile import LockBusy, acquire


def test_acquire_writes_pid(tmp_path):
    lock = tmp_path / "auto-fetch.lock"
    with acquire(lock) as fh:
        assert fh is not None
        contents = lock.read_text(encoding="utf-8").strip()
        assert contents == str(os.getpid())


def test_acquire_creates_parent_dir(tmp_path):
    lock = tmp_path / "deeply" / "nested" / "run" / "auto-fetch.lock"
    with acquire(lock):
        assert lock.exists()
        assert lock.parent.is_dir()


def test_second_acquire_in_same_process_busy(tmp_path):
    """Two `acquire()` calls in the same process MUST collide.

    `fcntl.flock` is per-OPEN-FILE-DESCRIPTION; a second open of the
    same path gets a separate FD that can't take the exclusive lock.
    """
    lock = tmp_path / "auto-fetch.lock"
    with acquire(lock):
        with pytest.raises(LockBusy):
            with acquire(lock):
                pytest.fail("should have been blocked")


def test_lock_released_on_context_exit(tmp_path):
    lock = tmp_path / "auto-fetch.lock"
    with acquire(lock):
        pass
    # After exit, we can re-acquire
    with acquire(lock):
        pass


def test_lock_concurrent_threads(tmp_path):
    """Acquire from a second thread while held — must raise LockBusy.

    `flock` IS per-OFD on Linux but pytest's threading model is good
    enough for this assertion when each thread opens its own FD.
    """
    lock = tmp_path / "auto-fetch.lock"
    busy_seen = threading.Event()
    failed = []

    def other_holder():
        try:
            with acquire(lock):
                failed.append("got_lock_unexpectedly")
        except LockBusy:
            busy_seen.set()

    with acquire(lock):
        t = threading.Thread(target=other_holder)
        t.start()
        t.join(timeout=2)
    # The other thread saw LockBusy
    assert busy_seen.is_set()
    assert failed == []
