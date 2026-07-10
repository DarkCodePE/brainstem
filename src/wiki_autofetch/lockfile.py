"""
Inter-process advisory lock for the auto-fetch one-shot tick.

Per issue #38: ``~/.sbw/run/auto-fetch.lock``; concurrent invocation is a no-op.

We use POSIX `fcntl.flock` (Linux/macOS) so the lock is held only as long
as the file descriptor is open — kernel-released on crash. Windows is
out of scope until the Tauri shell ships (M4 #42).

The lockfile is **not** in `~/.sbw/logs/` because we want a tmpfs-friendly
runtime directory; `~/.sbw/run/` is the canonical XDG-style location.
"""

from __future__ import annotations

import contextlib
import fcntl
import logging
import os
from pathlib import Path
from typing import IO

_log = logging.getLogger(__name__)

DEFAULT_LOCK_PATH = Path.home() / ".sbw" / "run" / "auto-fetch.lock"
"""Where the auto-fetch tick acquires its inter-process lock."""


class LockBusy(RuntimeError):  # noqa: N818 -- API surface; the "exception" suffix would obscure the busy-flag intent
    """Another auto-fetch tick is already running. Caller should exit 0."""


@contextlib.contextmanager
def acquire(path: Path = DEFAULT_LOCK_PATH):
    """Acquire a non-blocking exclusive `flock` on `path`.

    Yields the open file handle so the caller can write PID/start
    timestamp for diagnostics. Releases on context exit (or kernel
    crash).

    Raises
    ------
    LockBusy
        If the lock is held by another process. Per issue #38 AC
        ("concurrent invocation is a no-op"), the systemd-driven tick
        catches this and exits 0.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    # Best-effort 0o700 on the runtime dir so other local users can't
    # observe the lock holder PID. Failure here is non-fatal.
    try:
        os.chmod(path.parent, 0o700)
    except (OSError, PermissionError):
        pass

    fh: IO[str] = path.open("w", encoding="utf-8")
    try:
        fcntl.flock(fh.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError as exc:
        fh.close()
        raise LockBusy(f"auto-fetch lock held: {path}") from exc

    # We hold the lock — everything below must release it on exit.
    try:
        fh.seek(0)
        fh.truncate()
        fh.write(f"{os.getpid()}\n")
        fh.flush()
        yield fh
    finally:
        try:
            fcntl.flock(fh.fileno(), fcntl.LOCK_UN)
        except OSError:
            pass
        fh.close()


__all__ = ["DEFAULT_LOCK_PATH", "LockBusy", "acquire"]
