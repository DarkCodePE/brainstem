"""Security primitives for the wiki_ingest daemon.

Implements mitigations from SEC-ADR-006 (2026-04-18):
  SEC-01 path traversal, SEC-02 symlinks, SEC-03 TOCTOU,
  SEC-05 frontmatter / prompt injection, SEC-06 size bombs,
  SEC-07 log allowlist, SEC-10 atomic writes, SEC-11 DB perms.

Public surface (stable):
  - Exceptions: UnsafePathError, DangerousFrontmatterError,
    SymlinkRejectedError, SizeBombError, YamlBombError
  - Helpers: validate_safe_path, open_safe_fd, hash_and_stat_fd,
    parse_and_validate_frontmatter, wrap_untrusted_body,
    atomic_write_text, flock_slug, quarantine_symlink,
    SafeLogFormatter, apply_safe_logging
"""

from __future__ import annotations

import errno
import fcntl
import hashlib
import json
import logging
import os
import shutil
import tempfile
import time
from collections.abc import Iterator
from contextlib import contextmanager
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

# --------------------------------------------------------------------------- #
# Typed exceptions                                                            #
# --------------------------------------------------------------------------- #


class SecurityError(Exception):
    """Base class for all hardening-related rejections."""


class UnsafePathError(SecurityError):
    """Path escapes the raw root, contains NUL, or has control bytes."""


class SymlinkRejectedError(SecurityError):
    """Encountered a symlink where only regular files are allowed."""


class SizeBombError(SecurityError):
    """On-disk size exceeds the configured cap (checked via fstat)."""


class DangerousFrontmatterError(SecurityError):
    """Frontmatter contains keys that look like prompt-injection vectors."""


class YamlBombError(SecurityError):
    """Frontmatter YAML exceeds structural caps (depth or key count)."""


# --------------------------------------------------------------------------- #
# SEC-01 + SEC-02: path hardening                                             #
# --------------------------------------------------------------------------- #

_FRONTMATTER_BANNED_KEYS = frozenset(
    {"system", "tool", "assistant", "instructions", "instruction", "prompt", "role"}
)

_YAML_MAX_DEPTH = 6
_YAML_MAX_KEYS = 100
_FRONTMATTER_MAX_BYTES = 8 * 1024


def _has_control_chars(text: str) -> bool:
    if "\x00" in text:
        return True
    return any(ord(c) < 0x20 and c not in ("\t", "\n", "\r") for c in text)


def validate_safe_path(path: Path, root: Path) -> Path:
    """Resolve path and ensure it stays inside root.

    - Rejects NUL bytes and control characters in the raw path.
    - Rejects dangling paths via Path.resolve(strict=True).
    - Rejects paths that escape the canonical root.
    """
    raw = str(path)
    if _has_control_chars(raw):
        raise UnsafePathError(f"control character in path: {raw!r}")
    try:
        resolved = path.resolve(strict=True)
    except FileNotFoundError as e:
        raise UnsafePathError(f"path does not exist: {raw!r}") from e
    except OSError as e:
        raise UnsafePathError(f"cannot resolve path: {raw!r} ({e})") from e
    try:
        root_resolved = root.resolve(strict=True)
    except OSError as e:
        raise UnsafePathError(f"cannot resolve root: {root!r} ({e})") from e
    if not resolved.is_relative_to(root_resolved):
        raise UnsafePathError(f"path escapes root: {raw!r} (root={root_resolved})")
    if _has_control_chars(resolved.name):
        raise UnsafePathError(f"control character in resolved name: {resolved!r}")
    return resolved


def quarantine_symlink(path: Path, raw_root: Path, reason: str = "symlink") -> Path:
    """Move a rejected symlink to raw/_rejected/<reason>/<timestamp>_<name>.

    Returns the destination path. Uses os.rename where possible; falls back
    to unlink if rename is impossible (symlink crossing filesystems).
    """
    ts = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    dest_dir = raw_root / "_rejected" / reason
    dest_dir.mkdir(parents=True, exist_ok=True)
    dest = dest_dir / f"{ts}_{path.name}"
    try:
        os.rename(path, dest)
    except OSError:
        try:
            os.unlink(path)
        except OSError:
            pass
    return dest


# --------------------------------------------------------------------------- #
# SEC-02 + SEC-03 + SEC-06: single-fd open + streaming hash + size guard      #
# --------------------------------------------------------------------------- #


def open_safe_fd(path: Path) -> int:
    """Open path with O_RDONLY | O_NOFOLLOW | O_CLOEXEC.

    Raises SymlinkRejectedError if the path is a symlink (ELOOP).
    Caller is responsible for closing the returned fd.
    """
    try:
        return os.open(str(path), os.O_RDONLY | os.O_NOFOLLOW | os.O_CLOEXEC)
    except OSError as e:
        if e.errno in (errno.ELOOP, errno.EMLINK):
            raise SymlinkRejectedError(f"symlink refused: {path}") from e
        raise


def hash_and_stat_fd(fd: int, max_size_bytes: int, chunk: int = 1 << 16) -> dict[str, Any]:
    """Stream SHA256 + capture fstat metadata from a single file descriptor.

    Enforces the size cap via fstat BEFORE reading any bytes (SEC-06).
    Returns {'sha256', 'size', 'st_dev', 'st_ino', 'st_mtime_ns'}.
    """
    st = os.fstat(fd)
    if st.st_size > max_size_bytes:
        raise SizeBombError(f"file size {st.st_size} exceeds cap {max_size_bytes}")
    h = hashlib.sha256()
    remaining = st.st_size
    os.lseek(fd, 0, os.SEEK_SET)
    while remaining > 0:
        buf = os.read(fd, min(chunk, remaining))
        if not buf:
            break
        h.update(buf)
        remaining -= len(buf)
    return {
        "sha256": h.hexdigest(),
        "size": st.st_size,
        "st_dev": st.st_dev,
        "st_ino": st.st_ino,
        "st_mtime_ns": st.st_mtime_ns,
    }


# --------------------------------------------------------------------------- #
# SEC-05: frontmatter / prompt-injection firewall                             #
# --------------------------------------------------------------------------- #


def _yaml_structure_check(node: Any, depth: int = 0, state: dict | None = None) -> None:
    if state is None:
        state = {"keys": 0}
    if depth > _YAML_MAX_DEPTH:
        raise YamlBombError(f"frontmatter depth {depth} > {_YAML_MAX_DEPTH}")
    if isinstance(node, dict):
        state["keys"] += len(node)
        if state["keys"] > _YAML_MAX_KEYS:
            raise YamlBombError(f"frontmatter key count {state['keys']} > {_YAML_MAX_KEYS}")
        for v in node.values():
            _yaml_structure_check(v, depth + 1, state)
    elif isinstance(node, list):
        for v in node:
            _yaml_structure_check(v, depth + 1, state)


def _scan_banned_keys(node: Any) -> None:
    if isinstance(node, dict):
        for k, v in node.items():
            if isinstance(k, str) and k.strip().lower() in _FRONTMATTER_BANNED_KEYS:
                raise DangerousFrontmatterError(f"banned frontmatter key: {k!r}")
            _scan_banned_keys(v)
    elif isinstance(node, list):
        for v in node:
            _scan_banned_keys(v)


def parse_and_validate_frontmatter(body_bytes: bytes) -> tuple[dict, str]:
    """Split an ingested file into (frontmatter_dict, body_text).

    - Enforces strict UTF-8 transcoding.
    - Caps the frontmatter region at 8 KiB.
    - Rejects banned top-level/nested keys that look like prompt injection.
    - Rejects YAML depth > 6 or key count > 100.

    Returns an empty frontmatter dict if the file has no YAML preamble.
    """
    try:
        text = body_bytes.decode("utf-8", errors="strict")
    except UnicodeDecodeError as e:
        raise DangerousFrontmatterError(f"invalid utf-8: {e}") from e

    if not text.startswith("---\n") and not text.startswith("---\r\n"):
        return {}, text

    # Locate end of frontmatter within the 8 KiB cap.
    region_limit = _FRONTMATTER_MAX_BYTES
    head = text[: region_limit + 4]
    closing = head.find("\n---\n", 3)
    if closing < 0:
        closing = head.find("\n---\r\n", 3)
    if closing < 0:
        raise DangerousFrontmatterError(f"frontmatter not closed within {region_limit} bytes")
    frontmatter_raw = text[4:closing]
    body = text[closing + 5 :]
    if body.startswith("\n"):
        body = body[1:]

    try:
        import yaml  # local import; yaml is optional-at-import
    except ImportError as e:  # pragma: no cover
        raise DangerousFrontmatterError("pyyaml unavailable") from e

    try:
        data = yaml.safe_load(frontmatter_raw)
    except yaml.YAMLError as e:
        raise DangerousFrontmatterError(f"invalid yaml: {e}") from e

    if data is None:
        data = {}
    if not isinstance(data, dict):
        raise DangerousFrontmatterError("frontmatter root must be a mapping")

    _yaml_structure_check(data)
    _scan_banned_keys(data)
    return data, body


def wrap_untrusted_body(body: str, sha256: str, rel_path: str) -> str:
    """Wrap an ingested body in a trust-untrusted envelope (SEC-05)."""
    sha_attr = sha256.replace('"', "").replace("<", "").replace(">", "")
    rel_attr = rel_path.replace('"', "").replace("<", "").replace(">", "")
    return (
        f'<ingested_source trust="untrusted" sha256="{sha_attr}" '
        f'rel_path="{rel_attr}">\n'
        f"{body}\n"
        f"</ingested_source>\n"
    )


# --------------------------------------------------------------------------- #
# SEC-10: atomic filesystem writes + per-slug locks                           #
# --------------------------------------------------------------------------- #


def atomic_write_text(dest: Path, content: str, encoding: str = "utf-8", mode: int = 0o600) -> None:
    """Write content to dest atomically via tempfile + os.replace + fsync."""
    dest.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_path = tempfile.mkstemp(prefix=f".{dest.name}.", suffix=".tmp", dir=str(dest.parent))
    try:
        with os.fdopen(fd, "w", encoding=encoding) as f:
            f.write(content)
            f.flush()
            os.fsync(f.fileno())
        os.chmod(tmp_path, mode)
        os.replace(tmp_path, dest)
    except Exception:
        try:
            os.unlink(tmp_path)
        except OSError:
            pass
        raise
    # fsync the parent directory so the rename is durable.
    dir_fd = os.open(str(dest.parent), os.O_DIRECTORY | os.O_CLOEXEC)
    try:
        os.fsync(dir_fd)
    finally:
        os.close(dir_fd)


def atomic_move(src: Path, dest: Path) -> None:
    """Move src to dest using rename when possible, else copy+fsync+unlink."""
    dest.parent.mkdir(parents=True, exist_ok=True)
    try:
        os.replace(src, dest)
    except OSError as e:
        if e.errno != errno.EXDEV:
            raise
        # Cross-filesystem: copy to temp then rename.
        fd, tmp_path = tempfile.mkstemp(
            prefix=f".{dest.name}.", suffix=".tmp", dir=str(dest.parent)
        )
        os.close(fd)
        try:
            shutil.copy2(str(src), tmp_path)
            os.replace(tmp_path, dest)
            os.unlink(src)
        except Exception:
            try:
                os.unlink(tmp_path)
            except OSError:
                pass
            raise
    dir_fd = os.open(str(dest.parent), os.O_DIRECTORY | os.O_CLOEXEC)
    try:
        os.fsync(dir_fd)
    finally:
        os.close(dir_fd)


@contextmanager
def flock_slug(locks_dir: Path, slug: str) -> Iterator[int]:
    """Acquire an exclusive advisory lock on <locks_dir>/<slug>.lock."""
    locks_dir.mkdir(parents=True, exist_ok=True)
    safe = "".join(c if c.isalnum() or c in ("-", "_", ".") else "_" for c in slug)
    lock_path = locks_dir / f"{safe}.lock"
    fd = os.open(str(lock_path), os.O_RDWR | os.O_CREAT | os.O_CLOEXEC, 0o600)
    try:
        fcntl.flock(fd, fcntl.LOCK_EX)
        yield fd
    finally:
        try:
            fcntl.flock(fd, fcntl.LOCK_UN)
        finally:
            os.close(fd)


# --------------------------------------------------------------------------- #
# SEC-07: structured log allowlist                                            #
# --------------------------------------------------------------------------- #


_LOG_ALLOWED_FIELDS = frozenset(
    {
        "event_id",
        "rel_path",
        "page_path",
        "sha256",
        "size",
        "bucket",
        "event_type",
        "status",
        "latency_ms",
        "attempt",
        "error_class",
        "reason",
        "mime",
        "count",
        "queue_depth",
    }
)


class SafeLogFormatter(logging.Formatter):
    """JSON formatter that drops any extra field outside the allowlist.

    - Absolute paths are stripped (never emitted).
    - sha256 is truncated to the 12-char prefix.
    - Exception messages are dropped; only the class name survives as
      `error_class`.
    """

    def format(self, record: logging.LogRecord) -> str:
        payload: dict[str, Any] = {
            "ts": datetime.now(UTC).isoformat(timespec="milliseconds").replace("+00:00", "Z"),
            "level": record.levelname,
            "logger": record.name,
            "event": record.getMessage(),
        }
        extra = getattr(record, "extra_fields", None)
        if isinstance(extra, dict):
            for k, v in extra.items():
                if k not in _LOG_ALLOWED_FIELDS:
                    continue
                if k == "sha256" and isinstance(v, str):
                    v = v[:12]
                if k == "rel_path" and isinstance(v, str):
                    # Refuse anything that looks absolute.
                    if v.startswith("/"):
                        continue
                payload[k] = v
        if record.exc_info:
            exc_type = record.exc_info[0]
            payload.setdefault("error_class", exc_type.__name__ if exc_type else "Exception")
        return json.dumps(payload, ensure_ascii=False, default=str)


def apply_safe_logging(level: str = "INFO") -> None:
    """Replace root handlers with a single allowlist-filtered JSON handler."""
    import sys

    root = logging.getLogger()
    for h in list(root.handlers):
        root.removeHandler(h)
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(SafeLogFormatter())
    root.addHandler(handler)
    root.setLevel(getattr(logging, level.upper(), logging.INFO))


# --------------------------------------------------------------------------- #
# SEC-11: filesystem permissions                                              #
# --------------------------------------------------------------------------- #


def tighten_db_permissions(db_path: Path) -> None:
    """chmod db + WAL siblings to 0o600 if they exist."""
    for suffix in ("", "-wal", "-shm", "-journal"):
        p = db_path.with_name(db_path.name + suffix)
        if p.exists():
            try:
                os.chmod(p, 0o600)
            except OSError:
                pass


# --------------------------------------------------------------------------- #
# SEC-08: minute-bucket rate-limit helpers (SQLite-backed)                    #
# --------------------------------------------------------------------------- #


def minute_window_start(now: float | None = None) -> str:
    """Return an ISO timestamp for the current minute epoch (UTC).

    This is used as the logical window key for the persistent rate limit.
    """
    ts = now if now is not None else time.time()
    minute = int(ts // 60) * 60
    return (
        datetime.fromtimestamp(minute, tz=UTC).isoformat(timespec="seconds").replace("+00:00", "Z")
    )


__all__ = [
    "SecurityError",
    "UnsafePathError",
    "SymlinkRejectedError",
    "SizeBombError",
    "DangerousFrontmatterError",
    "YamlBombError",
    "validate_safe_path",
    "quarantine_symlink",
    "open_safe_fd",
    "hash_and_stat_fd",
    "parse_and_validate_frontmatter",
    "wrap_untrusted_body",
    "atomic_write_text",
    "atomic_move",
    "flock_slug",
    "SafeLogFormatter",
    "apply_safe_logging",
    "tighten_db_permissions",
    "minute_window_start",
]
