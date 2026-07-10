from __future__ import annotations

import fnmatch
from pathlib import Path

from wiki_ingest.config import Config
from wiki_ingest.security import UnsafePathError, validate_safe_path

_IGNORE_GLOBS: tuple[str, ...] = (
    "*/.git/*",
    ".git/*",
    "*/.obsidian/*",
    ".obsidian/*",
    "*/_ingested/*",
    "_ingested/*",
    # node_modules can be huge when a vault is also a JS project —
    # Tolaria adds it to the canonical watcher-skip list for the same
    # reason (vault_watcher.rs:22-43). Issue #127 sub-item 2.
    "*/node_modules/*",
    "node_modules/*",
    "*.sync-conflict-*",
    "*~",
    "*.tmp",
    "*.part",
    "*.crdownload",
    "*.swp",
    "*.swo",
)


def _matches_any(path_str: str, name: str, patterns: tuple[str, ...]) -> bool:
    for pat in patterns:
        if fnmatch.fnmatch(path_str, pat) or fnmatch.fnmatch(name, pat):
            return True
    return False


def should_ignore(
    path: Path,
    size: int | None = None,
    cfg: Config | None = None,
) -> tuple[bool, str]:
    name = path.name
    path_str = str(path)

    if name.startswith(".") and not path.is_dir():
        return True, "dotfile"

    if _matches_any(path_str, name, _IGNORE_GLOBS):
        return True, "ignore-glob"

    if path.is_dir():
        return True, "directory"

    if size is not None and cfg is not None and size > cfg.max_file_size_bytes:
        return True, f"size>{cfg.max_file_size_mb}MB"

    if size == 0:
        return True, "empty-file"

    return False, ""


def is_allowed_mime(mime: str | None, cfg: Config) -> bool:
    if not mime:
        return False
    for prefix in cfg.allowed_mime_prefixes:
        if mime.startswith(prefix):
            return True
    return False


def guard_event_path(path: Path, cfg: Config) -> Path | None:
    """SEC-01 enforcement: validate a watcher/catchup path against raw_dir.

    Returns the resolved, jailed path if safe; returns None (with the caller
    expected to skip/log) if the path escapes the raw root, contains control
    bytes, or cannot be resolved. Any UnsafePathError is swallowed here so
    callers stay free of exception handling for the fast-path.
    """
    try:
        return validate_safe_path(path, cfg.raw_dir)
    except UnsafePathError:
        return None
