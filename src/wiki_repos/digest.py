"""Local, no-clone gitingest-style digest builder (PRD-012 FR-3 / ADR-022).

This module walks an *already-extracted* repository directory on disk and
produces a :class:`~wiki_repos.types.Digest` — a summary, an indented file
tree, and a concatenated text blob of the included files plus accounting
:class:`~wiki_repos.types.DigestStats`.

It deliberately replaces the ``gitingest`` library: pure stdlib only
(``pathlib`` / ``os``), **no clone, no network, no GitPython**. The repo bytes
are assumed to already be on disk (produced by ``wiki_repos.fetcher`` from a
tarball). Truncation is *never silent* — when a byte/token cap is hit we stop
and append an explicit marker, and flip ``stats.truncated``.
"""

from __future__ import annotations

import os
from pathlib import Path

from .errors import DigestFailed
from .types import Digest, DigestStats, RepoRef

__all__ = ["build_digest"]

# --------------------------------------------------------------------------- #
# Skip policy
# --------------------------------------------------------------------------- #

#: Directory *names* skipped wholesale, anywhere in the tree.
SKIP_DIRS: frozenset[str] = frozenset(
    {
        ".git",
        "node_modules",
        "dist",
        "build",
        ".next",
        "out",
        "vendor",
        "__pycache__",
        ".venv",
        "venv",
        "target",
        ".cache",
        "coverage",
        ".idea",
        ".vscode",
    }
)

#: Binary / non-text file extensions skipped by name (lower-cased, no dot).
BINARY_EXTS: frozenset[str] = frozenset(
    {
        "png",
        "jpg",
        "jpeg",
        "gif",
        "webp",
        "svg",
        "ico",
        "pdf",
        "zip",
        "gz",
        "tar",
        "woff",
        "woff2",
        "ttf",
        "eot",
        "mp4",
        "mp3",
        "wav",
        "mov",
        "bin",
        "exe",
        "so",
        "dylib",
        "lock",
        "map",
        "node",
        "wasm",
    }
)

#: Compound suffixes (checked against the full lower-cased filename).
BINARY_SUFFIXES: tuple[str, ...] = (".min.js",)

#: Bytes read from the head of a file to sniff for NUL bytes.
_SNIFF_BYTES = 1024


def _is_binary_name(name: str) -> bool:
    """Return True if *name* matches the extension / suffix blocklist."""
    lower = name.lower()
    for suffix in BINARY_SUFFIXES:
        if lower.endswith(suffix):
            return True
    ext = lower.rsplit(".", 1)[-1] if "." in lower else ""
    return ext in BINARY_EXTS


def _looks_binary(path: Path) -> bool:
    """NUL-byte sniff of the first :data:`_SNIFF_BYTES` bytes of *path*.

    A read error is treated as "binary/unreadable" so we skip rather than crash.
    """
    try:
        with path.open("rb") as fh:
            chunk = fh.read(_SNIFF_BYTES)
    except OSError:
        return True
    return b"\x00" in chunk


def _read_text(path: Path) -> str | None:
    """Read *path* as UTF-8 (replacing undecodable bytes); None on OSError."""
    try:
        return path.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return None


# --------------------------------------------------------------------------- #
# Walk
# --------------------------------------------------------------------------- #


def _collect_files(repo_dir: Path, max_file_size: int) -> tuple[list[tuple[str, int]], set[str]]:
    """Walk *repo_dir*, returning ``(included, skipped_dir_names)``.

    ``included`` is a list of ``(relpath_posix, size_bytes)`` for every text
    file that passes the extension, NUL-sniff and per-file size filters, sorted
    by relative path for deterministic output. ``skipped_dir_names`` is the set
    of directory names pruned wholesale.
    """
    included: list[tuple[str, int]] = []
    skipped_dirs: set[str] = set()

    for dirpath, dirnames, filenames in os.walk(repo_dir):
        # Prune skip-listed directories in place so os.walk never descends them.
        kept: list[str] = []
        for d in dirnames:
            if d in SKIP_DIRS:
                skipped_dirs.add(d)
            else:
                kept.append(d)
        dirnames[:] = kept

        for fname in filenames:
            if _is_binary_name(fname):
                continue
            fpath = Path(dirpath) / fname
            try:
                size = fpath.stat().st_size
            except OSError:
                continue
            if size > max_file_size:
                continue
            if _looks_binary(fpath):
                continue
            rel = fpath.relative_to(repo_dir).as_posix()
            included.append((rel, size))

    included.sort(key=lambda item: item[0])
    return included, skipped_dirs


# --------------------------------------------------------------------------- #
# Tree rendering
# --------------------------------------------------------------------------- #


def _build_tree(relpaths: list[str]) -> str:
    """Render an indented file tree (dirs sorted first, then files) from a
    sorted list of posix relpaths."""
    # Nested dict: dir -> subtree; files map to None.
    root: dict[str, dict | None] = {}
    for rel in relpaths:
        parts = rel.split("/")
        node = root
        for part in parts[:-1]:
            child = node.get(part)
            if not isinstance(child, dict):
                child = {}
                node[part] = child
            node = child
        node[parts[-1]] = None  # leaf = file

    lines: list[str] = []

    def _walk(node: dict[str, dict | None], depth: int) -> None:
        dirs = sorted(k for k, v in node.items() if isinstance(v, dict))
        files = sorted(k for k, v in node.items() if v is None)
        indent = "  " * depth
        for name in dirs:
            lines.append(f"{indent}{name}/")
            child = node[name]
            assert isinstance(child, dict)
            _walk(child, depth + 1)
        for name in files:
            lines.append(f"{indent}{name}")

    _walk(root, 0)
    return "\n".join(lines)


# --------------------------------------------------------------------------- #
# Summary
# --------------------------------------------------------------------------- #


def _language_tally(relpaths: list[str]) -> str:
    """Return a ``".js: 120, .ts: 30"`` style top-languages string."""
    counts: dict[str, int] = {}
    for rel in relpaths:
        name = rel.rsplit("/", 1)[-1]
        ext = "." + name.rsplit(".", 1)[-1] if "." in name else "(none)"
        counts[ext] = counts.get(ext, 0) + 1
    # Sort by count desc, then extension asc for determinism.
    ranked = sorted(counts.items(), key=lambda kv: (-kv[1], kv[0]))
    return ", ".join(f"{ext}: {n}" for ext, n in ranked[:8])


def _build_summary(ref: RepoRef, n_files: int, n_bytes: int, langs: str, truncated: bool) -> str:
    lines = [
        f"Repository: {ref.canonical_url}",
        f"Included files: {n_files}",
        f"Total bytes: {n_bytes}",
        f"Top languages: {langs}",
        f"Truncated: {'yes' if truncated else 'no'}",
    ]
    return "\n".join(lines)


# --------------------------------------------------------------------------- #
# Public API
# --------------------------------------------------------------------------- #


def build_digest(
    repo_dir: Path,
    ref: RepoRef,
    *,
    max_file_size: int = 100_000,
    max_total_bytes: int = 2_000_000,
    max_total_tokens: int = 80_000,
) -> Digest:
    """Build a gitingest-shaped :class:`Digest` from a local repo directory.

    Walks *repo_dir* recursively, skipping vendored/build directories
    (:data:`SKIP_DIRS`), binary files (by extension blocklist and a NUL-byte
    sniff) and per-file oversize content (> *max_file_size*). The included text
    files — in deterministic sorted path order — are rendered into a file tree
    and concatenated into ``content`` until either *max_total_bytes* or
    *max_total_tokens* (estimated as ``bytes // 4``) is reached, at which point
    an explicit ``[TRUNCATED: …]`` marker is appended and ``stats.truncated`` is
    set. No network access, no clone, pure stdlib.

    Args:
        repo_dir: Path to an already-extracted repository directory on disk.
        ref: The validated :class:`RepoRef` the digest is for (used in summary).
        max_file_size: Per-file byte cap; larger files are skipped.
        max_total_bytes: Aggregate byte cap across all included content.
        max_total_tokens: Aggregate estimated-token cap (``bytes // 4``).

    Returns:
        A populated :class:`Digest`.

    Raises:
        DigestFailed: If *repo_dir* does not exist, is not a directory, or
            yields zero readable text files.
    """
    if not repo_dir.exists():
        raise DigestFailed(f"repo dir does not exist: {repo_dir}")
    if not repo_dir.is_dir():
        raise DigestFailed(f"repo path is not a directory: {repo_dir}")

    included, skipped_dirs = _collect_files(repo_dir, max_file_size)
    if not included:
        raise DigestFailed(f"no readable text files under: {repo_dir}")

    relpaths = [rel for rel, _ in included]
    tree = _build_tree(relpaths)

    # Concatenate content under the byte/token caps. est_tokens == bytes // 4,
    # so the token cap is equivalent to a byte cap of max_total_tokens * 4; we
    # take the tighter of the two as the effective byte budget.
    byte_budget = min(max_total_bytes, max_total_tokens * 4)

    blocks: list[str] = []
    total_bytes = 0
    emitted_files = 0
    truncated = False

    for rel, _ in included:
        text = _read_text(repo_dir / rel)
        if text is None:
            continue
        block = f"\n===== {rel} =====\n{text}\n"
        block_bytes = len(block.encode("utf-8"))
        if total_bytes + block_bytes > byte_budget and emitted_files > 0:
            truncated = True
            break
        blocks.append(block)
        total_bytes += block_bytes
        emitted_files += 1
        # A single first block can itself exceed the budget; stop after it.
        if total_bytes >= byte_budget:
            if emitted_files < len(included):
                truncated = True
            break

    if emitted_files == 0:
        # Every included path became unreadable between walk and read.
        raise DigestFailed(f"no readable text files under: {repo_dir}")

    if truncated:
        omitted_files = len(included) - emitted_files
        omitted_bytes = sum(size for rel, size in included[emitted_files:])
        blocks.append(
            f"\n... [TRUNCATED: {omitted_files} files / "
            f"{omitted_bytes} bytes omitted by digest caps] ...\n"
        )

    content = "".join(blocks)

    # Stats reflect the *emitted* set (what actually lives in `content`).
    emitted_relpaths = relpaths[:emitted_files]
    est_tokens = total_bytes // 4
    stats = DigestStats(
        n_files=emitted_files,
        n_bytes=total_bytes,
        est_tokens=est_tokens,
        truncated=truncated,
        skipped_dirs=tuple(sorted(skipped_dirs)),
    )

    langs = _language_tally(emitted_relpaths)
    summary = _build_summary(ref, emitted_files, total_bytes, langs, truncated)

    return Digest(summary=summary, tree=tree, content=content, stats=stats)
