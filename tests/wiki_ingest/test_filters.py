"""
Tests for `wiki_ingest.filters` — uniform allowlist/blocklist per ADR-006.

Contract:
  - `should_ignore(path) -> bool`        : True iff path matches blocklist
  - `is_size_allowed(size_bytes) -> bool`: False iff size > 25 MB
  - `is_allowed_mime(mime) -> bool`      : True for text/*, application/pdf,
                                           application/json, application/x-yaml
"""

from __future__ import annotations

from pathlib import Path

import pytest

filters = pytest.importorskip(
    "wiki_ingest.filters",
    reason="core not implemented yet",
)


MAX_SIZE = 25 * 1024 * 1024  # 25 MB per ADR-006


# --------------------------------------------------------------------------- #
# Blocklist: path-based                                                       #
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize(
    "rel_path",
    [
        ".git/objects/abcd1234",
        ".obsidian/workspace.json",
        "raw/articles/foo.sync-conflict-12.md",
        "raw/articles/backup~",
        "raw/articles/upload.tmp",
        "raw/articles/partial.part",
        "raw/articles/.hidden",
        "raw/_ingested/articles/already-processed.md",
    ],
)
def test_should_ignore_blocklisted_paths(tmp_wiki_root: Path, rel_path: str) -> None:
    target = tmp_wiki_root / rel_path
    assert filters.should_ignore(target) is True, f"{rel_path} must be ignored"


# --------------------------------------------------------------------------- #
# Allowlist: legitimate inbox drops                                           #
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize(
    "rel_path",
    [
        "raw/articles/ok.md",
        "raw/papers/doc.pdf",
        "raw/articles/notes-2026-04-18.md",
    ],
)
def test_should_not_ignore_valid_paths(tmp_wiki_root: Path, rel_path: str) -> None:
    target = tmp_wiki_root / rel_path
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_bytes(b"ok")
    assert filters.should_ignore(target) is False, f"{rel_path} must pass"


# --------------------------------------------------------------------------- #
# Size guard                                                                  #
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize(
    "size_bytes,expected_allowed",
    [
        (0, True),
        (24 * 1024 * 1024, True),  # 24 MB allowed
        (MAX_SIZE, True),  # exactly 25 MB allowed
        (MAX_SIZE + 1, False),  # 25 MB + 1 byte denied
        (26 * 1024 * 1024, False),  # 26 MB denied
    ],
)
def test_is_size_allowed(size_bytes: int, expected_allowed: bool) -> None:
    assert filters.is_size_allowed(size_bytes) is expected_allowed


# --------------------------------------------------------------------------- #
# MIME allowlist                                                              #
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize(
    "mime,expected_allowed",
    [
        ("text/markdown", True),
        ("text/plain", True),
        ("application/pdf", True),
        ("application/json", True),
        ("application/x-yaml", True),
        ("image/png", False),
        ("image/jpeg", False),
        ("application/octet-stream", False),
        ("video/mp4", False),
    ],
)
def test_is_allowed_mime(mime: str, expected_allowed: bool) -> None:
    assert filters.is_allowed_mime(mime) is expected_allowed


# --------------------------------------------------------------------------- #
# Cross-cutting: dotfiles and nested conflicts                                #
# --------------------------------------------------------------------------- #


def test_nested_dotfile_is_ignored(tmp_wiki_root: Path) -> None:
    target = tmp_wiki_root / "raw" / "articles" / "subdir" / ".DS_Store"
    assert filters.should_ignore(target) is True


def test_ingested_subtree_is_ignored(tmp_wiki_root: Path) -> None:
    target = tmp_wiki_root / "raw" / "_ingested" / "articles" / "done.md"
    assert filters.should_ignore(target) is True
