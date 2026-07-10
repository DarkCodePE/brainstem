"""
Focused tests for `wiki_ingest.filters` against the actual current API.

Replaces the broader, brownfield `test_filters.py` (quarantined) which
targeted a `is_allowed_mime(mime)` shape that never shipped — the real
signature is `is_allowed_mime(mime, cfg)` and `should_ignore` returns a
`(bool, reason)` tuple rather than a bare bool.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from wiki_ingest.config import Config
from wiki_ingest.filters import is_allowed_mime, should_ignore


@pytest.fixture
def cfg(tmp_path: Path) -> Config:
    """A minimum-valid Config bound to a tmp wiki root."""
    return Config(
        wiki_root=tmp_path,
        raw_dir=tmp_path / "raw",
        ingested_dir=tmp_path / "raw" / "_ingested",
        db_path=tmp_path / "ingest.db",
        debounce_seconds=3.0,
        worker_pool_size=2,
        rate_limit_per_minute=10,
        max_file_size_mb=25,
        allowed_mime_prefixes=(
            "text/",
            "application/pdf",
            "application/json",
            "application/x-yaml",
        ),
        mcp_command=("python", "-m", "wiki_agent.mcp_server"),
        metrics_path=tmp_path / "metrics.prom",
    )


class TestShouldIgnoreDotfilesAndIgnoreGlobs:
    @pytest.mark.parametrize(
        "filename",
        [".env", ".DS_Store", ".gitignore", ".hidden.md"],
    )
    def test_dotfile_at_root_is_ignored(self, tmp_path: Path, filename: str, cfg: Config) -> None:
        p = tmp_path / filename
        p.write_text("x")
        ignore, reason = should_ignore(p, size=10, cfg=cfg)
        assert ignore is True
        assert reason == "dotfile"

    @pytest.mark.parametrize(
        "pattern",
        ["sync-conflict-foo.md", "draft.swp", "transient.tmp", "incomplete.part"],
    )
    def test_dangerous_suffix_is_ignored(self, tmp_path: Path, pattern: str, cfg: Config) -> None:
        p = tmp_path / pattern
        if "sync-conflict" in pattern:
            p = tmp_path / f"file.sync-conflict-{pattern.split('-')[-1]}"
        p.write_text("x")
        ignore, reason = should_ignore(p, size=10, cfg=cfg)
        assert ignore is True
        assert "ignore-glob" in reason or "dotfile" in reason

    def test_ingested_subtree_is_ignored(self, tmp_path: Path, cfg: Config) -> None:
        nested = tmp_path / "raw" / "_ingested" / "x.md"
        nested.parent.mkdir(parents=True, exist_ok=True)
        nested.write_text("x")
        ignore, reason = should_ignore(nested, size=10, cfg=cfg)
        assert ignore is True


class TestShouldIgnoreSize:
    def test_oversized_is_ignored(self, tmp_path: Path, cfg: Config) -> None:
        p = tmp_path / "big.md"
        p.write_text("x")
        # 26 MB > 25 MB ceiling
        ignore, reason = should_ignore(p, size=26 * 1024 * 1024, cfg=cfg)
        assert ignore is True
        assert "size" in reason

    def test_at_ceiling_is_allowed(self, tmp_path: Path, cfg: Config) -> None:
        p = tmp_path / "exact.md"
        p.write_text("x")
        ignore, reason = should_ignore(p, size=25 * 1024 * 1024, cfg=cfg)
        assert ignore is False

    def test_empty_file_is_ignored(self, tmp_path: Path, cfg: Config) -> None:
        p = tmp_path / "empty.md"
        p.write_text("")
        ignore, reason = should_ignore(p, size=0, cfg=cfg)
        assert ignore is True
        assert reason == "empty-file"

    def test_no_size_info_passes(self, tmp_path: Path, cfg: Config) -> None:
        p = tmp_path / "ok.md"
        p.write_text("x")
        ignore, _ = should_ignore(p, size=None, cfg=cfg)
        assert ignore is False


class TestIsAllowedMime:
    @pytest.mark.parametrize(
        "mime,expected",
        [
            ("text/markdown", True),
            ("text/plain", True),
            ("text/html", True),
            ("application/pdf", True),
            ("application/json", True),
            ("application/x-yaml", True),
            ("image/png", False),
            ("image/jpeg", False),
            ("application/octet-stream", False),
            ("video/mp4", False),
            ("audio/mpeg", False),
        ],
    )
    def test_mime_allowlist(self, mime: str, expected: bool, cfg: Config) -> None:
        assert is_allowed_mime(mime, cfg) is expected

    def test_none_mime_is_rejected(self, cfg: Config) -> None:
        assert is_allowed_mime(None, cfg) is False

    def test_empty_mime_is_rejected(self, cfg: Config) -> None:
        assert is_allowed_mime("", cfg) is False
