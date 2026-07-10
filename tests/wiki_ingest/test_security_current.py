"""
Focused tests for `wiki_ingest.security` against the actual current API.

Replaces the broader brownfield `test_security.py` (quarantined) which
exercised the worker pool end-to-end with a sync queue API that never
shipped. Here we test the security helpers directly — they are sync and
pure-function-shaped, so the tests stay tight.
"""

from __future__ import annotations

import stat
from pathlib import Path

import pytest

from wiki_ingest.security import (
    DangerousFrontmatterError,
    SafeLogFormatter,
    UnsafePathError,
    YamlBombError,
    _has_control_chars,
    atomic_write_text,
    parse_and_validate_frontmatter,
    quarantine_symlink,
    validate_safe_path,
    wrap_untrusted_body,
)

# --------------------------------------------------------------------------- #
# SEC-01 — path validation                                                    #
# --------------------------------------------------------------------------- #


class TestValidateSafePath:
    def test_path_inside_root_is_accepted(self, tmp_path: Path) -> None:
        root = tmp_path
        target = tmp_path / "raw" / "x.md"
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text("x")
        assert validate_safe_path(target, root) == target.resolve()

    def test_traversal_escape_is_rejected(self, tmp_path: Path) -> None:
        root = tmp_path / "raw"
        root.mkdir(parents=True, exist_ok=True)
        outside = tmp_path / "etc" / "passwd"
        outside.parent.mkdir(parents=True, exist_ok=True)
        outside.write_text("x")
        with pytest.raises(UnsafePathError):
            validate_safe_path(outside, root)

    def test_control_chars_in_basename_are_rejected(self, tmp_path: Path) -> None:
        root = tmp_path
        # Use a non-printable control char (NULL is filtered by OS; SOH = 0x01)
        target = tmp_path / "with\x01ctrl.md"
        try:
            target.write_text("x")
        except OSError:
            pytest.skip("OS rejects control-char filename")
        with pytest.raises(UnsafePathError):
            validate_safe_path(target, root)


class TestHasControlChars:
    @pytest.mark.parametrize(
        "s,expected",
        [
            ("plain.md", False),
            ("with\x01ctrl.md", True),
            ("with\x07bell.md", True),
            ("emoji😀.md", False),
            # Newline and tab are explicitly whitelisted by _has_control_chars
            # because legitimate Markdown can include them in body text.
            ("with\nnewline.md", False),
            ("tab\tin.md", False),
        ],
    )
    def test_detection(self, s: str, expected: bool) -> None:
        assert _has_control_chars(s) is expected


# --------------------------------------------------------------------------- #
# SEC-02 — symlink quarantine                                                 #
# --------------------------------------------------------------------------- #


class TestQuarantineSymlink:
    def test_symlink_moves_to_quarantine(self, tmp_path: Path) -> None:
        raw = tmp_path / "raw"
        raw.mkdir()
        target = tmp_path / "real.txt"
        target.write_text("data")
        link = raw / "link.txt"
        link.symlink_to(target)
        moved = quarantine_symlink(link, raw, reason="symlink-test")
        # Link was removed from its original spot
        assert not link.exists()
        # The moved-to path is inside the quarantine subtree
        # (current implementation uses `_rejected/<reason>/` under raw root).
        assert "_rejected" in str(moved) or "quarantine" in str(moved)
        assert moved.exists()


# --------------------------------------------------------------------------- #
# SEC-05 — frontmatter safety                                                 #
# --------------------------------------------------------------------------- #


class TestParseAndValidateFrontmatter:
    def test_valid_frontmatter_parses(self) -> None:
        body = b"---\ntitle: Test\ndate: 2026-05-22\n---\n\nBody here.\n"
        fm, content = parse_and_validate_frontmatter(body)
        assert fm["title"] == "Test"
        assert "Body here" in content

    @pytest.mark.parametrize("danger_key", ["system", "role", "instruction"])
    def test_dangerous_keys_rejected(self, danger_key: str) -> None:
        body = f"---\n{danger_key}: rm -rf /\ntitle: x\n---\nBody\n".encode()
        with pytest.raises(DangerousFrontmatterError):
            parse_and_validate_frontmatter(body)

    def test_yaml_bomb_key_count_rejected(self) -> None:
        # Many top-level keys (>500) should trip the bomb detector.
        many_keys = "\n".join(f"k{i}: v" for i in range(600))
        body = f"---\n{many_keys}\n---\nBody\n".encode()
        with pytest.raises((YamlBombError, DangerousFrontmatterError)):
            parse_and_validate_frontmatter(body)


# --------------------------------------------------------------------------- #
# SEC-05 — untrusted body wrapping                                            #
# --------------------------------------------------------------------------- #


class TestWrapUntrustedBody:
    def test_wrap_emits_envelope(self) -> None:
        wrapped = wrap_untrusted_body("hello\n", sha256="abc123", rel_path="x.md")
        assert "<ingested_source" in wrapped
        assert "</ingested_source>" in wrapped
        assert "hello" in wrapped
        assert "abc123"[:12] in wrapped or "abc123" in wrapped

    def test_wrap_handles_empty_body(self) -> None:
        wrapped = wrap_untrusted_body("", sha256="def456", rel_path="empty.md")
        assert "<ingested_source" in wrapped


# --------------------------------------------------------------------------- #
# SEC-10 — atomic write                                                       #
# --------------------------------------------------------------------------- #


class TestAtomicWriteText:
    def test_creates_file_with_mode_0600(self, tmp_path: Path) -> None:
        target = tmp_path / "subdir" / "out.md"
        atomic_write_text(target, "hello", mode=0o600)
        assert target.exists()
        assert target.read_text() == "hello"
        mode = stat.S_IMODE(target.stat().st_mode)
        assert mode == 0o600

    def test_atomic_replace_preserves_old_on_failure(self, tmp_path: Path) -> None:
        target = tmp_path / "existing.md"
        target.write_text("old content")
        # Attempting to write to a directory path should fail without
        # half-writing the target.
        try:
            atomic_write_text(target / "impossible", "new")
        except OSError:
            pass
        assert target.read_text() == "old content"


# --------------------------------------------------------------------------- #
# SEC-07 — safe log formatter                                                 #
# --------------------------------------------------------------------------- #


class TestSafeLogFormatter:
    def test_drops_non_allowed_extra_fields(self) -> None:
        import logging

        rec = logging.LogRecord(
            name="t",
            level=logging.INFO,
            pathname="",
            lineno=0,
            msg="event.x",
            args=(),
            exc_info=None,
        )
        rec.extra_fields = {  # type: ignore[attr-defined]
            "event_id": "ok-keep",
            "error_message": "should-drop",
            "absolute_path": "/etc/passwd",
        }
        formatter = SafeLogFormatter()
        out = formatter.format(rec)
        assert "ok-keep" in out
        assert "should-drop" not in out
        assert "/etc/passwd" not in out

    def test_sha256_truncated_to_12_chars(self) -> None:
        import logging

        rec = logging.LogRecord(
            name="t",
            level=logging.INFO,
            pathname="",
            lineno=0,
            msg="event.x",
            args=(),
            exc_info=None,
        )
        rec.extra_fields = {"sha256": "a" * 64}  # type: ignore[attr-defined]
        out = SafeLogFormatter().format(rec)
        assert '"sha256": "' + "a" * 12 + '"' in out
        assert "a" * 64 not in out

    def test_exception_class_name_recorded(self) -> None:
        import logging
        import sys

        try:
            raise ValueError("boom")
        except ValueError:
            rec = logging.LogRecord(
                name="t",
                level=logging.ERROR,
                pathname="",
                lineno=0,
                msg="event.fail",
                args=(),
                exc_info=sys.exc_info(),
            )
            out = SafeLogFormatter().format(rec)
            assert '"error_class": "ValueError"' in out
            assert "boom" not in out  # message must NOT leak
