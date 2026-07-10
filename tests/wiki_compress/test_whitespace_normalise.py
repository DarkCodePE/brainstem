"""
Tests for `wiki_compress.stages.whitespace_normalise`.
"""

from __future__ import annotations

from wiki_compress.stages.preserve import mask_code_blocks, release_preserved
from wiki_compress.stages.whitespace_normalise import normalise_whitespace


class TestLineEndings:
    def test_crlf_to_lf(self) -> None:
        assert normalise_whitespace("a\r\nb\r\nc") == "a\nb\nc"

    def test_cr_to_lf(self) -> None:
        assert normalise_whitespace("a\rb\rc") == "a\nb\nc"

    def test_mixed_line_endings(self) -> None:
        assert normalise_whitespace("a\r\nb\rc\nd") == "a\nb\nc\nd"


class TestTrailingWhitespace:
    def test_trailing_spaces_stripped(self) -> None:
        assert normalise_whitespace("hello   \nworld   ") == "hello\nworld"

    def test_trailing_tabs_stripped(self) -> None:
        # Tabs first become spaces, then trailing-ws strip removes them.
        out = normalise_whitespace("hello\t\nworld")
        assert out == "hello\nworld"


class TestTabs:
    def test_tabs_become_four_spaces(self) -> None:
        out = normalise_whitespace("a\tb")
        # The 4 spaces then get squeezed by the multi-space rule:
        # "a    b" → "a b"
        assert out == "a b"


class TestBlankLines:
    def test_three_or_more_blanks_collapse(self) -> None:
        text = "a\n\n\n\n\nb"
        assert normalise_whitespace(text) == "a\n\nb"

    def test_two_blanks_preserved(self) -> None:
        assert normalise_whitespace("a\n\nb") == "a\n\nb"


class TestMultiSpace:
    def test_three_or_more_spaces_collapse_to_one(self) -> None:
        assert normalise_whitespace("a     b") == "a b"

    def test_two_spaces_collapse(self) -> None:
        # The pattern is "  +" (two or more), so two spaces collapse too.
        assert normalise_whitespace("a  b") == "a b"


class TestProtectedSpans:
    def test_code_block_whitespace_preserved(self) -> None:
        text = "```\n    indented    code   with     spaces\n```"
        wrapped, _ = mask_code_blocks(text)
        out = normalise_whitespace(wrapped)
        released = release_preserved(out)
        assert "    indented    code   with     spaces" in released


class TestIdempotent:
    def test_running_twice_is_noop(self) -> None:
        text = "a   b\r\nc\t\td   \n\n\n\ne"
        once = normalise_whitespace(text)
        twice = normalise_whitespace(once)
        assert once == twice


class TestUnicode:
    def test_cjk_preserved(self) -> None:
        text = "你好   世界 🌟"
        out = normalise_whitespace(text)
        assert "你好" in out
        assert "世界" in out
        assert "🌟" in out

    def test_zwj_emoji_preserved(self) -> None:
        # Family emoji with ZWJ sequences — must not be broken.
        text = "Family: 👨‍👩‍👧‍👦   here"
        out = normalise_whitespace(text)
        assert "👨‍👩‍👧‍👦" in out
