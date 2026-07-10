"""
Tests for `wiki_compress.stages.preserve` — sentinel-based span protection.

Coverage matrix:

| Behaviour                                          | Test                                        |
| -------------------------------------------------- | ------------------------------------------- |
| Wrap simple span; release restores it              | test_round_trip_simple                      |
| Wrap multiple spans, distinct ids                  | test_wrap_multiple_spans_distinct_ids       |
| Code blocks survive a full pipeline (fenced)       | test_fenced_code_block_survives_pipeline    |
| Inline code survives                               | test_inline_code_survives_pipeline          |
| Idempotent: mask twice is harmless                 | test_mask_then_mask_no_double_wrap          |
| iter_unprotected yields both kinds                 | test_iter_unprotected_yields_both           |
| apply_to_unprotected does not touch protected      | test_apply_to_unprotected_skips_protected   |
| CJK / emoji inside a protected span survive        | test_protected_span_keeps_cjk_emoji         |
| release on text without sentinels is identity      | test_release_on_plain_text_is_identity      |
"""

from __future__ import annotations

import re

from wiki_compress import build_default_pipeline
from wiki_compress.stages.preserve import (
    apply_to_unprotected,
    has_protected_spans,
    iter_unprotected,
    mask_code_blocks,
    preserve_spans,
    release_preserved,
)


class TestRoundTrip:
    def test_round_trip_simple(self) -> None:
        wrapped, _ = preserve_spans("hello WORLD bye", re.compile(r"WORLD"))
        assert "WORLD" in wrapped
        assert has_protected_spans(wrapped)
        assert release_preserved(wrapped) == "hello WORLD bye"

    def test_release_on_plain_text_is_identity(self) -> None:
        assert release_preserved("no sentinels here") == "no sentinels here"


class TestMultipleSpans:
    def test_wrap_multiple_spans_distinct_ids(self) -> None:
        wrapped, next_id = preserve_spans("a X b X c", re.compile(r"X"))
        assert next_id == 2
        assert release_preserved(wrapped) == "a X b X c"

    def test_iter_unprotected_yields_both(self) -> None:
        wrapped, _ = preserve_spans("AAA SECRET BBB", re.compile(r"SECRET"))
        items = list(iter_unprotected(wrapped))
        kinds = [is_p for is_p, _ in items]
        assert kinds == [False, True, False]
        # The unprotected halves must contain the surrounding text.
        assert items[0][1] == "AAA "
        assert items[2][1] == " BBB"


class TestApplyToUnprotected:
    def test_apply_to_unprotected_skips_protected(self) -> None:
        wrapped, _ = preserve_spans("hello DONTOUCH world", re.compile(r"DONTOUCH"))
        out = apply_to_unprotected(wrapped, str.upper)
        # The 'DONTOUCH' portion stays as the original — case preserved —
        # and surrounding text is uppercased.
        assert "DONTOUCH" in out
        assert "HELLO" in out
        assert "WORLD" in out
        # 'hello' and 'world' should NOT survive in lower-case:
        assert "hello" not in out
        assert "world" not in out


class TestCodeBlocks:
    def test_fenced_code_block_survives_pipeline(self) -> None:
        text = (
            "Some intro paragraph that is long enough to be worth keeping around.\n\n"
            "```python\n"
            "def hello():\n"
            "    return 'world'   # multiple   spaces  on   purpose\n"
            "```\n\n"
            "Trailing paragraph."
        )
        pipe, _ = build_default_pipeline()
        result = pipe.compress(text)
        # The exact code block body should appear in the compressed output.
        assert "def hello():" in result.body
        assert "return 'world'   # multiple   spaces  on   purpose" in result.body

    def test_inline_code_survives_pipeline(self) -> None:
        text = "Run `git status --porcelain` to see  the   state."
        pipe, _ = build_default_pipeline()
        result = pipe.compress(text)
        assert "`git status --porcelain`" in result.body

    def test_mask_then_mask_no_double_wrap(self) -> None:
        text = "intro\n\n```\ncode\n```\noutro"
        once, _ = mask_code_blocks(text)
        twice, _ = mask_code_blocks(once)
        # The second pass must not re-wrap the code block.
        assert release_preserved(twice) == text


class TestUnicode:
    def test_protected_span_keeps_cjk_emoji(self) -> None:
        text = "intro\n\n```\n你好 🌟 ZWJ 👨‍👩‍👧\n```\noutro"
        wrapped, _ = mask_code_blocks(text)
        released = release_preserved(wrapped)
        assert "你好 🌟 ZWJ 👨‍👩‍👧" in released
        assert released == text

    def test_protected_span_keeps_emoji_through_default_pipeline(self) -> None:
        text = "before\n\n```\n你好 🌟 ZWJ 👨‍👩‍👧 — 100% preserved\n```\nafter"
        pipe, _ = build_default_pipeline()
        result = pipe.compress(text)
        assert "你好 🌟 ZWJ 👨‍👩‍👧 — 100% preserved" in result.body
