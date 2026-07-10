"""
Tests for ``wiki_compress.stages.email_quotes``.

Coverage matrix:

| Behaviour                                              | Test                                  |
| ------------------------------------------------------ | ------------------------------------- |
| Single quoted block collapses to ``[quoted:hash8]``    | test_single_quoted_block_collapses    |
| Multiple distinct blocks get unique hashes             | test_multiple_blocks_unique_hashes    |
| Repeated identical block reuses the same hash          | test_identical_blocks_share_hash      |
| Nested ``>>`` quotes also collapse                     | test_nested_quotes_collapse           |
| Visible text is not touched                            | test_visible_text_preserved           |
| Empty input is a no-op                                 | test_empty_input_noop                 |
| CJK in visible text preserved                          | test_cjk_visible_preserved            |
| CJK inside quoted block preserved in the side map      | test_cjk_inside_quote_preserved       |
| Idempotency (function form)                            | test_function_idempotent              |
| Idempotency (class form)                               | test_class_idempotent                 |
| Stateful class accumulates across runs                 | test_class_accumulates_across_runs    |
| ``reset()`` clears the class state                     | test_class_reset_clears               |
| Protected code spans with ``>`` lines not collapsed    | test_protected_code_block_preserved   |
"""

from __future__ import annotations

import hashlib

from wiki_compress.stages.email_quotes import (
    LAST_RUN_QUOTE_MAP,
    QuoteCollapser,
    collapse_email_quotes,
)
from wiki_compress.stages.preserve import mask_code_blocks, release_preserved


def _hash8(text: str) -> str:
    return hashlib.sha1(text.encode("utf-8")).hexdigest()[:8]


class TestSingleBlock:
    def test_single_quoted_block_collapses(self) -> None:
        text = (
            "Reply body here.\n"
            "> Earlier message line one.\n"
            "> Earlier message line two.\n"
            "Final sentence."
        )
        out = collapse_email_quotes(text)
        quoted = "> Earlier message line one.\n> Earlier message line two."
        marker = f"[quoted:{_hash8(quoted)}]"
        assert marker in out
        assert "Earlier message line one" not in out
        assert "Reply body here." in out
        assert "Final sentence." in out
        assert LAST_RUN_QUOTE_MAP[_hash8(quoted)] == quoted


class TestMultipleBlocks:
    def test_multiple_blocks_unique_hashes(self) -> None:
        text = (
            "Top reply.\n"
            "> Block A line.\n"
            "> Block A line two.\n"
            "Middle interjection.\n"
            "> Different block B.\n"
            "> B line two.\n"
            "Tail."
        )
        out = collapse_email_quotes(text)
        marker_count = out.count("[quoted:")
        assert marker_count == 2
        # Two distinct hashes — pull them out and confirm they differ.
        hashes = [chunk for chunk in LAST_RUN_QUOTE_MAP.keys()]
        assert len(hashes) == 2
        assert len(set(hashes)) == 2

    def test_identical_blocks_share_hash(self) -> None:
        block = "> Repeated quoted line one.\n> Repeated quoted line two."
        text = f"Intro one.\n{block}\nIntro two.\n{block}\nDone."
        out = collapse_email_quotes(text)
        digest = _hash8(block)
        marker = f"[quoted:{digest}]"
        # Both occurrences collapse to the same marker.
        assert out.count(marker) == 2
        # Only one entry in the map — same hash → same key.
        assert len(LAST_RUN_QUOTE_MAP) == 1


class TestNested:
    def test_nested_quotes_collapse(self) -> None:
        text = "Outer reply.\n>> Deep level two.\n> Level one.\nTail."
        out = collapse_email_quotes(text)
        assert "[quoted:" in out
        assert ">> Deep level two." not in out


class TestVisibleText:
    def test_visible_text_preserved(self) -> None:
        text = (
            "Hello, this is the real body.\n"
            "It has multiple lines of substance.\n"
            "> But also a quoted reply chunk\n"
            "> that should disappear.\n"
            "Closing line."
        )
        out = collapse_email_quotes(text)
        assert "Hello, this is the real body." in out
        assert "It has multiple lines of substance." in out
        assert "Closing line." in out
        # Quoted body removed.
        assert "But also a quoted reply chunk" not in out


class TestEmpty:
    def test_empty_input_noop(self) -> None:
        assert collapse_email_quotes("") == ""
        assert LAST_RUN_QUOTE_MAP == {}

    def test_text_without_quotes_is_passthrough(self) -> None:
        text = "Plain prose with no quoted lines at all.\nSecond plain line."
        out = collapse_email_quotes(text)
        assert out == text


class TestCJK:
    def test_cjk_visible_preserved(self) -> None:
        text = (
            "你好世界,这是回复的正文。\n> 这是上一封信的引用块,应当折叠。\n> 第二行引用。\n结束。"
        )
        out = collapse_email_quotes(text)
        assert "你好世界,这是回复的正文。" in out
        assert "结束。" in out
        assert "这是上一封信的引用块" not in out

    def test_cjk_inside_quote_preserved(self) -> None:
        quoted = "> 中文引用第一行\n> 中文引用第二行"
        text = f"开场白\n{quoted}\n收尾"
        collapse_email_quotes(text)
        # The full CJK quote is preserved in the side-channel map.
        assert LAST_RUN_QUOTE_MAP[_hash8(quoted)] == quoted


class TestIdempotency:
    def test_function_idempotent(self) -> None:
        text = "Body line.\n> Quoted line one.\n> Quoted line two.\nClosing."
        once = collapse_email_quotes(text)
        twice = collapse_email_quotes(once)
        assert once == twice

    def test_class_idempotent(self) -> None:
        collapser = QuoteCollapser()
        text = "Body line.\n> Quote me.\n> Quote me too.\nEnd."
        once = collapser(text)
        twice = collapser(once)
        # Second run doesn't add a new mapping entry.
        assert once == twice


class TestStatefulCollapser:
    def test_class_accumulates_across_runs(self) -> None:
        collapser = QuoteCollapser()
        collapser("body1\n> chunk-a line\n> chunk-a more\ntail")
        collapser("body2\n> chunk-b different\n> chunk-b also\ntail")
        # Two distinct quoted blocks → two entries.
        assert len(collapser.quote_map) == 2

    def test_class_reset_clears(self) -> None:
        collapser = QuoteCollapser()
        collapser("body\n> q1\n> q2\nend")
        assert collapser.quote_map
        collapser.reset()
        assert collapser.quote_map == {}


class TestProtectedSpans:
    def test_protected_code_block_preserved(self) -> None:
        # A fenced code block that contains a `> ` line must not collapse:
        # mask_code_blocks wraps it before this stage runs.
        code_block = "```\n> not actually a quote, this is code\n> still code\n```"
        text = (
            "Real intro paragraph.\n"
            f"{code_block}\n"
            "> Actual quoted reply line.\n"
            "> Another quoted line.\n"
            "Trailing prose."
        )
        wrapped, _ = mask_code_blocks(text)
        collapsed = collapse_email_quotes(wrapped)
        released = release_preserved(collapsed)
        # The code block content survives verbatim.
        assert "> not actually a quote, this is code" in released
        assert "> still code" in released
        # The real quoted block was collapsed.
        assert "Actual quoted reply line." not in released
        assert "[quoted:" in released
