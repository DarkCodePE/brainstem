"""
Exhaustive tests for `wiki_memory.chunker` — deterministic ≤3k-token chunking
per [PRD-004 FR-1](../../docs/PRD-004-memory-tree.md).

Coverage matrix:

| Behaviour                                          | Test                                                    |
| -------------------------------------------------- | ------------------------------------------------------- |
| Empty / whitespace input                           | test_empty_text_yields_no_chunks, test_whitespace_only  |
| Single short paragraph                             | test_single_short_paragraph_one_chunk                   |
| Multiple paragraphs packed under target            | test_multiple_paragraphs_pack_to_one_chunk              |
| Paragraph boundary causes split                    | test_oversized_buffer_splits                            |
| Paragraph > hard cap → mid-paragraph split         | test_huge_paragraph_split_at_sentence                   |
| Sentence > hard cap → whitespace split             | test_huge_run_on_sentence_split_at_whitespace           |
| Identical input → identical chunks (idempotent)    | test_identical_input_identical_sha                      |
| Different input → different sha                    | test_different_input_different_sha                      |
| Chunks indexed monotonically from 0                | test_chunk_indices_are_monotonic                        |
| token_count is approximate but reasonable          | test_token_count_correlates_with_length                 |
| Unicode preserved (CJK, emoji)                     | test_unicode_preserved                                  |
| count_tokens(empty) returns 1 (floor)              | test_count_tokens_floor                                 |
"""

from __future__ import annotations

import pytest

from wiki_memory.chunker import (
    DEFAULT_HARD_CAP_TOKENS,
    DEFAULT_TARGET_TOKENS,
    PREAMBLE_HEADING,
    chunk_page,
    chunk_text,
    count_tokens,
)


class TestEmpty:
    def test_empty_text_yields_no_chunks(self) -> None:
        assert chunk_text("") == []

    def test_whitespace_only(self) -> None:
        assert chunk_text("   \n\n\t  \n  ") == []


class TestSingleParagraph:
    def test_single_short_paragraph_one_chunk(self) -> None:
        chunks = chunk_text("Hello world.")
        assert len(chunks) == 1
        assert chunks[0].chunk_index == 0
        assert chunks[0].body == "Hello world."

    def test_multiple_paragraphs_pack_to_one_chunk(self) -> None:
        text = "Para one.\n\nPara two.\n\nPara three."
        chunks = chunk_text(text)
        # Far under the 2500-token target → all in one chunk.
        assert len(chunks) == 1
        assert "Para one" in chunks[0].body
        assert "Para three" in chunks[0].body


class TestPacking:
    def test_oversized_buffer_splits(self) -> None:
        # Two distinct paragraphs each near the soft target → two chunks.
        # Use different content so the shas differ (chunker is content-keyed
        # by design — identical content produces identical sha).
        para_a = "a" * (DEFAULT_TARGET_TOKENS * 3) + "."
        para_b = "b" * (DEFAULT_TARGET_TOKENS * 3) + "."
        chunks = chunk_text(f"{para_a}\n\n{para_b}")
        assert len(chunks) == 2
        assert chunks[0].sha256 != chunks[1].sha256

    def test_oversized_identical_paragraphs_dedupe_shas(self) -> None:
        # Same content twice → two chunks but with identical shas (the
        # content_store will then skip the duplicate on insert).
        para = "x" * (DEFAULT_TARGET_TOKENS * 3) + "."
        chunks = chunk_text(f"{para}\n\n{para}")
        assert len(chunks) == 2
        assert chunks[0].sha256 == chunks[1].sha256

    def test_huge_paragraph_split_at_sentence(self) -> None:
        # One paragraph at 4x hard cap with sentence boundaries.
        sentence = ("word " * 600).strip() + "."  # ~3000 chars
        para = " ".join(sentence for _ in range(4))
        chunks = chunk_text(para)
        assert len(chunks) >= 2
        for c in chunks:
            # Loose check: each chunk under the hard cap.
            assert c.token_count <= DEFAULT_HARD_CAP_TOKENS

    def test_huge_run_on_sentence_split_at_whitespace(self) -> None:
        # One sentence (no boundary chars) at 2x hard cap.
        long_run = "a " * (DEFAULT_HARD_CAP_TOKENS * 8)
        chunks = chunk_text(long_run)
        assert len(chunks) >= 2
        for c in chunks:
            assert c.token_count <= DEFAULT_HARD_CAP_TOKENS


class TestIdempotency:
    def test_identical_input_identical_sha(self) -> None:
        text = "Some paragraph.\n\nAnother paragraph."
        a = chunk_text(text)
        b = chunk_text(text)
        assert [c.sha256 for c in a] == [c.sha256 for c in b]

    def test_different_input_different_sha(self) -> None:
        a = chunk_text("Text A")
        b = chunk_text("Text B")
        assert a[0].sha256 != b[0].sha256

    def test_chunk_indices_are_monotonic(self) -> None:
        para = "x" * (DEFAULT_TARGET_TOKENS * 3) + "."
        text = "\n\n".join(para for _ in range(5))
        chunks = chunk_text(text)
        for i, c in enumerate(chunks):
            assert c.chunk_index == i


class TestTokenCount:
    @pytest.mark.parametrize(
        "text,expected_min,expected_max",
        [
            ("hi", 1, 1),
            ("hello world", 2, 3),  # 11 chars / 4 = 2 (floored)
            ("x" * 4000, 1000, 1000),
            ("", 1, 1),  # floored to 1
        ],
    )
    def test_count_tokens_floor(self, text: str, expected_min: int, expected_max: int) -> None:
        assert expected_min <= count_tokens(text) <= expected_max

    def test_token_count_correlates_with_length(self) -> None:
        long_chunk = chunk_text("x" * 2000)[0]
        short_chunk = chunk_text("x" * 200)[0]
        assert long_chunk.token_count > short_chunk.token_count


class TestChunkPage:
    """ADR-036 D4: `## For future Claude` preamble isolated as its own chunk."""

    PAGE = (
        '---\ntitle: "T"\ndate: 2026-06-14\nsources: ["raw/x.md"]\n'
        'tags: ["ingested"]\norigin: llm-synthesized\ncategory: sources\n'
        "source_count: 1\n---\n\n# T\n\n## For future Claude\n\n"
        "This is the relevance note for a future AI reader.\n\n"
        "The body summary goes here with [[Some Entity]] linked.\n"
    )

    def test_preamble_isolated_as_own_chunk(self) -> None:
        chunks = chunk_page(self.PAGE)
        preamble = [c for c in chunks if c.body.startswith(PREAMBLE_HEADING)]
        assert len(preamble) == 1
        assert preamble[0].body == (
            "## For future Claude\n\nThis is the relevance note for a future AI reader."
        )
        # the body summary is NOT diluting the preamble's embedding target
        assert "body summary" not in preamble[0].body
        # nor is the YAML frontmatter
        assert "origin:" not in preamble[0].body

    def test_relevance_appears_in_exactly_one_chunk(self) -> None:
        bodies = [c.body for c in chunk_page(self.PAGE)]
        assert sum("relevance note for a future AI reader" in b for b in bodies) == 1

    def test_body_chunk_is_clean_of_preamble(self) -> None:
        chunks = chunk_page(self.PAGE)
        body_chunks = [c for c in chunks if "body summary goes here" in c.body]
        assert len(body_chunks) == 1
        assert PREAMBLE_HEADING not in body_chunks[0].body

    def test_no_preamble_falls_back_to_chunk_text(self) -> None:
        text = "Para one.\n\nPara two.\n\nPara three."
        assert [c.body for c in chunk_page(text)] == [c.body for c in chunk_text(text)]

    def test_indices_monotonic_and_sha_stable(self) -> None:
        a = chunk_page(self.PAGE)
        b = chunk_page(self.PAGE)
        assert [c.chunk_index for c in a] == list(range(len(a)))
        assert [c.sha256 for c in a] == [c.sha256 for c in b]


class TestUnicode:
    def test_unicode_preserved(self) -> None:
        text = "日本語のテキスト。\n\nWith emoji 😀 and 中文."
        chunks = chunk_text(text)
        assert len(chunks) == 1
        assert "日本語" in chunks[0].body
        assert "😀" in chunks[0].body
        assert "中文" in chunks[0].body
