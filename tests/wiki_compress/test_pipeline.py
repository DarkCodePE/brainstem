"""
End-to-end tests for `wiki_compress.pipeline`.

Includes grapheme-safety verification across the full default pipeline
(PRD-007 AC-4 hard guarantee: zero CJK / emoji corruption).

Coverage matrix:

| Behaviour                                             | Test                                |
| ----------------------------------------------------- | ----------------------------------- |
| Default pipeline reduces typical HTML payload         | test_default_pipeline_reduces_html  |
| Empty input → empty result, ratio=1.0                 | test_empty_input                    |
| Plain markdown passes through with minor changes      | test_markdown_round_trip            |
| Idempotency: compress(compress(x)) does not over-cut  | test_idempotency                    |
| Stages applied list mirrors construction              | test_stages_applied                 |
| Metrics: per-stage timing recorded                    | test_metrics_recorded               |
| Metrics: total_elapsed_ms ≥ sum of stages             | test_metrics_total_consistent       |
| Disabling metrics yields ``None``                     | test_metrics_off                    |
| URL map populated for long URLs                       | test_url_map_populated              |
| Custom pipeline: drop a stage                         | test_custom_pipeline_no_dedupe      |
| CJK preserved through full pipeline                   | test_cjk_preserved                  |
| Emoji + ZWJ preserved through full pipeline           | test_emoji_zwj_preserved            |
| Mixed multi-byte preserved                            | test_mixed_multibyte_preserved      |
| Code block survives end-to-end                        | test_code_block_round_trip          |
"""

from __future__ import annotations

import pytest

from wiki_compress import (
    CompressionPipeline,
    CompressionResult,
    build_default_pipeline,
)
from wiki_compress.stages import UrlShortener, html_to_markdown


@pytest.fixture
def html_payload() -> str:
    return (
        "<html><head><style>.x{color:red}</style></head><body>"
        "<h1>Article title</h1>"
        "<p>This is a substantive paragraph of content that talks about a topic "
        "in enough depth that compressing repeated copies saves real tokens.</p>"
        "<p>This is a substantive paragraph of content that talks about a topic "
        "in enough depth that compressing repeated copies saves real tokens.</p>"
        '<p>Reference: <a href="https://example.com/very/long/path?utm_source=foo'
        '&utm_medium=bar&utm_campaign=baz&id=12345&fbclid=xyz">link</a></p>'
        "<ul><li>First bullet</li><li>Second bullet</li></ul>"
        "<pre><code>print('preserve me')</code></pre>"
        "</body></html>"
    )


class TestDefaultPipeline:
    def test_default_pipeline_reduces_html(self, html_payload: str) -> None:
        pipe, _ = build_default_pipeline()
        result = pipe.compress(html_payload)
        assert isinstance(result, CompressionResult)
        assert result.ratio < 1.0
        assert result.compressed_tokens < result.original_tokens
        assert "Article title" in result.body

    def test_url_map_populated(self, html_payload: str) -> None:
        pipe, shortener = build_default_pipeline()
        pipe.compress(html_payload)
        # The hand-crafted long URL must have ended up in the map.
        assert any("utm_source" in url for url in shortener.url_map.values())

    def test_code_block_round_trip(self, html_payload: str) -> None:
        pipe, _ = build_default_pipeline()
        result = pipe.compress(html_payload)
        assert "print('preserve me')" in result.body

    def test_stages_applied(self) -> None:
        pipe, _ = build_default_pipeline()
        result = pipe.compress("hello")
        assert result.stages_applied == [
            "preserve",
            "html_to_md",
            "url_shorten",
            "whitespace",
            "dedupe",
            "release",
        ]


class TestEmpty:
    def test_empty_input(self) -> None:
        pipe, _ = build_default_pipeline()
        result = pipe.compress("")
        assert result.body == ""
        assert result.original_tokens == 0
        assert result.compressed_tokens == 0
        assert result.ratio == 1.0
        assert result.stages_applied == []


class TestPlainText:
    def test_markdown_round_trip(self) -> None:
        md = "# Heading\n\nSome paragraph.\n\nAnother."
        pipe, _ = build_default_pipeline()
        result = pipe.compress(md)
        # Plain markdown should still contain its content.
        assert "Heading" in result.body
        assert "Some paragraph." in result.body
        assert "Another." in result.body


class TestIdempotency:
    def test_idempotency(self, html_payload: str) -> None:
        pipe, _ = build_default_pipeline()
        once = pipe.compress(html_payload)
        pipe2, _ = build_default_pipeline()
        twice = pipe2.compress(once.body)
        # Second run on already-compressed text must not lose more than
        # a token-rounding amount.
        # Allow a small slack for the 4-char heuristic boundary.
        assert twice.compressed_tokens >= once.compressed_tokens - 1


class TestMetrics:
    def test_metrics_recorded(self, html_payload: str) -> None:
        pipe, _ = build_default_pipeline()
        result = pipe.compress(html_payload)
        assert result.metrics is not None
        assert len(result.metrics.stages) == 6
        for delta in result.metrics.stages:
            assert delta.elapsed_ms >= 0.0
            assert delta.tokens_before >= 1
            assert delta.tokens_after >= 1

    def test_metrics_total_consistent(self, html_payload: str) -> None:
        pipe, _ = build_default_pipeline()
        result = pipe.compress(html_payload)
        assert result.metrics is not None
        sum_per_stage = sum(d.elapsed_ms for d in result.metrics.stages)
        # Floating-point slack: total should be within 1e-6 of the sum.
        assert abs(result.metrics.total_elapsed_ms - sum_per_stage) < 1e-6

    def test_metrics_off(self) -> None:
        pipe, _ = build_default_pipeline()
        result = pipe.compress("hello world", with_metrics=False)
        assert result.metrics is None


class TestCustom:
    def test_custom_pipeline_no_dedupe(self) -> None:
        """Construct a custom pipeline that omits dedupe — duplicates should
        survive."""
        shortener = UrlShortener()
        pipe = CompressionPipeline(
            stages=[
                ("html_to_md", html_to_markdown),
                ("url_shorten", shortener),
            ]
        )
        text = (
            "Repeated long enough paragraph to qualify for the dedupe floor.\n\n"
            "Repeated long enough paragraph to qualify for the dedupe floor."
        )
        result = pipe.compress(text)
        assert result.body.count("Repeated long enough paragraph") == 2


class TestGraphemeSafety:
    """PRD-007 AC-4 hard guarantee: zero grapheme corruption."""

    def test_cjk_preserved(self) -> None:
        # CJK characters — both simplified and traditional, with punctuation.
        text = "中文测试段落,这段文字必须完整通过整条管道。\n\n繁體中文也要保留:你好世界,測試完成。"
        pipe, _ = build_default_pipeline()
        result = pipe.compress(text)
        for ch in "中文测试段落":
            assert ch in result.body
        for ch in "繁體中文":
            assert ch in result.body

    def test_emoji_zwj_preserved(self) -> None:
        # ZWJ sequences must not be split (U+200D is the Zero-Width Joiner).
        text = (
            "Family ZWJ sequence: 👨‍👩‍👧‍👦 — should arrive intact.\n\n"
            "Rainbow flag (also a ZWJ sequence): 🏳️‍🌈"
        )
        pipe, _ = build_default_pipeline()
        result = pipe.compress(text)
        assert "👨‍👩‍👧‍👦" in result.body
        assert "🏳️‍🌈" in result.body

    def test_mixed_multibyte_preserved(self) -> None:
        text = (
            "Mixed: 你好🌟 hello مرحبا שלום नमस्ते ☃️ — every grapheme cluster "
            "is one logical unit even though the byte counts differ."
        )
        pipe, _ = build_default_pipeline()
        result = pipe.compress(text)
        for fragment in ["你好", "🌟", "مرحبا", "שלום", "नमस्ते", "☃"]:
            assert fragment in result.body

    def test_long_cjk_payload_ratio_sane(self) -> None:
        """A CJK-heavy payload should still produce a sane ratio (no infinite
        loops, no exceptions, output non-empty)."""
        text = "这是中文段落,长度足够触发去重。" * 50
        pipe, _ = build_default_pipeline()
        result = pipe.compress(text)
        assert result.body  # non-empty
        assert result.ratio <= 1.0
