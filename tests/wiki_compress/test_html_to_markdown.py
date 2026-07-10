"""
Tests for `wiki_compress.stages.html_to_markdown`.

Covers the common HTML shapes a web-clip / tool result will throw at us:
tables, ordered lists, nested structures, code blocks, links, scripts.

Grapheme safety lives in `test_pipeline.py::TestGraphemeSafety` — this
file focuses on shape correctness.
"""

from __future__ import annotations

from wiki_compress.stages.html_to_markdown import html_to_markdown, is_html_like


class TestDetection:
    def test_is_html_like_true_for_html(self) -> None:
        assert is_html_like("<html><body>hi</body></html>")

    def test_is_html_like_false_for_markdown(self) -> None:
        assert not is_html_like("# Heading\n\nA paragraph.")

    def test_is_html_like_false_for_plain(self) -> None:
        assert not is_html_like("plain text 你好 🌟")

    def test_html_to_markdown_passes_markdown_through(self) -> None:
        md = "# Title\n\nBody."
        assert html_to_markdown(md) == md


class TestBlockShapes:
    def test_headings_collapse_to_atx(self) -> None:
        out = html_to_markdown("<h1>One</h1><h2>Two</h2>")
        assert "# One" in out
        assert "## Two" in out

    def test_ordered_list(self) -> None:
        html = "<ol><li>first</li><li>second</li><li>third</li></ol>"
        out = html_to_markdown(html)
        assert "1. first" in out
        assert "2. second" in out
        assert "3. third" in out

    def test_unordered_list(self) -> None:
        html = "<ul><li>alpha</li><li>beta</li></ul>"
        out = html_to_markdown(html)
        assert "- alpha" in out or "* alpha" in out
        assert "- beta" in out or "* beta" in out

    def test_nested_list(self) -> None:
        html = "<ul><li>outer<ul><li>inner</li></ul></li></ul>"
        out = html_to_markdown(html)
        assert "outer" in out
        assert "inner" in out

    def test_table(self) -> None:
        html = (
            "<table>"
            "<thead><tr><th>A</th><th>B</th></tr></thead>"
            "<tbody><tr><td>1</td><td>2</td></tr></tbody>"
            "</table>"
        )
        out = html_to_markdown(html)
        # markdownify renders tables as pipe-tables; just confirm the cells.
        assert "1" in out
        assert "2" in out
        assert "A" in out
        assert "B" in out

    def test_link(self) -> None:
        out = html_to_markdown('<a href="https://example.com">click</a>')
        assert "click" in out
        assert "https://example.com" in out

    def test_code_block(self) -> None:
        out = html_to_markdown("<pre><code>print('hi')</code></pre>")
        assert "print('hi')" in out


class TestDrop:
    def test_script_dropped(self) -> None:
        out = html_to_markdown("<p>kept</p><script>alert(1)</script>")
        assert "kept" in out
        assert "alert" not in out

    def test_style_dropped(self) -> None:
        out = html_to_markdown("<style>.x{color:red}</style><p>kept</p>")
        assert "kept" in out
        assert ".x" not in out
        assert "color" not in out

    def test_noscript_dropped(self) -> None:
        out = html_to_markdown("<noscript>enable js</noscript><p>kept</p>")
        assert "kept" in out
        assert "enable js" not in out

    def test_html_comment_dropped(self) -> None:
        out = html_to_markdown("<!-- hide me --><p>kept</p>")
        assert "kept" in out
        assert "hide me" not in out


class TestSize:
    def test_html_is_smaller_than_input(self) -> None:
        html = (
            "<html><head><title>T</title></head>"
            "<body>"
            "<h1>Header</h1>"
            "<p>One paragraph with <strong>bold</strong> text.</p>"
            "<p>Another paragraph with <em>emphasis</em>.</p>"
            "</body></html>"
        )
        out = html_to_markdown(html)
        assert len(out) < len(html)
