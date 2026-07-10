"""Template rendering matches the wiki frontmatter contract."""

from __future__ import annotations

import re

import yaml

from wiki_synthesis.templates import (
    render_concept_page,
    render_entity_page,
    render_source_page,
    slugify,
    wikilink_terms,
)

REQUIRED_KEYS = {"type", "title", "date", "sources", "tags", "origin"}


def parse_frontmatter(page: str) -> dict:
    match = re.match(r"\A---\s*\n(.*?)\n---\s*\n", page, re.DOTALL)
    assert match, "page must start with a YAML frontmatter block"
    return yaml.safe_load(match.group(1))


class TestOKFTypeBornConformant:
    """ADR-045: synthesis pages carry the OKF `type` without relying on the
    write_page injection (covers writers that bypass write_page)."""

    def test_source_page_type(self) -> None:
        page = render_source_page(
            title="T",
            date="2026-06-20",
            sources=["raw/x.md"],
            tags=["t"],
            origin="llm-synthesized",
            body="B",
        )
        assert parse_frontmatter(page)["type"] == "Source"

    def test_entity_page_type(self) -> None:
        page = render_entity_page(
            name="N",
            date="2026-06-20",
            source_page_path="sources/x.md",
            mention="M",
            origin="llm-synthesized",
        )
        assert parse_frontmatter(page)["type"] == "Entity"

    def test_concept_page_type(self) -> None:
        page = render_concept_page(
            name="N",
            date="2026-06-20",
            source_page_path="sources/x.md",
            mention="M",
            origin="llm-synthesized",
        )
        assert parse_frontmatter(page)["type"] == "Concept"


class TestSourcePage:
    def make(self) -> str:
        return render_source_page(
            title='My "quoted" Source',
            date="2026-06-10",
            sources=["https://example.com/a", "raw/articles/My Article (1).md"],
            tags=["ingested", "articles"],
            origin="llm-synthesized",
            body="Body with ![img](https://x/y.png) and ![[embed.png]].",
        )

    def test_required_frontmatter_keys(self) -> None:
        fm = parse_frontmatter(self.make())
        assert REQUIRED_KEYS <= set(fm)
        assert fm["category"] == "sources"
        assert fm["source_count"] == 1

    def test_sources_preserve_url_and_raw_path(self) -> None:
        fm = parse_frontmatter(self.make())
        assert "https://example.com/a" in fm["sources"]
        assert "raw/articles/My Article (1).md" in fm["sources"]

    def test_image_refs_survive_rendering(self) -> None:
        page = self.make()
        assert "![img](https://x/y.png)" in page
        assert "![[embed.png]]" in page

    def test_title_quoting_round_trips(self) -> None:
        fm = parse_frontmatter(self.make())
        assert fm["title"] == 'My "quoted" Source'

    def test_relevance_renders_for_future_claude_section(self) -> None:
        page = render_source_page(
            title="T",
            date="2026-06-14",
            sources=["raw/x.md"],
            tags=["ingested"],
            origin="llm-synthesized",
            body="Body here.",
            relevance="Why this matters to a future reader.",
        )
        assert "## For future Claude" in page
        assert "Why this matters to a future reader." in page
        # preamble precedes the body
        assert page.index("Why this matters") < page.index("Body here.")

    def test_no_relevance_keeps_plain_shape(self) -> None:
        page = render_source_page(
            title="T",
            date="2026-06-14",
            sources=["raw/x.md"],
            tags=["ingested"],
            origin="llm-synthesized",
            body="Body here.",
        )
        assert "## For future Claude" not in page


class TestEntityConceptPages:
    def test_entity_page_contract(self) -> None:
        page = render_entity_page(
            name="Claude Code",
            date="2026-06-10",
            source_page_path="wiki/sources/x.md",
            mention="Claude Code is a CLI agent.",
            origin="synthesized-deterministic",
        )
        fm = parse_frontmatter(page)
        assert REQUIRED_KEYS <= set(fm)
        assert fm["tags"][0] == "entity"
        assert fm["category"] == "entities"
        assert fm["sources"] == ["wiki/sources/x.md"]

    def test_concept_page_contract(self) -> None:
        page = render_concept_page(
            name="event sourcing pattern",
            date="2026-06-10",
            source_page_path="wiki/sources/x.md",
            mention="",
            origin="synthesized-deterministic",
        )
        fm = parse_frontmatter(page)
        assert fm["tags"][0] == "concept"
        assert fm["category"] == "concepts"


class TestWikilinks:
    def test_links_first_occurrence_only(self) -> None:
        body = "Claude Code rocks. Claude Code again."
        out = wikilink_terms(body, ["Claude Code"])
        assert out == "[[Claude Code]] rocks. Claude Code again."

    def test_longer_terms_win(self) -> None:
        out = wikilink_terms("Claude Code rocks.", ["Claude", "Claude Code"])
        assert "[[Claude Code]]" in out
        assert "[[Claude]] Code" not in out

    def test_existing_wikilinks_untouched(self) -> None:
        body = "[[Claude Code]] is here. Claude Code again."
        out = wikilink_terms(body, ["Claude Code"])
        assert out == "[[Claude Code]] is here. [[Claude Code]] again."


class TestSlugify:
    def test_spaces_and_punctuation_collapse(self) -> None:
        assert slugify("My Article (1)") == "my-article-1"

    def test_empty_falls_back(self) -> None:
        assert slugify("***") == "untitled"
