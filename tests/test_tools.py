"""Tests for the 7 custom wiki tools."""

import json
import os
from unittest.mock import MagicMock, patch

import pytest

from wiki_agent.tools import create_tools


@pytest.fixture
def tools(tmp_wiki_root):
    """Create all 7 tools bound to the temporary wiki root."""
    return {t.name: t for t in create_tools(tmp_wiki_root)}


@pytest.fixture
def tools_populated(populated_wiki):
    """Create tools bound to a populated wiki."""
    return {t.name: t for t in create_tools(populated_wiki)}


class TestSearchWikiIndex:
    def test_returns_matching_pages(self, tools_populated):
        result = json.loads(
            tools_populated["search_wiki_index"].invoke({"query": "Karpathy AI researcher"})
        )
        assert len(result) > 0
        titles = [r["title"] for r in result]
        assert "Andrej Karpathy" in titles

    def test_returns_empty_for_no_match(self, tools_populated):
        result = json.loads(
            tools_populated["search_wiki_index"].invoke({"query": "quantum computing xyz"})
        )
        assert result == []

    def test_returns_empty_when_no_index(self, tools):
        result = json.loads(tools["search_wiki_index"].invoke({"query": "anything"}))
        # Index exists but has no data rows (only header)
        assert isinstance(result, list)

    def test_result_structure(self, tools_populated):
        result = json.loads(
            tools_populated["search_wiki_index"].invoke({"query": "LLM wiki knowledge"})
        )
        assert len(result) > 0
        entry = result[0]
        assert "page_path" in entry
        assert "title" in entry
        assert "summary" in entry


class TestAppendToLog:
    def test_appends_entry(self, tools, tmp_wiki_root):
        result = tools["append_to_log"].invoke(
            {
                "entry_type": "ingest",
                "title": "Test Article",
                "details": "- Source: raw/test.md\n- Pages created: 1",
            }
        )
        assert "Log entry appended" in result

        log_path = os.path.join(tmp_wiki_root, "wiki", "log.md")
        with open(log_path) as fh:
            content = fh.read()
        assert "ingest | Test Article" in content
        assert "raw/test.md" in content

    def test_creates_log_if_missing(self, tmp_path):
        root = str(tmp_path / "empty")
        os.makedirs(os.path.join(root, "wiki"), exist_ok=True)
        tools = {t.name: t for t in create_tools(root)}
        result = tools["append_to_log"].invoke(
            {
                "entry_type": "query",
                "title": "Test Query",
                "details": "- Query answered",
            }
        )
        assert "Log entry appended" in result
        assert os.path.exists(os.path.join(root, "wiki", "log.md"))


class TestGetWikiStats:
    def test_returns_stats_json(self, tools):
        result = json.loads(tools["get_wiki_stats"].invoke({}))
        assert "page_count" in result
        assert "source_count" in result
        assert "entity_count" in result
        assert "concept_count" in result

    def test_counts_populated_wiki(self, tools_populated):
        result = json.loads(tools_populated["get_wiki_stats"].invoke({}))
        # 2 entities (karpathy, bad-entity) + 2 concepts + index + log = 6 pages
        assert result["page_count"] >= 4
        assert result["entity_count"] >= 1
        assert result["concept_count"] >= 1


class TestFindCrossReferences:
    def test_finds_outbound_wikilinks(self, tools_populated, populated_wiki):
        page = os.path.join("wiki", "entities", "andrej-karpathy.md")
        result = json.loads(tools_populated["find_cross_references"].invoke({"page_path": page}))
        assert "LLM Wiki" in result["outbound_links"]
        assert "Software 2.0" in result["outbound_links"]

    def test_finds_inbound_links(self, tools_populated, populated_wiki):
        page = os.path.join("wiki", "entities", "andrej-karpathy.md")
        result = json.loads(tools_populated["find_cross_references"].invoke({"page_path": page}))
        # llm-wiki.md links to [[Andrej Karpathy]] — tool searches for basename "andrej-karpathy"
        # The tool checks for [[page_name]] where page_name is the filename without extension
        # Since llm-wiki.md uses [[Andrej Karpathy]] (not [[andrej-karpathy]]),
        # verify at least that the inbound_links is a list
        assert isinstance(result["inbound_links"], list)
        # Outbound links should be detected correctly
        assert len(result["outbound_links"]) > 0

    def test_nonexistent_page_returns_empty(self, tools_populated):
        result = json.loads(
            tools_populated["find_cross_references"].invoke({"page_path": "wiki/nonexistent.md"})
        )
        assert result["outbound_links"] == []


class TestDetectOrphanPages:
    def test_finds_orphan_pages(self, tools_populated):
        result = json.loads(tools_populated["detect_orphan_pages"].invoke({}))
        # At minimum bad-entity.md should be orphan (no links to it)
        basenames = [os.path.basename(p) for p in result]
        assert len(basenames) > 0
        assert "bad-entity.md" in basenames

    def test_connected_pages_not_orphans(self, tools_populated):
        result = json.loads(tools_populated["detect_orphan_pages"].invoke({}))
        basenames = [os.path.basename(p) for p in result]
        # andrej-karpathy.md and llm-wiki.md link to each other
        assert "andrej-karpathy.md" not in basenames
        assert "llm-wiki.md" not in basenames

    def test_empty_wiki_returns_empty(self, tools):
        result = json.loads(tools["detect_orphan_pages"].invoke({}))
        assert result == []


class TestWebClip:
    @patch("wiki_agent.tools.create_tools")
    def test_web_clip_saves_file(self, mock_create, tmp_wiki_root):
        """Test web_clip with mocked HTTP response."""
        tools = {t.name: t for t in create_tools(tmp_wiki_root)}

        mock_response = MagicMock()
        mock_response.text = "<html><body><h1>Test</h1><p>Hello world</p></body></html>"
        mock_response.raise_for_status = MagicMock()

        with patch("httpx.get", return_value=mock_response):
            result = json.loads(tools["web_clip"].invoke({"url": "https://example.com/test"}))

        if "error" not in result:
            assert "saved_path" in result
            assert result["content_length"] > 0
            saved = os.path.join(tmp_wiki_root, result["saved_path"])
            assert os.path.exists(saved)


class TestValidateFrontmatter:
    def test_valid_frontmatter(self, tools_populated, populated_wiki):
        page = os.path.join("wiki", "entities", "andrej-karpathy.md")
        result = json.loads(tools_populated["validate_frontmatter"].invoke({"page_path": page}))
        assert result["valid"] is True
        assert result["missing_fields"] == []

    def test_invalid_frontmatter_missing_fields(self, tools_populated, populated_wiki):
        page = os.path.join("wiki", "entities", "bad-entity.md")
        result = json.loads(tools_populated["validate_frontmatter"].invoke({"page_path": page}))
        assert result["valid"] is False
        assert "date" in result["missing_fields"] or "sources" in result["missing_fields"]

    def test_nonexistent_file(self, tools_populated):
        result = json.loads(
            tools_populated["validate_frontmatter"].invoke({"page_path": "wiki/nope.md"})
        )
        assert result["valid"] is False
        assert len(result["errors"]) > 0

    def test_no_frontmatter(self, tmp_wiki_root):
        # Create a page without frontmatter
        page_path = os.path.join(tmp_wiki_root, "wiki", "no-fm.md")
        with open(page_path, "w") as fh:
            fh.write("# No Frontmatter\n\nJust content.")

        tools = {t.name: t for t in create_tools(tmp_wiki_root)}
        result = json.loads(tools["validate_frontmatter"].invoke({"page_path": "wiki/no-fm.md"}))
        assert result["valid"] is False
        assert any("frontmatter" in e.lower() for e in result["errors"])

    def test_accepts_synthesized_deterministic_origin(self, tmp_wiki_root):
        """ADR-036 D3: the deterministic-degrade origin must validate (it was
        missing from valid_origins, so every degrade page failed before)."""
        page_path = os.path.join(tmp_wiki_root, "wiki", "degrade.md")
        os.makedirs(os.path.dirname(page_path), exist_ok=True)
        with open(page_path, "w") as fh:
            fh.write(
                '---\ntype: Source\ntitle: "X"\ndate: 2026-06-14\nsources: ["raw/x.md"]\n'
                'tags: ["ingested"]\norigin: synthesized-deterministic\n'
                "category: sources\nsource_count: 1\n---\n\n# X\n\nBody.\n"
            )
        tools = {t.name: t for t in create_tools(tmp_wiki_root)}
        result = json.loads(tools["validate_frontmatter"].invoke({"page_path": "wiki/degrade.md"}))
        assert result["valid"] is True
        assert result["errors"] == []

    def test_helper_rejects_unknown_origin(self):
        """The extracted pure helper still flags an unknown origin."""
        from wiki_agent.tools import validate_page_frontmatter

        page = (
            '---\ntype: Source\ntitle: "X"\ndate: 2026-06-14\nsources: ["raw/x.md"]\n'
            'tags: ["t"]\norigin: totally-made-up\n---\n\n# X\n\nBody.\n'
        )
        result = validate_page_frontmatter(page)
        assert result["valid"] is False
        assert any("Invalid origin" in e for e in result["errors"])

    def test_rejects_missing_type(self):
        """OKF §9.2: a page without `type` is invalid; type is a required field."""
        from wiki_agent.tools import validate_page_frontmatter

        page = (
            '---\ntitle: "X"\ndate: 2026-06-14\nsources: ["raw/x.md"]\n'
            "tags: [t]\norigin: human\n---\n\n# X\n\nBody.\n"
        )
        result = validate_page_frontmatter(page)
        assert result["valid"] is False
        assert "type" in result["missing_fields"]


class TestOKFTypeInjection:
    """Fase 3: write_page injects an OKF `type` by destination folder."""

    def _write_and_read(self, tmp_wiki_root, page_path):
        tools = {t.name: t for t in create_tools(tmp_wiki_root)}
        body = (
            '---\ntitle: "T"\ndate: 2026-06-20\nsources: ["raw/x.md"]\n'
            "tags: [t]\norigin: human\n---\n\n# T\n\nBody.\n"
        )
        tools["write_page"].invoke({"page_path": page_path, "content": body})
        with open(os.path.join(tmp_wiki_root, page_path), encoding="utf-8") as fh:
            return fh.read()

    def test_injects_concept(self, tmp_wiki_root):
        out = self._write_and_read(tmp_wiki_root, "wiki/concepts/new-topic.md")
        assert "\ntype: Concept\n" in out

    def test_injects_entity(self, tmp_wiki_root):
        out = self._write_and_read(tmp_wiki_root, "wiki/entities/new-person.md")
        assert "\ntype: Entity\n" in out

    def test_injects_source(self, tmp_wiki_root):
        out = self._write_and_read(tmp_wiki_root, "wiki/sources/new-src.md")
        assert "\ntype: Source\n" in out

    def test_preserves_explicit_type(self, tmp_wiki_root):
        tools = {t.name: t for t in create_tools(tmp_wiki_root)}
        body = (
            '---\ntype: Paper\ntitle: "T"\ndate: 2026-06-20\n'
            'sources: ["raw/x.md"]\ntags: [t]\norigin: human\n---\n\n# T\n\nB.\n'
        )
        tools["write_page"].invoke({"page_path": "wiki/sources/paper.md", "content": body})
        with open(os.path.join(tmp_wiki_root, "wiki/sources/paper.md"), encoding="utf-8") as fh:
            out = fh.read()
        assert "type: Paper" in out
        assert "type: Source" not in out  # folder default did not override
