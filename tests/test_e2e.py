"""End-to-end tests for wiki agent tool chains.

Each class tests a complete workflow by invoking real tool chains
against temporary wiki directories -- no LLM mocking needed.
"""

import json
import os
import sys
from unittest.mock import MagicMock

import pytest

from wiki_agent.tools import create_tools

# -------------------------------------------------------------------
# Helpers
# -------------------------------------------------------------------


def _tools(root: str) -> dict:
    return {t.name: t for t in create_tools(root)}


def _invoke(tools, name, params):
    return json.loads(tools[name].invoke(params))


def _log(tools, entry_type, title, details):
    """Invoke append_to_log with its actual signature."""
    raw = tools["append_to_log"].invoke(
        {
            "entry_type": entry_type,
            "title": title,
            "details": details,
        }
    )
    return raw  # returns plain string, not JSON


# ===================================================================
# 1. Ingest flow
# ===================================================================


class TestIngestFlow:
    def test_write_update_log_read(self, tmp_wiki_root):
        tools = _tools(tmp_wiki_root)
        content = (
            '---\ntitle: "ML Intro"\ndate: 2026-04-15\n'
            "sources: []\ntags: [ml]\n---\n\n# ML Intro\n\nBody.\n"
        )
        w = _invoke(
            tools, "write_page", {"page_path": "wiki/sources/ml-intro.md", "content": content}
        )
        assert w["status"] == "created"

        idx = _invoke(
            tools,
            "update_index_entry",
            {
                "page_path": "sources/ml-intro.md",
                "category": "sources",
                "summary": "Intro to ML",
                "source_count": 1,
            },
        )
        assert idx["status"] == "added"

        result = _log(tools, "ingest", "ml-intro", "- Ingested ml-intro page")
        assert "Log entry appended" in result

        r = _invoke(tools, "read_wiki_file", {"file_path": "wiki/sources/ml-intro.md"})
        assert "ML Intro" in r["content"]

    def test_ingest_with_schema_lesson(self, tmp_wiki_root):
        tools = _tools(tmp_wiki_root)
        content = '---\ntitle: "X"\ndate: 2026-04-15\nsources: []\ntags: [x]\n---\n\n# X\n'
        _invoke(tools, "write_page", {"page_path": "wiki/sources/x.md", "content": content})
        sl = _invoke(tools, "update_schema_lessons", {"lesson": "Always slugify names"})
        assert sl["status"] == "added"

    def test_entity_concept_cross_references(self, tmp_wiki_root):
        tools = _tools(tmp_wiki_root)
        entity_content = (
            '---\ntitle: "Alice"\ndate: 2026-04-15\nsources: []\ntags: [person]\n---\n\n'
            "# Alice\n\nWorks on [[RAG Pattern]].\n"
        )
        concept_content = (
            '---\ntitle: "RAG Pattern"\ndate: 2026-04-15\nsources: []\ntags: [concept]\n---\n\n'
            "# RAG Pattern\n\nPioneered by [[Alice]].\n"
        )
        _invoke(
            tools, "write_page", {"page_path": "wiki/entities/alice.md", "content": entity_content}
        )
        _invoke(
            tools,
            "write_page",
            {"page_path": "wiki/concepts/rag-pattern.md", "content": concept_content},
        )
        _invoke(
            tools,
            "update_index_entry",
            {
                "page_path": "entities/alice.md",
                "category": "entities",
                "summary": "A person",
                "source_count": 1,
            },
        )
        _invoke(
            tools,
            "update_index_entry",
            {
                "page_path": "concepts/rag-pattern.md",
                "category": "concepts",
                "summary": "RAG",
                "source_count": 1,
            },
        )

        xrefs = _invoke(tools, "find_cross_references", {"page_path": "wiki/entities/alice.md"})
        assert "outbound_links" in xrefs
        assert len(xrefs["outbound_links"]) > 0


# ===================================================================
# 2. Query flow
# ===================================================================


class TestQueryFlow:
    def test_search_then_read(self, populated_wiki):
        tools = _tools(populated_wiki)
        sr = _invoke(tools, "search_wiki_index", {"query": "Karpathy"})
        # search_wiki_index returns a JSON array (list)
        assert isinstance(sr, list)
        assert len(sr) > 0
        first_path = sr[0]["page_path"]
        r = _invoke(tools, "read_wiki_file", {"file_path": f"wiki/{first_path}"})
        assert "Karpathy" in r["content"]

    def test_empty_search(self, tmp_wiki_root):
        """Search on a wiki with no index entries returns empty list."""
        tools = _tools(tmp_wiki_root)
        sr = _invoke(tools, "search_wiki_index", {"query": "zzz_nonexistent_zzz"})
        assert isinstance(sr, list)
        assert sr == []

    def test_search_and_file_answer(self, tmp_wiki_root):
        tools = _tools(tmp_wiki_root)
        page = (
            '---\ntitle: "Answer"\ndate: 2026-04-15\nsources: []\ntags: [answer]\n---\n\n'
            "# What is RAG?\n\nRetrieval-augmented generation.\n"
        )
        w = _invoke(tools, "write_page", {"page_path": "wiki/answers/rag.md", "content": page})
        assert w["status"] == "created"
        r = _invoke(tools, "read_wiki_file", {"file_path": "wiki/answers/rag.md"})
        assert "Retrieval-augmented generation" in r["content"]


# ===================================================================
# 3. Lint flow
# ===================================================================


class TestLintFlow:
    def test_detect_orphan_pages(self, populated_wiki):
        tools = _tools(populated_wiki)
        orphans = _invoke(tools, "detect_orphan_pages", {})
        # detect_orphan_pages returns a JSON array of path strings
        assert isinstance(orphans, list)
        assert len(orphans) > 0
        # bad-entity.md has no inbound links, so it should be detected
        assert any("bad-entity" in p for p in orphans)

    def test_validate_frontmatter_bad_page(self, populated_wiki):
        tools = _tools(populated_wiki)
        v = _invoke(tools, "validate_frontmatter", {"page_path": "wiki/entities/bad-entity.md"})
        dumped = json.dumps(v).lower()
        assert "missing" in dumped or "invalid" in dumped or v.get("valid") is False

    def test_full_lint_chain(self, populated_wiki):
        tools = _tools(populated_wiki)
        orphans = _invoke(tools, "detect_orphan_pages", {})
        assert isinstance(orphans, list)

        v = _invoke(tools, "validate_frontmatter", {"page_path": "wiki/entities/bad-entity.md"})
        assert v is not None

        stats = _invoke(tools, "get_wiki_stats", {})
        assert "page_count" in stats

        _log(tools, "lint", "Full lint", "- Scanned all pages")

    def test_cross_refs_connected_pages(self, populated_wiki):
        tools = _tools(populated_wiki)
        xrefs = _invoke(
            tools, "find_cross_references", {"page_path": "wiki/entities/andrej-karpathy.md"}
        )
        assert "outbound_links" in xrefs
        dumped = json.dumps(xrefs).lower()
        assert "llm" in dumped or "software" in dumped


# ===================================================================
# 4. Capture flow
# ===================================================================


class TestCaptureFlow:
    def test_write_observation_and_log(self, tmp_wiki_root):
        tools = _tools(tmp_wiki_root)
        obs_content = (
            '---\ntitle: "Observations 2026-04-15"\ndate: 2026-04-15\n'
            "sources: []\ntags: [observations]\n---\n\n"
            "# Observations for 2026-04-15\n\n"
            "### OBS-2026-04-15-001\n\n"
            "**Category:** product-gap\n"
            "**Confidence:** high\n"
            "**Graduated:** false\n"
            "**Text:** MCP needs retry logic\n\n"
        )
        w = _invoke(
            tools,
            "write_page",
            {"page_path": "wiki/sources/obs-2026-04-15.md", "content": obs_content},
        )
        assert w["status"] == "created"
        result = _log(tools, "capture", "OBS-2026-04-15-001", "- Captured observation")
        assert "Log entry appended" in result

    def test_write_and_read_back(self, tmp_wiki_root):
        tools = _tools(tmp_wiki_root)
        obs_content = (
            '---\ntitle: "Observations 2026-04-15"\ndate: 2026-04-15\n'
            "sources: []\ntags: [observations]\n---\n\n"
            "# Observations for 2026-04-15\n\n"
            "### OBS-2026-04-15-001\n\n"
            "**Category:** tool-learning\n"
            "**Confidence:** medium\n"
            "**Graduated:** false\n"
            "**Text:** fastembed rocks\n\n"
        )
        _invoke(
            tools, "write_page", {"page_path": "wiki/sources/obs-file.md", "content": obs_content}
        )
        r = _invoke(tools, "read_wiki_file", {"file_path": "wiki/sources/obs-file.md"})
        assert "fastembed" in r["content"]


# ===================================================================
# 5. Review flow
# ===================================================================


class TestReviewFlow:
    def test_read_observation_files(self, obs_wiki):
        tools = _tools(obs_wiki)
        r = _invoke(tools, "read_wiki_file", {"file_path": "observations/2026-04-13.md"})
        assert "OBS-2026-04-13-001" in r["content"]
        assert "product-gap" in r["content"]

    def test_search_index_for_themes(self, obs_wiki):
        tools = _tools(obs_wiki)
        sr = _invoke(tools, "search_wiki_index", {"query": "Karpathy"})
        assert isinstance(sr, list)
        assert len(sr) > 0


# ===================================================================
# 6. Index flow
# ===================================================================


class TestIndexFlow:
    def test_add_multiple_entries(self, tmp_wiki_root):
        tools = _tools(tmp_wiki_root)
        for i in range(3):
            r = _invoke(
                tools,
                "update_index_entry",
                {
                    "page_path": f"sources/page-{i}.md",
                    "category": "sources",
                    "summary": f"Page {i}",
                    "source_count": 1,
                },
            )
            assert r["status"] == "added"

    def test_update_existing_no_duplicates(self, tmp_wiki_root):
        tools = _tools(tmp_wiki_root)
        _invoke(
            tools,
            "update_index_entry",
            {
                "page_path": "entities/bob.md",
                "category": "entities",
                "summary": "Bob v1",
                "source_count": 1,
            },
        )
        r = _invoke(
            tools,
            "update_index_entry",
            {
                "page_path": "entities/bob.md",
                "category": "entities",
                "summary": "Bob v2",
                "source_count": 2,
            },
        )
        assert r["status"] == "updated"
        index_path = os.path.join(tmp_wiki_root, "wiki", "index.md")
        with open(index_path, encoding="utf-8") as fh:
            content = fh.read()
        assert content.count("entities/bob.md") == 1

    def test_add_write_cross_ref(self, tmp_wiki_root):
        tools = _tools(tmp_wiki_root)
        page = (
            '---\ntitle: "Node"\ndate: 2026-04-15\nsources: []\ntags: [concept]\n---\n\n'
            "# Node\n\nRelates to [[Edge]].\n"
        )
        _invoke(tools, "write_page", {"page_path": "wiki/concepts/node.md", "content": page})
        _invoke(
            tools,
            "update_index_entry",
            {
                "page_path": "concepts/node.md",
                "category": "concepts",
                "summary": "Node concept",
                "source_count": 0,
            },
        )
        xrefs = _invoke(tools, "find_cross_references", {"page_path": "wiki/concepts/node.md"})
        assert "outbound_links" in xrefs

    def test_stats_reflect_changes(self, tmp_wiki_root):
        tools = _tools(tmp_wiki_root)
        stats_before = _invoke(tools, "get_wiki_stats", {})
        page = "---\ntitle: T\ndate: 2026-04-15\nsources: []\ntags: []\n---\n\n# T\n"
        _invoke(tools, "write_page", {"page_path": "wiki/sources/new.md", "content": page})
        stats_after = _invoke(tools, "get_wiki_stats", {})
        assert stats_after["source_count"] > stats_before["source_count"]


# ===================================================================
# 7. Graduate observation
# ===================================================================


class TestGraduateObservation:
    def test_graduate_to_schema_rule(self, obs_wiki):
        tools = _tools(obs_wiki)
        r = _invoke(
            tools,
            "graduate_observation",
            {
                "observation_ids": "OBS-2026-04-13-001",
                "target_type": "schema-rule",
                "title": "MCP Error Handling Rule",
                "content": "Always implement retry logic for MCP tool calls.",
            },
        )
        assert r["status"] == "graduated"
        assert "schema" in r["target_path"]

    def test_graduate_to_concept_page(self, obs_wiki):
        tools = _tools(obs_wiki)
        r = _invoke(
            tools,
            "graduate_observation",
            {
                "observation_ids": "OBS-2026-04-14-002",
                "target_type": "concept-page",
                "title": "Event Sourcing",
                "content": "Event sourcing improves audit trail for wiki changes.",
            },
        )
        assert r["status"] == "graduated"
        assert "concepts" in r["target_path"]

    def test_graduate_to_entity_page(self, obs_wiki):
        tools = _tools(obs_wiki)
        r = _invoke(
            tools,
            "graduate_observation",
            {
                "observation_ids": "OBS-2026-04-13-002",
                "target_type": "entity-page",
                "title": "FastEmbed",
                "content": "FastEmbed is faster than Ollama for local embeddings.",
            },
        )
        assert r["status"] == "graduated"
        assert "entities" in r["target_path"]

    def test_marks_source_as_graduated(self, obs_wiki):
        tools = _tools(obs_wiki)
        _invoke(
            tools,
            "graduate_observation",
            {
                "observation_ids": "OBS-2026-04-15-001",
                "target_type": "schema-rule",
                "title": "MCP Retry",
                "content": "Implement retry logic.",
            },
        )
        obs_path = os.path.join(obs_wiki, "observations", "2026-04-15.md")
        with open(obs_path, encoding="utf-8") as fh:
            content = fh.read()
        assert "**Graduated:** true" in content

    def test_invalid_target_type(self, obs_wiki):
        tools = _tools(obs_wiki)
        r = _invoke(
            tools,
            "graduate_observation",
            {
                "observation_ids": "OBS-2026-04-13-001",
                "target_type": "invalid-type",
                "title": "X",
                "content": "Y",
            },
        )
        assert "error" in r

    def test_multiple_obs_ids(self, obs_wiki):
        tools = _tools(obs_wiki)
        r = _invoke(
            tools,
            "graduate_observation",
            {
                "observation_ids": "OBS-2026-04-13-001,OBS-2026-04-14-001,OBS-2026-04-15-001",
                "target_type": "concept-page",
                "title": "MCP Reliability",
                "content": "MCP needs better error handling, connection stability, and retry logic.",
            },
        )
        assert r["status"] == "graduated"
        assert len(r["graduated_ids"]) == 3


# ===================================================================
# 8. _build_llm fallback chain
# ===================================================================


class TestBuildLlmFallback:
    """Test _build_llm provider routing.

    Since _build_llm uses local imports inside the function body, we
    must patch at the sys.modules level.  We save and restore the
    original module entries to avoid poisoning the import state for
    deepagents (which imports langchain_anthropic at module level).
    """

    def test_explicit_openrouter_prefix(self, monkeypatch):
        monkeypatch.setenv("OPENROUTER_API_KEY", "sk-test-key")
        mock_cls = MagicMock(return_value="mock-llm")
        fake_mod = MagicMock(ChatOpenAI=mock_cls)
        saved = sys.modules.get("langchain_openai")
        sys.modules["langchain_openai"] = fake_mod
        try:
            from wiki_agent.agent import _build_llm

            result = _build_llm(model="openrouter:google/gemma-3")
            assert mock_cls.called
            assert result == "mock-llm"
        finally:
            if saved is None:
                sys.modules.pop("langchain_openai", None)
            else:
                sys.modules["langchain_openai"] = saved

    def test_explicit_anthropic_prefix(self, monkeypatch):
        monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
        mock_cls = MagicMock(return_value="mock-anthropic")
        fake_mod = MagicMock(ChatAnthropic=mock_cls)
        saved = sys.modules.get("langchain_anthropic")
        sys.modules["langchain_anthropic"] = fake_mod
        try:
            from wiki_agent.agent import _build_llm

            result = _build_llm(model="anthropic:claude-sonnet-4-5-20250929")
            assert mock_cls.called
            assert result == "mock-anthropic"
        finally:
            if saved is not None:
                sys.modules["langchain_anthropic"] = saved

    def test_no_providers_raises(self, monkeypatch):
        monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
        monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
        from wiki_agent.agent import _build_llm

        # Use a provider prefix that forces only ollama, then block it
        saved_ollama = sys.modules.get("langchain_ollama")
        sys.modules["langchain_ollama"] = None  # type: ignore[assignment]
        try:
            with pytest.raises((RuntimeError, ImportError)):
                _build_llm(model="ollama:nonexistent")
        finally:
            if saved_ollama is None:
                sys.modules.pop("langchain_ollama", None)
            else:
                sys.modules["langchain_ollama"] = saved_ollama

    def test_env_var_override(self, monkeypatch):
        monkeypatch.setenv("OPENROUTER_API_KEY", "sk-test")
        monkeypatch.setenv("OPENROUTER_MODEL", "custom/model-name")
        mock_cls = MagicMock(return_value="mock-llm")
        fake_mod = MagicMock(ChatOpenAI=mock_cls)
        saved = sys.modules.get("langchain_openai")
        sys.modules["langchain_openai"] = fake_mod
        try:
            from wiki_agent.agent import _build_llm

            _build_llm(model="some-default")
            call_kwargs = mock_cls.call_args
            assert call_kwargs[1]["model"] == "custom/model-name"
        finally:
            if saved is None:
                sys.modules.pop("langchain_openai", None)
            else:
                sys.modules["langchain_openai"] = saved


# ===================================================================
# 9. CLI integration
# ===================================================================


class TestCLIIntegration:
    def test_init_command(self, tmp_path):
        from wiki_agent.cli import main

        root = str(tmp_path / "cli-test-kb")
        rc = main(["--root", root, "init"])
        assert rc == 0
        assert os.path.isdir(os.path.join(root, "wiki", "entities"))

    def test_stats_command(self, tmp_wiki_root):
        from wiki_agent.cli import main

        rc = main(["--root", tmp_wiki_root, "stats"])
        assert rc == 0

    def test_parser_accepts_all_commands(self):
        from wiki_agent.cli import _build_parser

        parser = _build_parser()
        for cmd in ("init", "stats", "lint", "serve"):
            args = parser.parse_args(["--root", "/tmp/kb", cmd])
            assert args.command == cmd

    def test_parser_rejects_missing_args(self):
        from wiki_agent.cli import _build_parser

        parser = _build_parser()
        with pytest.raises(SystemExit):
            parser.parse_args(["ingest"])  # missing 'source' arg
