"""Tests for the 4 new tools (read_file, write_page, update_index_entry,
update_schema_lessons), the IndexState TypedDict, the INDEX_AGENT_PROMPT,
and the 9 new directories in init_wiki.py.
"""

import json
import os
import textwrap

import pytest

from wiki_agent.prompts import INDEX_AGENT_PROMPT, WIKI_ORCHESTRATOR_PROMPT
from wiki_agent.setup.init_wiki import init_knowledge_base
from wiki_agent.tools import create_tools

# -----------------------------------------------------------------------
# Fixtures
# -----------------------------------------------------------------------


@pytest.fixture
def tools(tmp_wiki_root):
    """All tools bound to tmp_wiki_root."""
    return {t.name: t for t in create_tools(tmp_wiki_root)}


@pytest.fixture
def tools_populated(populated_wiki):
    """Tools bound to a populated wiki."""
    return {t.name: t for t in create_tools(populated_wiki)}


# =======================================================================
# A) Tests for read_file
# =======================================================================


class TestReadFile:
    def test_reads_existing_wiki_file(self, tools_populated, populated_wiki):
        """read_file returns correct content for an existing wiki page."""
        result = json.loads(
            tools_populated["read_wiki_file"].invoke(
                {"file_path": "wiki/entities/andrej-karpathy.md"}
            )
        )
        assert "error" not in result
        assert "Andrej Karpathy" in result["content"]
        assert result["size_bytes"] > 0
        assert "entities" in result["file_path"]

    def test_parses_frontmatter(self, tools_populated, populated_wiki):
        """read_file parses YAML frontmatter when present."""
        result = json.loads(
            tools_populated["read_wiki_file"].invoke(
                {"file_path": "wiki/entities/andrej-karpathy.md"}
            )
        )
        fm = result["frontmatter"]
        assert fm is not None
        assert fm["title"] == "Andrej Karpathy"
        assert "person" in fm["tags"]

    def test_reads_raw_file(self, tools, tmp_wiki_root):
        """read_file works for files inside raw/."""
        raw_path = os.path.join(tmp_wiki_root, "raw", "articles", "test.md")
        with open(raw_path, "w", encoding="utf-8") as fh:
            fh.write("# Raw Test\n\nHello from raw.")
        result = json.loads(tools["read_wiki_file"].invoke({"file_path": "raw/articles/test.md"}))
        assert "error" not in result
        assert "Hello from raw" in result["content"]

    def test_rejects_path_traversal(self, tools, tmp_wiki_root):
        """read_file rejects paths outside wiki_root (path traversal)."""
        result = json.loads(tools["read_wiki_file"].invoke({"file_path": "../../etc/passwd"}))
        assert "error" in result
        assert "outside" in result["error"].lower() or "Path" in result["error"]

    def test_returns_error_for_missing_file(self, tools):
        """read_file returns a JSON error for a non-existent file."""
        result = json.loads(tools["read_wiki_file"].invoke({"file_path": "wiki/does-not-exist.md"}))
        assert "error" in result
        assert "not found" in result["error"].lower()


# =======================================================================
# B) Tests for write_page
# =======================================================================


class TestWritePage:
    def test_creates_new_page(self, tools, tmp_wiki_root):
        """write_page creates a new file in wiki/sources/."""
        content = textwrap.dedent("""\
            ---
            title: "New Page"
            date: 2026-04-07
            sources: []
            tags: [test]
            ---

            # New Page

            Body text.
        """)
        result = json.loads(
            tools["write_page"].invoke(
                {"page_path": "wiki/sources/new-page.md", "content": content}
            )
        )
        assert result["status"] == "created"

        file_path = os.path.join(tmp_wiki_root, "wiki", "sources", "new-page.md")
        assert os.path.exists(file_path)
        with open(file_path, encoding="utf-8") as fh:
            saved = fh.read()
        assert "New Page" in saved
        assert "Body text." in saved

    def test_updates_existing_page(self, tools_populated, populated_wiki):
        """write_page returns status 'updated' for an existing page."""
        updated_content = textwrap.dedent("""\
            ---
            title: "Andrej Karpathy"
            date: 2026-04-07
            sources: ["raw/articles/sample.md"]
            tags: [person, ai, researcher, updated]
            ---

            # Andrej Karpathy

            Updated content.
        """)
        result = json.loads(
            tools_populated["write_page"].invoke(
                {"page_path": "wiki/entities/andrej-karpathy.md", "content": updated_content}
            )
        )
        assert result["status"] == "updated"

        file_path = os.path.join(populated_wiki, "wiki", "entities", "andrej-karpathy.md")
        with open(file_path, encoding="utf-8") as fh:
            saved = fh.read()
        assert "Updated content." in saved

    def test_rejects_write_outside_wiki(self, tools, tmp_wiki_root):
        """write_page rejects paths that are not inside wiki/."""
        result = json.loads(
            tools["write_page"].invoke({"page_path": "raw/articles/hack.md", "content": "nope"})
        )
        assert "error" in result

    def test_creates_intermediate_directories(self, tools, tmp_wiki_root):
        """write_page creates parent directories if they don't exist."""
        result = json.loads(
            tools["write_page"].invoke(
                {
                    "page_path": "wiki/concepts/deep/nested/page.md",
                    "content": "# Nested\n\nNested content.",
                }
            )
        )
        assert result["status"] == "created"
        nested_path = os.path.join(tmp_wiki_root, "wiki", "concepts", "deep", "nested", "page.md")
        assert os.path.exists(nested_path)

    def test_content_written_exactly(self, tools, tmp_wiki_root):
        """write_page writes the exact content provided.

        The page already declares ``type`` so the OKF type-injection contract
        (Fase 3) is a no-op and byte-exact semantics hold.
        """
        exact_content = "---\ntype: Source\ntitle: Exact\n---\n\nExact body.\n"
        result = json.loads(
            tools["write_page"].invoke(
                {"page_path": "wiki/sources/exact.md", "content": exact_content}
            )
        )
        assert result["status"] == "created"

        file_path = os.path.join(tmp_wiki_root, "wiki", "sources", "exact.md")
        with open(file_path, encoding="utf-8") as fh:
            saved = fh.read()
        assert saved == exact_content


# =======================================================================
# B2) Source-dedup guard (INV-08 / Causa B, issue #140)
# =======================================================================


def _source_page(title: str, url: str) -> str:
    """A minimal source-page body referencing a single source URL."""
    return textwrap.dedent(
        f"""\
        ---
        title: "{title}"
        date: 2026-05-31
        sources:
          - "{url}"
        tags: [test]
        origin: llm-synthesized
        ---

        # {title}

        Body for {title}.
        """
    )


class TestWritePageSourceDedup:
    URL = "https://github.com/rowboatlabs/rowboat"

    def test_refuses_duplicate_source_under_new_slug(self, tools, tmp_wiki_root):
        """A new source page reusing an existing source URL is refused."""
        first = json.loads(
            tools["write_page"].invoke(
                {
                    "page_path": "wiki/sources/rowboat-ai-coworker.md",
                    "content": _source_page("Rowboat", self.URL),
                }
            )
        )
        assert first["status"] == "created"

        # Different slug, same source → must be refused, no file written.
        dup = json.loads(
            tools["write_page"].invoke(
                {
                    "page_path": "wiki/sources/rowboat.md",
                    "content": _source_page("Rowboat dup", self.URL),
                }
            )
        )
        assert dup["status"] == "refused"
        assert dup["reason"] == "duplicate_source"
        assert dup["existing_page"].endswith("rowboat-ai-coworker.md")
        assert not os.path.exists(os.path.join(tmp_wiki_root, "wiki", "sources", "rowboat.md"))

    def test_overwrite_true_bypasses_dedup(self, tools, tmp_wiki_root):
        """overwrite=True lets a duplicate-source page through."""
        tools["write_page"].invoke(
            {
                "page_path": "wiki/sources/rowboat-ai-coworker.md",
                "content": _source_page("Rowboat", self.URL),
            }
        )
        forced = json.loads(
            tools["write_page"].invoke(
                {
                    "page_path": "wiki/sources/rowboat.md",
                    "content": _source_page("Rowboat forced", self.URL),
                    "overwrite": True,
                }
            )
        )
        assert forced["status"] == "created"
        assert os.path.exists(os.path.join(tmp_wiki_root, "wiki", "sources", "rowboat.md"))

    def test_distinct_source_not_refused(self, tools, tmp_wiki_root):
        """A new source page with a non-overlapping source is created."""
        tools["write_page"].invoke(
            {
                "page_path": "wiki/sources/rowboat-ai-coworker.md",
                "content": _source_page("Rowboat", self.URL),
            }
        )
        other = json.loads(
            tools["write_page"].invoke(
                {
                    "page_path": "wiki/sources/other.md",
                    "content": _source_page("Other", "https://example.com/other"),
                }
            )
        )
        assert other["status"] == "created"

    def test_update_same_path_with_shared_source_allowed(self, tools, tmp_wiki_root):
        """Updating the SAME path is unaffected by the dedup guard."""
        tools["write_page"].invoke(
            {
                "page_path": "wiki/sources/rowboat-ai-coworker.md",
                "content": _source_page("Rowboat", self.URL),
            }
        )
        updated = json.loads(
            tools["write_page"].invoke(
                {
                    "page_path": "wiki/sources/rowboat-ai-coworker.md",
                    "content": _source_page("Rowboat v2", self.URL),
                }
            )
        )
        assert updated["status"] == "updated"

    def test_entity_sharing_source_not_refused(self, tools, tmp_wiki_root):
        """Entity/concept pages may share a source URL — guard is sources-only."""
        tools["write_page"].invoke(
            {
                "page_path": "wiki/sources/rowboat-ai-coworker.md",
                "content": _source_page("Rowboat", self.URL),
            }
        )
        entity = json.loads(
            tools["write_page"].invoke(
                {
                    "page_path": "wiki/entities/rowboat-labs.md",
                    "content": _source_page("Rowboat Labs", self.URL),
                }
            )
        )
        assert entity["status"] == "created"


# =======================================================================
# C) Tests for update_index_entry
# =======================================================================


class TestUpdateIndexEntry:
    def test_adds_new_entry(self, tools, tmp_wiki_root):
        """update_index_entry adds a new row to index.md."""
        result = json.loads(
            tools["update_index_entry"].invoke(
                {
                    "page_path": "sources/ml-intro.md",
                    "category": "sources",
                    "summary": "Introduction to machine learning",
                    "source_count": 3,
                }
            )
        )
        assert result["status"] == "added"

        index_path = os.path.join(tmp_wiki_root, "wiki", "index.md")
        with open(index_path, encoding="utf-8") as fh:
            content = fh.read()
        assert "Ml Intro" in content
        assert "sources" in content
        assert "Introduction to machine learning" in content

    def test_updates_existing_entry(self, tools, tmp_wiki_root):
        """update_index_entry updates an existing row when page_path matches."""
        # Add first
        tools["update_index_entry"].invoke(
            {
                "page_path": "entities/test-person.md",
                "category": "entities",
                "summary": "Original summary",
                "source_count": 1,
            }
        )
        # Update
        result = json.loads(
            tools["update_index_entry"].invoke(
                {
                    "page_path": "entities/test-person.md",
                    "category": "entities",
                    "summary": "Updated summary",
                    "source_count": 5,
                }
            )
        )
        assert result["status"] == "updated"

        index_path = os.path.join(tmp_wiki_root, "wiki", "index.md")
        with open(index_path, encoding="utf-8") as fh:
            content = fh.read()
        assert "Updated summary" in content
        # Original summary should be replaced
        assert content.count("entities/test-person.md") == 1

    def test_correct_category_in_entry(self, tools, tmp_wiki_root):
        """update_index_entry writes the category label correctly."""
        for category in ("sources", "entities", "concepts", "answers"):
            tools["update_index_entry"].invoke(
                {
                    "page_path": f"{category}/cat-test-{category}.md",
                    "category": category,
                    "summary": f"Test {category}",
                    "source_count": 1,
                }
            )

        index_path = os.path.join(tmp_wiki_root, "wiki", "index.md")
        with open(index_path, encoding="utf-8") as fh:
            content = fh.read()
        for category in ("sources", "entities", "concepts", "answers"):
            assert f"| {category} |" in content


# =======================================================================
# D) Tests for update_schema_lessons
# =======================================================================


class TestUpdateSchemaLessons:
    def test_adds_single_lesson(self, tools, tmp_wiki_root):
        """update_schema_lessons appends a lesson to wiki-schema.md."""
        result = json.loads(
            tools["update_schema_lessons"].invoke(
                {"lesson": "Always slugify entity names before creating pages."}
            )
        )
        assert result["status"] == "added"
        assert "slugify" in result["lesson"]

        schema_path = os.path.join(tmp_wiki_root, "schema", "wiki-schema.md")
        with open(schema_path, encoding="utf-8") as fh:
            content = fh.read()
        assert "Always slugify entity names before creating pages." in content

    def test_adds_multiple_lessons(self, tools, tmp_wiki_root):
        """update_schema_lessons can be called multiple times; all lessons present."""
        lessons = [
            "Use ISO dates everywhere.",
            "Cross-reference all entities mentioned in source pages.",
            "Keep summaries under 5 paragraphs.",
        ]
        for lesson in lessons:
            tools["update_schema_lessons"].invoke({"lesson": lesson})

        schema_path = os.path.join(tmp_wiki_root, "schema", "wiki-schema.md")
        with open(schema_path, encoding="utf-8") as fh:
            content = fh.read()
        for lesson in lessons:
            assert lesson in content

    def test_returns_error_when_schema_missing(self, tmp_path):
        """update_schema_lessons returns error if wiki-schema.md is absent."""
        root = str(tmp_path / "empty-root")
        os.makedirs(root)
        tools = {t.name: t for t in create_tools(root)}
        result = json.loads(tools["update_schema_lessons"].invoke({"lesson": "This should fail."}))
        assert "error" in result


# =======================================================================
# E) Tests for INDEX_AGENT_PROMPT
# =======================================================================


class TestIndexAgentPrompt:
    def test_prompt_not_empty(self):
        assert len(INDEX_AGENT_PROMPT) > 100

    def test_mentions_audit_index(self):
        assert "audit" in INDEX_AGENT_PROMPT.lower()

    def test_mentions_backlinks(self):
        assert "backlink" in INDEX_AGENT_PROMPT.lower()

    def test_mentions_stale_entries(self):
        assert "stale" in INDEX_AGENT_PROMPT.lower()

    def test_mentions_broken_links(self):
        assert "broken" in INDEX_AGENT_PROMPT.lower()

    def test_mentions_required_tools(self):
        """INDEX_AGENT_PROMPT references all the tools it should use."""
        for tool_name in (
            "read_wiki_file",
            "write_page",
            "update_index_entry",
            "find_cross_references",
            "get_wiki_stats",
        ):
            assert tool_name in INDEX_AGENT_PROMPT

    def test_orchestrator_mentions_index_agent(self):
        """The orchestrator prompt routes to index-agent."""
        assert (
            "index-agent" in WIKI_ORCHESTRATOR_PROMPT or "index_agent" in WIKI_ORCHESTRATOR_PROMPT
        )

    def test_orchestrator_mentions_index_routing(self):
        """The orchestrator prompt contains index routing rules."""
        assert "index" in WIKI_ORCHESTRATOR_PROMPT.lower()
        assert "reindex" in WIKI_ORCHESTRATOR_PROMPT.lower()


# =======================================================================
# G) Tests for init_wiki 9 new directories
# =======================================================================


class TestInitWikiNewDirectories:
    def test_creates_all_new_directories(self, tmp_path):
        """init_knowledge_base creates the 9 new directories added in the expansion."""
        root = str(tmp_path / "kb")
        init_knowledge_base(root)

        new_dirs = [
            "raw/repos",
            "raw/datasets",
            "raw/images",
            "raw/assets",
            "wiki/synthesis",
            "wiki/outputs/slides",
            "wiki/outputs/charts",
            "schema/templates",
            "schema/workflows",
        ]
        for d in new_dirs:
            full = os.path.join(root, d)
            assert os.path.isdir(full), f"Missing new directory: {d}"

    def test_total_directory_count(self, tmp_path):
        """init_knowledge_base creates at least 19 directories total."""
        root = str(tmp_path / "kb")
        result = init_knowledge_base(root)
        # Filter only directory entries
        dir_entries = [r for r in result if os.path.isdir(r)]
        assert len(dir_entries) >= 19

    def test_new_dirs_idempotent(self, tmp_path):
        """Running init twice does not fail or duplicate directories."""
        root = str(tmp_path / "kb")
        init_knowledge_base(root)
        init_knowledge_base(root)
        # All should still exist
        assert os.path.isdir(os.path.join(root, "wiki", "outputs", "charts"))
        assert os.path.isdir(os.path.join(root, "raw", "datasets"))
