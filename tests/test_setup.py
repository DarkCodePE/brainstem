"""Tests for knowledge base initialization."""

import os

from wiki_agent.setup.init_wiki import init_knowledge_base


class TestInitKnowledgeBase:
    def test_creates_directory_structure(self, tmp_path):
        root = str(tmp_path / "kb")
        init_knowledge_base(root)

        expected_dirs = [
            "raw/articles",
            "raw/papers",
            "raw/bookmarks",
            "raw/voice-notes",
            "wiki/sources",
            "wiki/entities",
            "wiki/concepts",
            "wiki/answers",
            "schema",
            "skills",
        ]
        for d in expected_dirs:
            assert os.path.isdir(os.path.join(root, d)), f"Missing directory: {d}"

    def test_creates_index_md(self, tmp_path):
        root = str(tmp_path / "kb")
        init_knowledge_base(root)

        index_path = os.path.join(root, "wiki", "index.md")
        assert os.path.exists(index_path)
        with open(index_path) as fh:
            content = fh.read()
        assert "Wiki Index" in content
        assert "| Page |" in content

    def test_creates_log_md(self, tmp_path):
        root = str(tmp_path / "kb")
        init_knowledge_base(root)

        log_path = os.path.join(root, "wiki", "log.md")
        assert os.path.exists(log_path)
        with open(log_path) as fh:
            content = fh.read()
        assert "Operation Log" in content
        assert "init | Knowledge base initialised" in content

    def test_creates_schema(self, tmp_path):
        root = str(tmp_path / "kb")
        init_knowledge_base(root)

        schema_path = os.path.join(root, "schema", "wiki-schema.md")
        assert os.path.exists(schema_path)
        with open(schema_path) as fh:
            content = fh.read()
        assert "Required frontmatter" in content

    def test_idempotent(self, tmp_path):
        root = str(tmp_path / "kb")
        result1 = init_knowledge_base(root)
        result2 = init_knowledge_base(root)

        # Second call should not fail
        assert len(result2) >= len(result1) - 3  # dirs created, files skipped

        # Content should be unchanged - heading appears once
        with open(os.path.join(root, "wiki", "index.md")) as fh:
            content = fh.read()
        assert content.count("# Wiki Index") == 1

    def test_returns_created_items(self, tmp_path):
        root = str(tmp_path / "kb")
        result = init_knowledge_base(root)
        assert isinstance(result, list)
        assert len(result) >= 10  # 10 dirs + 3 files
