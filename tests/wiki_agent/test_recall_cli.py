"""Tests for the deterministic recall CLI (ADR-034 D4, wiki_agent/recall_cli.py).

Hermetic: a tmp wiki with an index table and one page; no LLM, no network.
"""

from __future__ import annotations

from pathlib import Path

from wiki_agent.recall_cli import build_recall_context, parse_index, score_entry

INDEX_MD = """# Wiki Index

## sources

| Page | Category | Summary | Sources | Updated |
|------|----------|---------|---------|---------|
| [ReasoningBank](sources/reasoningbank.md) | sources | Memory framework distilling agent strategies | arxiv | 2026-06-01 |
| [Tombstone fix](sources/tombstone.md) | sources | Real forgetting: tombstone removes chunks from recall | repo | 2026-06-06 |
| [Legacy row](wiki/sources/legacy.md) | sources | Legacy prefixed link kept for fallback coverage | repo | 2026-06-06 |
"""

PAGE_MD = """---
title: "ReasoningBank"
date: 2026-06-01
sources:
  - https://arxiv.org/abs/2509.25140
tags: [memory, agents]
origin: llm-synthesized
---

# ReasoningBank

A memory framework that distills reasoning strategies from agent successes
and failures, retrieving them by embedding similarity before each new task.
"""


def _wiki(tmp_path: Path) -> Path:
    (tmp_path / "wiki" / "sources").mkdir(parents=True)
    (tmp_path / "wiki" / "index.md").write_text(INDEX_MD, encoding="utf-8")
    (tmp_path / "wiki" / "sources" / "reasoningbank.md").write_text(PAGE_MD, encoding="utf-8")
    return tmp_path


def test_parse_index_reads_five_column_rows() -> None:
    entries = parse_index(INDEX_MD)
    assert len(entries) == 3
    assert entries[0].page_path == "sources/reasoningbank.md"
    assert entries[0].summary.startswith("Memory framework")


def test_score_entry_is_term_overlap() -> None:
    entries = parse_index(INDEX_MD)
    assert score_entry("memory framework strategies", entries[0]) > score_entry(
        "memory framework strategies", entries[1]
    )
    assert score_entry("", entries[0]) == 0.0


def test_context_carries_caption_metadata_and_body(tmp_path: Path) -> None:
    root = _wiki(tmp_path)
    ctx = build_recall_context(root, "memory framework agent strategies")
    assert len(ctx["results"]) == 1
    result = ctx["results"][0]
    assert result["title"] == "ReasoningBank"
    assert result["tags"] == ["memory", "agents"]
    assert result["sources"] == ["https://arxiv.org/abs/2509.25140"]
    assert "distills reasoning strategies" in result["body"]
    assert result["truncated"] is False


def test_token_budget_truncates_body(tmp_path: Path) -> None:
    root = _wiki(tmp_path)
    ctx = build_recall_context(root, "memory framework agent strategies", token_budget=10)
    result = ctx["results"][0]
    assert result["truncated"] is True
    assert len(result["body"]) <= 10 * 4


def test_missing_index_degrades_to_empty(tmp_path: Path) -> None:
    ctx = build_recall_context(tmp_path, "anything")
    assert ctx == {"query": "anything", "results": []}


def test_zero_score_pages_are_omitted(tmp_path: Path) -> None:
    root = _wiki(tmp_path)
    ctx = build_recall_context(root, "zzz qqq xxx")
    assert ctx["results"] == []


def test_missing_page_file_is_skipped_with_warning(tmp_path: Path, capsys) -> None:
    root = _wiki(tmp_path)
    ctx = build_recall_context(root, "tombstone forgetting recall", limit=3)
    paths = [r["page_path"] for r in ctx["results"]]
    assert "sources/tombstone.md" not in paths  # file does not exist on disk
    err = capsys.readouterr().err
    assert "sources/tombstone.md" in err  # silent degradation is no longer silent


def test_legacy_wiki_prefixed_links_resolve_via_root_fallback(tmp_path: Path) -> None:
    root = _wiki(tmp_path)
    (root / "wiki" / "sources" / "legacy.md").write_text(PAGE_MD, encoding="utf-8")
    ctx = build_recall_context(root, "legacy prefixed link fallback coverage", limit=3)
    paths = [r["page_path"] for r in ctx["results"]]
    assert "wiki/sources/legacy.md" in paths
