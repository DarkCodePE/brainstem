"""Tests for `wiki_qa.linter` — orphans, broken wikilinks, duplicate slugs."""

from __future__ import annotations

from wiki_qa.graph import KnowledgeGraph
from wiki_qa.linter import (
    find_broken_wikilinks,
    find_duplicate_slugs,
    find_orphans,
    lint,
)


class TestOrphans:
    def test_disconnected_articles_are_orphans(self, graph: KnowledgeGraph) -> None:
        orphans = find_orphans(graph)
        assert orphans == ("article:orphans/lonely", "article:other/foo")

    def test_linked_articles_are_not_orphans(self, graph: KnowledgeGraph) -> None:
        orphans = find_orphans(graph)
        assert "article:concepts/foo" not in orphans
        assert "article:entities/bar" not in orphans

    def test_topics_and_sources_are_ignored(self, graph: KnowledgeGraph) -> None:
        orphans = find_orphans(graph)
        assert all(o.startswith("article:") for o in orphans)


class TestBrokenWikilinks:
    def test_unresolvable_target_is_broken(self, graph: KnowledgeGraph) -> None:
        broken = find_broken_wikilinks(graph)
        keys = {b.key() for b in broken}
        assert "concepts/foo.md -> missing-page" in keys

    def test_unique_basename_resolves(self, graph: KnowledgeGraph) -> None:
        # ``[[bar]]`` -> entities/bar.md (unique basename) is NOT broken.
        broken = {b.target for b in find_broken_wikilinks(graph)}
        assert "bar" not in broken

    def test_suffix_resolution(self, graph: KnowledgeGraph) -> None:
        # ``[[foo]]`` from bar resolves to concepts/foo.md via the ``/foo`` suffix.
        broken = {(b.source_file, b.target) for b in find_broken_wikilinks(graph)}
        assert ("entities/bar.md", "foo") not in broken

    def test_only_expected_link_is_broken(self, graph: KnowledgeGraph) -> None:
        assert len(find_broken_wikilinks(graph)) == 1

    def test_shell_flag_targets_skipped(self) -> None:
        from wiki_qa.graph import graph_from_dict

        g = graph_from_dict(
            {
                "nodes": [
                    {
                        "id": "article:a",
                        "type": "article",
                        "name": "A",
                        "filePath": "a.md",
                        "knowledgeMeta": {"wikilinks": ["--full", ""]},
                    }
                ],
                "edges": [],
            }
        )
        assert find_broken_wikilinks(g) == ()


class TestDuplicateSlugs:
    def test_shared_basename_flagged(self, graph: KnowledgeGraph) -> None:
        dups = find_duplicate_slugs(graph)
        assert len(dups) == 1
        assert dups[0].slug == "foo"
        assert dups[0].paths == ("concepts/foo.md", "other/foo.md")

    def test_unique_slugs_not_flagged(self, graph: KnowledgeGraph) -> None:
        slugs = {d.slug for d in find_duplicate_slugs(graph)}
        assert "bar" not in slugs
        assert "lonely" not in slugs


class TestLintReport:
    def test_report_aggregates_all_checks(self, graph: KnowledgeGraph) -> None:
        report = lint(graph)
        assert len(report.orphans) == 2
        assert len(report.broken_wikilinks) == 1
        assert len(report.duplicate_slugs) == 1
        assert report.total_issues == 4

    def test_report_to_dict_is_serialisable(self, graph: KnowledgeGraph) -> None:
        import json

        report = lint(graph)
        payload = report.to_dict()
        assert payload["summary"]["total"] == 4
        # Round-trips through JSON without error.
        assert json.loads(json.dumps(payload))["summary"]["orphans"] == 2
