"""Wiki-health linter over the UA knowledge graph (issue UA-3).

Flags three classes of wiki-health problem that SBW's ingest pipeline can
silently introduce — the QA signal Understand-Anything surfaces visually, here
made machine-checkable so it can gate CI:

- **orphan pages**     — articles with no inbound *or* outbound ``related`` link
- **broken wikilinks** — ``[[target]]`` references that resolve to no page
- **duplicate slugs**  — the same filename stem under two directories
  (the duplicate-slug regression class noted in project memory: cron dup,
  source-dedup)

All checks are deterministic and pure — same graph in, same report out — so the
report can be compared against a committed baseline (see :mod:`wiki_qa.baseline`).
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import PurePosixPath
from typing import Any

from wiki_qa.graph import KnowledgeGraph

RELATED_EDGE = "related"


@dataclass(frozen=True)
class BrokenWikilink:
    """A wikilink in ``source_file`` whose ``target`` resolves to no page."""

    source_file: str
    target: str

    def key(self) -> str:
        """Stable identity used for baseline diffing."""
        return f"{self.source_file} -> {self.target}"


@dataclass(frozen=True)
class DuplicateSlug:
    """A filename stem (``slug``) that maps to more than one page ``paths``."""

    slug: str
    paths: tuple[str, ...]

    def key(self) -> str:
        """Stable identity used for baseline diffing."""
        return self.slug


@dataclass(frozen=True)
class HealthReport:
    """The full set of wiki-health findings for one graph."""

    orphans: tuple[str, ...]
    broken_wikilinks: tuple[BrokenWikilink, ...]
    duplicate_slugs: tuple[DuplicateSlug, ...]

    @property
    def total_issues(self) -> int:
        return len(self.orphans) + len(self.broken_wikilinks) + len(self.duplicate_slugs)

    def to_dict(self) -> dict[str, Any]:
        """JSON-serialisable representation (sorted, stable ordering)."""
        return {
            "orphans": list(self.orphans),
            "broken_wikilinks": [
                {"source_file": b.source_file, "target": b.target} for b in self.broken_wikilinks
            ],
            "duplicate_slugs": [
                {"slug": d.slug, "paths": list(d.paths)} for d in self.duplicate_slugs
            ],
            "summary": {
                "orphans": len(self.orphans),
                "broken_wikilinks": len(self.broken_wikilinks),
                "duplicate_slugs": len(self.duplicate_slugs),
                "total": self.total_issues,
            },
        }


def _stem(file_path: str) -> str:
    """Lower-cased POSIX stem, e.g. ``concepts/foo.md`` -> ``concepts/foo``."""
    return PurePosixPath(file_path).with_suffix("").as_posix().lower()


def _basename(file_path: str) -> str:
    """Lower-cased bare filename stem, e.g. ``concepts/foo.md`` -> ``foo``."""
    return PurePosixPath(file_path).with_suffix("").name.lower()


@dataclass(frozen=True)
class _ResolveIndex:
    stems: frozenset[str]
    unique_basenames: frozenset[str]


def _build_resolve_index(graph: KnowledgeGraph) -> _ResolveIndex:
    """Index article file paths so wikilink targets can be resolved."""
    stems: set[str] = set()
    basename_counts: dict[str, int] = {}
    for node in graph.articles:
        if not node.file_path:
            continue
        stems.add(_stem(node.file_path))
        base = _basename(node.file_path)
        basename_counts[base] = basename_counts.get(base, 0) + 1
    unique = {b for b, c in basename_counts.items() if c == 1}
    return _ResolveIndex(stems=frozenset(stems), unique_basenames=frozenset(unique))


def _resolves(target: str, index: _ResolveIndex) -> bool:
    """Mirror UA's wikilink resolution: full-stem, unique-basename, or suffix."""
    key = target.strip().lower()
    if key in index.stems or key in index.unique_basenames:
        return True
    suffix = "/" + key
    return any(stem.endswith(suffix) for stem in index.stems)


def find_orphans(graph: KnowledgeGraph) -> tuple[str, ...]:
    """Article ids isolated in the article-to-article (``related``) subgraph."""
    linked: set[str] = set()
    for edge in graph.edges:
        if edge.type == RELATED_EDGE:
            linked.add(edge.source)
            linked.add(edge.target)
    return tuple(sorted(n.id for n in graph.articles if n.id not in linked))


def find_broken_wikilinks(graph: KnowledgeGraph) -> tuple[BrokenWikilink, ...]:
    """Wikilink targets that resolve to no known article page."""
    index = _build_resolve_index(graph)
    seen: set[tuple[str, str]] = set()
    broken: list[BrokenWikilink] = []
    for node in graph.articles:
        source = node.file_path or node.id
        for raw_target in node.wikilinks:
            target = raw_target.strip()
            # Skip empties and shell-flag false positives (e.g. ``[[--full]]``).
            if not target or target.startswith("-"):
                continue
            if _resolves(target, index):
                continue
            dedup = (source, target)
            if dedup in seen:
                continue
            seen.add(dedup)
            broken.append(BrokenWikilink(source_file=source, target=target))
    return tuple(broken)


def find_duplicate_slugs(graph: KnowledgeGraph) -> tuple[DuplicateSlug, ...]:
    """Filename stems shared by more than one page across directories."""
    by_slug: dict[str, set[str]] = {}
    for node in graph.articles:
        if not node.file_path:
            continue
        by_slug.setdefault(_basename(node.file_path), set()).add(node.file_path)
    dups = [
        DuplicateSlug(slug=slug, paths=tuple(sorted(paths)))
        for slug, paths in by_slug.items()
        if len(paths) > 1
    ]
    return tuple(sorted(dups, key=lambda d: d.slug))


def lint(graph: KnowledgeGraph) -> HealthReport:
    """Run every health check and assemble a :class:`HealthReport`."""
    return HealthReport(
        orphans=find_orphans(graph),
        broken_wikilinks=find_broken_wikilinks(graph),
        duplicate_slugs=find_duplicate_slugs(graph),
    )
