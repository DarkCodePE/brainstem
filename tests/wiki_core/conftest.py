"""Fixtures for the wiki_core protocol-contract tests.

These tests are NOT under tests/wiki_ingest/ (which is quarantined for the
brownfield API drift) — they assert on the new typed Protocols from
`wiki_core.protocols` and the adapters that satisfy them.
"""

from __future__ import annotations

from collections.abc import Iterator
from datetime import UTC, datetime
from pathlib import Path

import pytest


@pytest.fixture
def event_factory() -> Iterator[callable]:  # type: ignore[type-arg]
    """Build minimal-valid wiki_core.IngestEvent instances."""
    from wiki_core.protocols import IngestEvent

    counter = 0

    def make(
        sha256: str = "f" * 64,
        path: str = "/tmp/sample.md",
        bucket: str = "articles",
        size: int = 1024,
    ) -> IngestEvent:
        nonlocal counter
        counter += 1
        return IngestEvent(
            event_id=f"019e5130-0000-7000-8000-{counter:012x}",
            source=f"watcher:{bucket}",
            path_or_uri=path,
            sha256=sha256,
            received_at=datetime.now(UTC),
            metadata={
                "rel_path": Path(path).name,
                "bucket": bucket,
                "event_type": "created",
                "mtime": "2026-05-22T20:00:00Z",
                "size": size,
                "mime": "text/markdown",
            },
        )

    yield make


@pytest.fixture
def tmp_db_path(tmp_path: Path) -> Path:
    return tmp_path / "test_memory_store.db"


@pytest.fixture
def page_factory() -> Iterator[callable]:  # type: ignore[type-arg]
    """Build wiki_core.Page instances for WriteSink tests."""
    from wiki_core.protocols import Page, PageRef

    def make(
        page_path: str = "wiki/sources/sample.md",
        category: str = "sources",
        title: str = "Sample",
        body: str = "Body of the page.\n",
        sha256: str | None = None,
    ) -> Page:
        frontmatter: dict[str, object] = {
            "title": title,
            "date": "2026-05-22",
            "tags": ["test"],
        }
        if sha256:
            frontmatter["sha256"] = sha256
        return Page(
            ref=PageRef(page_path=page_path, category=category),  # type: ignore[arg-type]
            frontmatter=frontmatter,
            body=body,
        )

    yield make


class FakeClock:
    """Manually-advanced monotonic clock — same shape as the one in
    ``tests/wiki_autofetch/conftest.py``.

    Duplicated here (rather than imported) so the wiki_core suite stays
    self-contained: it shouldn't need the wiki_autofetch test package on
    its sys.path to run.
    """

    def __init__(self, start: float = 0.0) -> None:
        self._now = float(start)

    def __call__(self) -> float:
        return self._now

    def tick(self, seconds: float) -> None:
        self._now += float(seconds)

    def set(self, t: float) -> None:
        self._now = float(t)


@pytest.fixture
def fake_clock() -> FakeClock:
    return FakeClock(start=1000.0)


@pytest.fixture
def search_hit_factory() -> Iterator[callable]:  # type: ignore[type-arg]
    from wiki_core.protocols import PageRef, SearchHit

    def make(
        page_path: str = "wiki/concepts/example.md",
        category: str = "concepts",
        title: str = "Example",
        snippet: str = "An example page.",
        score: float = 0.75,
        components: dict[str, float] | None = None,
    ) -> SearchHit:
        return SearchHit(
            ref=PageRef(page_path=page_path, category=category),  # type: ignore[arg-type]
            title=title,
            snippet=snippet,
            score=score,
            score_components=components or {"keyword": 0.3, "semantic": 0.45},
        )

    yield make
