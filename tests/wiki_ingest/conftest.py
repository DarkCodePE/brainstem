"""Shared fixtures for the `wiki_ingest` test suite.

Per ADR-006, the daemon under `src/wiki_ingest/` wraps a watchdog Observer,
a SQLite WAL queue, and an asyncio worker pool that calls the
`wiki-knowledge-engine` MCP.

## M2 Sprint 2 — selective quarantine

The original test files in this directory (`test_queue.py`,
`test_worker.py`, `test_watcher.py`, `test_e2e.py`, plus partial
`test_filters.py` / `test_security.py`) were authored against a
*hypothetical* synchronous API (`IngestQueue.enqueue(...)` returning an
id, `q.get(eid)`, `q.mark_failed(eid, err=..., retryable=True)`) that
never shipped. The actual implementation is async with different
signatures.

Rather than rewriting ~1500 LOC of brownfield tests to chase the real
API, we replaced them with focused contract tests:

- **MemoryStore semantics** → `tests/wiki_core/test_sqlite_memory_store.py`
  (48 tests covering enqueue/claim/mark_done/sha_seen/recover_stuck/
  concurrency/error paths).
- **WriteSink policy + serialisation** → `tests/wiki_core/test_write_sink.py`.
- **Search adapter shape** → `tests/wiki_core/test_search_adapter.py`.
- **Protocol-shape isinstance() checks** → `tests/wiki_core/test_protocols.py`.
- **Filters at the real signature** → `test_filters_current.py` here.
- **Security helpers** → `test_security_current.py` here.

The legacy files in *this* directory remain on-disk as a record of intent
but are skipped at collect time. To revive any, rewrite against the
current `src/wiki_ingest/queue.py` async API and remove the entry from
`collect_ignore` below.
"""

from __future__ import annotations

import hashlib
import uuid
from collections.abc import Callable, Iterator
from datetime import UTC, datetime
from pathlib import Path
from typing import Any
from unittest.mock import AsyncMock, MagicMock

import pytest

# --- Quarantine (M2 Sprint 2) ---------------------------------------------- #
collect_ignore = [
    "test_queue.py",
    "test_worker.py",
    "test_watcher.py",
    "test_e2e.py",
    "test_security.py",
    "test_filters.py",
]


# --------------------------------------------------------------------------- #
# Filesystem fixtures                                                         #
# --------------------------------------------------------------------------- #


@pytest.fixture
def tmp_wiki_root(tmp_path: Path) -> Path:
    """An empty raw/+wiki/ layout for end-to-end exercises."""
    raw = tmp_path / "knowledge-base" / "raw"
    wiki = tmp_path / "knowledge-base" / "wiki"
    for d in (raw, raw / "_ingested", raw / "articles", wiki):
        d.mkdir(parents=True, exist_ok=True)
    return tmp_path / "knowledge-base"


@pytest.fixture
def ingest_db_path(tmp_path: Path) -> Path:
    return tmp_path / "wiki-ingest.db"


@pytest.fixture
def event_factory() -> Iterator[Callable[..., dict[str, Any]]]:
    """Legacy dict-shaped event factory. Retained so the quarantined
    fixtures still resolve at collect time; new tests should build
    dataclasses directly."""

    def make(
        *,
        path: str = "/tmp/x.md",
        bucket: str = "articles",
        size: int = 1024,
        sha256: str | None = None,
    ) -> dict[str, Any]:
        return {
            "event_id": str(uuid.uuid4()),
            "path": path,
            "rel_path": Path(path).name,
            "bucket": bucket,
            "event_type": "created",
            "mtime": datetime.now(UTC).isoformat(),
            "size": size,
            "sha256": sha256 or hashlib.sha256(str(uuid.uuid4()).encode()).hexdigest(),
            "mime": "text/markdown",
        }

    yield make


@pytest.fixture
def mock_mcp_client() -> MagicMock:
    m = MagicMock()
    m.write_page = AsyncMock(return_value=None)
    m.append_to_log = AsyncMock(return_value=None)
    return m
