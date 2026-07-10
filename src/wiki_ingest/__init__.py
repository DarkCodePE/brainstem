from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

from wiki_ingest.config import Config

if TYPE_CHECKING:
    from wiki_ingest.adapter import SqliteMemoryStore

__all__ = ["Config", "Daemon", "open_memory_store"]


def __getattr__(name: str):
    if name == "Daemon":
        from wiki_ingest.daemon import Daemon

        return Daemon
    if name == "open_memory_store":
        return open_memory_store
    raise AttributeError(f"module 'wiki_ingest' has no attribute {name!r}")


async def open_memory_store(db_path: Path | str) -> SqliteMemoryStore:
    """Factory that opens a `wiki_core.MemoryStore`-shaped handle backed by
    the canonical wiki_ingest SQLite WAL queue.

    Use from any consumer that wants the protocol surface without
    importing the concrete `EventQueue` (e.g. Memory Tree workers per
    PRD-004, ad-hoc CLI scripts, future test harnesses). The factory
    awaits `init()` so the returned store is immediately usable.

    Example
    -------
    >>> from wiki_ingest import open_memory_store
    >>> store = await open_memory_store("~/.local/state/wiki_ingest/wiki-ingest.db")
    >>> await store.enqueue(event)
    """
    # Local import to keep the import graph lean for callers that only
    # need `Config`.
    from wiki_ingest.adapter import SqliteMemoryStore

    store = SqliteMemoryStore(Path(db_path).expanduser())
    await store.init()
    return store
