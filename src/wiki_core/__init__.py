"""
`wiki_core` — shared protocol contracts for the Second Brain Wiki harness.

Per [ADR-008](../../docs/ADR-008-language-strategy.md) and the M1 sprint 1
foundation work, this package defines the **typed Protocols** the rest of
the codebase implements. The point is to decouple the orchestrator
(`wiki_agent`) and the ingest pipeline (`wiki_ingest`) from each other via
small, mockable surfaces, and to give the rebuilt test suite (see
[#21](https://github.com/DarkCodePE/second-brain-wiki/issues/21)) a stable
target to assert against.

Re-exports the public Protocol types so callers can write::

    from wiki_core import IngestSource, MemoryStore, Search, WriteSink

instead of reaching into ``wiki_core.protocols``.
"""

from wiki_core.protocols import (
    IngestEvent,
    IngestSource,
    MemoryStore,
    Page,
    PageRef,
    RateLimiter,
    Search,
    SearchHit,
    WriteSink,
)

__all__ = [
    "IngestEvent",
    "IngestSource",
    "MemoryStore",
    "Page",
    "PageRef",
    "RateLimiter",
    "Search",
    "SearchHit",
    "WriteSink",
]
