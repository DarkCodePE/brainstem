"""
`wiki_memory` — Memory Tree v1 storage substrate (M2 Sprint 3).

Per [PRD-004 Memory Tree](../../docs/PRD-004-memory-tree.md), this package
implements the hierarchical summary compression machinery that PRD-004 calls
out. The v1 surface ships the **storage substrate** only:

- `chunker` — deterministic ≤3k-token chunking with stable sha256-keyed IDs
- `content_store` — durable SQLite store for chunk bodies
- `tree_nodes` — SQLite-backed tree node CRUD (source/topic/global kinds)
- `recall` — token-budgeted retrieval skeleton (leaf chunks; scoring stub)
- `seal_worker` — composes child chunks into parent summaries (M2-S4)
- `summariser` — `Summariser` Protocol + `NullSummariser` reference
- `summariser_factory` — picks the right `Summariser` from env (M3-S2)

The LLM-dependent pieces (seal jobs, RAGAS-graded summarisation, scoring
weights) ship in M2 Sprint 4 once the substrate is exercised in flight.

This package depends on `wiki_core.protocols` for the `MemoryStore` /
`WriteSink` surfaces it consumes, and on `wiki_ingest.open_memory_store`
for getting an event-queue handle. No direct dependency on `EventQueue` or
the MCP write handlers — everything flows through the protocols substrate
landed in PR #87.

### Constructing a `SealWorker` for production

The canonical entry point for assembling a fully-wired seal worker is
`build_default_seal_worker(content_store, tree_store, write_sink)`. It
delegates to `summariser_factory.build_default_summariser()` so the
seal flow uses an LLM-backed summariser when provider keys are present
in the environment and falls back to `NullSummariser` cleanly when
they aren't.

Callers that need a custom summariser (tests, dry-runs, or deployments
that pin a specific strategy) construct `SealWorker(...)` directly
with the desired `summariser=` argument — the constructor still
defaults to `NullSummariser` per M2-S4 behaviour so this is fully
backwards compatible.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from wiki_memory.chunker import Chunk, chunk_page, chunk_text, count_tokens
from wiki_memory.content_store import ContentStore
from wiki_memory.seal_worker import SealError, SealResult, SealWorker
from wiki_memory.summariser_factory import build_default_summariser
from wiki_memory.tree_nodes import TreeNode, TreeNodeStore

if TYPE_CHECKING:
    from wiki_core.protocols import WriteSink


def build_default_seal_worker(
    content_store: ContentStore,
    tree_store: TreeNodeStore,
    write_sink: WriteSink,
) -> SealWorker:
    """Assemble a `SealWorker` wired with the environment's preferred
    summariser.

    This is the canonical construction path for production callers
    (daemon, agent factory, CLI). It picks an LLM-backed summariser via
    `build_default_summariser()` when provider keys are available and
    falls back to `NullSummariser` otherwise, so the same call works
    online and offline.

    Callers that need a different `Summariser` should construct
    `SealWorker(...)` directly with the `summariser=` argument; the
    constructor still defaults to `NullSummariser` for that path.

    Parameters
    ----------
    content_store:
        Durable chunk storage; the seal worker reads child chunks by
        source.
    tree_store:
        Tree node CRUD; the seal worker marks the sealed row.
    write_sink:
        Vault mirror; the seal worker writes the parent summary
        markdown here.

    Returns
    -------
    SealWorker
        Ready to seal — no further wiring needed.
    """
    return SealWorker(
        content_store=content_store,
        tree_store=tree_store,
        write_sink=write_sink,
        summariser=build_default_summariser(),
    )


__all__ = [
    "Chunk",
    "ContentStore",
    "SealError",
    "SealResult",
    "SealWorker",
    "TreeNode",
    "TreeNodeStore",
    "build_default_seal_worker",
    "build_default_summariser",
    "chunk_page",
    "chunk_text",
    "count_tokens",
]
