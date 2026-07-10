"""
Token-budgeted retrieval per [PRD-004 FR-6](../../docs/PRD-004-memory-tree.md)
and [ADR-027](../../docs/ADR-027-activate-recall-scoring-and-relevance-ordering.md).

`recall_leaves` packs leaf chunks under a hard token budget. Two modes:

- **Legacy (no `scores`)** — greedy in `chunk_index` order. Kept as the
  default so pre-ADR-027 callers (and `source_id`-scoped recall, where a
  single source's chunks are naturally read in order) are unchanged.
- **Scored (`scores` provided)** — *select* the chunks that survive the
  budget in **descending score order** (so the most relevant chunks win
  the truncation cut, not merely the earliest), then **present** the
  selected subset back in `chunk_index` order (so the LLM still sees the
  source paragraphs in written order). This is ADR-027 step 4.

The skeleton is stateless — callers pass already-loaded chunks (and,
for the scored path, a sha->score map built via `build_chunk_scores`).
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime

from wiki_memory.content_store import StoredChunk
from wiki_memory.scoring import ScoreInputs, ScoreWeights, score_node


@dataclass(frozen=True, slots=True)
class RecallBundle:
    """A token-budgeted slice of chunks ready for an LLM prompt."""

    chunks: list[StoredChunk]
    """Selected leaves, always presented in chunk_index ascending order."""

    total_tokens: int
    """Sum of `token_count` across `chunks`. Always ≤ `token_budget`."""

    truncated: bool
    """True iff one or more candidates were dropped to stay under budget."""


def build_chunk_scores(
    chunks: Sequence[StoredChunk],
    *,
    max_reuse: int,
    in_degrees: Mapping[str, int] | None = None,
    max_in_degree: int = 1,
    now: datetime | None = None,
    weights: ScoreWeights | None = None,
) -> dict[str, float]:
    """Compute a recall score in [0, 1] per chunk via `scoring.score_node`.

    Combines the three live signals (ADR-027): recency from `created_at`,
    reuse from the chunk's `reuse_count` normalised by `max_reuse`, and
    pagerank-proxy from the chunk's citation in-degree normalised by
    `max_in_degree`. Returns a `{sha256: score}` map.

    Pure given its inputs — the caller (MCP recall handler) supplies the
    normalisers and in-degree map from the content store.
    """
    in_degrees = in_degrees or {}
    norm_reuse = max(max_reuse, 1)
    norm_in_degree = max(max_in_degree, 1)
    out: dict[str, float] = {}
    for c in chunks:
        out[c.sha256] = score_node(
            ScoreInputs(
                created_at_iso=c.created_at,
                reuse_count=c.reuse_count,
                in_degree=in_degrees.get(c.sha256, 0),
                tree_max_reuse=norm_reuse,
                tree_max_in_degree=norm_in_degree,
            ),
            weights=weights,
            now=now,
        )
    return out


def recall_leaves(
    chunks: Sequence[StoredChunk],
    *,
    token_budget: int,
    scores: Mapping[str, float] | None = None,
) -> RecallBundle:
    """Token-budgeted packer.

    Without `scores`: greedy selection in `chunk_index` order (legacy).

    With `scores`: select chunks in **descending score** order until the
    budget is exhausted (relevance wins the truncation cut), then present
    the selected subset in `chunk_index` order (local coherence). A chunk
    missing from `scores` is treated as score 0.0. Ties break by
    `chunk_index` ascending so selection is deterministic.
    """
    if token_budget <= 0:
        return RecallBundle(chunks=[], total_tokens=0, truncated=bool(chunks))

    if scores is None:
        selection_order = sorted(chunks, key=lambda c: c.chunk_index)
    else:
        selection_order = sorted(
            chunks,
            key=lambda c: (-scores.get(c.sha256, 0.0), c.chunk_index),
        )

    selected: list[StoredChunk] = []
    total = 0
    truncated = False
    for chunk in selection_order:
        if total + chunk.token_count > token_budget:
            truncated = True
            continue
        selected.append(chunk)
        total += chunk.token_count

    # Present in chunk_index order regardless of selection order so the
    # LLM reads source paragraphs as written (ADR-027 step 4).
    selected.sort(key=lambda c: c.chunk_index)
    return RecallBundle(chunks=selected, total_tokens=total, truncated=truncated)
