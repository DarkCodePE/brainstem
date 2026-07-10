"""
Memory Tree scoring per [PRD-004 FR-5](../../docs/PRD-004-memory-tree.md).

Each chunk and tree node carries a `score` in [0, 1] computed as a weighted
combination of three signals:

- **recency** — decays exponentially from the chunk's `created_at`.
  Fresh chunks ≈ 1.0; very old chunks → 0.
- **reuse** — how many recall calls have surfaced this chunk recently,
  normalised against a global maximum.
- **pagerank_proxy** — degree-in-tree (how many topic / global nodes
  cite this chunk via `summary_sha256` references), normalised against
  the tree-wide max in-degree.

The real pagerank computation needs the seal worker to have run at least
once (otherwise no parent summaries exist to count citations from), so
v1 of this module returns a `pagerank_proxy` of 0.0 if the tree has zero
sealed nodes. The other two signals (`recency`, `reuse`) work end-to-end
from day one.

Weights are configurable but default to (0.5, 0.3, 0.2) — recency
dominates while the tree is shallow, reuse takes over once the agent
has been querying for a while, and pagerank kicks in once the seal
worker has populated topic summaries.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Final

DEFAULT_RECENCY_WEIGHT: Final[float] = 0.5
DEFAULT_REUSE_WEIGHT: Final[float] = 0.3
DEFAULT_PAGERANK_WEIGHT: Final[float] = 0.2

#: Recency half-life in days. After this many days a chunk's recency
#: component drops to 0.5 of its initial value.
DEFAULT_RECENCY_HALFLIFE_DAYS: Final[float] = 30.0


@dataclass(frozen=True, slots=True)
class ScoreInputs:
    """All the signals required to compute a node score.

    Constructed by the scoring caller (worker or recall path) from data
    on hand. Keeping this a value type makes the scoring functions
    trivially testable with no DB dependency.
    """

    created_at_iso: str
    """ISO-8601 with 'Z' suffix (matches our codebase convention)."""

    reuse_count: int = 0
    """Number of times this node has been surfaced by recall. Bounded
    by the caller; we just normalise."""

    in_degree: int = 0
    """Number of parent summaries citing this node by `summary_sha256`."""

    tree_max_reuse: int = 1
    """Max reuse_count across the tree; used to normalise to [0, 1]."""

    tree_max_in_degree: int = 1
    """Max in_degree across the tree; used to normalise to [0, 1]."""


@dataclass(frozen=True, slots=True)
class ScoreWeights:
    recency: float = DEFAULT_RECENCY_WEIGHT
    reuse: float = DEFAULT_REUSE_WEIGHT
    pagerank: float = DEFAULT_PAGERANK_WEIGHT


def _parse_iso(s: str) -> datetime:
    if s.endswith("Z"):
        s = s[:-1] + "+00:00"
    return datetime.fromisoformat(s)


def recency_score(
    created_at_iso: str,
    *,
    now: datetime | None = None,
    halflife_days: float = DEFAULT_RECENCY_HALFLIFE_DAYS,
) -> float:
    """Exponential decay from creation.

    `score = 0.5 ** (age_days / halflife_days)`.

    A chunk created `halflife_days` ago scores 0.5; one created
    `2 * halflife_days` ago scores 0.25. Today's chunks score ≈ 1.0.
    """
    if now is None:
        now = datetime.now(UTC)
    created = _parse_iso(created_at_iso)
    if created.tzinfo is None:
        created = created.replace(tzinfo=UTC)
    age_days = max((now - created).total_seconds() / 86400.0, 0.0)
    return 0.5 ** (age_days / halflife_days)


def reuse_score(reuse_count: int, *, tree_max: int) -> float:
    """Linear normalisation against `tree_max`. Returns 0 if either
    input is 0 (no information yet)."""
    if tree_max <= 0 or reuse_count <= 0:
        return 0.0
    # Use log1p to compress the long tail — a chunk surfaced 100 times
    # shouldn't beat one surfaced 50 times by 2x.
    return min(math.log1p(reuse_count) / math.log1p(tree_max), 1.0)


def pagerank_proxy_score(in_degree: int, *, tree_max: int) -> float:
    """Degree-in-tree normalised against `tree_max`. Returns 0 if either
    input is 0 — this is the "tree not yet sealed" case."""
    if tree_max <= 0 or in_degree <= 0:
        return 0.0
    return min(in_degree / tree_max, 1.0)


def score_node(
    inputs: ScoreInputs,
    *,
    weights: ScoreWeights | None = None,
    now: datetime | None = None,
    halflife_days: float = DEFAULT_RECENCY_HALFLIFE_DAYS,
) -> float:
    """Compose the final score in [0, 1]."""
    w = weights or ScoreWeights()
    total_w = w.recency + w.reuse + w.pagerank
    if total_w <= 0:
        return 0.0
    recency = recency_score(inputs.created_at_iso, now=now, halflife_days=halflife_days)
    reuse = reuse_score(inputs.reuse_count, tree_max=inputs.tree_max_reuse)
    pagerank = pagerank_proxy_score(inputs.in_degree, tree_max=inputs.tree_max_in_degree)
    weighted = w.recency * recency + w.reuse * reuse + w.pagerank * pagerank
    return min(max(weighted / total_w, 0.0), 1.0)
