"""
Per-stage and pipeline-level compression metrics.

`CompressionMetrics` records the deltas a single pipeline run produces:
the running token count after each stage, the time each stage took,
and the final ratio (compressed / original).

The token counting heuristic matches `wiki_memory.chunker` — 4 chars/token.
That is fine for *chunking* and *audit ratios* but is **not** billing-grade.
When [PRD-008](../../docs/PRD-008-model-routing.md) lands a real tokeniser
(tiktoken cl100k_base), the call sites should swap in that counter.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Final

#: Conservative heuristic — matches `wiki_memory.chunker.CHARS_PER_TOKEN`.
#: Documented at the call sites; not for billing.
CHARS_PER_TOKEN: Final[int] = 4


def count_tokens(text: str) -> int:
    """Estimate token count for *text* using the 4-char heuristic.

    Floors at 1 so an empty stage delta still has a divisible token count
    (mirrors `wiki_memory.chunker.count_tokens`).
    """
    return max(1, len(text) // CHARS_PER_TOKEN)


@dataclass(frozen=True, slots=True)
class StageDelta:
    """The recorded effect of a single stage."""

    name: str
    """Stage name (matches the pipeline's stage label)."""

    tokens_before: int
    """Token estimate immediately before this stage ran."""

    tokens_after: int
    """Token estimate immediately after this stage ran."""

    elapsed_ms: float
    """Wall-clock duration in milliseconds."""

    @property
    def delta(self) -> int:
        """Tokens removed (positive) or added (negative) by this stage."""
        return self.tokens_before - self.tokens_after


@dataclass(slots=True)
class CompressionMetrics:
    """Roll-up of a single pipeline invocation.

    Mutates as the pipeline runs (each stage appends a `StageDelta`),
    then is sealed at the end with `total_elapsed_ms` + `ratio`.
    """

    original_tokens: int
    compressed_tokens: int = 0
    total_elapsed_ms: float = 0.0
    stages: list[StageDelta] = field(default_factory=list)

    @property
    def ratio(self) -> float:
        """compressed / original — a value < 1.0 means tokens shrank."""
        if self.original_tokens == 0:
            return 1.0
        return self.compressed_tokens / self.original_tokens

    def record(self, delta: StageDelta) -> None:
        """Append a `StageDelta` and update the running total."""
        self.stages.append(delta)
        self.compressed_tokens = delta.tokens_after
        self.total_elapsed_ms += delta.elapsed_ms
