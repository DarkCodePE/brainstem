"""
Memory Tree seal worker per [PRD-004 FR-4](../../docs/PRD-004-memory-tree.md).

The seal worker turns N child chunks (or sub-summaries) into a single
parent summary node. It's an async coroutine you can call manually or
schedule from a background task in the daemon.

This v1 ships **without the LLM call** — the wiring takes a
`Summariser` (protocol), and `NullSummariser` is the deterministic
default. Swap in an LLM-backed Summariser once PRD-008 model routing
lands; no other code in the seal worker changes.

What the seal worker does:

1. Resolve the child chunks for a source node (or topic node) from
   `ContentStore`.
2. Build `SummaryPart`s and call the Summariser.
3. Faithfulness gate: refuse the seal if the summary cites shas that
   weren't passed in (PRD-004 R-1 hallucination mitigation).
4. Persist the parent summary body via a `wiki_core.WriteSink`
   (PRD-004 FR-7 vault mirror) at a deterministic path.
5. Update the corresponding `tree_nodes` row with
   `summary_sha256 + sealed_at + score`.

The worker is **stateless** beyond the stores it's constructed with.
Multiple workers can run in parallel against the same DB (SQLite WAL
handles concurrent writes; tree node upsert is idempotent).
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING

from wiki_memory.content_store import ContentStore, StoredChunk
from wiki_memory.scoring import ScoreInputs, score_node
from wiki_memory.summariser import NullSummariser, Summariser, SummaryPart, SummaryResult
from wiki_memory.tree_nodes import TreeNodeStore

if TYPE_CHECKING:
    from wiki_core.protocols import WriteSink


VAULT_TREES_PREFIX = "wiki/trees/"  # PRD-004 FR-7 vault mirror location


@dataclass(frozen=True, slots=True)
class SealResult:
    node_id: str
    summary_sha256: str
    parent_token_count: int
    children_count: int
    page_path: str
    """The on-disk path under the WriteSink-allowed prefix."""


class SealError(RuntimeError):
    """Raised when a seal request can't be honoured (no children, summary
    cites a sha not passed in, or the WriteSink refuses the path)."""


class SealWorker:
    """Compose a parent summary from a node's children and persist it.

    Constructor wiring (DI all the way down):
    - `content_store` — pulls child chunk bodies.
    - `tree_store` — updates `summary_sha256` + `sealed_at` + `score`.
    - `write_sink` — emits the summary markdown into the vault.
    - `summariser` — defaults to `NullSummariser` for the no-LLM path.
    """

    def __init__(
        self,
        *,
        content_store: ContentStore,
        tree_store: TreeNodeStore,
        write_sink: WriteSink,
        summariser: Summariser | None = None,
    ) -> None:
        self._content = content_store
        self._tree = tree_store
        self._write = write_sink
        self._summariser: Summariser = summariser or NullSummariser()

    async def seal_source(self, *, source_id: str, node_id: str) -> SealResult:
        """Seal a source-level node: summarise all of its chunks into one
        summary, mirror to the vault, mark sealed in `tree_nodes`."""
        chunks = await self._content.list_by_source(source_id)
        if not chunks:
            raise SealError(f"no chunks for source_id={source_id!r}; cannot seal")

        parts = [
            SummaryPart(sha256=c.sha256, body=c.body, token_count=c.token_count) for c in chunks
        ]
        summary = await self._call_summariser(parts)

        page_path = f"{VAULT_TREES_PREFIX}sources/{node_id}.md"
        await self._mirror_summary(summary=summary, page_path=page_path, kind="source")

        # ADR-027 #156: persist full cited shas so chunk in-degree (the
        # pagerank-proxy scoring signal) is computable at recall time.
        # The vault frontmatter only keeps an 8-char prefix, unusable for
        # counting. Recorded BEFORE scoring so the fresh summary's own
        # citations feed the node score immediately.
        await self._content.record_citations(
            summary_sha256=summary.sha256,
            cited_shas=list(summary.cited_shas),
        )
        # ADR-027 #157: persist a live score into tree_nodes.score via the
        # `score` parameter mark_sealed already accepted (previously the
        # row kept its 0.0 placeholder forever).
        score = await self._score_source(chunks)
        await self._tree.mark_sealed(node_id, summary_sha256=summary.sha256, score=score)
        return SealResult(
            node_id=node_id,
            summary_sha256=summary.sha256,
            parent_token_count=summary.parent_token_count,
            children_count=len(parts),
            page_path=page_path,
        )

    async def seal_topic(
        self,
        *,
        topic_node_id: str,
        child_sub_summaries: Sequence[SummaryPart],
    ) -> SealResult:
        """Seal a topic node from a pre-collected set of child sub-summaries.

        The caller is responsible for choosing which children to fold in.
        This decouples the worker from any clustering policy (which the
        roadmap defers to a `topic_router` module that doesn't exist yet)."""
        if not child_sub_summaries:
            raise SealError(f"no children passed to seal_topic({topic_node_id!r})")

        summary = await self._call_summariser(child_sub_summaries)
        page_path = f"{VAULT_TREES_PREFIX}topics/{topic_node_id}.md"
        await self._mirror_summary(summary=summary, page_path=page_path, kind="topic")
        await self._tree.mark_sealed(topic_node_id, summary_sha256=summary.sha256)
        return SealResult(
            node_id=topic_node_id,
            summary_sha256=summary.sha256,
            parent_token_count=summary.parent_token_count,
            children_count=len(child_sub_summaries),
            page_path=page_path,
        )

    async def _score_source(self, chunks: Sequence[StoredChunk]) -> float | None:
        """Compose the source node's score (recency × reuse × pagerank-proxy)
        from its chunks and the tree-wide normalisers (ADR-027 #157).

        Aggregation per signal: recency from the newest chunk, reuse and
        in-degree from the strongest chunk — a source is as alive as its
        liveliest paragraph. Best-effort: returns None on any failure so a
        scoring hiccup never blocks the seal (`mark_sealed` keeps the
        previous score when passed None).
        """
        try:
            shas = [c.sha256 for c in chunks]
            in_degrees = await self._content.in_degrees(shas)
            max_reuse = await self._content.max_reuse()
            max_in_degree = await self._content.max_in_degree()
            return score_node(
                ScoreInputs(
                    created_at_iso=max(c.created_at for c in chunks),
                    reuse_count=max((c.reuse_count for c in chunks), default=0),
                    in_degree=max(in_degrees.values(), default=0),
                    tree_max_reuse=max(max_reuse, 1),
                    tree_max_in_degree=max(max_in_degree, 1),
                )
            )
        except Exception:  # noqa: BLE001
            return None

    async def _call_summariser(self, parts: Sequence[SummaryPart]) -> SummaryResult:
        summary = await self._summariser.summarise(parts)
        # PRD-004 R-1 faithfulness gate: refuse a summary that cites
        # shas we didn't pass in.
        passed_in = {p.sha256 for p in parts}
        for cited in summary.cited_shas:
            if cited not in passed_in:
                raise SealError(f"summariser cited sha not present in inputs: {cited[:12]}…")
        return summary

    async def _mirror_summary(
        self,
        *,
        summary: SummaryResult,
        page_path: str,
        kind: str,
    ) -> None:
        from wiki_core.protocols import Page, PageRef

        # PRD-004 FR-7: the summary lives in the vault as a real
        # Obsidian-compatible page with frontmatter pointing at the
        # cited children.
        frontmatter = {
            "title": f"Auto-summary ({kind})",
            "kind": kind,
            "summary_sha256": summary.sha256,
            "cited": [f"chunk:{s[:8]}" for s in summary.cited_shas],
            "date": _utcnow_iso(),
        }
        ref = PageRef(page_path=page_path, category="synthesis")  # type: ignore[arg-type]
        page = Page(ref=ref, frontmatter=frontmatter, body=summary.body)
        await self._write.write_page(page, mode="upsert")


def _utcnow_iso() -> str:
    from datetime import UTC, datetime

    return datetime.now(UTC).isoformat(timespec="seconds").replace("+00:00", "Z")


__all__ = ["SealError", "SealResult", "SealWorker"]
