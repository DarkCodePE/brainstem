"""
Summariser protocol + reference implementations for the Memory Tree seal
worker.

The seal worker (`wiki_memory.seal_worker`) needs a function that turns N
child chunk bodies into a single ≤3k-token parent summary. The
production implementation calls an LLM through PRD-008 routing — that
wiring is deferred until PRD-008 lands. For now we ship:

- `Summariser` — `typing.Protocol` declaring `async summarise(parts) ->
  SummaryResult`. Both the LLM path and the test stubs satisfy this.
- `NullSummariser` — concatenates children with a header. No LLM calls;
  zero cost; deterministic. Production-safe-ish for the M2 Sprint 4
  ship because it preserves citations (PRD-004 §"Summary drift"
  mitigation) and the seal worker's downstream wiring still gets
  exercised end-to-end.
- `CompositeSummariser` — chain of summarisers; first one to succeed
  wins. Useful for "try LLM, fall back to Null" deployments.

The seal worker takes a `Summariser` in its constructor — no global
state, no hidden coupling.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Protocol, runtime_checkable


@dataclass(frozen=True, slots=True)
class SummaryPart:
    """One child chunk or sub-summary headed for the parent summary."""

    sha256: str
    body: str
    token_count: int


@dataclass(frozen=True, slots=True)
class SummaryResult:
    """Output of a summariser call."""

    body: str
    """Markdown body of the parent summary. PRD-004 FR-7 wants this to
    cite child chunk IDs as `[[chunk:SHA8]]` so faithfulness can be
    audited later."""

    sha256: str
    """sha256(body). Used by the seal worker to populate
    `tree_nodes.summary_sha256`."""

    parent_token_count: int
    """Estimated token count of `body` for budgeting."""

    cited_shas: tuple[str, ...]
    """The set of child shas (PRD-004 FR-3) that appear in the
    summary. The faithfulness gate at seal time uses this — if the
    summary cites a sha we didn't pass in, refuse the seal."""


@runtime_checkable
class Summariser(Protocol):
    async def summarise(self, parts: Sequence[SummaryPart]) -> SummaryResult: ...


# --------------------------------------------------------------------------- #
# NullSummariser — deterministic, no LLM                                      #
# --------------------------------------------------------------------------- #


def _sha8(sha256: str) -> str:
    return sha256[:8]


class NullSummariser:
    """Concatenates children with a deterministic header.

    Useful for M2 Sprint 4 because it lets us land the seal worker and
    everything downstream (sealed_at population, vault mirror,
    `mark_sealed` flow) before PRD-008 model routing exists. Once
    routing lands, swap in an LLM-backed `Summariser` and delete this
    from the production path (keep it for tests).

    The output preserves child sha citations so any downstream
    faithfulness check has something to work with.
    """

    header: str

    def __init__(self, *, header: str = "Auto-generated stub summary") -> None:
        self.header = header

    async def summarise(self, parts: Sequence[SummaryPart]) -> SummaryResult:
        import hashlib

        cited = tuple(p.sha256 for p in parts)
        body_lines = [f"# {self.header}", ""]
        for part in parts:
            body_lines.append(f"- [[chunk:{_sha8(part.sha256)}]] {part.body[:200].strip()}")
        body = "\n".join(body_lines) + "\n"
        sha = hashlib.sha256(body.encode("utf-8")).hexdigest()
        return SummaryResult(
            body=body,
            sha256=sha,
            parent_token_count=max(1, len(body) // 4),
            cited_shas=cited,
        )


# --------------------------------------------------------------------------- #
# CompositeSummariser                                                         #
# --------------------------------------------------------------------------- #


class CompositeSummariser:
    """Chain of summarisers; the first one that returns wins.

    A summariser is considered to have failed if it raises. Useful for
    "try LLM, fall back to Null" deployments without leaking the
    fallback policy through the seal worker."""

    def __init__(self, *summarisers: Summariser) -> None:
        if not summarisers:
            raise ValueError("CompositeSummariser needs at least one delegate")
        self._delegates = summarisers

    async def summarise(self, parts: Sequence[SummaryPart]) -> SummaryResult:
        last_err: Exception | None = None
        for delegate in self._delegates:
            try:
                return await delegate.summarise(parts)
            except Exception as e:  # noqa: BLE001
                last_err = e
                continue
        raise RuntimeError("all summarisers in the chain failed") from last_err


__all__ = [
    "CompositeSummariser",
    "NullSummariser",
    "SummaryPart",
    "SummaryResult",
    "Summariser",
]
