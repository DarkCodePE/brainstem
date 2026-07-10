"""Page-level deterministic accretion (ADR-036).

Two layers under test:

1. The pure ``reconcile`` functions — ``accrete_source_page`` (``## History``
   provenance ledger) and ``accrete_mention_page`` (entity/concept ``sources``
   union + per-source ``## Mentions`` bullet, with the legacy-rich-body guard).
2. The ``SynthesisAgent`` wiring — and the load-bearing guarantee that with
   NO ``read_page`` injected the output is byte-identical to today.
"""

from __future__ import annotations

import json
from datetime import UTC, datetime
from types import SimpleNamespace

import pytest

from wiki_synthesis.agent import SynthesisAgent
from wiki_synthesis.reconcile import (
    Accretion,
    accrete_mention_page,
    accrete_source_page,
)
from wiki_synthesis.templates import render_entity_page, render_source_page

FIXED_CLOCK = lambda: datetime(2026, 6, 10, 12, 0, 0, tzinfo=UTC)  # noqa: E731
LATER_CLOCK = lambda: datetime(2026, 6, 14, 9, 0, 0, tzinfo=UTC)  # noqa: E731


def _entity(*, name, source, mention, date="2026-06-10", origin="llm-synthesized") -> str:
    return render_entity_page(
        name=name, date=date, source_page_path=source, mention=mention, origin=origin
    )


def _source(*, title, body, sources, date="2026-06-10") -> str:
    return render_source_page(
        title=title,
        date=date,
        sources=sources,
        tags=["ingested"],
        origin="llm-synthesized",
        body=body,
    )


# --------------------------------------------------------------------------- #
# accrete_mention_page — entity/concept union (D1)                             #
# --------------------------------------------------------------------------- #


class TestAccreteMentionPage:
    def test_different_source_appends_mention_and_unions_sources(self) -> None:
        prior = _entity(name="Claude Code", source="wiki/sources/a.md", mention="A's take.")
        fresh = _entity(name="Claude Code", source="wiki/sources/b.md", mention="B's take.")

        acc = accrete_mention_page(prior, fresh, now=LATER_CLOCK())

        assert isinstance(acc, Accretion)
        assert acc.accreted is True
        assert acc.source_count == 2
        # both sources present in frontmatter, source_count bumped
        assert 'sources: ["wiki/sources/a.md", "wiki/sources/b.md"]' in acc.text
        assert "source_count: 2" in acc.text
        # prior body preserved as the canonical overview; new source ledgered
        assert "A's take." in acc.text
        assert "## Mentions" in acc.text
        assert "- [[wiki/sources/b.md]] (2026-06-14): B's take." in acc.text

    def test_same_source_reingest_replaces_its_own_bullet(self) -> None:
        prior = _entity(name="Claude Code", source="wiki/sources/a.md", mention="canonical body")
        # first different source adds a bullet
        once = accrete_mention_page(
            prior,
            _entity(name="Claude Code", source="wiki/sources/b.md", mention="old B"),
            now=FIXED_CLOCK(),
        )
        # re-ingest the SAME source b with a new mention
        twice = accrete_mention_page(
            once.text,
            _entity(name="Claude Code", source="wiki/sources/b.md", mention="new B"),
            now=LATER_CLOCK(),
        )

        assert twice.source_count == 2  # union of {a, b} — b not double-counted
        assert twice.text.count("[[wiki/sources/b.md]]") == 1  # own bullet replaced, not duplicated
        assert "new B" in twice.text
        assert "old B" not in twice.text

    def test_legacy_rich_body_is_preserved_not_clobbered(self) -> None:
        # A rich legacy entity page (Hermes-era), no ## Mentions ledger yet.
        legacy = (
            '---\ntitle: "Cole Medin"\ndate: 2026-04-14\n'
            'sources: ["wiki/sources/legacy.md"]\ntags: ["entity"]\n'
            "origin: llm-synthesized\ncategory: entities\nsource_count: 1\n---\n\n"
            "# Cole Medin\n\n## Overview\nA prolific builder.\n\n"
            "## Key Contributions\n- archon\n- context engineering\n"
        )
        thin_stub = _entity(
            name="Cole Medin", source="wiki/sources/new.md", mention="Mentioned briefly."
        )

        acc = accrete_mention_page(legacy, thin_stub, now=LATER_CLOCK())

        # the rich overview survives intact
        assert "## Overview" in acc.text
        assert "A prolific builder." in acc.text
        assert "## Key Contributions" in acc.text
        assert "- archon" in acc.text
        # and the new thin mention is ledgered, not promoted over the overview
        assert "- [[wiki/sources/new.md]] (2026-06-14): Mentioned briefly." in acc.text
        assert acc.source_count == 2

    def test_unparseable_prior_degrades_to_fresh(self) -> None:
        fresh = _entity(name="X", source="wiki/sources/b.md", mention="m")
        acc = accrete_mention_page("not a page at all", fresh, now=FIXED_CLOCK())
        assert acc.accreted is False
        assert acc.text == fresh


# --------------------------------------------------------------------------- #
# accrete_source_page — ## History ledger (D2)                                 #
# --------------------------------------------------------------------------- #


class TestAccreteSourcePage:
    def test_appends_history_with_prior_summary_count_stays_one(self) -> None:
        prior = _source(title="Doc", body="The original synthesis prose.", sources=["raw/x.md"])
        fresh = _source(
            title="Doc", body="A fresher synthesis prose.", sources=["raw/x.md"], date="2026-06-14"
        )

        acc = accrete_source_page(prior, fresh, now=LATER_CLOCK())

        assert acc.accreted is True
        assert acc.source_count == 1  # a source page references exactly one source doc
        assert "A fresher synthesis prose." in acc.text  # fresh body on top
        assert "## History" in acc.text
        assert "- 2026-06-10: The original synthesis prose." in acc.text

    def test_history_carries_forward_prior_entries(self) -> None:
        prior = _source(title="Doc", body="v1 prose.", sources=["raw/x.md"])
        mid = accrete_source_page(
            prior,
            _source(title="Doc", body="v2 prose.", sources=["raw/x.md"], date="2026-06-12"),
            now=datetime(2026, 6, 12, tzinfo=UTC),
        )
        latest = accrete_source_page(
            mid.text,
            _source(title="Doc", body="v3 prose.", sources=["raw/x.md"], date="2026-06-14"),
            now=LATER_CLOCK(),
        )

        # all prior summaries retained, newest first
        assert "v3 prose." in latest.text
        assert "- 2026-06-12: v2 prose." in latest.text
        assert "- 2026-06-10: v1 prose." in latest.text
        assert latest.text.index("v2 prose.") < latest.text.index("v1 prose.")

    def test_unparseable_prior_degrades_to_fresh(self) -> None:
        fresh = _source(title="Doc", body="b", sources=["raw/x.md"])
        acc = accrete_source_page("", fresh, now=FIXED_CLOCK())
        assert acc.accreted is False
        assert acc.text == fresh


# --------------------------------------------------------------------------- #
# SynthesisAgent wiring — byte-identity guarantee + integration                #
# --------------------------------------------------------------------------- #

RAW_ONE = "# Doc One\n\nClaude Code is Anthropic's CLI agent.\n"
RAW_TWO = "# Doc Two\n\nClaude Code keeps improving.\n"

GOOD = {
    "summary": "Claude Code is Anthropic's CLI agent for building the wiki.",
    "entities": [{"name": "Claude Code", "type": "tool", "description": "Anthropic's CLI agent."}],
    "concepts": [],
}


class _Sink:
    def __init__(self) -> None:
        self.pages: dict[str, str] = {}
        self.index: list[tuple[str, str, str, int]] = []
        self.logs: list[tuple[str, str, str]] = []

    async def write_page(self, path: str, content: str) -> str:
        self.pages[path] = content
        return path

    async def update_index(self, path: str, category: str, summary: str, n: int) -> None:
        self.index.append((path, category, summary, n))

    async def append_log(self, t: str, title: str, details: str) -> None:
        self.logs.append((t, title, details))

    async def read_page(self, path: str) -> str | None:
        return self.pages.get(path)


class _StubRouter:
    def __init__(self, payload: str) -> None:
        self.payload = payload

    async def call(self, task, *, messages):
        return SimpleNamespace(text=self.payload)


def _agent(sink: _Sink, *, read_page=None, router=None) -> SynthesisAgent:
    return SynthesisAgent(
        write_page=sink.write_page,
        update_index=sink.update_index,
        append_log=sink.append_log,
        router=router,
        read_page=read_page,
        clock=FIXED_CLOCK,
    )


class TestAgentByteIdentity:
    @pytest.mark.asyncio
    async def test_no_read_page_is_byte_identical_to_read_page_returning_none(self) -> None:
        """The accretion seam must not perturb the existing output: an agent
        with no ``read_page`` and an agent whose ``read_page`` finds nothing
        must produce exactly the same pages (the degrade contract)."""
        without = _Sink()
        with_empty = _Sink()

        async def no_prior(_path: str) -> str | None:
            return None

        await _agent(without).synthesize(raw_text=RAW_ONE, rel_path="raw/doc-one.md")
        await _agent(with_empty, read_page=no_prior).synthesize(
            raw_text=RAW_ONE, rel_path="raw/doc-one.md"
        )

        assert without.pages == with_empty.pages
        # and source_count is the untouched default
        assert all("source_count: 1" in p for p in without.pages.values())


class TestAgentAccretionIntegration:
    @pytest.mark.asyncio
    async def test_second_source_accretes_shared_entity_page(self) -> None:
        sink = _Sink()
        router = _StubRouter(json.dumps(GOOD))
        agent = _agent(sink, read_page=sink.read_page, router=router)

        await agent.synthesize(raw_text=RAW_ONE, rel_path="raw/doc-one.md")
        await agent.synthesize(raw_text=RAW_TWO, rel_path="raw/doc-two.md")

        entity = sink.pages["wiki/entities/claude-code.md"]
        assert "source_count: 2" in entity
        assert "wiki/sources/doc-one.md" in entity
        assert "wiki/sources/doc-two.md" in entity
        assert "## Mentions" in entity
        # the index entry for the entity reflects the bumped count
        entity_counts = [n for (p, _c, _s, n) in sink.index if p == "wiki/entities/claude-code.md"]
        assert entity_counts[-1] == 2

    @pytest.mark.asyncio
    async def test_reingesting_same_source_adds_history_to_source_page(self) -> None:
        sink = _Sink()
        router = _StubRouter(json.dumps(GOOD))
        agent = _agent(sink, read_page=sink.read_page, router=router)

        await agent.synthesize(raw_text=RAW_ONE, rel_path="raw/doc-one.md")
        await agent.synthesize(raw_text=RAW_ONE, rel_path="raw/doc-one.md")

        source = sink.pages["wiki/sources/doc-one.md"]
        assert "## History" in source
        assert "source_count: 1" in source  # still one source document
