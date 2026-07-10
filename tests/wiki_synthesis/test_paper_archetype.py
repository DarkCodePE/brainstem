"""Paper synthesis archetype (PRD-015 FR-6): detection + prompt routing.

`type: paper` frontmatter (the FR-4 sidecar contract) or a
``raw/papers/`` rel_path must route the single structured router call
to the paper-shaped prompt (problem / method / key results with
numbers / limitations / why-it-matters); everything else keeps the
default extraction prompt unchanged.
"""

from __future__ import annotations

import json
from datetime import UTC, datetime
from types import SimpleNamespace

import pytest

from wiki_synthesis.agent import SynthesisAgent
from wiki_synthesis.structured import (
    ARCHETYPE_DEFAULT,
    ARCHETYPE_PAPER,
    EXTRACTION_SYSTEM_PROMPT,
    PAPER_EXTRACTION_SYSTEM_PROMPT,
    detect_archetype,
    extract_structured,
)

PAPER_RAW = """---
type: paper
arxiv_id: '2605.23904'
title: 'GEPA: Reflective Prompt Evolution'
---

# GEPA: Reflective Prompt Evolution

GEPA outperforms GRPO by 10% while using 35x fewer rollouts.
"""

BOOKMARK_RAW = "# Plain Bookmark\n\nJarvis is a personal AI assistant project.\n"

VALID = {
    "summary": (
        "**Problem** — prompt optimization is sample-hungry.\n"
        "**Key results** — GEPA beats GRPO by 10% with 35x fewer rollouts."
    ),
    "entities": [{"name": "GEPA", "type": "project", "description": "A prompt optimizer."}],
    "concepts": [{"name": "reflective prompt evolution", "description": "Evolving prompts."}],
}


class StubRouter:
    def __init__(self, payload: str = json.dumps(VALID)) -> None:
        self.payload = payload
        self.calls: list[tuple] = []

    async def call(self, task, *, messages):
        self.calls.append((task, messages))
        return SimpleNamespace(text=self.payload)


class TestDetectArchetype:
    def test_frontmatter_type_paper(self) -> None:
        assert detect_archetype(PAPER_RAW) == ARCHETYPE_PAPER

    def test_quoted_type_value(self) -> None:
        raw = '---\ntype: "paper"\n---\n\nbody\n'
        assert detect_archetype(raw) == ARCHETYPE_PAPER

    def test_papers_rel_path_wins_when_frontmatter_stripped(self) -> None:
        # The worker's SEC-05 envelope strips raw frontmatter before the
        # synthesis hook sees the body — the path is the honest signal.
        assert detect_archetype("# GEPA\n\nbody", "raw/papers/2605.23904.md") == ARCHETYPE_PAPER
        assert detect_archetype("# GEPA\n\nbody", "papers/2605.23904.md") == ARCHETYPE_PAPER

    def test_default_for_everything_else(self) -> None:
        assert detect_archetype(BOOKMARK_RAW) == ARCHETYPE_DEFAULT
        assert detect_archetype(BOOKMARK_RAW, "raw/bookmarks/x.md") == ARCHETYPE_DEFAULT

    def test_type_paper_in_body_without_frontmatter_is_default(self) -> None:
        assert detect_archetype("Some prose.\ntype: paper\nMore prose.") == ARCHETYPE_DEFAULT

    def test_other_frontmatter_type_is_default(self) -> None:
        assert detect_archetype("---\ntype: bookmark\n---\n\nbody\n") == ARCHETYPE_DEFAULT


class TestPromptRouting:
    @pytest.mark.asyncio
    async def test_paper_archetype_uses_paper_prompt(self) -> None:
        router = StubRouter()
        extraction = await extract_structured(
            PAPER_RAW, title="GEPA", router=router, archetype=ARCHETYPE_PAPER
        )
        assert extraction is not None
        _task, messages = router.calls[0]
        assert messages[0].content == PAPER_EXTRACTION_SYSTEM_PROMPT

    @pytest.mark.asyncio
    async def test_default_archetype_keeps_existing_prompt(self) -> None:
        router = StubRouter()
        await extract_structured(BOOKMARK_RAW, title="t", router=router)
        _task, messages = router.calls[0]
        assert messages[0].content == EXTRACTION_SYSTEM_PROMPT

    @pytest.mark.asyncio
    async def test_empty_router_text_degrades_with_a_warning(self, caplog) -> None:
        # The call "succeeds" but yields no text (reasoning models can
        # return empty content) — must degrade AND say so in the journal.
        router = StubRouter()
        router.payload = None  # SimpleNamespace(text=None)
        with caplog.at_level("WARNING", logger="wiki_synthesis.structured"):
            extraction = await extract_structured(
                PAPER_RAW, title="GEPA", router=router, archetype=ARCHETYPE_PAPER
            )
        assert extraction is None
        assert any("extraction_empty_text" in r.message for r in caplog.records)

    def test_paper_prompt_demands_the_paper_shape(self) -> None:
        for required in (
            "**Problem**",
            "**Method**",
            "**Key results**",
            "**Limitations**",
            "**Why it matters**",
            "CONCRETE NUMBERS",
            "wikilinked",
        ):
            assert required in PAPER_EXTRACTION_SYSTEM_PROMPT


class FakeSink:
    def __init__(self) -> None:
        self.pages: dict[str, str] = {}

    async def write_page(self, page_path: str, content: str) -> str:
        self.pages[page_path] = content
        return page_path

    async def update_index(self, page_path: str, category: str, summary: str, n: int) -> None:
        return None

    async def append_log(self, entry_type: str, title: str, details: str) -> None:
        return None


def make_agent(sink: FakeSink, router: StubRouter) -> SynthesisAgent:
    return SynthesisAgent(
        write_page=sink.write_page,
        update_index=sink.update_index,
        append_log=sink.append_log,
        router=router,
        clock=lambda: datetime(2026, 6, 10, 12, 0, 0, tzinfo=UTC),
    )


class TestAgentWiring:
    @pytest.mark.asyncio
    async def test_agent_routes_paper_file_to_paper_prompt(self) -> None:
        router = StubRouter()
        outcome = await make_agent(FakeSink(), router).synthesize(
            raw_text=PAPER_RAW, rel_path="raw/papers/2605.23904.md"
        )
        assert outcome.llm_extracted
        _task, messages = router.calls[0]
        assert messages[0].content == PAPER_EXTRACTION_SYSTEM_PROMPT

    @pytest.mark.asyncio
    async def test_agent_keeps_default_prompt_for_bookmarks(self) -> None:
        router = StubRouter()
        await make_agent(FakeSink(), router).synthesize(
            raw_text=BOOKMARK_RAW, rel_path="raw/bookmarks/My Bookmark.md"
        )
        _task, messages = router.calls[0]
        assert messages[0].content == EXTRACTION_SYSTEM_PROMPT
