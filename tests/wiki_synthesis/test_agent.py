"""SynthesisAgent contract tests (mock-first, London school)."""

from __future__ import annotations

import json
from datetime import UTC, datetime
from types import SimpleNamespace

import pytest

from wiki_synthesis.agent import SynthesisAgent

RAW = """---
clip: true
---

# AI Second Brain Notes

Claude Code is Anthropic's CLI agent. Claude Code pairs well with an
event sourcing pattern for the wiki.

![diagram](https://img.example/arch.png)

Canonical: https://example.com/original-post
"""

REL_PATH = "raw/bookmarks/My Bookmark (with spaces).md"

FIXED_CLOCK = lambda: datetime(2026, 6, 10, 12, 0, 0, tzinfo=UTC)  # noqa: E731


class FakeSink:
    """Records every write/index/log call the agent makes."""

    def __init__(self) -> None:
        self.pages: dict[str, str] = {}
        self.index_calls: list[tuple[str, str, str, int]] = []
        self.log_calls: list[tuple[str, str, str]] = []

    async def write_page(self, page_path: str, content: str) -> str:
        self.pages[page_path] = content
        return page_path

    async def update_index(self, page_path: str, category: str, summary: str, n: int) -> None:
        self.index_calls.append((page_path, category, summary, n))

    async def append_log(self, entry_type: str, title: str, details: str) -> None:
        self.log_calls.append((entry_type, title, details))


def make_agent(sink: FakeSink, router=None) -> SynthesisAgent:
    return SynthesisAgent(
        write_page=sink.write_page,
        update_index=sink.update_index,
        append_log=sink.append_log,
        router=router,
        clock=FIXED_CLOCK,
    )


class FailingRouter:
    def __init__(self) -> None:
        self.calls: list[list] = []

    async def call(self, task, *, messages):
        self.calls.append(messages)
        raise ConnectionError("LLM down")


class StubRouter:
    """Returns a fixed payload and records every call (budget audit)."""

    def __init__(self, payload: str) -> None:
        self.payload = payload
        self.calls: list[list] = []

    async def call(self, task, *, messages):
        self.calls.append(messages)
        return SimpleNamespace(text=self.payload)


GOOD_EXTRACTION = {
    "summary": (
        "Claude Code is Anthropic's CLI agent and pairs with an event "
        "sourcing pattern to build the second-brain wiki."
    ),
    "entities": [
        {
            "name": "Claude Code",
            "type": "tool",
            "description": "Anthropic's CLI agent used to build the wiki.",
        }
    ],
    "concepts": [
        {
            "name": "event sourcing pattern",
            "description": "Keeps every wiki state change replayable.",
        }
    ],
}


class TestDeterministicPath:
    @pytest.mark.asyncio
    async def test_creates_source_entity_concept_pages(self) -> None:
        sink = FakeSink()
        outcome = await make_agent(sink).synthesize(raw_text=RAW, rel_path=REL_PATH)

        assert outcome.source_page_path == "wiki/sources/ai-second-brain-notes.md"
        assert outcome.source_page_path in sink.pages
        assert any(p.startswith("wiki/entities/") for p in outcome.entity_page_paths)
        assert any(p.startswith("wiki/concepts/") for p in outcome.concept_page_paths)
        assert not outcome.llm_extracted

    @pytest.mark.asyncio
    async def test_source_page_preserves_images_urls_and_raw_path(self) -> None:
        sink = FakeSink()
        outcome = await make_agent(sink).synthesize(raw_text=RAW, rel_path=REL_PATH)
        page = sink.pages[outcome.source_page_path]
        assert "![diagram](https://img.example/arch.png)" in page
        assert "https://example.com/original-post" in page
        assert REL_PATH in page  # raw path in sources
        assert "origin: synthesized-deterministic" in page

    @pytest.mark.asyncio
    async def test_source_body_wikilinks_extracted_terms(self) -> None:
        sink = FakeSink()
        outcome = await make_agent(sink).synthesize(raw_text=RAW, rel_path=REL_PATH)
        page = sink.pages[outcome.source_page_path]
        assert "[[Claude Code]]" in page
        assert "[[event sourcing pattern]]" in page

    @pytest.mark.asyncio
    async def test_index_updated_per_page(self) -> None:
        sink = FakeSink()
        outcome = await make_agent(sink).synthesize(raw_text=RAW, rel_path=REL_PATH)
        indexed = {c[0] for c in sink.index_calls}
        assert outcome.source_page_path in indexed
        assert set(outcome.entity_page_paths) <= indexed
        assert set(outcome.concept_page_paths) <= indexed
        categories = {c[1] for c in sink.index_calls}
        assert categories == {"sources", "entities", "concepts"}

    @pytest.mark.asyncio
    async def test_log_entry_contains_raw_path_verbatim(self) -> None:
        """The Hermes batch detector marks files processed by substring
        match of the raw path in log/index — keep that contract."""
        sink = FakeSink()
        await make_agent(sink).synthesize(raw_text=RAW, rel_path=REL_PATH)
        assert len(sink.log_calls) == 1
        entry_type, _title, details = sink.log_calls[0]
        assert entry_type == "ingest"
        assert REL_PATH in details  # verbatim, spaces and all
        assert "Created 1 source," in details

    @pytest.mark.asyncio
    async def test_determinism_same_input_same_pages(self) -> None:
        sink_a, sink_b = FakeSink(), FakeSink()
        await make_agent(sink_a).synthesize(raw_text=RAW, rel_path=REL_PATH)
        await make_agent(sink_b).synthesize(raw_text=RAW, rel_path=REL_PATH)
        assert sink_a.pages == sink_b.pages


class TestRouterPath:
    """Primary path: ONE structured extraction call per file (#180)."""

    @pytest.mark.asyncio
    async def test_structured_extraction_happy_path(self) -> None:
        sink = FakeSink()
        router = StubRouter(json.dumps(GOOD_EXTRACTION))
        outcome = await make_agent(sink, router=router).synthesize(raw_text=RAW, rel_path=REL_PATH)

        assert outcome.llm_extracted
        page = sink.pages[outcome.source_page_path]
        assert "origin: llm-synthesized" in page
        assert GOOD_EXTRACTION["summary"].split(".")[0] in page.replace("[[", "").replace("]]", "")

        entity_page = sink.pages["wiki/entities/claude-code.md"]
        assert "Anthropic's CLI agent used to build the wiki." in entity_page
        assert "origin: llm-synthesized" in entity_page

        concept_page = sink.pages["wiki/concepts/event-sourcing-pattern.md"]
        assert "Keeps every wiki state change replayable." in concept_page
        assert "origin: llm-synthesized" in concept_page

    @pytest.mark.asyncio
    async def test_budget_exactly_one_router_call_per_file(self) -> None:
        sink = FakeSink()
        router = StubRouter(json.dumps(GOOD_EXTRACTION))
        await make_agent(sink, router=router).synthesize(raw_text=RAW, rel_path=REL_PATH)
        assert len(router.calls) == 1

    @pytest.mark.asyncio
    async def test_json_wrapped_in_prose_and_fences_still_parses(self) -> None:
        sink = FakeSink()
        payload = f"Here is the analysis:\n```json\n{json.dumps(GOOD_EXTRACTION)}\n```\nDone."
        outcome = await make_agent(sink, router=StubRouter(payload)).synthesize(
            raw_text=RAW, rel_path=REL_PATH
        )
        assert outcome.llm_extracted

    @pytest.mark.asyncio
    async def test_images_and_urls_reappended_when_model_omits_them(self) -> None:
        """The summary in GOOD_EXTRACTION contains neither the image
        ref nor the canonical URL — CODE must re-append both."""
        sink = FakeSink()
        outcome = await make_agent(sink, router=StubRouter(json.dumps(GOOD_EXTRACTION))).synthesize(
            raw_text=RAW, rel_path=REL_PATH
        )
        page = sink.pages[outcome.source_page_path]
        assert outcome.llm_extracted
        assert "![diagram](https://img.example/arch.png)" in page
        assert "https://example.com/original-post" in page
        assert REL_PATH in page  # raw path in frontmatter sources

    @pytest.mark.asyncio
    async def test_entity_concept_caps_enforced_in_code(self) -> None:
        oversized = {
            "summary": "A document naming many things.",
            "entities": [
                {"name": f"Tool Number {i}", "type": "tool", "description": "d"} for i in range(9)
            ],
            "concepts": [{"name": f"idea pattern {i}", "description": "d"} for i in range(8)],
        }
        sink = FakeSink()
        outcome = await make_agent(sink, router=StubRouter(json.dumps(oversized))).synthesize(
            raw_text=RAW, rel_path=REL_PATH
        )
        assert outcome.llm_extracted
        assert len(outcome.entity_page_paths) == 5
        assert len(outcome.concept_page_paths) == 5

    @pytest.mark.asyncio
    async def test_router_exception_degrades_to_heuristic_single_call(self) -> None:
        sink = FakeSink()
        router = FailingRouter()
        outcome = await make_agent(sink, router=router).synthesize(raw_text=RAW, rel_path=REL_PATH)
        page = sink.pages[outcome.source_page_path]
        assert not outcome.llm_extracted
        assert "origin: synthesized-deterministic" in page
        assert "![diagram](https://img.example/arch.png)" in page
        assert len(router.calls) == 1  # no second (refine) call after failure

    @pytest.mark.asyncio
    async def test_bad_json_degrades_to_heuristic(self) -> None:
        sink = FakeSink()
        outcome = await make_agent(sink, router=StubRouter("sorry, no JSON today")).synthesize(
            raw_text=RAW, rel_path=REL_PATH
        )
        page = sink.pages[outcome.source_page_path]
        assert not outcome.llm_extracted
        assert "origin: synthesized-deterministic" in page
        assert "[[Claude Code]]" in page  # heuristic extraction still ran

    @pytest.mark.asyncio
    async def test_empty_response_degrades_to_heuristic(self) -> None:
        sink = FakeSink()
        outcome = await make_agent(sink, router=StubRouter("")).synthesize(
            raw_text=RAW, rel_path=REL_PATH
        )
        assert not outcome.llm_extracted
        assert outcome.source_page_path in sink.pages

    @pytest.mark.asyncio
    async def test_empty_extraction_degrades_to_heuristic(self) -> None:
        empty = {"summary": "Something.", "entities": [], "concepts": []}
        sink = FakeSink()
        outcome = await make_agent(sink, router=StubRouter(json.dumps(empty))).synthesize(
            raw_text=RAW, rel_path=REL_PATH
        )
        assert not outcome.llm_extracted
        assert "origin: synthesized-deterministic" in sink.pages[outcome.source_page_path]


class TestAIFirstPreamble:
    """ADR-036 D4: every source page carries a `## For future Claude` note."""

    @pytest.mark.asyncio
    async def test_llm_path_source_page_has_preamble(self) -> None:
        sink = FakeSink()
        router = StubRouter(json.dumps({**GOOD_EXTRACTION, "relevance": "What and why."}))
        outcome = await make_agent(sink, router=router).synthesize(raw_text=RAW, rel_path=REL_PATH)
        page = sink.pages[outcome.source_page_path]
        assert "## For future Claude" in page
        assert "What and why." in page

    @pytest.mark.asyncio
    async def test_degrade_path_source_page_has_default_preamble(self) -> None:
        sink = FakeSink()
        outcome = await make_agent(sink).synthesize(raw_text=RAW, rel_path=REL_PATH)
        page = sink.pages[outcome.source_page_path]
        assert "## For future Claude" in page
        assert "origin: synthesized-deterministic" in page

    @pytest.mark.asyncio
    async def test_entity_pages_have_no_preamble(self) -> None:
        """Entity/concept pages stay stub/ledger-shaped (accretion owns them)."""
        sink = FakeSink()
        router = StubRouter(json.dumps(GOOD_EXTRACTION))
        outcome = await make_agent(sink, router=router).synthesize(raw_text=RAW, rel_path=REL_PATH)
        for path in outcome.entity_page_paths:
            assert "## For future Claude" not in sink.pages[path]

    @pytest.mark.asyncio
    async def test_index_summary_prefers_relevance(self) -> None:
        sink = FakeSink()
        payload = {**GOOD_EXTRACTION, "relevance": "Purpose-built index one-liner here."}
        router = StubRouter(json.dumps(payload))
        await make_agent(sink, router=router).synthesize(raw_text=RAW, rel_path=REL_PATH)
        source_summary = next(c[2] for c in sink.index_calls if c[1] == "sources")
        assert "Purpose-built index one-liner here." in source_summary


# --------------------------------------------------------------------------- #
# ADR-048 Fase 3: paper degrade distillation + per-page quality skips         #
# --------------------------------------------------------------------------- #

PAPER_RAW = (
    "# Attention Is Not Enough\n\n"
    "## Abstract\n\n"
    "We study the failure modes of long-context attention and show that a "
    "hierarchical memory beats brute-force context windows on retrieval-heavy "
    "tasks, reducing token cost by 20x while keeping accuracy.\n\n"
    "## 2. Contributions\n\n"
    "- A hierarchical memory tree with sealed summaries.\n"
    "- A deterministic distillation pass for ingest.\n\n"
    "## 5. Results\n\n"
    "On the vault benchmark the tree reaches 76.2% genuine rate versus 52% "
    "for the dump baseline, at 1/20th the token cost.\n\n"
    "## References\n\n"
    "[1] Vaswani et al. Attention is all you need. " + "Filler reference. " * 400
)

PAPER_REL_PATH = "raw/papers/attention-is-not-enough.md"


class TestPaperDegradeDistillation:
    @pytest.mark.asyncio
    async def test_paper_degrade_body_is_distilled_not_dumped(self) -> None:
        """D5: with the router down, a paper's body is its own distilled
        Abstract/Contributions/Results — never the full extraction dump."""
        sink = FakeSink()
        outcome = await make_agent(sink, router=FailingRouter()).synthesize(
            raw_text=PAPER_RAW, rel_path=PAPER_REL_PATH
        )
        page = sink.pages[outcome.source_page_path]
        assert "## Abstract" in page
        assert "## Results" in page
        assert "ADR-048 D5" in page
        # The References tail (the dump) must NOT be the body.
        assert "Filler reference." not in page
        assert "origin: synthesized-deterministic" in page

    @pytest.mark.asyncio
    async def test_non_paper_degrade_keeps_full_prose_body(self) -> None:
        """Bookmarks keep the classic never-summarise degrade contract."""
        sink = FakeSink()
        outcome = await make_agent(sink, router=FailingRouter()).synthesize(
            raw_text=RAW, rel_path=REL_PATH
        )
        page = sink.pages[outcome.source_page_path]
        assert "![diagram](https://img.example/arch.png)" in page  # refs preserved
        assert "ADR-048 D5" not in page

    @pytest.mark.asyncio
    async def test_shapeless_paper_falls_back_to_full_prose(self) -> None:
        """A paper-path file with no distillable shape degrades as before."""
        sink = FakeSink()
        outcome = await make_agent(sink, router=FailingRouter()).synthesize(
            raw_text="# Tiny\n\nTiny.", rel_path="raw/papers/tiny.md"
        )
        page = sink.pages[outcome.source_page_path]
        assert "Tiny." in page
        assert "ADR-048 D5" not in page


class SkippingSink(FakeSink):
    """write_page declines entity/concept stubs like the ADR-048 D4 skip tier."""

    async def write_page(self, page_path: str, content: str) -> str:
        if "/entities/" in page_path or "/concepts/" in page_path:
            from wiki_synthesis.agent import PageWriteSkippedError

            raise PageWriteSkippedError("quality-no_signal")
        return await super().write_page(page_path, content)


class TestPageWriteSkipped:
    @pytest.mark.asyncio
    async def test_skipped_stub_pages_do_not_abort_synthesis(self) -> None:
        sink = SkippingSink()
        router = StubRouter(json.dumps(GOOD_EXTRACTION))
        outcome = await make_agent(sink, router=router).synthesize(raw_text=RAW, rel_path=REL_PATH)
        # The source page still writes + indexes + logs.
        assert outcome.source_page_path in sink.pages
        assert any(c[1] == "sources" for c in sink.index_calls)
        assert sink.log_calls, "ingest log entry must still be appended"
        # Skipped stubs are absent from the outcome AND from the index.
        assert outcome.entity_page_paths == ()
        assert outcome.concept_page_paths == ()
        assert not any(c[1] in ("entities", "concepts") for c in sink.index_calls)
