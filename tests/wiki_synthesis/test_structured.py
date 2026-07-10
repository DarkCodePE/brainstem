"""Structured router extraction: defensive parse + degrade contract (#180)."""

from __future__ import annotations

import json
from types import SimpleNamespace

import pytest

from wiki_synthesis.structured import (
    EXTRACTION_SYSTEM_PROMPT,
    extract_structured,
)

RAW = """---
clip: true
---

# Jarvis Notes

Jarvis is a personal AI assistant project by Hugo Bowne-Anderson.
"""

VALID = {
    "summary": "Jarvis is a personal AI assistant project.",
    "entities": [
        {"name": "Jarvis", "type": "project", "description": "A personal AI assistant."},
        {"name": "Hugo Bowne-Anderson", "type": "person", "description": "Its author."},
    ],
    "concepts": [
        {"name": "hybrid llm approach", "description": "Mixing local and cloud models."},
    ],
}


class StubRouter:
    def __init__(self, payload: str) -> None:
        self.payload = payload
        self.calls: list[tuple] = []

    async def call(self, task, *, messages):
        self.calls.append((task, messages))
        return SimpleNamespace(text=self.payload)


async def run(payload: str, **kwargs):
    return await extract_structured(RAW, title="Jarvis Notes", router=StubRouter(payload), **kwargs)


class TestHappyPath:
    @pytest.mark.asyncio
    async def test_valid_json_parses(self) -> None:
        extraction = await run(json.dumps(VALID))
        assert extraction is not None
        assert extraction.summary == VALID["summary"]
        assert [e.name for e in extraction.entities] == ["Jarvis", "Hugo Bowne-Anderson"]
        assert extraction.entities[0].type == "project"
        assert extraction.concepts[0].description == "Mixing local and cloud models."

    @pytest.mark.asyncio
    async def test_first_json_object_extracted_from_prose(self) -> None:
        extraction = await run(f"Sure! Here it is:\n```json\n{json.dumps(VALID)}\n```")
        assert extraction is not None
        assert extraction.entities[0].name == "Jarvis"

    @pytest.mark.asyncio
    async def test_call_sends_prompt_and_document(self) -> None:
        router = StubRouter(json.dumps(VALID))
        await extract_structured(RAW, title="Jarvis Notes", router=router)
        assert len(router.calls) == 1
        task, messages = router.calls[0]
        assert task.intent == "ingest"
        assert messages[0].content == EXTRACTION_SYSTEM_PROMPT
        assert "Title: Jarvis Notes" in messages[1].content
        assert "Jarvis is a personal AI assistant" in messages[1].content
        assert "clip: true" not in messages[1].content  # frontmatter stripped

    @pytest.mark.asyncio
    async def test_unknown_entity_type_coerced_and_duplicates_skipped(self) -> None:
        payload = {
            "summary": "s",
            "entities": [
                {"name": "Jarvis", "type": "BANANA", "description": "d"},
                {"name": "jarvis", "type": "project", "description": "dup"},
                "not-a-dict",
                {"type": "tool", "description": "nameless"},
            ],
            "concepts": [],
        }
        extraction = await run(json.dumps(payload))
        assert extraction is not None
        assert len(extraction.entities) == 1
        assert extraction.entities[0].type == "tool"  # coerced default

    @pytest.mark.asyncio
    async def test_caps_enforced_in_code(self) -> None:
        payload = {
            "summary": "s",
            "entities": [
                {"name": f"E{i} Tool", "type": "tool", "description": "d"} for i in range(9)
            ],
            "concepts": [{"name": f"c{i} pattern", "description": "d"} for i in range(9)],
        }
        extraction = await run(json.dumps(payload), max_entities=5, max_concepts=5)
        assert extraction is not None
        assert len(extraction.entities) == 5
        assert len(extraction.concepts) == 5

    @pytest.mark.asyncio
    async def test_malformed_concept_items_skipped(self) -> None:
        payload = {
            "summary": "s",
            "entities": [],
            "concepts": [
                "not-a-dict",
                {"description": "nameless"},
                {"name": "event sourcing", "description": "good"},
                {"name": "Event Sourcing", "description": "dup"},
            ],
        }
        extraction = await run(json.dumps(payload))
        assert extraction is not None
        assert [c.name for c in extraction.concepts] == ["event sourcing"]
        assert extraction.concepts[0].description == "good"


class TestRelevance:
    """ADR-036 D4: the AI-first `relevance` preamble (zero new LLM call)."""

    @pytest.mark.asyncio
    async def test_relevance_parsed_when_present(self) -> None:
        payload = {**VALID, "relevance": "What this is and why a future reader cares."}
        extraction = await run(json.dumps(payload))
        assert extraction is not None
        assert extraction.relevance == "What this is and why a future reader cares."

    @pytest.mark.asyncio
    async def test_relevance_falls_back_to_summary_when_absent(self) -> None:
        extraction = await run(json.dumps(VALID))  # VALID has no `relevance`
        assert extraction is not None
        assert "Jarvis is a personal AI assistant project." in extraction.relevance

    @pytest.mark.asyncio
    async def test_relevance_fallback_takes_first_two_sentences(self) -> None:
        payload = {**VALID, "summary": "One. Two. Three. Four."}
        extraction = await run(json.dumps(payload))
        assert extraction is not None
        assert extraction.relevance == "One. Two."

    @pytest.mark.asyncio
    async def test_relevance_falls_back_when_non_string(self) -> None:
        payload = {**VALID, "relevance": 123}
        extraction = await run(json.dumps(payload))
        assert extraction is not None
        assert "Jarvis" in extraction.relevance  # fell back, did not stringify 123


class TestDegrade:
    @pytest.mark.asyncio
    async def test_no_router_returns_none(self) -> None:
        assert await extract_structured(RAW, title="t", router=None) is None

    @pytest.mark.asyncio
    async def test_router_exception_returns_none(self) -> None:
        class Boom:
            async def call(self, task, *, messages):
                raise ConnectionError("down")

        assert await extract_structured(RAW, title="t", router=Boom()) is None

    @pytest.mark.asyncio
    async def test_router_without_callable_returns_none(self) -> None:
        assert await extract_structured(RAW, title="t", router=object()) is None

    @pytest.mark.asyncio
    async def test_non_string_text_returns_none(self) -> None:
        class WeirdRouter:
            async def call(self, task, *, messages):
                return SimpleNamespace(text=42)

        assert await extract_structured(RAW, title="t", router=WeirdRouter()) is None

    @pytest.mark.asyncio
    async def test_empty_response_returns_none(self) -> None:
        assert await run("   ") is None

    @pytest.mark.asyncio
    async def test_bad_json_returns_none(self) -> None:
        assert await run("no json here") is None
        assert await run("{broken json") is None

    @pytest.mark.asyncio
    async def test_non_object_json_returns_none(self) -> None:
        assert await run('["a", "list"]') is None

    @pytest.mark.asyncio
    async def test_missing_or_blank_summary_returns_none(self) -> None:
        assert await run(json.dumps({"entities": VALID["entities"], "concepts": []})) is None
        assert await run(json.dumps({"summary": "  ", "entities": VALID["entities"]})) is None

    @pytest.mark.asyncio
    async def test_empty_extraction_returns_none(self) -> None:
        assert await run(json.dumps({"summary": "s", "entities": [], "concepts": []})) is None
        assert await run(json.dumps({"summary": "s", "entities": "nope", "concepts": None})) is None
