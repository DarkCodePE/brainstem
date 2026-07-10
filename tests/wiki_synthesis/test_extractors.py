"""Deterministic extraction contract (ADR-035 D3 — London school)."""

from __future__ import annotations

from wiki_synthesis.extractors import (
    extract_concepts,
    extract_entities,
    extract_image_refs,
    extract_urls,
    strip_frontmatter,
)

SAMPLE = """---
title: raw note
---

# Building a Second Brain with Claude Code

Claude Code is Anthropic's CLI agent. I paired Claude Code with an
event sourcing pattern and a knowledge graph to build the wiki.
The `wiki_ingest` daemon and `SynthesisAgent` do the heavy lifting.

The event sourcing pattern keeps every state change replayable.

![diagram](https://img.example/arch.png)
![[local-embed.png]]

Source: https://example.com/post?id=1.
"""


class TestDeterminism:
    def test_same_input_same_entities(self) -> None:
        runs = [tuple(extract_entities(SAMPLE)) for _ in range(3)]
        assert len(set(runs)) == 1

    def test_same_input_same_concepts(self) -> None:
        runs = [tuple(extract_concepts(SAMPLE)) for _ in range(3)]
        assert len(set(runs)) == 1


class TestEntities:
    def test_finds_proper_nouns_and_identifiers(self) -> None:
        entities = extract_entities(SAMPLE, limit=10)
        assert "Claude Code" in entities
        assert "wiki_ingest" in entities
        assert "SynthesisAgent" in entities

    def test_sentence_start_words_filtered(self) -> None:
        text = "The cat sat. The dog ran. The bird flew. It was fine."
        assert extract_entities(text) == []

    def test_limit_respected(self) -> None:
        assert len(extract_entities(SAMPLE, limit=2)) <= 2

    def test_generic_acronyms_blacklisted(self) -> None:
        """Junk pages from the 2026-06-10 live batch (#180): LLM, GPU,
        KV must never become entities, however often they appear."""
        text = (
            "LLM inference on a GPU uses a KV cache. The LLM streams "
            "tokens while the GPU fills the KV store. LLM, GPU and KV "
            "again: LLM GPU KV."
        )
        entities = extract_entities(text, limit=10)
        assert "LLM" not in entities
        assert "GPU" not in entities
        assert "KV" not in entities

    def test_blacklist_is_case_insensitive_for_identifiers(self) -> None:
        text = "Run `llm` and `gpu` from the shell. Use `llm` twice."
        entities = extract_entities(text, limit=10)
        assert "llm" not in entities
        assert "gpu" not in entities

    def test_single_word_needs_three_consistent_mentions(self) -> None:
        twice = "Jarvis answered. Jarvis slept."
        thrice = "Jarvis answered. Jarvis slept. Jarvis woke."
        assert "Jarvis" not in extract_entities(twice, limit=10)
        assert "Jarvis" in extract_entities(thrice, limit=10)

    def test_default_cap_is_five(self) -> None:
        text = " ".join(f"Tool{i} Kit{i} works with Tool{i} Kit{i}." for i in range(12))
        assert len(extract_entities(text)) <= 5


class TestConcepts:
    def test_finds_pattern_phrases(self) -> None:
        concepts = extract_concepts(SAMPLE, limit=10)
        assert "event sourcing pattern" in concepts
        assert "knowledge graph" in concepts

    def test_frequency_ranks_first(self) -> None:
        concepts = extract_concepts(SAMPLE, limit=10)
        # "event sourcing pattern" appears twice, "knowledge graph" once.
        assert concepts.index("event sourcing pattern") < concepts.index("knowledge graph")

    def test_generic_led_phrases_blacklisted(self) -> None:
        text = "The llm pipeline feeds the gpu workflow and the event sourcing pattern."
        concepts = extract_concepts(text, limit=10)
        assert "llm pipeline" not in concepts
        assert "gpu workflow" not in concepts
        assert "event sourcing pattern" in concepts


class TestImagesAndUrls:
    def test_both_image_forms_extracted(self) -> None:
        refs = extract_image_refs(SAMPLE)
        assert "![diagram](https://img.example/arch.png)" in refs
        assert "![[local-embed.png]]" in refs

    def test_urls_extracted_and_trimmed(self) -> None:
        urls = extract_urls(SAMPLE)
        assert "https://example.com/post?id=1" in urls  # trailing '.' trimmed

    def test_strip_frontmatter(self) -> None:
        assert strip_frontmatter(SAMPLE).startswith("# Building")
