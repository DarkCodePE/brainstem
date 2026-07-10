"""
Tests for the terminal `KeywordOnlyBackend` fallback.
"""

from __future__ import annotations

import pytest

from wiki_routing.providers.keyword_only import KeywordOnlyBackend
from wiki_routing.router import Message


@pytest.mark.asyncio
async def test_returns_keyword_tagged_response():
    backend = KeywordOnlyBackend()
    msg = Message(
        role="user", content="The Transformer architecture replaces recurrence with attention."
    )
    response = await backend.generate([msg])
    assert response.text.startswith("[keyword-only]")
    assert "transformer" in response.text.lower()
    assert response.cost_usd == 0.0


@pytest.mark.asyncio
async def test_extracts_first_sentence():
    backend = KeywordOnlyBackend()
    msg = Message(role="user", content="First sentence. Second sentence. Third.")
    response = await backend.generate([msg])
    assert "First sentence" in response.text


@pytest.mark.asyncio
async def test_returns_keywords_section():
    backend = KeywordOnlyBackend(max_keywords=3)
    msg = Message(
        role="user",
        content="Memory tree memory tree summary summary summary architecture",
    )
    response = await backend.generate([msg])
    # summary appears 3 times → first keyword
    assert "summary" in response.text.lower()


@pytest.mark.asyncio
async def test_empty_input():
    backend = KeywordOnlyBackend()
    msg = Message(role="user", content="")
    response = await backend.generate([msg])
    assert "[keyword-only]" in response.text


@pytest.mark.asyncio
async def test_picks_last_user_message():
    backend = KeywordOnlyBackend()
    msgs = [
        Message(role="user", content="early irrelevant text"),
        Message(role="assistant", content="some reply"),
        Message(role="user", content="memory tree summary"),
    ]
    response = await backend.generate(msgs)
    assert "memory" in response.text.lower() or "tree" in response.text.lower()


def test_quote_always_zero():
    backend = KeywordOnlyBackend()
    q = backend.quote([Message(role="user", content="anything")])
    assert q.estimated_usd == 0.0
    assert q.backend_label == backend.label


def test_max_keywords_validation():
    with pytest.raises(ValueError):
        KeywordOnlyBackend(max_keywords=0)
