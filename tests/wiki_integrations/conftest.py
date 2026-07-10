"""Fixtures for the `wiki_integrations` substrate tests.

These tests assert on the new OAuth integrations layer landing for M3
(per PRD-005). They are isolated from the brownfield `tests/wiki_ingest/`
suite — no real network, no real Composio account, deterministic stubs
only.
"""

from __future__ import annotations

from collections.abc import AsyncIterator, Iterable
from datetime import timedelta
from typing import Any

import pytest


@pytest.fixture
def fetch_window() -> timedelta:
    """Default look-back window for provider sources under test."""
    return timedelta(hours=24)


class RecordingCallback:
    """Helper for capturing `on_event` invocations in tests.

    Behaves as an awaitable callable; tests inspect ``events`` after a
    `fetch_batch` to assert on the captured shape.
    """

    def __init__(self) -> None:
        self.events: list[Any] = []

    async def __call__(self, event: Any) -> None:  # noqa: D401 — callback shape
        self.events.append(event)


@pytest.fixture
def recording_callback() -> RecordingCallback:
    return RecordingCallback()


class FakeWalker:
    """A stand-in for `ComposioBridge` exposing only the `walk` surface.

    Tests pass a mapping of provider -> iterable of raw payloads; the
    walker yields each dict on demand.
    """

    def __init__(self, payloads: dict[str, Iterable[dict[str, Any]]]) -> None:
        # Materialise + defensive-copy so tests can mutate the source
        # without invalidating a previous walker.
        self._payloads = {k: [dict(item) for item in v] for k, v in payloads.items()}
        self.walked: list[str] = []

    async def walk(self, provider: str) -> AsyncIterator[dict[str, Any]]:
        self.walked.append(provider)
        for item in self._payloads.get(provider, []):
            yield dict(item)


@pytest.fixture
def fake_walker_factory() -> type[FakeWalker]:
    return FakeWalker
