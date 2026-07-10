"""
Tests for `wiki_integrations.registry.IntegrationRegistry`.

Coverage matrix:

| Behaviour                                  | Test                                  |
| ------------------------------------------ | ------------------------------------- |
| register() adds the source                 | test_register_adds_source             |
| Duplicate register() raises ValueError     | test_register_duplicate_raises        |
| `name in registry` works                   | test_contains_operator                |
| len(registry) reflects size                | test_len_reflects_count               |
| get() returns source                       | test_get_returns_registered           |
| get() returns None on miss                 | test_get_returns_none_for_unknown     |
| active() lists registered                  | test_active_returns_in_order          |
| active() returns a snapshot copy           | test_active_returns_copy              |
| unregister() returns True on hit           | test_unregister_returns_true_on_hit   |
| unregister() returns False on miss         | test_unregister_returns_false_on_miss |
| unregister() calls stop()                  | test_unregister_stops_source          |
| start_all() starts every registered source | test_start_all_starts_each            |
| stop_all() stops every registered source   | test_stop_all_stops_each              |
| start_all() tolerates a failing source     | test_start_all_tolerates_failure      |
| stop_all() tolerates a failing source      | test_stop_all_tolerates_failure       |
"""

from __future__ import annotations

from datetime import timedelta

import pytest

from wiki_integrations.base import OAuthIntegrationSource
from wiki_integrations.registry import IntegrationRegistry


class _RecordingSource(OAuthIntegrationSource):
    """Test double that counts start/stop calls and lets tests inject failures."""

    def __init__(
        self,
        name: str,
        *,
        on_event,
        fetch_window: timedelta = timedelta(hours=1),
        fail_on_start: bool = False,
        fail_on_stop: bool = False,
    ) -> None:
        super().__init__(name, fetch_window=fetch_window, on_event=on_event)
        self.start_calls = 0
        self.stop_calls = 0
        self._fail_on_start = fail_on_start
        self._fail_on_stop = fail_on_stop

    async def start(self) -> None:
        self.start_calls += 1
        if self._fail_on_start:
            raise RuntimeError("boom on start")
        await super().start()

    async def stop(self) -> None:
        self.stop_calls += 1
        if self._fail_on_stop:
            raise RuntimeError("boom on stop")
        await super().stop()

    async def fetch_batch(self):  # pragma: no cover — unused by registry tests
        return []


@pytest.fixture
def source_factory(recording_callback):
    def make(name: str, **kwargs) -> _RecordingSource:
        return _RecordingSource(name, on_event=recording_callback, **kwargs)

    return make


def test_register_adds_source(source_factory) -> None:
    reg = IntegrationRegistry()
    reg.register(source_factory("gmail"))
    assert "gmail" in reg
    assert len(reg) == 1


def test_register_duplicate_raises(source_factory) -> None:
    reg = IntegrationRegistry()
    reg.register(source_factory("gmail"))
    with pytest.raises(ValueError, match="already registered"):
        reg.register(source_factory("gmail"))


def test_contains_operator(source_factory) -> None:
    reg = IntegrationRegistry()
    reg.register(source_factory("github"))
    assert "github" in reg
    assert "gmail" not in reg
    # Non-string membership tests should return False, not raise.
    assert 123 not in reg


def test_len_reflects_count(source_factory) -> None:
    reg = IntegrationRegistry()
    assert len(reg) == 0
    reg.register(source_factory("gmail"))
    reg.register(source_factory("github"))
    assert len(reg) == 2


def test_get_returns_registered(source_factory) -> None:
    reg = IntegrationRegistry()
    src = source_factory("gmail")
    reg.register(src)
    assert reg.get("gmail") is src


def test_get_returns_none_for_unknown(source_factory) -> None:
    reg = IntegrationRegistry()
    assert reg.get("slack") is None


def test_active_returns_in_order(source_factory) -> None:
    reg = IntegrationRegistry()
    gmail = source_factory("gmail")
    github = source_factory("github")
    reg.register(gmail)
    reg.register(github)
    actives = reg.active()
    assert [s.name() for s in actives] == ["gmail", "github"]


def test_active_returns_copy(source_factory) -> None:
    reg = IntegrationRegistry()
    reg.register(source_factory("gmail"))
    first = reg.active()
    first.clear()  # mutate the returned list
    assert len(reg) == 1
    assert reg.active() != []  # internal state untouched


@pytest.mark.asyncio
async def test_unregister_returns_true_on_hit(source_factory) -> None:
    reg = IntegrationRegistry()
    reg.register(source_factory("gmail"))
    assert await reg.unregister("gmail") is True
    assert "gmail" not in reg


@pytest.mark.asyncio
async def test_unregister_returns_false_on_miss() -> None:
    reg = IntegrationRegistry()
    assert await reg.unregister("ghost") is False


@pytest.mark.asyncio
async def test_unregister_stops_source(source_factory) -> None:
    reg = IntegrationRegistry()
    src = source_factory("gmail")
    reg.register(src)
    await src.start()
    await reg.unregister("gmail")
    assert src.stop_calls >= 1
    assert src.started is False


@pytest.mark.asyncio
async def test_start_all_starts_each(source_factory) -> None:
    reg = IntegrationRegistry()
    a = source_factory("gmail")
    b = source_factory("github")
    reg.register(a)
    reg.register(b)
    await reg.start_all()
    assert a.start_calls == 1
    assert b.start_calls == 1
    assert a.started and b.started


@pytest.mark.asyncio
async def test_stop_all_stops_each(source_factory) -> None:
    reg = IntegrationRegistry()
    a = source_factory("gmail")
    b = source_factory("github")
    reg.register(a)
    reg.register(b)
    await reg.start_all()
    await reg.stop_all()
    assert a.stop_calls == 1
    assert b.stop_calls == 1
    assert not a.started and not b.started


@pytest.mark.asyncio
async def test_start_all_tolerates_failure(source_factory) -> None:
    reg = IntegrationRegistry()
    bad = source_factory("gmail", fail_on_start=True)
    good = source_factory("github")
    reg.register(bad)
    reg.register(good)
    # Must not raise — failed start is logged and the loop continues.
    await reg.start_all()
    assert good.started is True
    assert bad.started is False


@pytest.mark.asyncio
async def test_stop_all_tolerates_failure(source_factory) -> None:
    reg = IntegrationRegistry()
    bad = source_factory("gmail", fail_on_stop=True)
    good = source_factory("github")
    reg.register(bad)
    reg.register(good)
    await reg.start_all()
    # Must not raise.
    await reg.stop_all()
    assert good.started is False
