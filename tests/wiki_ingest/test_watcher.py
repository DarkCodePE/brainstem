"""
Tests for `wiki_ingest.watcher` — watchdog-based observer with per-path
debounce (3 s window) per ADR-006 §Debouncing.

Contract:
    watcher = IngestWatcher(
        root: Path,
        on_debounced: Callable[[Path, str], Awaitable[None] | None],
        debounce_seconds: float = 3.0,
        filters: Filters,
    )
    await watcher.start()
    await watcher.stop()
    watcher.handle_raw(event)   # test seam for synthetic events
"""

from __future__ import annotations

import asyncio
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

wiki_ingest = pytest.importorskip(
    "wiki_ingest",
    reason="core not implemented yet",
)
watcher_mod = pytest.importorskip(
    "wiki_ingest.watcher",
    reason="core not implemented yet",
)


# --------------------------------------------------------------------------- #
# Fixtures local to watcher tests                                             #
# --------------------------------------------------------------------------- #


@pytest.fixture
def fake_observer():
    with patch.object(watcher_mod, "Observer", autospec=True) as mock_obs:
        instance = MagicMock(name="Observer")
        mock_obs.return_value = instance
        yield instance


@pytest.fixture
def recorder():
    calls: list[tuple[str, str]] = []

    async def _on_debounced(path: Path, event_type: str) -> None:
        calls.append((str(path), event_type))

    _on_debounced.calls = calls  # type: ignore[attr-defined]
    return _on_debounced


# --------------------------------------------------------------------------- #
# Observer wiring                                                             #
# --------------------------------------------------------------------------- #


@pytest.mark.asyncio
async def test_start_schedules_recursive_watch_on_raw(
    tmp_wiki_root: Path, fake_observer: MagicMock, recorder
) -> None:
    w = watcher_mod.IngestWatcher(root=tmp_wiki_root, on_debounced=recorder, debounce_seconds=0.2)
    await w.start()
    # Expect a single recursive schedule on raw/
    call = fake_observer.schedule.call_args_list[0]
    (
        _handler,
        path,
    ) = call.args[:2]
    kwargs = call.kwargs
    assert Path(path) == tmp_wiki_root / "raw"
    assert kwargs.get("recursive", False) is True
    fake_observer.start.assert_called_once()
    await w.stop()


@pytest.mark.asyncio
async def test_stop_shuts_observer_cleanly(
    tmp_wiki_root: Path, fake_observer: MagicMock, recorder
) -> None:
    w = watcher_mod.IngestWatcher(root=tmp_wiki_root, on_debounced=recorder, debounce_seconds=0.2)
    await w.start()
    await w.stop()
    fake_observer.stop.assert_called_once()
    fake_observer.join.assert_called_once()


# --------------------------------------------------------------------------- #
# Debouncing                                                                  #
# --------------------------------------------------------------------------- #


@pytest.mark.asyncio
async def test_burst_on_same_path_collapses_to_one_callback(
    tmp_wiki_root: Path, fake_observer: MagicMock, recorder
) -> None:
    w = watcher_mod.IngestWatcher(root=tmp_wiki_root, on_debounced=recorder, debounce_seconds=0.3)
    await w.start()
    target = tmp_wiki_root / "raw" / "articles" / "same.md"
    target.write_bytes(b"x")
    for _ in range(5):
        w.handle_raw_event(path=target, event_type="modified")
        await asyncio.sleep(0.05)
    await asyncio.sleep(0.6)  # let debounce window expire
    await w.stop()
    assert len(recorder.calls) == 1
    assert recorder.calls[0][0] == str(target)


@pytest.mark.asyncio
async def test_different_paths_do_not_merge(
    tmp_wiki_root: Path, fake_observer: MagicMock, recorder
) -> None:
    w = watcher_mod.IngestWatcher(root=tmp_wiki_root, on_debounced=recorder, debounce_seconds=0.2)
    await w.start()
    p1 = tmp_wiki_root / "raw" / "articles" / "a.md"
    p2 = tmp_wiki_root / "raw" / "articles" / "b.md"
    p1.write_bytes(b"a")
    p2.write_bytes(b"b")
    w.handle_raw_event(path=p1, event_type="created")
    w.handle_raw_event(path=p2, event_type="created")
    await asyncio.sleep(0.5)
    await w.stop()
    assert sorted(c[0] for c in recorder.calls) == sorted([str(p1), str(p2)])


# --------------------------------------------------------------------------- #
# Event type handling                                                         #
# --------------------------------------------------------------------------- #


@pytest.mark.asyncio
async def test_moved_to_triggers_same_pipeline_as_close_write(
    tmp_wiki_root: Path, fake_observer: MagicMock, recorder
) -> None:
    w = watcher_mod.IngestWatcher(root=tmp_wiki_root, on_debounced=recorder, debounce_seconds=0.15)
    await w.start()
    target = tmp_wiki_root / "raw" / "articles" / "moved.md"
    target.write_bytes(b"m")
    w.handle_raw_event(path=target, event_type="moved")
    await asyncio.sleep(0.4)
    await w.stop()
    assert any(event_type == "moved" for _, event_type in recorder.calls)


@pytest.mark.asyncio
async def test_filtered_paths_never_reach_callback(
    tmp_wiki_root: Path, fake_observer: MagicMock, recorder
) -> None:
    w = watcher_mod.IngestWatcher(root=tmp_wiki_root, on_debounced=recorder, debounce_seconds=0.15)
    await w.start()
    junk = tmp_wiki_root / "raw" / ".git" / "HEAD"
    junk.parent.mkdir(parents=True, exist_ok=True)
    junk.write_bytes(b"ref: refs/heads/main\n")
    w.handle_raw_event(path=junk, event_type="created")
    await asyncio.sleep(0.35)
    await w.stop()
    assert recorder.calls == []
