"""
Tests for the ADR-035 D2 oneshot (`--once`) path of the ingest daemon:
`Daemon.run(catchup_only=True)` must release the pool + queue so the
process can exit, and `_async_main` must build through the composition
root so env-gated hooks are wired in the ephemeral activation too.
"""

from __future__ import annotations

import os
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from wiki_ingest import daemon as daemon_mod
from wiki_ingest.config import Config
from wiki_ingest.daemon import Daemon


class TestRunCatchupOnly:
    @pytest.mark.asyncio
    async def test_oneshot_closes_pool_and_queue_then_returns(self) -> None:
        d = Daemon.__new__(Daemon)  # skip __init__: stub collaborators
        d.pool = SimpleNamespace(close=AsyncMock())
        d.queue = SimpleNamespace(close=AsyncMock())
        started: dict[str, bool] = {}

        async def fake_start(catchup_only: bool = False) -> None:
            started["catchup_only"] = catchup_only

        d.start = fake_start  # type: ignore[method-assign]

        await d.run(catchup_only=True)  # returns without awaiting shutdown

        assert started == {"catchup_only": True}
        d.pool.close.assert_awaited_once()
        d.queue.close.assert_awaited_once()


class TestAsyncMainOnce:
    @pytest.mark.asyncio
    async def test_once_builds_via_composition_root(self, monkeypatch, tmp_path: Path) -> None:
        raw = tmp_path / "kb" / "raw"
        raw.mkdir(parents=True)
        monkeypatch.setenv("WIKI_ROOT", str(tmp_path / "kb"))
        monkeypatch.setenv("WIKI_RAW_DIR", str(raw))
        monkeypatch.setenv("INGEST_DB", str(tmp_path / "ingest.db"))
        # Keep global logging config untouched for the rest of the suite.
        monkeypatch.setattr(daemon_mod, "_setup_logging", lambda: None)

        runs: dict[str, bool] = {}
        built: dict[str, Config] = {}

        class _StubDaemon:
            def request_shutdown(self) -> None:  # signal-handler surface
                pass

            async def run(self, catchup_only: bool = False) -> None:
                runs["catchup_only"] = catchup_only

        async def fake_build_daemon(cfg: Config) -> _StubDaemon:
            built["cfg"] = cfg
            return _StubDaemon()

        # _async_main imports build_daemon from the composition module at
        # call time — patch it there.
        monkeypatch.setattr("wiki_ingest.composition.build_daemon", fake_build_daemon)

        prev_umask = os.umask(0o022)  # _async_main sets 0o077; restore after
        os.umask(prev_umask)
        try:
            rc = await daemon_mod._async_main(["--once"])
        finally:
            os.umask(prev_umask)

        assert rc == 0
        assert runs == {"catchup_only": True}
        assert built["cfg"].raw_dir == raw
