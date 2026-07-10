"""
Tests for the synthesis-on-ingest wire-in inside `wiki_ingest.composition`
(ADR-035 D3): router construction, `build_synthesis_hook` tool adapters,
and the `build_daemon` flag handling / composite assembly.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any
from unittest.mock import patch

import pytest

from wiki_ingest import composition
from wiki_ingest.composition import (
    _build_synthesis_router,
    build_daemon,
    build_synthesis_hook,
)
from wiki_ingest.config import Config
from wiki_routing.config import RouterConfig
from wiki_routing.tiers import Tier
from wiki_synthesis.hooks import CompositePostWriteHook, SynthesisOnIngestHook


def _make_cfg(
    tmp_path: Path,
    *,
    seal_enabled: bool = False,
    synthesis_enabled: bool = False,
    synthesis_router_tier: str = "REASONING",
) -> Config:
    wiki_root = tmp_path / "knowledge-base"
    raw = wiki_root / "raw"
    ingested = raw / "_ingested"
    db = tmp_path / "ingest.db"
    for d in (wiki_root, raw, ingested, wiki_root / "wiki" / "trees"):
        d.mkdir(parents=True, exist_ok=True)
    return Config(
        wiki_root=wiki_root,
        raw_dir=raw,
        ingested_dir=ingested,
        db_path=db,
        debounce_seconds=3.0,
        worker_pool_size=2,
        rate_limit_per_minute=10,
        max_file_size_mb=25,
        mcp_command=("python", "-m", "wiki_agent.mcp_server"),
        seal_on_ingest_enabled=seal_enabled,
        synthesis_enabled=synthesis_enabled,
        synthesis_router_tier=synthesis_router_tier,
    )


# --------------------------------------------------------------------------- #
# _build_synthesis_router                                                      #
# --------------------------------------------------------------------------- #


class TestBuildSynthesisRouter:
    def test_pins_ingest_intent_to_configured_tier(self, tmp_path: Path) -> None:
        cfg = _make_cfg(tmp_path, synthesis_router_tier="FAST")
        sentinel = object()
        captured: dict[str, Any] = {}

        def fake_default_router(*, config: RouterConfig) -> Any:
            captured["config"] = config
            return sentinel

        # Imports are local inside _build_synthesis_router, so patch at
        # the wiki_routing source modules (re-fetched on every call).
        with (
            patch("wiki_routing.config.load", return_value=RouterConfig()),
            patch("wiki_routing.factory.default_router", side_effect=fake_default_router),
        ):
            router = _build_synthesis_router(cfg)

        assert router is sentinel
        assert captured["config"].overrides["ingest"] is Tier.FAST

    def test_preserves_pre_existing_overrides(self, tmp_path: Path) -> None:
        cfg = _make_cfg(tmp_path)  # default tier REASONING
        base = RouterConfig(overrides={"seal": Tier.VISION})
        captured: dict[str, Any] = {}

        def fake_default_router(*, config: RouterConfig) -> Any:
            captured["config"] = config
            return object()

        with (
            patch("wiki_routing.config.load", return_value=base),
            patch("wiki_routing.factory.default_router", side_effect=fake_default_router),
        ):
            assert _build_synthesis_router(cfg) is not None

        overrides = captured["config"].overrides
        assert overrides["ingest"] is Tier.REASONING
        assert overrides["seal"] is Tier.VISION
        # The loaded config must not be mutated in place.
        assert base.overrides == {"seal": Tier.VISION}

    def test_returns_none_when_router_build_fails(self, tmp_path: Path) -> None:
        cfg = _make_cfg(tmp_path)
        with (
            patch("wiki_routing.config.load", return_value=RouterConfig()),
            patch("wiki_routing.factory.default_router", side_effect=RuntimeError("boom")),
        ):
            assert _build_synthesis_router(cfg) is None

    def test_returns_none_when_config_load_fails(self, tmp_path: Path) -> None:
        cfg = _make_cfg(tmp_path)
        with patch("wiki_routing.config.load", side_effect=OSError("no config")):
            assert _build_synthesis_router(cfg) is None


# --------------------------------------------------------------------------- #
# build_synthesis_hook — tool adapter closures                                 #
# --------------------------------------------------------------------------- #


class _StubTool:
    """Mimics the LangChain tool surface used by the factory:
    `.name` + synchronous `.invoke(dict) -> str`."""

    def __init__(self, name: str, response: Any = None) -> None:
        self.name = name
        self.calls: list[dict[str, Any]] = []
        self._response = response

    def invoke(self, payload: dict[str, Any]) -> str:
        self.calls.append(payload)
        return json.dumps(self._response if self._response is not None else {})


def _build_hook_with_stub_tools(
    cfg: Config, *, write_response: Any = None, read_response: Any = None
) -> tuple[SynthesisOnIngestHook, dict[str, _StubTool]]:
    responses = {"write_page": write_response, "read_wiki_file": read_response}
    tools = {
        name: _StubTool(name, responses.get(name))
        for name in ("write_page", "update_index_entry", "append_to_log", "read_wiki_file")
    }
    with (
        patch("wiki_agent.tools.create_tools", return_value=list(tools.values())),
        patch.object(composition, "_build_synthesis_router", return_value=None),
    ):
        hook = build_synthesis_hook(cfg)
    return hook, tools


class TestBuildSynthesisHook:
    def test_returns_hook_with_agent_assembled(self, tmp_path: Path) -> None:
        cfg = _make_cfg(tmp_path, synthesis_enabled=True)
        hook, _tools = _build_hook_with_stub_tools(cfg)
        assert isinstance(hook, SynthesisOnIngestHook)
        assert hook._agent.router is None  # router build stubbed to None

    @pytest.mark.asyncio
    async def test_write_page_returns_confirmed_path(self, tmp_path: Path) -> None:
        cfg = _make_cfg(tmp_path)
        hook, tools = _build_hook_with_stub_tools(
            cfg, write_response={"page_path": "wiki/sources/confirmed.md"}
        )
        got = await hook._agent.write_page("wiki/sources/requested.md", "body")
        assert got == "wiki/sources/confirmed.md"
        assert tools["write_page"].calls == [
            {"page_path": "wiki/sources/requested.md", "content": "body"}
        ]

    @pytest.mark.asyncio
    async def test_write_page_falls_back_to_requested_path(self, tmp_path: Path) -> None:
        cfg = _make_cfg(tmp_path)
        hook, _tools = _build_hook_with_stub_tools(cfg, write_response={})
        got = await hook._agent.write_page("wiki/sources/requested.md", "body")
        assert got == "wiki/sources/requested.md"

    @pytest.mark.asyncio
    async def test_write_page_error_raises(self, tmp_path: Path) -> None:
        cfg = _make_cfg(tmp_path)
        hook, _tools = _build_hook_with_stub_tools(cfg, write_response={"error": "disk full"})
        with pytest.raises(RuntimeError, match="write_page failed"):
            await hook._agent.write_page("wiki/sources/x.md", "body")

    @pytest.mark.asyncio
    async def test_write_page_refused_returns_existing_page(self, tmp_path: Path) -> None:
        cfg = _make_cfg(tmp_path)
        hook, _tools = _build_hook_with_stub_tools(
            cfg,
            write_response={
                "status": "refused",
                "reason": "duplicate_source",
                "existing_page": "wiki/sources/old.md",
            },
        )
        got = await hook._agent.write_page("wiki/sources/new.md", "body")
        assert got == "wiki/sources/old.md"

    @pytest.mark.asyncio
    async def test_write_page_refused_without_existing_raises(self, tmp_path: Path) -> None:
        cfg = _make_cfg(tmp_path)
        hook, _tools = _build_hook_with_stub_tools(
            cfg, write_response={"status": "refused", "reason": "policy"}
        )
        with pytest.raises(RuntimeError, match="write_page refused"):
            await hook._agent.write_page("wiki/sources/x.md", "body")

    @pytest.mark.asyncio
    async def test_update_index_forwards_args(self, tmp_path: Path) -> None:
        cfg = _make_cfg(tmp_path)
        hook, tools = _build_hook_with_stub_tools(cfg)
        await hook._agent.update_index("wiki/sources/x.md", "sources", "a summary", 3)
        assert tools["update_index_entry"].calls == [
            {
                "page_path": "wiki/sources/x.md",
                "category": "sources",
                "summary": "a summary",
                "source_count": 3,
            }
        ]

    @pytest.mark.asyncio
    async def test_append_log_forwards_args(self, tmp_path: Path) -> None:
        cfg = _make_cfg(tmp_path)
        hook, tools = _build_hook_with_stub_tools(cfg)
        await hook._agent.append_log("created", "Title", "details")
        assert tools["append_to_log"].calls == [
            {"entry_type": "created", "title": "Title", "details": "details"}
        ]

    def test_read_page_wired_on_agent(self, tmp_path: Path) -> None:
        """ADR-036: the accretion read-back is wired from read_wiki_file."""
        cfg = _make_cfg(tmp_path)
        hook, _tools = _build_hook_with_stub_tools(cfg)
        assert hook._agent.read_page is not None

    @pytest.mark.asyncio
    async def test_read_page_returns_content(self, tmp_path: Path) -> None:
        cfg = _make_cfg(tmp_path)
        hook, tools = _build_hook_with_stub_tools(cfg, read_response={"content": "PAGE BODY"})
        got = await hook._agent.read_page("wiki/entities/x.md")
        assert got == "PAGE BODY"
        assert tools["read_wiki_file"].calls == [{"file_path": "wiki/entities/x.md"}]

    @pytest.mark.asyncio
    async def test_read_page_none_on_file_not_found(self, tmp_path: Path) -> None:
        cfg = _make_cfg(tmp_path)
        hook, _tools = _build_hook_with_stub_tools(
            cfg, read_response={"error": "File not found: x"}
        )
        got = await hook._agent.read_page("wiki/entities/missing.md")
        assert got is None


# --------------------------------------------------------------------------- #
# build_daemon — flag handling + composite                                     #
# --------------------------------------------------------------------------- #


class TestBuildDaemonSynthesisWiring:
    @pytest.mark.asyncio
    async def test_synthesis_enabled_wires_hook(self, tmp_path: Path, monkeypatch) -> None:
        cfg = _make_cfg(tmp_path, synthesis_enabled=True)
        sentinel = object()
        monkeypatch.setattr(composition, "build_synthesis_hook", lambda _cfg: sentinel)
        daemon = await build_daemon(cfg)
        try:
            assert daemon.pool._post_write_hook is sentinel
        finally:
            await daemon.queue.close()

    @pytest.mark.asyncio
    async def test_synthesis_disabled_means_no_hook(self, tmp_path: Path) -> None:
        cfg = _make_cfg(tmp_path, synthesis_enabled=False)
        daemon = await build_daemon(cfg)
        try:
            assert daemon.pool._post_write_hook is None
        finally:
            await daemon.queue.close()

    @pytest.mark.asyncio
    async def test_seal_and_synthesis_compose(self, tmp_path: Path, monkeypatch) -> None:
        cfg = _make_cfg(tmp_path, seal_enabled=True, synthesis_enabled=True)

        async def fake_seal_builder(_cfg: Config) -> tuple[Any, list[Any]]:
            return object(), []  # real store wiring covered in test_composition_seal_hook

        monkeypatch.setattr(composition, "build_seal_on_ingest_hook", fake_seal_builder)
        monkeypatch.setattr(composition, "build_synthesis_hook", lambda _cfg: object())
        daemon = await build_daemon(cfg)
        try:
            assert isinstance(daemon.pool._post_write_hook, CompositePostWriteHook)
            assert len(daemon.pool._post_write_hook._hooks) == 2
        finally:
            await daemon.queue.close()

    @pytest.mark.asyncio
    async def test_synthesis_build_failure_degrades_to_no_hook(
        self, tmp_path: Path, monkeypatch
    ) -> None:
        cfg = _make_cfg(tmp_path, synthesis_enabled=True)

        def boom(_cfg: Config) -> Any:
            raise RuntimeError("synthesis wiring exploded")

        monkeypatch.setattr(composition, "build_synthesis_hook", boom)
        daemon = await build_daemon(cfg)
        try:
            # Failure is isolated: the daemon still builds, hook-less.
            assert daemon.pool._post_write_hook is None
        finally:
            await daemon.queue.close()
