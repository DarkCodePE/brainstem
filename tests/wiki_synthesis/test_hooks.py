"""Hook protocol tests: never raises into the worker; flag gates wiring."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest

from wiki_ingest.config import Config
from wiki_synthesis.hooks import (
    CompositePostWriteHook,
    SynthesisOnIngestHook,
    unwrap_envelope,
)

WRAPPED_PAGE = """---
title: "x"
origin: ingested-untrusted
---

# x

<ingested_source trust="untrusted" sha256="abc" rel_path="raw/articles/x.md">
# Real Title

Claude Code body text.
</ingested_source>
"""


def make_event(rel_path: str = "raw/articles/x.md") -> SimpleNamespace:
    return SimpleNamespace(
        event_id="ev-1",
        metadata={"rel_path": rel_path},
        path_or_uri=f"/kb/{rel_path}",
    )


class TestUnwrapEnvelope:
    def test_recovers_raw_body(self) -> None:
        body = unwrap_envelope(WRAPPED_PAGE)
        assert body.startswith("# Real Title")
        assert "<ingested_source" not in body

    def test_no_envelope_returns_input(self) -> None:
        assert unwrap_envelope("plain text") == "plain text"


class TestHookNeverRaises:
    @pytest.mark.asyncio
    async def test_agent_crash_is_swallowed(self, tmp_path: Path) -> None:
        class CrashingAgent:
            async def synthesize(self, **kwargs):
                raise RuntimeError("boom")

        hook = SynthesisOnIngestHook(
            agent=CrashingAgent(),
            vault_root=tmp_path,
            read_page=_const_page(),
        )
        await hook(make_event(), "wiki/sources/x.md")  # must not raise

    @pytest.mark.asyncio
    async def test_missing_page_is_swallowed(self, tmp_path: Path) -> None:
        class NeverCalledAgent:
            async def synthesize(self, **kwargs):  # pragma: no cover
                raise AssertionError("must not be reached")

        hook = SynthesisOnIngestHook(agent=NeverCalledAgent(), vault_root=tmp_path)
        await hook(make_event(), "wiki/sources/missing.md")  # must not raise

    @pytest.mark.asyncio
    async def test_path_escape_is_swallowed(self, tmp_path: Path) -> None:
        class NeverCalledAgent:
            async def synthesize(self, **kwargs):  # pragma: no cover
                raise AssertionError("must not be reached")

        hook = SynthesisOnIngestHook(agent=NeverCalledAgent(), vault_root=tmp_path)
        await hook(make_event(), "../../etc/passwd")  # must not raise


class TestHookHappyPath:
    @pytest.mark.asyncio
    async def test_agent_receives_unwrapped_body_and_rel_path(self, tmp_path: Path) -> None:
        calls: list[dict] = []

        class RecordingAgent:
            async def synthesize(self, **kwargs):
                calls.append(kwargs)
                return SimpleNamespace(
                    source_page_path="wiki/sources/real-title.md",
                    entity_page_paths=(),
                    concept_page_paths=(),
                    llm_extracted=False,
                )

        hook = SynthesisOnIngestHook(
            agent=RecordingAgent(),
            vault_root=tmp_path,
            read_page=_const_page(),
        )
        await hook(make_event("raw/articles/x.md"), "wiki/sources/x.md")
        assert len(calls) == 1
        assert calls[0]["rel_path"] == "raw/articles/x.md"
        assert calls[0]["raw_text"].startswith("# Real Title")


class TestCompositeHook:
    @pytest.mark.asyncio
    async def test_all_hooks_run_and_failures_are_isolated(self) -> None:
        ran: list[str] = []

        async def good_hook(event, page_path):
            ran.append("good")

        async def bad_hook(event, page_path):
            ran.append("bad")
            raise RuntimeError("boom")

        composite = CompositePostWriteHook([bad_hook, good_hook])
        await composite(make_event(), "wiki/x.md")  # must not raise
        assert ran == ["bad", "good"]


class TestFlagGating:
    def make_cfg(self, tmp_path: Path, **kwargs) -> Config:
        kb = tmp_path / "kb"
        (kb / "raw").mkdir(parents=True, exist_ok=True)
        return Config(
            wiki_root=kb,
            raw_dir=kb / "raw",
            ingested_dir=kb / "raw" / "_ingested",
            db_path=tmp_path / "q.db",
            mcp_command=("/bin/true",),
            metrics_path=None,
            **kwargs,
        )

    @pytest.mark.asyncio
    async def test_flag_off_hook_not_wired(self, tmp_path: Path) -> None:
        from wiki_ingest.composition import build_daemon

        cfg = self.make_cfg(tmp_path, synthesis_enabled=False)
        daemon = await build_daemon(cfg)
        assert daemon.pool._post_write_hook is None

    @pytest.mark.asyncio
    async def test_flag_on_hook_wired(self, tmp_path: Path, monkeypatch) -> None:
        import wiki_ingest.composition as composition

        sentinel = object()
        monkeypatch.setattr(composition, "build_synthesis_hook", lambda cfg: sentinel)
        cfg = self.make_cfg(tmp_path, synthesis_enabled=True)
        daemon = await composition.build_daemon(cfg)
        assert daemon.pool._post_write_hook is sentinel

    @pytest.mark.asyncio
    async def test_both_hooks_compose(self, tmp_path: Path, monkeypatch) -> None:
        import wiki_ingest.composition as composition

        synth = object()
        seal = object()

        async def fake_seal(cfg):
            return seal, []

        monkeypatch.setattr(composition, "build_seal_on_ingest_hook", fake_seal)
        monkeypatch.setattr(composition, "build_synthesis_hook", lambda cfg: synth)
        cfg = self.make_cfg(tmp_path, synthesis_enabled=True, seal_on_ingest_enabled=True)
        daemon = await composition.build_daemon(cfg)
        hook = daemon.pool._post_write_hook
        assert isinstance(hook, CompositePostWriteHook)
        assert hook._hooks == [seal, synth]

    def test_invalid_tier_rejected(self, tmp_path: Path) -> None:
        with pytest.raises(ValueError, match="synthesis_router_tier"):
            self.make_cfg(tmp_path, synthesis_router_tier="TURBO")

    def test_tier_normalised(self, tmp_path: Path) -> None:
        cfg = self.make_cfg(tmp_path, synthesis_router_tier="reasoning")
        assert cfg.synthesis_router_tier == "REASONING"


def _const_page():
    async def read_page(page_path: str) -> str:
        return WRAPPED_PAGE

    return read_page
