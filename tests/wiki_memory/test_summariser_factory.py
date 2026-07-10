"""Tests for ``wiki_memory.summariser_factory.build_default_summariser``.

The factory is the wire-in seam that swaps the seal worker's
``NullSummariser`` default for an LLM-backed ``RouterSummariser`` when
provider keys are configured. The tests here cover:

- Behaviour with **no provider keys** (offline / dev): bare
  ``NullSummariser``.
- Behaviour with **at least one provider key**: ``CompositeSummariser``
  wrapping ``RouterSummariser`` + ``NullSummariser`` so an LLM failure
  falls back deterministically.
- ``prefer_llm=False`` opt-out: even with keys set, the factory must
  return ``NullSummariser`` so deterministic dry-runs are possible.
- **No network at construction time**: building the factory must not
  hit ``httpx`` or run a real LLM call.
- End-to-end seal flow via ``build_default_seal_worker`` works with a
  mocked router and also falls back to ``NullSummariser`` when the
  router raises.

All tests use ``monkeypatch`` to clear / set env vars in isolation and
mock the router's ``call`` method to avoid real LLM dispatch. The seal
worker fixtures (``content_store``, ``tree_store``) come from
``tests/wiki_memory/conftest.py``.

Since PR #103 wired ``build_default_summariser`` to
``wiki_routing.factory.default_router()`` (which reads
``~/.sbw/config.toml`` and opens ``~/.sbw/state/router_telemetry.db``),
this module pins ``Path.home`` to a tmp directory so tests never touch
the developer's real HOME.
"""

from __future__ import annotations

import hashlib
from pathlib import Path

import pytest

from wiki_agent.write_sink import NullWriteSink
from wiki_memory import build_default_seal_worker
from wiki_memory.chunker import chunk_text
from wiki_memory.seal_worker import VAULT_TREES_PREFIX, SealWorker
from wiki_memory.summariser import (
    CompositeSummariser,
    NullSummariser,
    Summariser,
)
from wiki_memory.summariser_factory import build_default_summariser


@pytest.fixture(autouse=True)
def _isolate_home(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """Redirect the two HOME-derived module constants the router factory
    relies on (``wiki_routing.config.DEFAULT_CONFIG_PATH`` and
    ``wiki_routing.telemetry.DEFAULT_DB_PATH``) to a tmp directory for
    every test in this module. Those constants are evaluated at import
    time from ``Path.home()`` so monkeypatching ``Path.home`` after the
    fact wouldn't help; we patch the constants directly."""
    from wiki_routing import config as _routing_config
    from wiki_routing import telemetry as _routing_telemetry

    fake_home = tmp_path / "home"
    fake_home.mkdir()
    monkeypatch.setattr(
        _routing_config,
        "DEFAULT_CONFIG_PATH",
        fake_home / ".sbw" / "config.toml",
    )
    monkeypatch.setattr(
        _routing_telemetry,
        "DEFAULT_DB_PATH",
        fake_home / ".sbw" / "state" / "router_telemetry.db",
    )
    return fake_home


# --------------------------------------------------------------------------- #
# Helpers                                                                     #
# --------------------------------------------------------------------------- #


def _clear_provider_keys(monkeypatch: pytest.MonkeyPatch) -> None:
    """Strip every provider key from the test environment.

    Keeps the factory paths deterministic regardless of the developer's
    local ``.env``."""
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)


@pytest.fixture
def write_sink() -> NullWriteSink:
    return NullWriteSink()


# --------------------------------------------------------------------------- #
# Branch selection                                                            #
# --------------------------------------------------------------------------- #


class TestFactoryBranching:
    """Exercises the factory's env-driven selection between
    ``NullSummariser`` and the router composite."""

    def test_factory_returns_null_summariser_when_no_keys(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        _clear_provider_keys(monkeypatch)
        s = build_default_summariser()
        assert isinstance(s, NullSummariser)
        # And it satisfies the Protocol the seal worker depends on.
        assert isinstance(s, Summariser)

    def test_factory_returns_composite_when_anthropic_key_present(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        _clear_provider_keys(monkeypatch)
        monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-anthropic-stub")
        s = build_default_summariser()
        assert isinstance(s, CompositeSummariser)
        # First delegate is the router summariser, last is the Null fallback.
        from wiki_routing.router_summariser import RouterSummariser

        delegates = s._delegates  # type: ignore[attr-defined]
        assert isinstance(delegates[0], RouterSummariser)
        assert isinstance(delegates[-1], NullSummariser)

    def test_factory_returns_composite_when_openrouter_key_present(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        _clear_provider_keys(monkeypatch)
        monkeypatch.setenv("OPENROUTER_API_KEY", "sk-openrouter-stub")
        s = build_default_summariser()
        assert isinstance(s, CompositeSummariser)

    def test_factory_treats_empty_string_key_as_unset(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # A stray ``export ANTHROPIC_API_KEY=`` should not trigger the
        # router branch. The factory's helper trims whitespace so the
        # blank value is rejected.
        _clear_provider_keys(monkeypatch)
        monkeypatch.setenv("ANTHROPIC_API_KEY", "   ")
        monkeypatch.setenv("OPENROUTER_API_KEY", "")
        s = build_default_summariser()
        assert isinstance(s, NullSummariser)

    def test_factory_prefer_llm_false_skips_router(self, monkeypatch: pytest.MonkeyPatch) -> None:
        # Even with both keys set, prefer_llm=False must short-circuit.
        monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-stub-1")
        monkeypatch.setenv("OPENROUTER_API_KEY", "sk-stub-2")
        s = build_default_summariser(prefer_llm=False)
        assert isinstance(s, NullSummariser)


# --------------------------------------------------------------------------- #
# No-network guarantee                                                        #
# --------------------------------------------------------------------------- #


class TestNoNetworkAtConstruction:
    """The factory must build pure-Python objects only — no httpx, no
    sockets. This guards the daemon's startup latency budget and keeps
    test runs hermetic."""

    def test_factory_does_not_invoke_provider_generate(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # Patch each provider's ``generate`` to record any call. The
        # factory must not trigger them.
        from wiki_routing.providers import anthropic as anthropic_mod
        from wiki_routing.providers import ollama as ollama_mod
        from wiki_routing.providers import openrouter as openrouter_mod

        invoked: list[str] = []

        async def _trap_anthropic(self, messages):  # type: ignore[no-untyped-def]
            invoked.append("anthropic")
            raise RuntimeError("anthropic.generate should not have been called")

        async def _trap_openrouter(self, messages):  # type: ignore[no-untyped-def]
            invoked.append("openrouter")
            raise RuntimeError("openrouter.generate should not have been called")

        async def _trap_ollama(self, messages):  # type: ignore[no-untyped-def]
            invoked.append("ollama")
            raise RuntimeError("ollama.generate should not have been called")

        monkeypatch.setattr(anthropic_mod.AnthropicBackend, "generate", _trap_anthropic)
        monkeypatch.setattr(openrouter_mod.OpenRouterBackend, "generate", _trap_openrouter)
        monkeypatch.setattr(ollama_mod.OllamaBackend, "generate", _trap_ollama)

        # Construct with both keys present so every provider is wired.
        monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-trap-anthropic")
        monkeypatch.setenv("OPENROUTER_API_KEY", "sk-trap-openrouter")
        s = build_default_summariser()
        assert isinstance(s, CompositeSummariser)
        # No provider's ``generate`` ran during construction.
        assert invoked == []

    def test_factory_does_not_import_httpx(self, monkeypatch: pytest.MonkeyPatch) -> None:
        # Trip ``httpx.AsyncClient`` instantiation as a proxy for "no
        # network client was set up". If the factory ever sneaks one in,
        # this test will fire.
        import httpx

        original = httpx.AsyncClient

        instances: list[httpx.AsyncClient] = []

        def _record_async_client(*args, **kwargs):  # type: ignore[no-untyped-def]
            inst = original(*args, **kwargs)
            instances.append(inst)
            return inst

        monkeypatch.setattr(httpx, "AsyncClient", _record_async_client)
        monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-trap")
        s = build_default_summariser()
        assert isinstance(s, CompositeSummariser)
        # Nothing should have instantiated an AsyncClient.
        assert instances == []


# --------------------------------------------------------------------------- #
# Seal-worker integration                                                     #
# --------------------------------------------------------------------------- #


def _seed_chunks(content_store, tree_store, source_id: str = "src-factory"):
    """Common setup: drop two chunks into the content store and create
    a tree node for them. Returns the chunks for citation building."""

    async def _do():
        chunks = chunk_text("alpha alpha.\n\nbeta beta.", target_tokens=5, hard_cap_tokens=20)
        await content_store.insert_many(source_id=source_id, chunks=chunks)
        await tree_store.create_source_node(node_id=source_id)
        return chunks

    return _do


def _summary_text_citing(chunks) -> str:
    """Render a substring-mode summary body — used by tests that pin the
    legacy parser path (passing ``output_format="substring"`` explicitly
    when constructing the RouterSummariser they care about)."""
    lines = ["# Summary"]
    for c in chunks:
        lines.append(f"- fact from [[chunk:{c.sha256[:8]}]]")
    return "\n".join(lines)


def _summary_json_citing(chunks) -> str:
    """Render the JSON-mode response shape the factory's default
    RouterSummariser expects. Cites every chunk by full sha so the
    parser preserves them end-to-end."""
    import json

    body = "\n".join(["# Summary", *(f"- fact from [[chunk:{c.sha256[:8]}]]" for c in chunks)])
    return json.dumps({"body": body, "cited_shas": [c.sha256 for c in chunks]})


class TestSealWorkerViaFactory:
    """End-to-end: build a SealWorker via the factory and confirm it
    routes through the LLM path when keys are set, and falls back to
    ``NullSummariser`` when the router raises."""

    @pytest.mark.asyncio
    async def test_seal_worker_via_factory_uses_router(
        self,
        content_store,
        tree_store,
        write_sink: NullWriteSink,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-test-key")

        # Capture router calls — we patch ``ModelRouter.call`` to return a
        # canned RouterResult so the seal flow exercises the RouterSummariser
        # branch of the composite without touching a real provider.
        from wiki_routing.router import ModelRouter, RouterResult
        from wiki_routing.tiers import Tier

        router_invocations: list[tuple[object, ...]] = []

        chunks = await _seed_chunks(content_store, tree_store, source_id="src-router")()
        # Factory defaults RouterSummariser to JSON mode (SPEC-009 OQ-2
        # close-out). The canned router response is the JSON envelope;
        # the parser extracts the inner body and ends up on the seal.
        inner_body = _summary_text_citing(chunks)
        canned_response = _summary_json_citing(chunks)

        async def _fake_call(self, task, *, messages):  # type: ignore[no-untyped-def]
            router_invocations.append((task, messages))
            return RouterResult(
                text=canned_response,
                tier=Tier.REASONING,
                tokens_in=42,
                tokens_out=84,
                cost_usd=0.001,
                latency_ms=12.3,
                backend_label="stub:reasoning",
                fallback_steps=0,
            )

        monkeypatch.setattr(ModelRouter, "call", _fake_call)

        worker = build_default_seal_worker(content_store, tree_store, write_sink)
        result = await worker.seal_source(source_id="src-router", node_id="src-router")

        # Router was hit exactly once with the seal-intent task.
        assert len(router_invocations) == 1
        task, messages = router_invocations[0]
        assert task.intent == "seal"  # type: ignore[attr-defined]
        # The JSON-mode prompt has a per-chunk ``### CHUNK <full-sha>``
        # header — assert the full sha appears + the chunk body was
        # forwarded.
        user_msg = messages[-1]
        for c in chunks:
            assert c.sha256 in user_msg.content
            # And the actual chunk body was forwarded to the model.
            assert c.body in user_msg.content

        # The seal landed using the inner body extracted from the JSON
        # envelope (not the raw response text).
        page_path = result.page_path
        assert page_path.startswith(VAULT_TREES_PREFIX)
        assert "sources/" in page_path
        assert len(write_sink.calls) == 1
        mode, page = write_sink.calls[0]
        assert mode == "upsert"
        # Body equals the model's *inner* body — the RouterSummariser
        # JSON path was taken, not the NullSummariser fallback.
        assert page.body == inner_body

        # Tree node was sealed.
        node = await tree_store.get("src-router")
        assert node is not None
        assert node.sealed_at is not None
        assert node.summary_sha256 == result.summary_sha256

    @pytest.mark.asyncio
    async def test_seal_worker_via_factory_falls_back_to_null(
        self,
        content_store,
        tree_store,
        write_sink: NullWriteSink,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        # Keys are set so the factory builds the composite, but the
        # router will raise — the composite must walk to NullSummariser
        # and the seal must still succeed.
        monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-test-key")

        from wiki_routing.fallback import BackendError
        from wiki_routing.router import ModelRouter

        async def _boom(self, task, *, messages):  # type: ignore[no-untyped-def]
            raise BackendError("simulated provider outage", kind="server")

        monkeypatch.setattr(ModelRouter, "call", _boom)

        await _seed_chunks(content_store, tree_store, source_id="src-fallback")()

        worker = build_default_seal_worker(content_store, tree_store, write_sink)
        result = await worker.seal_source(source_id="src-fallback", node_id="src-fallback")

        # Seal still succeeded — the composite's NullSummariser caught
        # the BackendError and produced the deterministic stub.
        assert result.summary_sha256
        assert len(write_sink.calls) == 1
        mode, page = write_sink.calls[0]
        assert mode == "upsert"
        # NullSummariser embeds the [[chunk:SHA8]] markers so the
        # faithfulness gate has something to verify.
        assert "[[chunk:" in page.body
        # And the tree node was sealed despite the router blow-up.
        node = await tree_store.get("src-fallback")
        assert node is not None
        assert node.sealed_at is not None

    @pytest.mark.asyncio
    async def test_seal_worker_via_factory_offline_uses_null_directly(
        self,
        content_store,
        tree_store,
        write_sink: NullWriteSink,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        # No keys at all — the factory returns NullSummariser directly,
        # no router involved. The seal worker should behave exactly like
        # the M2-S4 default path.
        _clear_provider_keys(monkeypatch)

        chunks = await _seed_chunks(content_store, tree_store, source_id="src-offline")()
        worker = build_default_seal_worker(content_store, tree_store, write_sink)
        # Sanity: the underlying summariser is the bare NullSummariser.
        assert isinstance(worker._summariser, NullSummariser)  # type: ignore[attr-defined]

        result = await worker.seal_source(source_id="src-offline", node_id="src-offline")
        assert result.summary_sha256
        # Same shape as the M2-S4 ``NullSummariser`` integration test.
        page = write_sink.calls[0][1]
        assert isinstance(page.frontmatter.get("cited"), list)
        assert len(page.frontmatter["cited"]) == len(chunks)


# --------------------------------------------------------------------------- #
# Backwards compatibility                                                     #
# --------------------------------------------------------------------------- #


class TestBackwardsCompat:
    """The factory wire-in must not regress the M2-S4 default: a bare
    ``SealWorker(...)`` (no ``summariser=`` kwarg) still uses
    ``NullSummariser``."""

    @pytest.mark.asyncio
    async def test_bare_seal_worker_still_defaults_to_null_summariser(
        self,
        content_store,
        tree_store,
        write_sink: NullWriteSink,
    ) -> None:
        worker = SealWorker(
            content_store=content_store,
            tree_store=tree_store,
            write_sink=write_sink,
        )
        # The constructor's default lives here, not in the factory.
        assert isinstance(worker._summariser, NullSummariser)  # type: ignore[attr-defined]


# --------------------------------------------------------------------------- #
# Protocol conformance                                                        #
# --------------------------------------------------------------------------- #


class TestProtocolConformance:
    """The factory's return value, whichever branch fires, satisfies the
    ``Summariser`` Protocol the seal worker depends on. The orchestrator
    handoff to follow-up Sprints relies on this."""

    def test_null_branch_is_a_summariser(self, monkeypatch: pytest.MonkeyPatch) -> None:
        _clear_provider_keys(monkeypatch)
        s = build_default_summariser()
        assert isinstance(s, Summariser)

    def test_composite_branch_is_a_summariser(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-stub")
        s = build_default_summariser()
        assert isinstance(s, Summariser)

    def test_prefer_llm_false_is_a_summariser(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-stub")
        s = build_default_summariser(prefer_llm=False)
        assert isinstance(s, Summariser)


# --------------------------------------------------------------------------- #
# Production router wire-in (PR #103 — closes #37 follow-up)                  #
# --------------------------------------------------------------------------- #


class TestProductionRouterWireIn:
    """PR #103 swapped the ad-hoc router that ``build_default_summariser``
    used to build for ``wiki_routing.factory.default_router()``. These
    tests pin the contract: the seal flow now honours ``~/.sbw/config.toml``,
    receives a ``CostBudget``, and records to the production telemetry DB —
    none of which the previous ad-hoc router did."""

    def test_factory_delegates_to_default_router(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """The composite's RouterSummariser must wrap the exact router
        returned by ``wiki_routing.factory.default_router`` — not a
        re-built one with parallel defaults."""
        monkeypatch.setenv("OPENROUTER_API_KEY", "sk-test")

        from wiki_routing import factory as routing_factory
        from wiki_routing.router_summariser import RouterSummariser

        sentinel_calls: list[tuple[object, ...]] = []
        real_default_router = routing_factory.default_router

        def _spy_default_router(*args, **kwargs):  # type: ignore[no-untyped-def]
            sentinel_calls.append((args, kwargs))
            return real_default_router(*args, **kwargs)

        monkeypatch.setattr(routing_factory, "default_router", _spy_default_router)
        # ``summariser_factory`` does ``from wiki_routing.factory import
        # default_router`` at call-time (inside the function), so patching
        # the module attribute is enough — no need to patch the import
        # site.

        s = build_default_summariser()
        assert isinstance(s, CompositeSummariser)
        # The spy fired exactly once.
        assert len(sentinel_calls) == 1
        # First delegate is a RouterSummariser, second is the NullSummariser.
        delegates = s._delegates  # type: ignore[attr-defined]
        assert isinstance(delegates[0], RouterSummariser)
        assert isinstance(delegates[-1], NullSummariser)

    def test_factory_honours_config_toml_provider_overrides(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        """Writing a ``~/.sbw/config.toml`` with an Ollama-only FAST
        provider should propagate all the way: the router built inside
        ``build_default_summariser`` exposes that exact backend on the
        FAST tier."""
        monkeypatch.setenv("OPENROUTER_API_KEY", "sk-test")

        # _isolate_home redirected DEFAULT_CONFIG_PATH; write into that path.
        from wiki_routing import config as routing_config

        cfg_path = routing_config.DEFAULT_CONFIG_PATH
        cfg_path.parent.mkdir(parents=True, exist_ok=True)
        cfg_path.write_text(
            "[router]\n"
            "max_per_day_usd = 5.0\n"
            "max_per_task_usd = 0.25\n\n"
            "[router.providers.fast]\n"
            'backend = "ollama"\n'
            'model = "qwen2.5:7b"\n',
            encoding="utf-8",
        )

        s = build_default_summariser()
        assert isinstance(s, CompositeSummariser)

        # Reach into the router and confirm the FAST chain's first link is Ollama.
        from wiki_routing.providers.ollama import OllamaProvider
        from wiki_routing.tiers import Tier

        router = s._delegates[0]._router  # type: ignore[attr-defined]
        fast_chain = router._tiers[Tier.FAST].chain  # type: ignore[attr-defined]
        first_backend = fast_chain._backends[0]  # type: ignore[attr-defined]
        assert isinstance(first_backend, OllamaProvider)

        # And the per-task budget came through as well.
        assert router._budget.max_per_task_usd == 0.25  # type: ignore[attr-defined]
        assert router._budget.max_per_day_usd == 5.0  # type: ignore[attr-defined]

    def test_factory_router_has_telemetry_attached(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """``default_router()`` attaches a ``RouterTelemetry`` by default;
        the seal flow must use that same router so calls are recorded to
        the per-user SQLite DB (the PR #102 promise)."""
        monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-test")

        from wiki_routing.telemetry import RouterTelemetry

        s = build_default_summariser()
        assert isinstance(s, CompositeSummariser)
        router = s._delegates[0]._router  # type: ignore[attr-defined]
        # The router exposes its telemetry under ``_telemetry``; assert it
        # is a real RouterTelemetry (not ``None``) so ``sbw doctor`` can
        # see seal-flow calls.
        assert isinstance(router._telemetry, RouterTelemetry)  # type: ignore[attr-defined]

    @pytest.mark.asyncio
    async def test_seal_call_writes_router_telemetry_row(
        self,
        content_store,
        tree_store,
        write_sink: NullWriteSink,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Issue #115 / PR #102 follow-up: an actual seal call must land
        a row in ``router_telemetry.db``. The earlier
        ``test_factory_router_has_telemetry_attached`` only verifies the
        object exists; this one exercises the recording path end-to-end
        by mocking the backend's ``generate`` (rather than the router's
        ``call``) so the router's recording branch runs."""
        monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-test-telemetry")

        from wiki_routing.providers import anthropic as anthropic_mod
        from wiki_routing.providers import ollama as ollama_mod
        from wiki_routing.providers import openrouter as openrouter_mod
        from wiki_routing.router import BackendResponse

        chunks = await _seed_chunks(content_store, tree_store, source_id="src-telemetry")()
        canned_json = _summary_json_citing(chunks)

        # Mock the backend's ``generate`` (not the router's ``call``) so
        # the router's own dispatch + telemetry.record() path runs
        # end-to-end. Earlier tests in this file mock at the router level
        # which bypasses the recording branch.
        async def _fake_generate(self, messages):  # type: ignore[no-untyped-def]
            return BackendResponse(
                text=canned_json,
                tokens_in=20,
                tokens_out=40,
                cost_usd=0.0002,
            )

        monkeypatch.setattr(anthropic_mod.AnthropicBackend, "generate", _fake_generate)
        monkeypatch.setattr(openrouter_mod.OpenRouterBackend, "generate", _fake_generate)
        monkeypatch.setattr(ollama_mod.OllamaBackend, "generate", _fake_generate)

        worker = build_default_seal_worker(content_store, tree_store, write_sink)
        await worker.seal_source(source_id="src-telemetry", node_id="src-telemetry")

        # Reach the router's telemetry on the worker's actual summariser
        # so we're looking at the same router that just ran the seal.
        router = worker._summariser._delegates[0]._router  # type: ignore[attr-defined]
        telemetry = router._telemetry  # type: ignore[attr-defined]
        assert telemetry.total_calls() >= 1, (
            "expected at least one router_calls row after seal — telemetry not wired"
        )


# --------------------------------------------------------------------------- #
# Hash sanity                                                                 #
# --------------------------------------------------------------------------- #


@pytest.mark.asyncio
async def test_null_summariser_via_factory_hashes_body(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Cheap end-to-end smoke: build the offline factory, run summarise
    over a tiny part list, and assert the returned sha matches the body.
    Catches any future regression where the factory returns a broken
    Summariser instance."""
    _clear_provider_keys(monkeypatch)
    from wiki_memory.summariser import SummaryPart

    s = build_default_summariser()
    parts = [
        SummaryPart(
            sha256=hashlib.sha256(b"a").hexdigest(),
            body="alpha",
            token_count=1,
        ),
        SummaryPart(
            sha256=hashlib.sha256(b"b").hexdigest(),
            body="beta",
            token_count=1,
        ),
    ]
    result = await s.summarise(parts)
    assert result.sha256 == hashlib.sha256(result.body.encode("utf-8")).hexdigest()
    assert set(result.cited_shas) == {p.sha256 for p in parts}
