"""
Daemon composition root.

M3 Sprint 2 (PRD-006) wires the `AutoFetchWorker` into the unified
ingest daemon as a sibling background task next to the filesystem
`WatcherService`. M3 Sprint 4 extends the same composition root to
wire the seal-on-ingest hook (`wiki_memory.seal_hook.SealOnIngestHook`)
into the daemon's `WorkerPool` so every successfully ingested page
triggers chunk-and-index into the Memory Tree.

This module is the seam that keeps `daemon.py` free of construction
logic for either feature: the daemon class itself merely accepts
optional dependencies (`autofetch_worker`, `post_write_hook`); this
factory decides *how* to build them.

Backwards compatible by design: callers can keep building `Daemon(cfg)`
directly. New callers use `build_daemon(cfg, registry=..., enable_seal_hook=...)`
and the factory handles the wiring.

## Env knobs (alphabetical)

- `WIKI_AUTOFETCH_ENABLED` — gates `AutoFetchWorker` wire-in (M3-S2).
- `WIKI_SEAL_ON_INGEST_ENABLED` — gates the post-write seal hook
  (M3-S4). Defaults to off; flip to 1 to activate.
- `WIKI_SYNTHESIS_ENABLED` — gates the post-write synthesis hook
  (ADR-035 D3). Defaults to off. When both seal and synthesis are
  enabled they compose via `CompositePostWriteHook` (both run; each
  failure is isolated).
- `WIKI_SYNTHESIS_ROUTER_TIER` — router tier for the structured
  synthesis extraction call (default REASONING per ADR-035).

## Storage layout

When seal-on-ingest is wired the factory provisions two extra SQLite
stores beside the main ingest queue:

  ~/.local/state/wiki_ingest/content_store.db   chunks (memory tree)
  ~/.local/state/wiki_ingest/tree_nodes.db      source/topic/global nodes

Both paths are configurable via `cfg.memory_content_store_path` and
`cfg.memory_tree_nodes_path` (defaults computed from `cfg.db_path`).
"""

from __future__ import annotations

import asyncio
import json
import logging
from dataclasses import replace
from pathlib import Path
from typing import TYPE_CHECKING, Any

from wiki_autofetch.metrics import AutoFetchMetrics
from wiki_autofetch.rate_limiter import TokenBucket
from wiki_autofetch.worker import AutoFetchWorker
from wiki_core.protocols import MemoryStore
from wiki_ingest.config import Config
from wiki_ingest.daemon import Daemon

if TYPE_CHECKING:
    from wiki_integrations.registry import IntegrationRegistry

log = logging.getLogger("wiki_ingest.composition")


async def build_seal_on_ingest_hook(cfg: Config) -> tuple[Any, list[Any]]:
    """Construct a `SealOnIngestHook` from a `Config`.

    Returns `(hook, resources)` where `resources` is the list of opened
    stores the caller must `await close()` on shutdown. Tests inject
    their own stores via the lower-level `SealOnIngestHook(...)`
    constructor; this factory is the deployment-side default.

    The hook reads pages from `cfg.wiki_root / wiki/`. Content + tree
    nodes persist under `cfg.memory_content_store_path` and
    `cfg.memory_tree_nodes_path` (defaults: sibling files of
    `cfg.db_path`). The write sink wraps an in-process callable that
    forwards to the MCP `write_page` handler so seal summaries land in
    the same vault path the agent reads from.
    """
    # Local imports keep wiki_ingest free of a hard dep on wiki_memory
    # at module-import time. The dep is real once seal-on-ingest is
    # enabled, but skipping it at import keeps `Daemon(cfg)` working
    # in deployments that haven't installed wiki_memory yet.
    from wiki_agent.write_sink import LocalWriteSink
    from wiki_memory.content_store import ContentStore
    from wiki_memory.seal_hook import build_default_seal_hook
    from wiki_memory.tree_nodes import TreeNodeStore

    content_path: Path = getattr(
        cfg, "memory_content_store_path", cfg.db_path.parent / "content_store.db"
    )
    tree_path: Path = getattr(cfg, "memory_tree_nodes_path", cfg.db_path.parent / "tree_nodes.db")
    content_store = ContentStore(content_path)
    tree_store = TreeNodeStore(tree_path)
    await content_store.init()
    await tree_store.init()

    # In-process LocalWriteSink with no-op handlers: the SealWorker
    # writes the summary markdown to disk via this sink. The sink
    # forwards to `write_page` if the daemon ever wires a real MCP
    # client; today the file write itself is handled by the inner
    # handler. This is a placeholder until we surface the real MCP
    # write_page call here — flagged for M3-S5 follow-up. For now we
    # write directly to disk under `cfg.wiki_root` so the vault stays
    # consistent.
    async def _file_write(page_path: str, content: str) -> str:
        full = cfg.wiki_root / page_path
        full.parent.mkdir(parents=True, exist_ok=True)
        full.write_text(content, encoding="utf-8")
        return page_path

    async def _file_log(entry_type: str, title: str, details: str) -> str:
        log_path = cfg.wiki_root / "wiki" / "log.md"
        log_path.parent.mkdir(parents=True, exist_ok=True)
        with log_path.open("a", encoding="utf-8") as f:
            f.write(f"\n## {entry_type} — {title}\n{details}\n")
        return "ok"

    write_sink = LocalWriteSink(_file_write, _file_log)
    hook = build_default_seal_hook(
        content_store=content_store,
        tree_store=tree_store,
        write_sink=write_sink,
        vault_root=cfg.wiki_root,
    )
    return hook, [content_store, tree_store]


def _build_synthesis_router(cfg: Config) -> Any | None:
    """Build the ADR-013 router for the structured synthesis extraction
    call (issue #180), pinning the ``ingest`` intent to
    ``cfg.synthesis_router_tier`` (REASONING per ADR-035 D3). Returns
    None on any construction failure — the agent then runs
    deterministic-only (honest ``origin``)."""
    try:
        from wiki_routing.config import load as load_router_config
        from wiki_routing.factory import default_router
        from wiki_routing.tiers import Tier

        router_cfg = load_router_config()
        overrides = dict(router_cfg.overrides)
        overrides["ingest"] = Tier[cfg.synthesis_router_tier]
        return default_router(config=replace(router_cfg, overrides=overrides))
    except Exception as e:  # noqa: BLE001
        log.warning(
            "composition.synthesis_router_skipped",
            extra={"extra_fields": {"error_class": type(e).__name__}},
        )
        return None


def build_synthesis_hook(cfg: Config) -> Any:
    """Construct a ``SynthesisOnIngestHook`` from a ``Config`` (ADR-035 D3).

    The write/index/log legs reuse the SAME LangChain tools the MCP
    server exposes (`wiki_agent.tools.create_tools`), invoked in-process
    — identical behaviour (duplicate-source guard, index format, log
    format) to what the Hermes batch got over MCP, without a subprocess.
    Tests inject their own callables via the lower-level
    ``SynthesisAgent(...)`` constructor; this factory is the
    deployment-side default.
    """
    from wiki_agent.tools import create_tools
    from wiki_synthesis.agent import SynthesisAgent
    from wiki_synthesis.hooks import SynthesisOnIngestHook

    tools = {t.name: t for t in create_tools(str(cfg.wiki_root))}

    async def _write_page(page_path: str, content: str) -> str:
        from wiki_synthesis.agent import PageWriteSkippedError

        raw = await asyncio.to_thread(
            tools["write_page"].invoke, {"page_path": page_path, "content": content}
        )
        payload = json.loads(raw)
        if payload.get("error"):
            raise RuntimeError(f"write_page failed: {payload['error']}")
        if payload.get("status") == "refused":
            existing = payload.get("existing_page")
            if existing:
                return str(existing)
            raise RuntimeError(f"write_page refused: {payload.get('reason')}")
        if payload.get("status") == "skipped":
            # ADR-048 D4: quality declined the page. The agent skips it
            # per-page (a stub entity must not abort the whole synthesis).
            raise PageWriteSkippedError(str(payload.get("reason") or "skipped"))
        return str(payload.get("page_path") or page_path)

    async def _update_index(page_path: str, category: str, summary: str, count: int) -> None:
        await asyncio.to_thread(
            tools["update_index_entry"].invoke,
            {
                "page_path": page_path,
                "category": category,
                "summary": summary,
                "source_count": count,
            },
        )

    async def _append_log(entry_type: str, title: str, details: str) -> None:
        await asyncio.to_thread(
            tools["append_to_log"].invoke,
            {"entry_type": entry_type, "title": title, "details": details},
        )

    # ADR-036: read-back for page-level accretion. Wired to the same
    # read_wiki_file tool the MCP server exposes; returns None when the page
    # is absent (or the tool is unavailable), so synthesis degrades to the
    # prior overwrite behaviour rather than failing.
    read_tool = tools.get("read_wiki_file")

    async def _read_page(page_path: str) -> str | None:
        if read_tool is None:
            return None
        raw = await asyncio.to_thread(read_tool.invoke, {"file_path": page_path})
        payload = json.loads(raw)
        if payload.get("error"):
            return None
        content = payload.get("content")
        return content if isinstance(content, str) else None

    agent = SynthesisAgent(
        write_page=_write_page,
        update_index=_update_index,
        append_log=_append_log,
        router=_build_synthesis_router(cfg),
        read_page=_read_page,
    )
    return SynthesisOnIngestHook(agent=agent, vault_root=cfg.wiki_root)


async def build_daemon(
    cfg: Config,
    *,
    registry: IntegrationRegistry | None = None,
    memory_store: MemoryStore | None = None,
    enable_seal_hook: bool | None = None,
) -> Daemon:
    """Construct a `Daemon`, optionally with an `AutoFetchWorker` wired
    over the sources registered in `registry`.

    The autofetch worker pulls `IngestEvent`s from registered OAuth
    sources every `cfg.autofetch_interval_seconds` (default 1200 = 20 min
    per PRD-006 FR-1) and pushes them into the same `MemoryStore` the
    watcher feeds. Per-source error isolation (PRD-006 US-002): a Gmail
    outage doesn't stop the GitHub poll, doesn't stop the filesystem
    watcher.

    Parameters
    ----------
    cfg:
        Daemon configuration. `cfg.autofetch_enabled`,
        `cfg.autofetch_interval_seconds`, and
        `cfg.autofetch_rate_limit_per_minute` control the worker.
    registry:
        Live `IntegrationRegistry`. If None, empty, or
        `cfg.autofetch_enabled is False`, no worker is built — the
        returned daemon is identical to `Daemon(cfg)`.
    memory_store:
        The sink to which fetched events are pushed. Defaults to the
        daemon's own `EventQueue` (which already implements the
        `MemoryStore` protocol). Tests can inject an in-memory stub.

    Returns
    -------
    Daemon
        Either the watcher-only daemon (when no registry / autofetch
        disabled / no active sources) or a daemon with `autofetch`
        wired.
    """
    # M3 Sprint 4: build the seal-on-ingest hook if requested. Resolves
    # the enable_seal_hook decision: explicit kwarg wins; otherwise read
    # cfg.seal_on_ingest_enabled (defaults to False).
    hooks: list[Any] = []
    if enable_seal_hook is None:
        enable_seal_hook = getattr(cfg, "seal_on_ingest_enabled", False)
    if enable_seal_hook:
        try:
            seal_hook, _resources = await build_seal_on_ingest_hook(cfg)
            hooks.append(seal_hook)
            log.info(
                "composition.seal_hook_wired",
                extra={
                    "extra_fields": {
                        "content_store": str(getattr(cfg, "memory_content_store_path", "default")),
                        "tree_nodes": str(getattr(cfg, "memory_tree_nodes_path", "default")),
                    }
                },
            )
        except Exception as e:  # noqa: BLE001
            # Misconfiguration on the memory side should not block the
            # ingest daemon from running. Log + fall back to no-hook.
            log.warning(
                "composition.seal_hook_skipped",
                extra={
                    "extra_fields": {
                        "reason": "build_failed",
                        "error_class": type(e).__name__,
                    }
                },
            )

    # ADR-035 D3: synthesis-on-ingest hook. Same posture as the seal
    # hook: a build failure logs and degrades; it never blocks ingest.
    if getattr(cfg, "synthesis_enabled", False):
        try:
            hooks.append(build_synthesis_hook(cfg))
            log.info(
                "composition.synthesis_hook_wired",
                extra={"extra_fields": {"reason": cfg.synthesis_router_tier}},
            )
        except Exception as e:  # noqa: BLE001
            log.warning(
                "composition.synthesis_hook_skipped",
                extra={
                    "extra_fields": {
                        "reason": "build_failed",
                        "error_class": type(e).__name__,
                    }
                },
            )

    post_write_hook: Any = None
    if len(hooks) == 1:
        post_write_hook = hooks[0]
    elif hooks:
        from wiki_synthesis.hooks import CompositePostWriteHook

        post_write_hook = CompositePostWriteHook(hooks)

    daemon = Daemon(cfg, post_write_hook=post_write_hook)
    if not cfg.autofetch_enabled:
        log.info("composition.autofetch_disabled", extra={"extra_fields": {"reason": "config"}})
        return daemon
    if registry is None:
        log.info("composition.autofetch_skipped", extra={"extra_fields": {"reason": "no_registry"}})
        return daemon

    sources = registry.active()
    if not sources:
        log.info(
            "composition.autofetch_skipped",
            extra={"extra_fields": {"reason": "no_sources"}},
        )
        return daemon

    sink: MemoryStore = memory_store if memory_store is not None else daemon.queue
    bucket = TokenBucket(
        capacity=cfg.autofetch_rate_limit_per_minute,
        refill_per_second=cfg.autofetch_rate_limit_per_minute / 60.0,
    )
    metrics = AutoFetchMetrics()
    worker = AutoFetchWorker(
        sources=sources,
        store=sink,
        interval_seconds=cfg.autofetch_interval_seconds,
        rate_limiter=bucket,
        metrics=metrics,
    )

    # Rebuild the daemon with the worker wired. We can't mutate `daemon`
    # after construction without reaching into private state — clean
    # constructor injection keeps the Daemon API honest.
    daemon = Daemon(cfg, autofetch_worker=worker, post_write_hook=post_write_hook)
    log.info(
        "composition.autofetch_wired",
        extra={
            "extra_fields": {
                "sources": [s.name() for s in sources],
                "interval_seconds": cfg.autofetch_interval_seconds,
                "rate_limit_per_minute": cfg.autofetch_rate_limit_per_minute,
            }
        },
    )
    return daemon


__all__ = ["build_daemon", "build_seal_on_ingest_hook", "build_synthesis_hook"]
