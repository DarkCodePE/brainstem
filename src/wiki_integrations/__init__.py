"""
`wiki_integrations` — OAuth integrations substrate for M3 per
[PRD-005 OAuth Integrations Layer](../../docs/PRD-005-oauth-integrations-layer.md)
and [ADR-009 OAuth integrations strategy](../../docs/ADR-009-oauth-integrations-strategy.md).

Phase 1 ships the **substrate** only:

- `base` — `OAuthIntegrationSource`, the abstract `wiki_core.IngestSource` that
  every provider extends.
- `composio_bridge` — managed-mode bridge over Composio's REST API. Falls back
  to a deterministic stub iterator when `COMPOSIO_API_KEY` is unset so the
  test suite stays hermetic.
- `providers.gmail`, `providers.github` — concrete sources for the two
  highest-value providers. Each translates its provider's items into
  `wiki_core.IngestEvent`s with the metadata keys required by the
  `wiki_ingest.adapter._to_storage` bridge (`bucket`, `rel_path`,
  `event_type`, `mtime`, `size`, optional `mime`).
- `registry` — `IntegrationRegistry` tracks live sources and offers a
  `start_all`/`stop_all` pair the daemon wiring can call at boot/shutdown.

Phase 2 (per ADR-009) layers self-hosted token vault adapters; Phase 3 swaps
Composio for native OAuth. The abstraction here is the lock-in mitigation.

This package depends on `wiki_core.protocols` only. It does not touch the
filesystem watcher, SQLite, or the LLM tool layer; integration with the
event queue happens at the orchestrator level through the `on_event`
callback each source is constructed with.
"""

from __future__ import annotations

from wiki_integrations.base import OAuthIntegrationSource
from wiki_integrations.composio_bridge import ComposioBridge
from wiki_integrations.cursor_store import CursorStore
from wiki_integrations.registry import IntegrationRegistry

__all__ = [
    "ComposioBridge",
    "CursorStore",
    "IntegrationRegistry",
    "OAuthIntegrationSource",
]
