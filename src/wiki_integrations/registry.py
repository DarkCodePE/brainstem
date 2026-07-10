"""
`IntegrationRegistry` — in-memory directory of live `OAuthIntegrationSource`s.

The registry is the seam the daemon (and CLI commands like
``wiki-agent integrations list`` per PRD-005 US-005) interacts with. It
holds nothing that needs disk persistence — the source of truth for which
providers a user has connected is Composio (Phase 1) or the local vault
(Phase 2/3). The registry tracks **what's wired in this process right now**.

Contract:

- `register(source)` — add a source. Raises `ValueError` on duplicate name.
- `unregister(name)` — drop a source. Calls `stop()` on it if started.
  Returns `True` if a source was removed, `False` otherwise.
- `get(name)` — fetch a source by name. Returns `None` if missing.
- `active()` — list currently-registered sources.
- `start_all()` / `stop_all()` — bulk lifecycle. Idempotent; safe to call
  during boot/shutdown without inspecting state first.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from wiki_integrations.base import OAuthIntegrationSource

log = logging.getLogger("wiki_integrations.registry")


class IntegrationRegistry:
    """In-memory registry of live OAuth integration sources."""

    def __init__(self) -> None:
        self._sources: dict[str, OAuthIntegrationSource] = {}

    # ------------------------------------------------------------------ #
    # CRUD                                                               #
    # ------------------------------------------------------------------ #

    def register(self, source: OAuthIntegrationSource) -> None:
        """Add `source` to the registry. Raises if a source with the same
        name is already registered — names must be unique within a process."""
        name = source.name()
        if name in self._sources:
            raise ValueError(f"integration {name!r} already registered")
        self._sources[name] = source

    async def unregister(self, name: str) -> bool:
        """Drop a source. Returns `True` if removed, `False` if no such
        name was registered. Stops the source if it was started."""
        source = self._sources.pop(name, None)
        if source is None:
            return False
        try:
            await source.stop()
        except Exception:  # noqa: BLE001 — stop must not propagate during unregister
            log.exception(
                "integration.stop_failed_during_unregister",
                extra={"extra_fields": {"integration": name}},
            )
        return True

    def get(self, name: str) -> OAuthIntegrationSource | None:
        return self._sources.get(name)

    def active(self) -> list[OAuthIntegrationSource]:
        """Snapshot of currently-registered sources, in registration order."""
        return list(self._sources.values())

    def __contains__(self, name: object) -> bool:
        return isinstance(name, str) and name in self._sources

    def __len__(self) -> int:
        return len(self._sources)

    # ------------------------------------------------------------------ #
    # Bulk lifecycle                                                     #
    # ------------------------------------------------------------------ #

    async def start_all(self) -> None:
        """Call `start()` on every registered source. Best-effort: if one
        source fails to start, log and continue — the orchestrator's
        health endpoint surfaces the per-source status."""
        for source in self._sources.values():
            try:
                await source.start()
            except Exception:  # noqa: BLE001
                log.exception(
                    "integration.start_failed",
                    extra={"extra_fields": {"integration": source.name()}},
                )

    async def stop_all(self) -> None:
        """Call `stop()` on every registered source. Best-effort, same as
        `start_all`."""
        for source in self._sources.values():
            try:
                await source.stop()
            except Exception:  # noqa: BLE001
                log.exception(
                    "integration.stop_failed",
                    extra={"extra_fields": {"integration": source.name()}},
                )
