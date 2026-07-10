"""
Abstract base class for OAuth-backed `wiki_core.IngestSource` implementations.

Every provider (Gmail, GitHub, Slack, Notion, Drive, Calendar, …) shares the
same lifecycle and the same emit shape: poll the upstream API on a fixed
`fetch_window`, translate each item into a `wiki_core.IngestEvent`, and hand
it to the orchestrator via the `on_event` callback. The concrete subclass
only owns `fetch_batch()` — everything else (start/stop idempotency, name,
the polling loop scaffolding for later phases) lives here.

The class deliberately does NOT inherit from `wiki_core.IngestSource` —
`IngestSource` is a `@runtime_checkable` `Protocol`, so structural
conformance is checked at call sites via `isinstance(x, IngestSource)`.
Subclassing would buy nothing and would force `__init_subclass__` boilerplate
on every provider.

Subclasses should call `super().__init__(...)` and override `fetch_batch`.
They MAY override `start`/`stop` if they need to spin up provider-specific
state (e.g. an httpx.AsyncClient pool); they MUST call super's
implementation in that case to keep the started-flag honest.
"""

from __future__ import annotations

from collections.abc import Awaitable, Callable
from datetime import timedelta
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from wiki_core.protocols import IngestEvent


EventCallback = Callable[["IngestEvent"], Awaitable[None]]
"""Coroutine called once per produced `IngestEvent`.

The orchestrator wires this to `MemoryStore.enqueue` (or a queue middleware
that wraps it) so the rest of the pipeline doesn't have to know an event came
from a polled OAuth source rather than the local watcher.
"""


class OAuthIntegrationSource:
    """Base for every OAuth integration `IngestSource`.

    Parameters
    ----------
    name:
        Stable identifier for telemetry/logging — e.g. ``"gmail"`` or
        ``"github"``. Same value returned by ``name()``.
    fetch_window:
        Look-back window for the next ``fetch_batch`` call. Subclasses use
        this to bound the upstream query (e.g. Gmail's ``newer_than:24h``).
        Not consumed by this class directly; surfaced for subclasses.
    on_event:
        Awaitable invoked once per produced ``IngestEvent``. The base class
        keeps the reference but does NOT pump events itself; the orchestrator
        is in charge of when ``fetch_batch`` runs and what to do with the
        returned events. This keeps the source side-effect-free except for
        the upstream API call, which makes it trivial to unit-test.
    """

    def __init__(
        self,
        name: str,
        *,
        fetch_window: timedelta,
        on_event: EventCallback,
    ) -> None:
        if not name:
            raise ValueError("OAuthIntegrationSource requires a non-empty name")
        if fetch_window <= timedelta(0):
            raise ValueError("fetch_window must be positive")
        self._name = name
        self._fetch_window = fetch_window
        self._on_event = on_event
        self._started: bool = False

    # ------------------------------------------------------------------ #
    # `wiki_core.IngestSource` surface                                   #
    # ------------------------------------------------------------------ #

    async def start(self) -> None:
        """Mark the source live. Idempotent — repeated calls are no-ops.

        Subclasses with provider-specific setup (HTTP clients, token
        refresh primers) should override and call ``await super().start()``
        first; that way `self._started` stays the single source of truth.
        """
        self._started = True

    async def stop(self) -> None:
        """Shut down cleanly. Idempotent — repeated calls are no-ops.

        Per the `IngestSource` contract, in-flight events are queued before
        returning; subclasses owning long-lived resources (sessions, timers)
        clean them up here and then call ``await super().stop()``.
        """
        self._started = False

    def name(self) -> str:
        """Stable identifier for telemetry/logging."""
        return self._name

    # ------------------------------------------------------------------ #
    # Subclass surface                                                   #
    # ------------------------------------------------------------------ #

    @property
    def fetch_window(self) -> timedelta:
        """Look-back window subclasses use to bound the upstream query."""
        return self._fetch_window

    @property
    def started(self) -> bool:
        """True iff `start()` has been called more recently than `stop()`."""
        return self._started

    async def emit(self, event: IngestEvent) -> None:
        """Hand `event` to the orchestrator via the wired callback.

        Subclasses call this from `fetch_batch` or any future streaming
        path. Centralising it here gives a single seam for future hooks
        (deduplication, rate-limit slowdown, prompt-injection guard per
        ADR-015).
        """
        await self._on_event(event)

    async def fetch_batch(self) -> list[IngestEvent]:
        """Pull one window's worth of items, translate, and return.

        Subclasses MUST override. The orchestrator (or a polling worker
        landed in a follow-up sprint) is in charge of scheduling these
        calls; this method is a pure function of the provider's current
        state plus `self.fetch_window`.

        Implementations SHOULD also `await self.emit(event)` for each
        produced event so downstream consumers wired via `on_event` see
        the batch. Returning the list allows tests to assert on the shape
        without subscribing to the callback.
        """
        raise NotImplementedError(f"{type(self).__name__} must override fetch_batch()")
