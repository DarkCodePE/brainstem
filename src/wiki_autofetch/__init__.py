"""
`wiki_autofetch` — periodic auto-fetch scheduler for M3 integrations.

Per [PRD-006 Auto-fetch Worker](../../docs/PRD-006-autofetch-worker.md), this
package polls every active `wiki_core.IngestSource` on a configurable
interval (default 20 minutes), rate-limits the resulting events through a
token bucket, and pushes them into a `wiki_core.MemoryStore` sink.

Surfaces shipped here:

- `AutoFetchScheduler` — asyncio periodic task that ticks N sources with
  jittered interval and per-source error isolation.
- `TokenBucket` — deterministic rate limiter (injectable clock) shared
  between auto-fetch and file-drop ingest per PRD-006 FR-7.
- `AutoFetchMetrics` — sink-agnostic value object that telemetry
  consumers (Prometheus textfile, etc.) can serialise via `to_dict()`.
- `AutoFetchWorker` — composes the three above with a `MemoryStore`
  and a per-source fetch callable.

This package depends ONLY on `wiki_core.protocols`. Concrete integrations
(Gmail, Slack, GitHub, ...) live in `wiki_integrations/` and wire into
the scheduler through the `IngestSource` contract.
"""

from __future__ import annotations

from wiki_autofetch.dlq import AutoFetchDLQ, DLQEntry
from wiki_autofetch.lockfile import LockBusy, acquire
from wiki_autofetch.metrics import AutoFetchMetrics, SourceMetrics
from wiki_autofetch.rate_limiter import TokenBucket
from wiki_autofetch.scheduler import (
    AutoFetchResult,
    AutoFetchScheduler,
    FetchCallable,
    SourceTickResult,
)
from wiki_autofetch.tick import SourceTick, TickReport, run_tick
from wiki_autofetch.worker import AutoFetchWorker

__all__ = [
    "AutoFetchDLQ",
    "AutoFetchMetrics",
    "AutoFetchResult",
    "AutoFetchScheduler",
    "AutoFetchWorker",
    "DLQEntry",
    "FetchCallable",
    "LockBusy",
    "SourceMetrics",
    "SourceTick",
    "SourceTickResult",
    "TickReport",
    "TokenBucket",
    "acquire",
    "run_tick",
]
