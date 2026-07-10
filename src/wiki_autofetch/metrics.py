"""
Sink-agnostic auto-fetch counters.

Per [PRD-006 FR-8], the worker exposes Prometheus-flavoured counters
(`autofetch_items_total{provider,status}`, `autofetch_tick_duration_seconds`,
`autofetch_lag_seconds`). This module is **transport-agnostic**: it holds
the numbers in memory; emitting them as Prometheus textfile, OTLP, or
plain logs is the caller's job via `to_dict()`.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Any


@dataclass(slots=True)
class SourceMetrics:
    """Per-source counters."""

    name: str
    events_fetched: int = 0
    errors: int = 0
    rate_limited: int = 0
    last_tick_at: datetime | None = None
    last_tick_duration_seconds: float = 0.0
    last_error_class: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "events_fetched": self.events_fetched,
            "errors": self.errors,
            "rate_limited": self.rate_limited,
            "last_tick_at": (
                self.last_tick_at.isoformat() if self.last_tick_at is not None else None
            ),
            "last_tick_duration_seconds": self.last_tick_duration_seconds,
            "last_error_class": self.last_error_class,
        }


@dataclass(slots=True)
class AutoFetchMetrics:
    """Aggregate counters across all sources.

    The scheduler / worker call the `record_*` mutators after each
    per-source tick. The whole object is safe to snapshot via
    `to_dict()` from another asyncio task because we never hold a
    reference to mutable state outside this object.
    """

    total_ticks: int = 0
    total_events_fetched: int = 0
    total_errors: int = 0
    total_rate_limited: int = 0
    last_tick_at: datetime | None = None
    last_tick_duration_seconds: float = 0.0
    sources: dict[str, SourceMetrics] = field(default_factory=dict)

    # ------------------------------------------------------------------ #
    # Mutators                                                           #
    # ------------------------------------------------------------------ #

    def _source(self, name: str) -> SourceMetrics:
        m = self.sources.get(name)
        if m is None:
            m = SourceMetrics(name=name)
            self.sources[name] = m
        return m

    def record_source_success(
        self, name: str, *, events: int, duration_seconds: float, at: datetime | None = None
    ) -> None:
        s = self._source(name)
        s.events_fetched += events
        s.last_tick_at = at or datetime.now(UTC)
        s.last_tick_duration_seconds = duration_seconds
        s.last_error_class = None
        self.total_events_fetched += events

    def record_source_error(
        self,
        name: str,
        *,
        error_class: str,
        duration_seconds: float,
        at: datetime | None = None,
    ) -> None:
        s = self._source(name)
        s.errors += 1
        s.last_tick_at = at or datetime.now(UTC)
        s.last_tick_duration_seconds = duration_seconds
        s.last_error_class = error_class
        self.total_errors += 1

    def record_source_rate_limited(self, name: str) -> None:
        s = self._source(name)
        s.rate_limited += 1
        self.total_rate_limited += 1

    def record_tick(self, *, duration_seconds: float, at: datetime | None = None) -> None:
        self.total_ticks += 1
        self.last_tick_at = at or datetime.now(UTC)
        self.last_tick_duration_seconds = duration_seconds

    # ------------------------------------------------------------------ #
    # Serialisation                                                      #
    # ------------------------------------------------------------------ #

    def to_dict(self) -> dict[str, Any]:
        """Stable JSON-friendly snapshot for telemetry consumers."""
        return {
            "total_ticks": self.total_ticks,
            "total_events_fetched": self.total_events_fetched,
            "total_errors": self.total_errors,
            "total_rate_limited": self.total_rate_limited,
            "last_tick_at": (
                self.last_tick_at.isoformat() if self.last_tick_at is not None else None
            ),
            "last_tick_duration_seconds": self.last_tick_duration_seconds,
            "sources": {name: m.to_dict() for name, m in self.sources.items()},
        }
