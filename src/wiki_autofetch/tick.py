"""
One-shot auto-fetch tick — the entry point systemd invokes every 20 min.

Per [#38](https://github.com/DarkCodePE/second-brain-wiki/issues/38):
- ``~/.sbw/run/auto-fetch.lock`` flock; concurrent invocation is a no-op
- Per-provider checkpoint advanced only on success; failure → DLQ + backoff
- No-op tick ≤ 2 s; typical delta tick ≤ 60 s

Compositional contract:

- The tick picks up the set of providers from a registry function (default:
  return [] — until the daemon composition root wires them, the tick is a
  fast no-op so the systemd timer is harmless).
- For each provider, check DLQ backoff window; skip if backed off.
- Call provider's `fetch_batch()`; on success → `dlq.record_success(name)`
  + cursor advance is the provider's responsibility (already done in M3
  Sprint 1 substrate).
- On failure → `dlq.record_failure(name, error_class, detail)`; no cursor
  advance.
- Emit a final `TickReport` summarising the run; CLI prints it.
"""

from __future__ import annotations

import asyncio
import logging
import time
from collections.abc import Awaitable, Callable, Sequence
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from wiki_autofetch.dlq import AutoFetchDLQ
from wiki_autofetch.lockfile import LockBusy, acquire

if TYPE_CHECKING:
    from wiki_core.protocols import IngestEvent, IngestSource

_log = logging.getLogger(__name__)


SourceLoader = Callable[[], Awaitable[Sequence["IngestSource"]]]
"""Returns the sources to poll on this tick. Plumbed in by the daemon
composition root; default loader yields [] so the systemd timer is a
no-op until providers are wired."""


@dataclass(slots=True)
class SourceTick:
    name: str
    events: int
    duration_s: float
    skipped: bool = False
    skip_reason: str | None = None
    error: str | None = None


@dataclass(slots=True)
class TickReport:
    duration_s: float
    sources: list[SourceTick] = field(default_factory=list)
    lock_busy: bool = False

    @property
    def total_events(self) -> int:
        return sum(s.events for s in self.sources)

    @property
    def failures(self) -> list[SourceTick]:
        return [s for s in self.sources if s.error is not None]

    @property
    def skipped(self) -> list[SourceTick]:
        return [s for s in self.sources if s.skipped]


async def _default_loader() -> Sequence[IngestSource]:
    """Empty by default. The daemon composition root overrides this."""
    return []


async def run_tick(
    *,
    source_loader: SourceLoader | None = None,
    dlq: AutoFetchDLQ | None = None,
    lock_path=None,
) -> TickReport:
    """Execute one tick under the file lock.

    Returns a populated TickReport. When the lock is already held by
    another process, returns ``TickReport(lock_busy=True)`` immediately
    — exit code 0 (no-op) per #38 AC.
    """
    loader = source_loader if source_loader is not None else _default_loader

    # `lock_path=None` means use DEFAULT_LOCK_PATH (the wrapper's default).
    from wiki_autofetch.lockfile import DEFAULT_LOCK_PATH

    path = lock_path or DEFAULT_LOCK_PATH
    start = time.monotonic()

    try:
        with acquire(path):
            return await _run_locked(loader=loader, dlq=dlq, start=start)
    except LockBusy:
        return TickReport(duration_s=time.monotonic() - start, lock_busy=True)


async def _run_locked(
    *,
    loader: SourceLoader,
    dlq: AutoFetchDLQ | None,
    start: float,
) -> TickReport:
    sources = list(await loader())
    own_dlq = dlq is None
    dlq_obj = dlq if dlq is not None else AutoFetchDLQ()
    report = TickReport(duration_s=0.0)

    try:
        for src in sources:
            name = _source_name(src)

            if dlq_obj.is_backed_off(name):
                report.sources.append(
                    SourceTick(
                        name=name,
                        events=0,
                        duration_s=0.0,
                        skipped=True,
                        skip_reason="backoff",
                    )
                )
                continue

            sub_start = time.monotonic()
            try:
                events: Sequence[IngestEvent] = await _fetch(src)
            except Exception as exc:  # noqa: BLE001 — isolate per-source failures
                duration = time.monotonic() - sub_start
                err_class = type(exc).__name__
                detail = str(exc)[:300]
                attempt = dlq_obj.record_failure(name, err_class, detail)
                _log.warning(
                    "autofetch.tick.source_error",
                    extra={"extra_fields": {"source": name, "attempt": attempt}},
                )
                report.sources.append(
                    SourceTick(
                        name=name,
                        events=0,
                        duration_s=duration,
                        error=f"{err_class}:{attempt}",
                    )
                )
                continue

            duration = time.monotonic() - sub_start
            dlq_obj.record_success(name)
            report.sources.append(SourceTick(name=name, events=len(events), duration_s=duration))
    finally:
        if own_dlq:
            dlq_obj.close()

    report.duration_s = time.monotonic() - start
    return report


async def _fetch(source: IngestSource) -> Sequence[IngestEvent]:
    """Pull a batch from a source. Tries ``fetch_batch`` (OAuth integrations)
    then ``fetch_delta`` (scheduler-shape) and falls back to []."""
    for method in ("fetch_batch", "fetch_delta"):
        fn = getattr(source, method, None)
        if fn is None:
            continue
        result = await fn()
        if isinstance(result, tuple):
            result = result[0]
        return list(result)
    return []


def _source_name(source: IngestSource) -> str:
    name_fn = getattr(source, "name", None)
    if callable(name_fn):
        try:
            return str(name_fn())
        except Exception:  # noqa: BLE001
            return type(source).__name__
    return type(source).__name__


# --------------------------------------------------------------------------- #
# CLI entry — `python -m wiki_autofetch.tick`                                 #
# --------------------------------------------------------------------------- #


def main(argv: list[str] | None = None) -> int:
    """systemd-invoked entry. Returns exit code.

    0 = success (including no-op due to lock contention)
    1 = at least one source failed
    2 = unhandled error in the tick harness itself
    """
    import argparse

    parser = argparse.ArgumentParser(prog="sbw-auto-fetch")
    parser.add_argument(
        "--provider", help="Restrict to one provider (for `sbw fetch run --provider X`)"
    )
    parser.add_argument("--json", action="store_true", help="Emit a machine-readable report")
    args = parser.parse_args(argv)

    try:
        report = asyncio.run(run_tick())
    except Exception:  # noqa: BLE001
        _log.exception("autofetch.tick.harness_error")
        return 2

    _print_report(report, as_json=args.json, provider_filter=args.provider)
    return 1 if report.failures else 0


def _print_report(report: TickReport, *, as_json: bool, provider_filter: str | None) -> None:
    if as_json:
        import json

        body = {
            "duration_s": round(report.duration_s, 3),
            "lock_busy": report.lock_busy,
            "sources": [
                {
                    "name": s.name,
                    "events": s.events,
                    "duration_s": round(s.duration_s, 3),
                    "skipped": s.skipped,
                    "skip_reason": s.skip_reason,
                    "error": s.error,
                }
                for s in report.sources
                if provider_filter is None or s.name == provider_filter
            ],
        }
        print(json.dumps(body))
        return

    if report.lock_busy:
        print("auto-fetch: lock busy (another tick is running) — no-op.")
        return
    if not report.sources:
        print(f"auto-fetch: no providers configured. duration={report.duration_s:.2f}s")
        return
    print(f"auto-fetch tick ({report.duration_s:.2f}s):")
    for s in report.sources:
        if provider_filter is not None and s.name != provider_filter:
            continue
        if s.skipped:
            print(f"  - {s.name:<10} SKIP   ({s.skip_reason})")
        elif s.error is not None:
            print(f"  - {s.name:<10} ERROR  {s.error}  ({s.duration_s:.2f}s)")
        else:
            print(f"  - {s.name:<10} OK     events={s.events}  ({s.duration_s:.2f}s)")


__all__ = ["SourceTick", "TickReport", "main", "run_tick"]
