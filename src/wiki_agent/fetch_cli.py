"""
CLI handler for ``sbw fetch {status,run,clear}`` per issue #38 AC.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys

from wiki_autofetch.dlq import AutoFetchDLQ
from wiki_autofetch.tick import run_tick


def run_fetch_cli(args: argparse.Namespace) -> int:
    action = getattr(args, "fetch_action", None)
    if action == "status":
        return _cmd_status(as_json=getattr(args, "json", False))
    if action == "run":
        return _cmd_run(provider=args.provider, as_json=args.json)
    if action == "clear":
        return _cmd_clear(provider=args.provider)
    print(f"Unknown fetch action: {action!r}", file=sys.stderr)
    return 1


def _cmd_status(*, as_json: bool = False) -> int:
    dlq = AutoFetchDLQ()
    try:
        failures = dlq.list_failures()
        if as_json:
            print(
                json.dumps(
                    {
                        "failure_count": len(failures),
                        "entries": [
                            {
                                "source": e.source_name,
                                "ts": e.ts,
                                "error": e.error_class,
                                "attempt": e.attempt,
                            }
                            for e in failures[:50]
                        ],
                    }
                )
            )
            return 0
        if not failures:
            print("auto-fetch: DLQ empty. All providers in green state.")
            return 0
        print(f"auto-fetch: {len(failures)} DLQ entries (showing last 20)")
        for entry in failures[:20]:
            print(
                f"  {entry.ts}  {entry.source_name:<10}  "
                f"attempt={entry.attempt:<3} {entry.error_class}: {entry.error_detail[:80]}"
            )
        return 1
    finally:
        dlq.close()


def _cmd_run(*, provider: str | None, as_json: bool) -> int:
    report = asyncio.run(run_tick())
    if report.lock_busy:
        if as_json:
            print(json.dumps({"lock_busy": True, "duration_s": report.duration_s}))
        else:
            print("auto-fetch: lock busy (another tick is running) — no-op.")
        return 0

    if as_json:
        body = {
            "duration_s": round(report.duration_s, 3),
            "lock_busy": False,
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
                if provider is None or s.name == provider
            ],
        }
        print(json.dumps(body))
        return 1 if report.failures else 0

    if not report.sources:
        print(f"auto-fetch: no providers configured. duration={report.duration_s:.2f}s")
        print("(The daemon composition root has not registered any IngestSources.)")
        return 0

    print(f"auto-fetch tick ({report.duration_s:.2f}s):")
    for s in report.sources:
        if provider is not None and s.name != provider:
            continue
        if s.skipped:
            print(f"  - {s.name:<10} SKIP   ({s.skip_reason})")
        elif s.error is not None:
            print(f"  - {s.name:<10} ERROR  {s.error}  ({s.duration_s:.2f}s)")
        else:
            print(f"  - {s.name:<10} OK     events={s.events}  ({s.duration_s:.2f}s)")
    return 1 if report.failures else 0


def _cmd_clear(*, provider: str | None) -> int:
    dlq = AutoFetchDLQ()
    try:
        removed = dlq.clear(provider)
        scope = provider if provider else "ALL"
        print(f"auto-fetch: cleared {removed} DLQ entries (scope: {scope}).")
        return 0
    finally:
        dlq.close()


__all__ = ["run_fetch_cli"]
