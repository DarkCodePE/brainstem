from __future__ import annotations

import argparse
import asyncio
import logging
import mimetypes
import os
import signal
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any

from wiki_ingest.config import Config
from wiki_ingest.filters import should_ignore
from wiki_ingest.models import IngestEvent
from wiki_ingest.queue import EventQueue
from wiki_ingest.security import (
    SafeLogFormatter,
    UnsafePathError,
    apply_safe_logging,
    tighten_db_permissions,
    validate_safe_path,
)
from wiki_ingest.watcher import WatcherService
from wiki_ingest.worker import WorkerPool

if TYPE_CHECKING:
    from wiki_autofetch.worker import AutoFetchWorker

log = logging.getLogger("wiki_ingest.daemon")


# Kept as an alias so callers that imported the legacy name still work; the
# actual implementation comes from wiki_ingest.security (SEC-07 allowlist).
JsonFormatter = SafeLogFormatter


def _setup_logging() -> None:
    level_name = os.environ.get("INGEST_LOG_LEVEL", "INFO").upper()
    apply_safe_logging(level_name)


def _rel_to_raw(rel_path: str, cfg: Config) -> str:
    """Normalise `rel_path` so it is relative to raw/, never absolute."""
    raw_name = cfg.raw_dir.name
    parts = rel_path.split(os.sep)
    if parts and parts[0] == raw_name:
        parts = parts[1:]
    stripped = os.sep.join(parts) if parts else rel_path
    return stripped.lstrip(os.sep)


def _sd_notify(msg: str) -> None:
    try:
        from systemd import daemon as sd_daemon  # type: ignore

        sd_daemon.notify(msg)
        return
    except ImportError:
        pass
    sock_path = os.environ.get("NOTIFY_SOCKET")
    if not sock_path:
        return
    try:
        import socket

        family = socket.AF_UNIX
        addr = sock_path
        if sock_path.startswith("@"):
            addr = "\0" + sock_path[1:]
        with socket.socket(family, socket.SOCK_DGRAM) as s:
            s.sendto(msg.encode("utf-8"), addr)
    except OSError:
        pass


class Daemon:
    def __init__(
        self,
        cfg: Config,
        *,
        autofetch_worker: AutoFetchWorker | None = None,
        post_write_hook: Any = None,
    ) -> None:
        """Build a daemon.

        Parameters
        ----------
        cfg:
            Daemon configuration.
        autofetch_worker:
            Optional `AutoFetchWorker` to run alongside the filesystem
            watcher. When supplied, the daemon manages its lifecycle
            (start with the watcher, stop on shutdown) and surfaces
            per-source metrics in `status()`. When `None`, the daemon
            behaves exactly as it did before M3 Sprint 2 — pure
            filesystem watcher.
        post_write_hook:
            Optional callable `(domain_event, page_path) -> Awaitable[None]`
            invoked after each successful `write_page`. M3 Sprint 4
            wire-in surface for `wiki_memory.seal_hook.SealOnIngestHook`
            so the Memory Tree indexes + seals every ingested source.
            When `None`, the daemon ingests without seal-on-ingest
            (M2 + M3-S1..S3 behaviour).
        """
        self.cfg = cfg
        self.queue = EventQueue(cfg.db_path)
        self.pool = WorkerPool(cfg, self.queue, post_write_hook=post_write_hook)
        self.watcher = WatcherService(cfg, self._on_debounced)
        self.autofetch: AutoFetchWorker | None = autofetch_worker
        self._shutdown = asyncio.Event()
        self._tasks: set[asyncio.Task] = set()

    async def start(self, catchup_only: bool = False) -> None:
        await self.queue.init()
        # SEC-11: lock down the SQLite DB and its WAL siblings.
        tighten_db_permissions(self.cfg.db_path)
        recovered = await self.queue.recover_stuck()
        if recovered:
            log.info(
                "daemon.recovered_stuck",
                extra={"extra_fields": {"count": recovered}},
            )

        await self._catchup()

        if catchup_only:
            await self._drain_queue()
            return

        await self.watcher.start()
        # M3 Sprint 2 (PRD-006): start AutoFetchWorker as a sibling
        # background task. A failure to spin up autofetch must NOT take
        # the watcher down — the daemon's primary contract is filesystem
        # ingest. We log the start failure and surface it via status().
        self._autofetch_start_error: str | None = None
        if self.autofetch is not None:
            try:
                await self.autofetch.start()
                log.info(
                    "daemon.autofetch_started",
                    extra={
                        "extra_fields": {
                            "interval_seconds": self.cfg.autofetch_interval_seconds,
                        }
                    },
                )
            except Exception as e:  # noqa: BLE001
                err_class = type(e).__name__
                self._autofetch_start_error = err_class
                log.error(
                    "daemon.autofetch_start_failed",
                    extra={"extra_fields": {"error_class": err_class}},
                )
        self._spawn(self._dispatcher_loop(), "dispatcher")
        self._spawn(self._metrics_loop(), "metrics")
        _sd_notify("READY=1")
        log.info("daemon.ready")

    async def run(self, catchup_only: bool = False) -> None:
        await self.start(catchup_only=catchup_only)
        if catchup_only:
            # ADR-035 D2: the oneshot path must release its resources or
            # the process never exits — the aiosqlite connection thread
            # is non-daemon and the MCP stdio subprocess stays alive.
            await self.pool.close()
            await self.queue.close()
            return
        await self._shutdown.wait()
        await self.stop()

    async def stop(self) -> None:
        log.info("daemon.shutdown_begin")
        _sd_notify("STOPPING=1")
        await self.watcher.stop()
        # M3 Sprint 2: stop autofetch worker before the dispatcher loop
        # so any in-flight tick that pushes into the queue completes
        # before pool.close().
        if self.autofetch is not None:
            try:
                await self.autofetch.stop()
            except Exception as e:  # noqa: BLE001
                log.error(
                    "daemon.autofetch_stop_failed",
                    extra={"extra_fields": {"error_class": type(e).__name__}},
                )
        for t in list(self._tasks):
            t.cancel()
        for t in list(self._tasks):
            try:
                await t
            except (asyncio.CancelledError, Exception):  # noqa: BLE001
                pass
        await self.pool.close()
        await self.queue.close()
        log.info("daemon.shutdown_done")

    # ------------------------------------------------------------------ #
    # Status RPC                                                         #
    # ------------------------------------------------------------------ #

    async def status(self) -> dict[str, Any]:
        """Snapshot of daemon state for telemetry / status RPC.

        When `autofetch` is wired, the `autofetch:` key is populated with
        the worker's metrics snapshot (per-source counters + totals).
        Otherwise the key is `None` so callers can distinguish "disabled"
        from "enabled but quiet".
        """
        counts = await self.queue.counts_by_status()
        depth = await self.queue.queue_depth()
        out: dict[str, Any] = {
            "queue": {
                "depth": depth,
                "counts": counts,
            },
            "pool": {
                "busy": self.pool.busy_count,
            },
            "watcher": {
                "running": self.watcher is not None,
            },
            "autofetch": None,
        }
        if self.autofetch is not None:
            snap = self.autofetch.metrics.to_dict()
            per_source: dict[str, dict[str, Any]] = {}
            for name, m in snap.get("sources", {}).items():
                per_source[name] = {
                    "events_fetched": m.get("events_fetched", 0),
                    "errors": m.get("errors", 0),
                    "rate_limited": m.get("rate_limited", 0),
                    "last_tick_at": m.get("last_tick_at"),
                    "last_error_class": m.get("last_error_class"),
                }
            out["autofetch"] = {
                "running": self.autofetch.is_running,
                "fetched_total": snap.get("total_events_fetched", 0),
                "errors_total": snap.get("total_errors", 0),
                "rate_limited_total": snap.get("total_rate_limited", 0),
                "ticks_total": snap.get("total_ticks", 0),
                "last_tick_at": snap.get("last_tick_at"),
                "last_tick_duration_seconds": snap.get("last_tick_duration_seconds", 0.0),
                "start_error": getattr(self, "_autofetch_start_error", None),
                "sources": per_source,
            }
        return out

    def request_shutdown(self) -> None:
        self._shutdown.set()

    def _spawn(self, coro, name: str) -> asyncio.Task:
        task = asyncio.create_task(coro, name=name)
        self._tasks.add(task)
        task.add_done_callback(self._tasks.discard)
        return task

    async def _on_debounced(self, path: Path) -> None:
        # SEC-01: re-jail the path before any stat call. The watcher already
        # validated it but the file may have been swapped between debounce
        # scheduling and callback firing.
        try:
            safe_path = validate_safe_path(path, self.cfg.raw_dir)
        except UnsafePathError:
            log.warning(
                "daemon.path_rejected",
                extra={"extra_fields": {"reason": "unsafe-path"}},
            )
            return

        # SEC-02: symlinks are rejected by the watcher, but double-check.
        try:
            if safe_path.is_symlink():
                log.warning(
                    "daemon.symlink_rejected",
                    extra={"extra_fields": {"reason": "symlink-rejected"}},
                )
                return
        except OSError:
            return

        if not safe_path.exists():
            return
        try:
            stat = safe_path.stat()
        except OSError as e:
            log.warning(
                "daemon.stat_failed",
                extra={"extra_fields": {"error_class": type(e).__name__}},
            )
            return

        ignore, reason = should_ignore(safe_path, size=stat.st_size, cfg=self.cfg)
        if ignore:
            log.debug(
                "daemon.ignored",
                extra={"extra_fields": {"reason": reason}},
            )
            return

        event = self._build_event(safe_path, stat.st_mtime, stat.st_size, "modified")
        if event is None:
            return
        await self.queue.enqueue(event)
        log.info(
            "daemon.enqueued",
            extra={
                "extra_fields": {
                    "event_id": event.event_id,
                    "rel_path": _rel_to_raw(event.rel_path, self.cfg),
                    "bucket": event.bucket,
                    "size": event.size,
                }
            },
        )

    def _build_event(
        self, path: Path, mtime: float, size: int, event_type: str
    ) -> IngestEvent | None:
        try:
            rel = path.relative_to(self.cfg.wiki_root)
        except ValueError:
            return None
        parts = rel.parts
        if len(parts) < 2 or parts[0] != self.cfg.raw_dir.name:
            return None
        bucket = parts[1] if len(parts) >= 3 else "uncategorized"
        mtime_iso = (
            datetime.fromtimestamp(mtime, tz=UTC)
            .isoformat(timespec="seconds")
            .replace("+00:00", "Z")
        )
        mime, _ = mimetypes.guess_type(str(path))
        return IngestEvent(
            path=str(path),
            rel_path=str(rel),
            bucket=bucket,
            event_type=event_type,
            mtime=mtime_iso,
            size=size,
            mime=mime,
        )

    async def _catchup(self) -> None:
        raw = self.cfg.raw_dir
        if not raw.exists():
            return
        count = 0
        for entry in raw.rglob("*"):
            if not entry.is_file():
                continue
            # SEC-02: skip symlinks entirely during catchup.
            try:
                if entry.is_symlink():
                    continue
            except OSError:
                continue
            if str(entry).startswith(str(self.cfg.ingested_dir)):
                continue
            # SEC-01: jail every catchup candidate.
            try:
                safe_entry = validate_safe_path(entry, self.cfg.raw_dir)
            except UnsafePathError:
                continue
            try:
                stat = safe_entry.stat()
            except OSError:
                continue
            ignore, _ = should_ignore(safe_entry, size=stat.st_size, cfg=self.cfg)
            if ignore:
                continue
            event = self._build_event(safe_entry, stat.st_mtime, stat.st_size, "catchup")
            if event is None:
                continue
            await self.queue.enqueue(event)
            count += 1
        if count:
            log.info("daemon.catchup", extra={"extra_fields": {"count": count}})

    async def _dispatcher_loop(self) -> None:
        while not self._shutdown.is_set():
            event = await self.queue.claim_next()
            if event is None:
                try:
                    await asyncio.wait_for(self._shutdown.wait(), timeout=1.0)
                except TimeoutError:
                    pass
                continue
            self._spawn(self.pool.dispatch(event), f"worker:{event.event_id[:8]}")

    async def _drain_queue(self) -> None:
        while True:
            depth = await self.queue.queue_depth()
            if depth == 0:
                break
            event = await self.queue.claim_next()
            if event is None:
                await asyncio.sleep(0.1)
                continue
            await self.pool.dispatch(event)

    async def _metrics_loop(self) -> None:
        metrics_path = self.cfg.metrics_path
        if metrics_path is None:
            return
        while not self._shutdown.is_set():
            try:
                await self._write_metrics(metrics_path)
            except Exception as e:  # noqa: BLE001
                log.error(
                    "daemon.metrics_failed",
                    extra={"extra_fields": {"error_class": type(e).__name__}},
                )
            _sd_notify("WATCHDOG=1")
            try:
                await asyncio.wait_for(self._shutdown.wait(), timeout=15.0)
            except TimeoutError:
                pass

    async def _write_metrics(self, path: Path) -> None:
        counts = await self.queue.counts_by_status()
        depth = await self.queue.queue_depth()
        p50 = await self.queue.lag_p(0.5)
        p90 = await self.queue.lag_p(0.9)
        p99 = await self.queue.lag_p(0.99)
        lines = [
            "# HELP wiki_ingest_events_total Events grouped by terminal status",
            "# TYPE wiki_ingest_events_total counter",
        ]
        for status in ("pending", "processing", "done", "skipped", "failed"):
            lines.append(f'wiki_ingest_events_total{{status="{status}"}} {counts.get(status, 0)}')
        lines += [
            "# TYPE wiki_ingest_queue_depth gauge",
            f"wiki_ingest_queue_depth {depth}",
            "# TYPE wiki_ingest_lag_seconds gauge",
            f'wiki_ingest_lag_seconds{{quantile="0.5"}} {p50:.3f}',
            f'wiki_ingest_lag_seconds{{quantile="0.9"}} {p90:.3f}',
            f'wiki_ingest_lag_seconds{{quantile="0.99"}} {p99:.3f}',
            "# TYPE wiki_ingest_worker_pool_busy gauge",
            f"wiki_ingest_worker_pool_busy {self.pool.busy_count}",
            "",
        ]
        from wiki_ingest.security import atomic_write_text as _atomic_write_text

        _atomic_write_text(path, "\n".join(lines), mode=0o600)


def _install_signal_handlers(loop: asyncio.AbstractEventLoop, daemon: Daemon) -> None:
    for sig in (signal.SIGTERM, signal.SIGINT):
        try:
            loop.add_signal_handler(sig, daemon.request_shutdown)
        except NotImplementedError:  # pragma: no cover
            pass


def _parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(prog="wiki-ingest")
    parser.add_argument("--once", action="store_true", help="Run catch-up over raw/ then exit")
    return parser.parse_args(argv)


async def _async_main(argv: list[str]) -> int:
    # SEC-11: restrict created-file mode bits to the owner BEFORE any
    # filesystem I/O touches the SQLite DB or the metrics file.
    os.umask(0o077)
    _setup_logging()
    args = _parse_args(argv)
    cfg = Config.from_env()
    # ADR-035: build through the composition root so env-gated hooks
    # (seal-on-ingest, synthesis-on-ingest) are wired in BOTH the
    # resident-daemon mode and the ephemeral `--once` activation (D2).
    # Local import: composition imports this module at its top level.
    from wiki_ingest.composition import build_daemon

    daemon = await build_daemon(cfg)
    loop = asyncio.get_running_loop()
    _install_signal_handlers(loop, daemon)
    log.info(
        "daemon.starting",
        extra={"extra_fields": {"bucket": cfg.raw_dir.name}},
    )
    try:
        await daemon.run(catchup_only=args.once)
    except Exception as e:  # noqa: BLE001
        # Issue #18 follow-up: surface the full traceback to stderr so
        # systemd journal shows what actually failed. The SafeLogFormatter
        # (security.py:372) intentionally drops error messages from the
        # structured payload, so we bypass it for this fatal-path event.
        import sys as _sys
        import traceback as _tb

        log.error(
            "daemon.fatal",
            extra={"extra_fields": {"error_class": type(e).__name__}},
        )
        print(
            f"DAEMON_FATAL_DEBUG type={type(e).__name__} msg={e!r}",
            file=_sys.stderr,
            flush=True,
        )
        _tb.print_exc(file=_sys.stderr)
        return 1
    return 0


def main() -> int:
    return asyncio.run(_async_main(sys.argv[1:]))


if __name__ == "__main__":
    raise SystemExit(main())
