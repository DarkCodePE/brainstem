from __future__ import annotations

import asyncio
import logging
from collections.abc import Awaitable, Callable
from pathlib import Path

try:
    from watchdog.events import FileSystemEvent, PatternMatchingEventHandler
    from watchdog.observers import Observer
except ImportError as e:  # pragma: no cover
    raise ImportError(
        "wiki_ingest.watcher requires the `watchdog` package. "
        "Install via `pip install watchdog>=4.0`."
    ) from e

from wiki_ingest.config import Config
from wiki_ingest.security import (
    UnsafePathError,
    quarantine_symlink,
    validate_safe_path,
)

log = logging.getLogger("wiki_ingest.watcher")

DebounceCallback = Callable[[Path], Awaitable[None]]


class _Handler(PatternMatchingEventHandler):
    def __init__(
        self,
        loop: asyncio.AbstractEventLoop,
        on_event: Callable[[Path, str], None],
    ) -> None:
        super().__init__(
            patterns=["*"],
            ignore_patterns=[
                "*/.git/*",
                "*/.obsidian/*",
                "*/_ingested/*",
                "*.sync-conflict-*",
                "*~",
                "*.tmp",
                "*.part",
                "*.swp",
                "*.crdownload",
            ],
            ignore_directories=True,
            case_sensitive=True,
        )
        self._loop = loop
        self._on_event = on_event

    def _dispatch(self, path: str, event_type: str) -> None:
        p = Path(path)
        self._loop.call_soon_threadsafe(self._on_event, p, event_type)

    def on_created(self, event: FileSystemEvent) -> None:
        if not event.is_directory:
            self._dispatch(event.src_path, "created")

    def on_modified(self, event: FileSystemEvent) -> None:
        if not event.is_directory:
            self._dispatch(event.src_path, "modified")

    def on_moved(self, event: FileSystemEvent) -> None:
        if not event.is_directory:
            dest = getattr(event, "dest_path", event.src_path)
            self._dispatch(dest, "moved")

    def on_closed(self, event: FileSystemEvent) -> None:  # pragma: no cover
        if not event.is_directory:
            self._dispatch(event.src_path, "closed")


class WatcherService:
    """Recursive watchdog observer with per-path debouncing on asyncio loop."""

    def __init__(self, cfg: Config, on_debounced: DebounceCallback) -> None:
        self._cfg = cfg
        self._on_debounced = on_debounced
        self._observer: Observer | None = None
        self._timers: dict[str, asyncio.TimerHandle] = {}
        self._loop: asyncio.AbstractEventLoop | None = None

    async def start(self) -> None:
        self._loop = asyncio.get_running_loop()
        if not self._cfg.raw_dir.exists():
            self._cfg.raw_dir.mkdir(parents=True, exist_ok=True)
        handler = _Handler(self._loop, self._schedule)
        observer = Observer()
        observer.schedule(handler, str(self._cfg.raw_dir), recursive=True)
        observer.daemon = True
        observer.start()
        self._observer = observer
        log.info(
            "watcher.started",
            extra={"extra_fields": {"raw_dir": str(self._cfg.raw_dir)}},
        )

    def _schedule(self, path: Path, event_type: str) -> None:
        # SEC-02: reject symlinks on the raw event path (before resolution).
        try:
            if path.is_symlink():
                quarantine_symlink(path, self._cfg.raw_dir, reason="symlinks")
                log.warning(
                    "watcher.symlink_rejected",
                    extra={"extra_fields": {"reason": "symlink-rejected"}},
                )
                return
        except OSError:
            return

        # SEC-01: validate-and-jail the path against raw_dir (strict resolve).
        try:
            safe_path = validate_safe_path(path, self._cfg.raw_dir)
        except UnsafePathError:
            log.warning(
                "watcher.path_rejected",
                extra={"extra_fields": {"reason": "unsafe-path"}},
            )
            return

        key = str(safe_path)
        timer = self._timers.pop(key, None)
        if timer is not None:
            timer.cancel()

        loop = self._loop
        if loop is None:
            return

        def _fire() -> None:
            self._timers.pop(key, None)
            asyncio.create_task(self._safe_callback(safe_path))

        self._timers[key] = loop.call_later(self._cfg.debounce_seconds, _fire)

    async def _safe_callback(self, path: Path) -> None:
        try:
            await self._on_debounced(path)
        except Exception as e:  # noqa: BLE001
            log.error(
                "watcher.callback_failed",
                extra={"extra_fields": {"error_class": type(e).__name__}},
            )

    async def stop(self) -> None:
        for timer in list(self._timers.values()):
            timer.cancel()
        self._timers.clear()
        if self._observer is not None:
            self._observer.stop()
            self._observer.join(timeout=5.0)
            self._observer = None
        log.info("watcher.stopped")


# Backward-compat alias — tests in tests/wiki_ingest/ import IngestWatcher.
# Canonical name is WatcherService.
IngestWatcher = WatcherService
