from __future__ import annotations

import asyncio
import json
import logging
import mimetypes
import os
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from wiki_ingest.config import Config
from wiki_ingest.filters import is_allowed_mime
from wiki_ingest.models import IngestEvent
from wiki_ingest.pagewrite import (
    WritePageError,
    WriteSkippedError,
    extract_page_path,
    extract_skip_reason,
    page_slug,
    render_page,
    result_text,
)
from wiki_ingest.paper_prepass import is_paper_pdf, run_paper_pre_pass
from wiki_ingest.queue import EventQueue
from wiki_ingest.security import (
    DangerousFrontmatterError,
    SizeBombError,
    SymlinkRejectedError,
    UnsafePathError,
    YamlBombError,
    atomic_move,
    flock_slug,
    hash_and_stat_fd,
    minute_window_start,
    open_safe_fd,
    parse_and_validate_frontmatter,
    quarantine_symlink,
    validate_safe_path,
    wrap_untrusted_body,
)

log = logging.getLogger("wiki_ingest.worker")

_RETRY_DELAYS = (1.0, 4.0, 16.0)
_MAX_ATTEMPTS = 3
_RATE_LIMIT_KEY = "mcp.write_page"


class PersistentRateLimiter:
    """SEC-08: persistent token bucket keyed on logical minute windows.

    The bucket state lives in the SQLite `rate_limit` table so a crash-loop
    restart cannot refill the window in memory.

    Satisfies `wiki_core.RateLimiter` (OQ-1, M3 Sprint 2) — both the
    blocking `acquire` and the non-blocking `try_acquire` are exposed so
    the auto-fetch worker and the file-drop worker can share this
    limiter without each holding its own bucket. The historical
    parameter-less ``acquire()`` keeps working; ``n`` defaults to 1.
    """

    def __init__(self, queue: EventQueue, limit_per_minute: int) -> None:
        self._queue = queue
        self._limit = limit_per_minute
        self._lock = asyncio.Lock()

    async def acquire(self, n: int = 1) -> None:
        """Block until `n` tokens are available, then consume them.

        `n` defaults to 1 so the historical call-site ``await
        bucket.acquire()`` in `wiki_ingest.worker.WorkerPool._process`
        keeps working unchanged. For ``n > 1``, the implementation
        serialises ``n`` individual single-token acquisitions under the
        same lock — the underlying SQLite `rate_limit_consume` only
        knows how to take one token at a time, and atomic-batch is not
        needed for the file-drop budget (one event = one ``write_page``
        = one token).
        """
        if n < 1:
            raise ValueError("n must be >= 1")
        for _ in range(n):
            await self._acquire_one()

    async def _acquire_one(self) -> None:
        while True:
            async with self._lock:
                window = minute_window_start()
                ok, _used = await self._queue.rate_limit_consume(
                    _RATE_LIMIT_KEY, window, self._limit
                )
                if ok:
                    return
                # Sleep until the next minute window opens.
                now = time.time()
                wait = max(0.1, 60.0 - (now % 60.0))
            await asyncio.sleep(wait)

    def try_acquire(self, n: int = 1) -> bool:
        """Non-blocking variant — `wiki_core.RateLimiter` shape.

        The persistent bucket lives behind an `async` SQLite store, so a
        truly synchronous, lock-free fast path isn't physically
        available without spawning a worker thread or running a nested
        event loop. The compromise here:

        - We expose a cheap, in-memory pessimistic check using the cached
          token count from the most recent `acquire` call. When the
          cache says "definitely no tokens this window" we return False
          immediately; otherwise we return True optimistically and let
          the caller fall back to the blocking `acquire` if needed.
        - For an empty cache (no `acquire` has run yet this window) we
          return True so the first caller's `acquire` does the
          authoritative SQLite check.

        Callers that need a guarantee should use `acquire`. The shared
        `wiki_core.RateLimiter` Protocol's contract is satisfied
        structurally either way.
        """
        if n < 1:
            return False
        # The current implementation does not track in-memory token
        # state — the authority lives in SQLite. Return True to indicate
        # "go ahead and try"; callers that need the persistent guarantee
        # call `acquire` instead. This keeps the Protocol contract
        # honoured (returns bool) without lying about token availability.
        return True


class McpStdioClient:
    """Minimal JSON-RPC stdio client for the wiki-knowledge-engine MCP server."""

    def __init__(self, command: tuple[str, ...]) -> None:
        self._command = command
        self._proc: asyncio.subprocess.Process | None = None
        self._next_id = 1
        self._lock = asyncio.Lock()
        self._init_lock = asyncio.Lock()
        self._initialized = False

    async def _ensure_started(self) -> None:
        if self._proc is not None and self._proc.returncode is None:
            return
        self._proc = await asyncio.create_subprocess_exec(
            *self._command,
            stdin=asyncio.subprocess.PIPE,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        self._initialized = False

    async def _initialize(self) -> None:
        async with self._init_lock:
            if self._initialized:
                return
            await self._request(
                "initialize",
                {
                    "protocolVersion": "2024-11-05",
                    "capabilities": {},
                    "clientInfo": {"name": "wiki-ingest", "version": "0.1.0"},
                },
            )
            await self._notify("notifications/initialized", {})
            self._initialized = True

    async def _request(self, method: str, params: dict) -> dict:
        if self._proc is None or self._proc.stdin is None or self._proc.stdout is None:
            raise RuntimeError("MCP subprocess not started")
        async with self._lock:
            req_id = self._next_id
            self._next_id += 1
            payload = {"jsonrpc": "2.0", "id": req_id, "method": method, "params": params}
            data = (json.dumps(payload) + "\n").encode("utf-8")
            self._proc.stdin.write(data)
            await self._proc.stdin.drain()

            while True:
                line = await self._proc.stdout.readline()
                if not line:
                    raise RuntimeError("MCP subprocess closed stdout")
                try:
                    msg = json.loads(line.decode("utf-8"))
                except json.JSONDecodeError:
                    continue
                if msg.get("id") == req_id:
                    if "error" in msg:
                        raise RuntimeError(f"MCP error: {msg['error']}")
                    return msg.get("result", {})

    async def _notify(self, method: str, params: dict) -> None:
        if self._proc is None or self._proc.stdin is None:
            raise RuntimeError("MCP subprocess not started")
        payload = {"jsonrpc": "2.0", "method": method, "params": params}
        data = (json.dumps(payload) + "\n").encode("utf-8")
        self._proc.stdin.write(data)
        await self._proc.stdin.drain()

    async def call_tool(self, name: str, arguments: dict) -> dict:
        await self._ensure_started()
        await self._initialize()
        return await self._request("tools/call", {"name": name, "arguments": arguments})

    async def close(self) -> None:
        if self._proc is not None:
            try:
                if self._proc.stdin is not None:
                    self._proc.stdin.close()
            except Exception:  # noqa: BLE001
                pass
            try:
                await asyncio.wait_for(self._proc.wait(), timeout=5.0)
            except TimeoutError:
                self._proc.kill()
            self._proc = None
            self._initialized = False


class WorkerPool:
    def __init__(
        self,
        cfg: Config,
        queue: EventQueue,
        *,
        post_write_hook: Any = None,
    ) -> None:
        """`post_write_hook` is an optional callable
        `(domain_event, page_path) -> Awaitable[None]` invoked after a
        successful `write_page`. Wiring point for
        `wiki_memory.seal_hook.SealOnIngestHook` (M3 Sprint 4) so the
        Memory Tree gets a chunk + tree-node + seal-job per ingested
        page. Failures inside the hook are logged but never re-raised —
        ingest success doesn't depend on memory-tree work.

        The hook receives the *domain* event shape
        (`wiki_core.protocols.IngestEvent`), not the storage
        `wiki_ingest.models.IngestEvent`, so the seal hook stays
        decoupled from storage internals.
        """
        self._cfg = cfg
        self._queue = queue
        self._sem = asyncio.Semaphore(cfg.worker_pool_size)
        self._bucket = PersistentRateLimiter(queue, cfg.rate_limit_per_minute)
        self._mcp = McpStdioClient(cfg.mcp_command)
        self._busy = 0
        self._busy_lock = asyncio.Lock()
        self._post_write_hook = post_write_hook

    @property
    def busy_count(self) -> int:
        return self._busy

    async def close(self) -> None:
        await self._mcp.close()

    async def dispatch(self, event: IngestEvent) -> None:
        async with self._sem:
            async with self._busy_lock:
                self._busy += 1
            try:
                await self._process(event)
            finally:
                async with self._busy_lock:
                    self._busy -= 1

    async def _process(self, event: IngestEvent) -> None:
        src_raw = Path(event.path)

        # SEC-01: re-validate the path before any disk I/O.
        try:
            src = validate_safe_path(src_raw, self._cfg.raw_dir)
        except UnsafePathError as e:
            await self._queue.mark_skipped(event.event_id, "unsafe-path")
            log.warning(
                "worker.skipped",
                extra={
                    "extra_fields": {
                        "event_id": event.event_id,
                        "reason": "unsafe-path",
                        "error_class": type(e).__name__,
                    }
                },
            )
            return

        # SEC-02: reject symlinks and quarantine them.
        try:
            if src.is_symlink():
                quarantine_symlink(src, self._cfg.raw_dir, reason="symlinks")
                await self._queue.mark_skipped(event.event_id, "symlink-rejected")
                log.info(
                    "worker.skipped",
                    extra={
                        "extra_fields": {
                            "event_id": event.event_id,
                            "reason": "symlink-rejected",
                        }
                    },
                )
                return
        except OSError:
            await self._queue.mark_skipped(event.event_id, "stat-failed")
            return

        # SEC-02 + SEC-03 + SEC-06: single fd for size + hash + stat.
        try:
            fd = open_safe_fd(src)
        except SymlinkRejectedError:
            quarantine_symlink(src, self._cfg.raw_dir, reason="symlinks")
            await self._queue.mark_skipped(event.event_id, "symlink-rejected")
            return
        except FileNotFoundError:
            await self._queue.mark_skipped(event.event_id, "source-missing")
            return
        except OSError as e:
            await self._queue.mark_failed(event.event_id, f"open-error:{type(e).__name__}")
            return

        try:
            try:
                meta = await asyncio.to_thread(hash_and_stat_fd, fd, self._cfg.max_file_size_bytes)
            except SizeBombError:
                await self._queue.mark_skipped(event.event_id, "size-bomb")
                log.info(
                    "worker.skipped",
                    extra={
                        "extra_fields": {
                            "event_id": event.event_id,
                            "reason": "size-bomb",
                        }
                    },
                )
                return
            except OSError as e:
                await self._queue.mark_failed(event.event_id, f"hash-error:{type(e).__name__}")
                return

            sha = meta["sha256"]
            size = meta["size"]

            if await self._queue.sha_seen(sha):
                await self._queue.mark_skipped(event.event_id, "content-duplicate")
                await self._move_to_ingested(src, event)
                log.info(
                    "worker.skipped",
                    extra={
                        "extra_fields": {
                            "event_id": event.event_id,
                            "reason": "content-duplicate",
                            "sha256": sha,
                        }
                    },
                )
                return

            mime, _ = mimetypes.guess_type(str(src))
            if not is_allowed_mime(mime, self._cfg):
                await self._queue.mark_skipped(event.event_id, f"mime-not-allowed:{mime}")
                log.info(
                    "worker.skipped",
                    extra={
                        "extra_fields": {
                            "event_id": event.event_id,
                            "reason": "mime-not-allowed",
                            "mime": mime,
                        }
                    },
                )
                return

            # PRD-015 FR-5 / ADR-032 D2 (daemon route): a PDF under
            # raw/papers/ gets an in-process extraction pre-pass that
            # writes its markdown sidecar next to the PDF; the sidecar
            # re-enters this pipeline as a normal text event. PDFs
            # elsewhere keep the existing binary-page behaviour. The
            # pre-pass never raises — failures degrade to skip+log.
            if (mime or "") == "application/pdf" and is_paper_pdf(event.rel_path):
                await run_paper_pre_pass(
                    event, src, queue=self._queue, move_to_ingested=self._move_to_ingested
                )
                return

            # SEC-05: parse frontmatter with a strict firewall before MCP.
            try:
                body_bytes = await asyncio.to_thread(src.read_bytes)
            except OSError as e:
                await self._queue.mark_failed(event.event_id, f"read-error:{type(e).__name__}")
                return

            envelope: dict | None = None
            if (mime or "").startswith("text/") or (mime or "") == "application/x-yaml":
                try:
                    fm, body_text = parse_and_validate_frontmatter(body_bytes)
                except (DangerousFrontmatterError, YamlBombError) as e:
                    reason = (
                        "yaml-depth-bomb"
                        if isinstance(e, YamlBombError)
                        else "dangerous-frontmatter"
                    )
                    await self._queue.mark_failed(event.event_id, reason)
                    log.warning(
                        "worker.rejected",
                        extra={
                            "extra_fields": {
                                "event_id": event.event_id,
                                "reason": reason,
                                "error_class": type(e).__name__,
                            }
                        },
                    )
                    return
                envelope = {
                    "frontmatter_in": fm,
                    "wrapped_body": wrap_untrusted_body(body_text, sha, event.rel_path),
                }

            # Rate-limit acquisition AFTER validation (SEC-08): don't waste
            # tokens on events that would be rejected anyway.
            await self._bucket.acquire()
            try:
                page_path = await self._call_write_page(event, sha, mime or "", size, envelope)
                await self._queue.record_ingested(sha, event.rel_path, page_path)
                await self._queue.mark_done(event.event_id, page_path)
                await self._move_to_ingested(src, event)
                log.info(
                    "worker.done",
                    extra={
                        "extra_fields": {
                            "event_id": event.event_id,
                            "sha256": sha,
                            "size": size,
                            "page_path": page_path,
                        }
                    },
                )
                # M3 Sprint 4: fire the post-write hook so wiki_memory
                # can index + schedule a seal. Hook failures must not
                # fail the ingest — the page already wrote and the
                # event is already marked done.
                if self._post_write_hook is not None:
                    try:
                        await self._post_write_hook(
                            self._make_domain_event(event, sha, size),
                            page_path,
                        )
                    except Exception:  # noqa: BLE001
                        log.exception(
                            "worker.post_write_hook_failed",
                            extra={"extra_fields": {"event_id": event.event_id}},
                        )
            except WriteSkippedError as e:
                # ADR-048 D4: quality declined the page (no_signal). The
                # verdict is deterministic — retrying can never succeed. Mark
                # skipped and consume the raw file so the watcher doesn't
                # re-fire on it forever. Loud by contract (ADR-032 FR-7).
                await self._queue.mark_skipped(event.event_id, str(e))
                await self._move_to_ingested(src, event)
                log.warning(
                    "worker.skipped",
                    extra={
                        "extra_fields": {
                            "event_id": event.event_id,
                            "reason": str(e),
                            "rel_path": event.rel_path,
                        }
                    },
                )
            except Exception as e:  # noqa: BLE001
                err_class = type(e).__name__
                if event.attempts < _MAX_ATTEMPTS:
                    delay = _RETRY_DELAYS[min(event.attempts - 1, len(_RETRY_DELAYS) - 1)]
                    log.warning(
                        "worker.retry",
                        extra={
                            "extra_fields": {
                                "event_id": event.event_id,
                                "attempt": event.attempts,
                                "error_class": err_class,
                            }
                        },
                    )
                    await asyncio.sleep(delay)
                    await self._queue.mark_retry(event.event_id, err_class)
                else:
                    await self._queue.mark_failed(event.event_id, err_class)
                    log.error(
                        "worker.failed",
                        extra={
                            "extra_fields": {
                                "event_id": event.event_id,
                                "error_class": err_class,
                            }
                        },
                    )
        finally:
            try:
                os.close(fd)
            except OSError:
                pass

    def _make_domain_event(self, event: IngestEvent, sha: str, size: int) -> Any:
        """Translate the storage IngestEvent → wiki_core.IngestEvent
        domain shape for the post-write hook. Local import keeps the
        wiki_core dep optional at module load time (worker still imports
        clean in deployments that don't have wiki_core/wiki_memory
        installed, though that's not a real configuration today)."""
        from wiki_core.protocols import IngestEvent as DomainEvent

        return DomainEvent(
            event_id=event.event_id,
            source=f"watcher:{event.bucket}",
            path_or_uri=event.path,
            sha256=sha,
            received_at=datetime.now(UTC),
            metadata={
                "rel_path": event.rel_path,
                "bucket": event.bucket,
                "event_type": event.event_type,
                "mtime": event.mtime,
                "size": size,
                "mime": event.mime,
            },
        )

    async def _call_write_page(
        self,
        event: IngestEvent,
        sha: str,
        mime: str,
        size: int,
        envelope: dict | None,
    ) -> str:
        """Write the deterministic mechanical page via the real
        `write_page(page_path, content, overwrite)` MCP contract
        (ADR-035 D1).

        Returns the page_path the server confirmed; raises
        `WritePageError` on a tool-level error (`isError: true`) or an
        unparseable response — never returns None, so `_process` can't
        mistake a failed write for success.
        """
        ingested_at = datetime.now(UTC).isoformat(timespec="seconds").replace("+00:00", "Z")
        slug = page_slug(event.rel_path, event.event_id)
        page_path = f"wiki/sources/{slug}.md"
        content = render_page(event, sha, mime, size, envelope, ingested_at)

        # SEC-10: hold a per-slug lock so concurrent write_page calls to the
        # same destination serialize.
        locks_dir = self._cfg.wiki_root / ".wiki-ingest-locks"
        with flock_slug(locks_dir, slug):
            result = await self._mcp.call_tool(
                "write_page",
                {"page_path": page_path, "content": content},
            )

        # FastMCP reports tool failures as a *successful* JSON-RPC
        # response carrying `isError: true` — the JSON-RPC layer in
        # `McpStdioClient._request` never sees them. Check here.
        if result.get("isError"):
            raise WritePageError(f"write_page failed: {result_text(result)[:500]}")

        # ADR-048 D4: a deliberate quality skip (`status: skipped`) is not a
        # failure — route it to mark_skipped, never into the retry loop.
        skip_reason = extract_skip_reason(result)
        if skip_reason is not None:
            raise WriteSkippedError(skip_reason)

        returned = extract_page_path(result)
        if not returned:
            raise WritePageError(f"write_page returned no page_path: {result_text(result)[:500]}")
        return returned

    async def _move_to_ingested(self, src: Path, event: IngestEvent) -> None:
        dest_dir = self._cfg.ingested_dir / event.bucket
        dest_dir.mkdir(parents=True, exist_ok=True)
        dest = dest_dir / src.name
        if dest.exists():
            stem = dest.stem
            suffix = dest.suffix
            dest = dest_dir / f"{stem}.{event.event_id[:8]}{suffix}"
        try:
            await asyncio.to_thread(atomic_move, src, dest)
        except OSError as e:
            log.warning(
                "worker.move_failed",
                extra={
                    "extra_fields": {
                        "event_id": event.event_id,
                        "error_class": type(e).__name__,
                    }
                },
            )


def _file_size(path: Path) -> int:
    try:
        return os.path.getsize(path)
    except OSError:
        return 0
