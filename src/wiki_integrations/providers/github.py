"""
GitHub `OAuthIntegrationSource` — per [PRD-005](../../../docs/PRD-005-oauth-integrations-layer.md) MVP scope.

Pulls issues + PRs via a `ComposioBridge.walk("github")` iterator and
translates each into a `wiki_core.IngestEvent`. Translation rules:

- ``source`` = ``"github"``.
- ``path_or_uri`` = ``html_url`` (the canonical issue/PR URL).
- ``sha256`` = ``sha256(title + "\\n" + body)`` so dedup at the MemoryStore
  level catches the common "PR description re-saved twice" path. Title +
  body is the minimum that uniquely identifies content; ids alone collide
  across repos in some Composio responses.
- ``metadata`` keys required by `wiki_ingest.adapter._to_storage`:
  ``bucket`` (= ``"github-issues"``), ``rel_path`` (``"<kind>/<number>"``),
  ``event_type`` (= ``"created"``), ``mtime`` (the upstream ``updated_at``),
  ``size`` (``len(payload.encode())``), optional ``mime`` (= ``"text/markdown"``).
"""

from __future__ import annotations

import hashlib
import logging
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta
from typing import TYPE_CHECKING, Any, Protocol
from uuid import uuid4

from wiki_integrations.base import OAuthIntegrationSource

if TYPE_CHECKING:
    from collections.abc import AsyncIterator

    from wiki_core.protocols import IngestEvent
    from wiki_integrations.cursor_store import CursorStore

log = logging.getLogger("wiki_integrations.providers.github")


class _Walker(Protocol):
    """Structural type for the `walk` surface we depend on."""

    def walk(self, provider: str) -> AsyncIterator[dict[str, Any]]: ...


@dataclass(frozen=True, slots=True)
class GitHubItem:
    """Normalised GitHub issue or PR used inside the provider."""

    item_id: str
    kind: str  # "issue" | "pull_request"
    number: int
    title: str
    body: str
    html_url: str
    updated_at: str
    state: str
    metadata: dict[str, Any] = field(default_factory=dict)


class GitHubIntegrationSource(OAuthIntegrationSource):
    """GitHub provider implementation."""

    PROVIDER = "github"
    BUCKET = "github-issues"
    DEFAULT_FETCH_WINDOW = timedelta(hours=24)

    def __init__(
        self,
        *,
        on_event: Callable[[IngestEvent], Awaitable[None]],
        walker: _Walker,
        fetch_window: timedelta | None = None,
        cursor_store: CursorStore | None = None,
    ) -> None:
        super().__init__(
            self.PROVIDER,
            fetch_window=fetch_window or self.DEFAULT_FETCH_WINDOW,
            on_event=on_event,
        )
        self._walker = walker
        self._cursor_store = cursor_store
        self._cursor: str | None = None

    async def fetch_batch(self) -> list[IngestEvent]:
        """Pull a window's worth of GitHub issues/PRs and emit one event each.

        Cursor handling (OQ-2, PRD-006 FR-3): if a `CursorStore` was
        passed at construction time, the provider reads its persisted
        cursor before walking and writes the new cursor after a
        successful walk. The cursor for GitHub is the highest
        ``updated_at`` seen in the batch — GitHub's search API accepts
        that as a ``since=`` boundary. When no `CursorStore` is wired,
        every tick pulls the configured window with no persistence.
        """
        if self._cursor_store is not None and self._cursor is None:
            self._cursor = await self._cursor_store.get(self.PROVIDER)

        events: list[IngestEvent] = []
        latest_updated_at: str | None = None
        async for raw in self._walker.walk(self.PROVIDER):
            try:
                item = _normalise(raw)
            except (KeyError, TypeError, ValueError) as exc:
                log.warning(
                    "github.malformed_payload",
                    extra={"extra_fields": {"error_class": type(exc).__name__}},
                )
                continue
            event = _to_event(item)
            events.append(event)
            await self.emit(event)
            if latest_updated_at is None or item.updated_at > latest_updated_at:
                latest_updated_at = item.updated_at

        if self._cursor_store is not None and latest_updated_at is not None:
            self._cursor = latest_updated_at
            await self._cursor_store.set(self.PROVIDER, latest_updated_at)
        return events

    @property
    def cursor(self) -> str | None:
        """Last persisted cursor value, or None if nothing fetched yet."""
        return self._cursor


# --------------------------------------------------------------------------- #
# Translation helpers                                                         #
# --------------------------------------------------------------------------- #


def _normalise(raw: dict[str, Any]) -> GitHubItem:
    """Translate Composio's GitHub payload to the internal `GitHubItem`.

    Handles three shapes (#151): ``issue`` / ``pull_request`` (have a
    ``number``, ``title``, ``body``) and ``repository`` — the shape the wired
    ``GITHUB_LIST_REPOSITORIES_FOR_AUTHENTICATED_USER`` action returns
    (``full_name``/``name``, ``description``, no ``number``). When ``kind`` is
    absent it is inferred: a payload with a name but no number is a repository.
    Tests may pass ``item_id``/``url`` variants; we accept both. An explicit
    unknown ``kind`` still raises (caller skips it).
    """
    item_id = str(raw.get("item_id") or raw["id"])
    kind = raw.get("kind") or raw.get("type")
    if not kind:
        kind = (
            "repository"
            if (raw.get("full_name") or raw.get("name")) and "number" not in raw
            else "issue"
        )
    kind = str(kind)
    if kind not in {"issue", "pull_request", "repository"}:
        raise ValueError(f"unsupported github item kind: {kind!r}")

    if kind == "repository":
        number = int(raw.get("number") or 0)
        title = str(raw.get("full_name") or raw.get("name", ""))
        body = str(raw.get("description") or "")
        state = str(raw.get("state") or ("archived" if raw.get("archived") else "active"))
        updated_at = str(raw.get("updated_at") or raw.get("pushed_at") or _utcnow_iso())
    else:
        number = int(raw["number"])
        title = str(raw.get("title", ""))
        body = str(raw.get("body", ""))
        state = str(raw.get("state", "unknown"))
        updated_at = str(raw.get("updated_at") or _utcnow_iso())

    return GitHubItem(
        item_id=item_id,
        kind=kind,
        number=number,
        title=title,
        body=body,
        html_url=str(raw.get("html_url") or raw.get("url", "")),
        updated_at=updated_at,
        state=state,
        metadata={k: v for k, v in raw.items() if k not in {"body", "title", "description"}},
    )


def _to_event(item: GitHubItem) -> IngestEvent:
    """Translate a normalised `GitHubItem` into a `wiki_core.IngestEvent`."""
    from wiki_core.protocols import IngestEvent

    payload = f"{item.title}\n{item.body}"
    payload_bytes = payload.encode("utf-8")
    sha = hashlib.sha256(payload_bytes).hexdigest()
    if item.kind == "repository":
        bucket = "github-repos"
        rel_path = f"repo/{item.title.replace('/', '--')}"
    else:
        bucket = GitHubIntegrationSource.BUCKET
        rel_path = f"{item.kind}/{item.number}"
    return IngestEvent(
        event_id=str(uuid4()),
        source=GitHubIntegrationSource.PROVIDER,
        path_or_uri=item.html_url,
        sha256=sha,
        received_at=datetime.now(UTC),
        metadata={
            "bucket": bucket,
            "rel_path": rel_path,
            "event_type": "created",
            "mtime": item.updated_at,
            "size": len(payload_bytes),
            "mime": "text/markdown",
            "kind": item.kind,
            "state": item.state,
            "number": item.number,
        },
    )


def _utcnow_iso() -> str:
    return datetime.now(UTC).isoformat(timespec="seconds").replace("+00:00", "Z")
