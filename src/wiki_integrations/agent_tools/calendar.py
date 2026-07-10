"""
Google Calendar `IIntegration` impl per issue #32 / PRD-006.

Read-only — locked scope is ``calendar.readonly`` per
`wiki_core.secrets.policy_for("calendar")`. Returns a bounded **[-7d, +14d]
window** of events relative to ``datetime.now(UTC)`` by default; callers
may tighten the lower bound by passing ``since=`` to ``list``.

Recurring events are expected to arrive **already expanded** into per-instance
rows (Google Calendar's ``singleEvents=true`` flag, which Composio is
documented as setting by default). Each instance carries its own ``id`` and
``start`` so dedup keys are per-instance and the window filter works on the
instance start time, not the master recurrence rule.

Per-event idempotency is sha256 of ``f"{event_id}|{updated_ts}"`` carried in
``metadata.sha256``. The MemoryStore can reject duplicates downstream without
re-parsing.
"""

from __future__ import annotations

import hashlib
from datetime import UTC, datetime, timedelta
from typing import Any

from wiki_core.integrations.protocol import IntegrationItem, SearchResult
from wiki_integrations.agent_tools.base import ComposioBackedIntegration

# Window per issue #32: 7 days back, 14 days forward of "now".
WINDOW_BACK = timedelta(days=7)
WINDOW_FORWARD = timedelta(days=14)


class CalendarIntegration(ComposioBackedIntegration):
    """Google Calendar agent-tool surface backed by ComposioBridge."""

    PROVIDER = "calendar"

    def _to_item(self, raw: dict[str, Any]) -> IntegrationItem:
        event_id = str(raw.get("id") or raw["event_id"])
        summary = str(raw.get("summary") or raw.get("title") or "(no title)")
        description = str(raw.get("description", ""))
        location = str(raw.get("location", ""))
        # snippet stitches description + location so a search over snippet
        # picks up "Zoom" / "Office" / room names alongside body text.
        snippet_parts = [p for p in (description, location) if p]
        snippet = " — ".join(snippet_parts)[:500]
        html_link = str(raw.get("htmlLink") or raw.get("html_link") or "")
        start_ts = _parse_event_ts(raw.get("start"))
        end_ts = _parse_event_ts(raw.get("end"))
        updated_raw = raw.get("updated") or raw.get("updated_at") or raw.get("etag", "")
        # Idempotency key per #32 AC: sha256(event_id, updated).
        idem = hashlib.sha256(f"{event_id}|{updated_raw}".encode()).hexdigest()
        return IntegrationItem(
            id=event_id,
            title=summary,
            snippet=snippet,
            uri=html_link or f"calendar:{event_id}",
            updated_at=start_ts,
            metadata={
                "start": start_ts.isoformat(),
                "end": end_ts.isoformat() if end_ts is not None else None,
                "location": location,
                "organizer": _organizer_email(raw.get("organizer")),
                "recurring_event_id": str(raw.get("recurringEventId", "")) or None,
                "calendar_id": str(raw.get("calendarId", "primary")),
                "sha256": idem,
            },
        )

    async def list(
        self,
        *,
        since: datetime | None = None,
        limit: int = 50,
    ) -> tuple[IntegrationItem, ...]:
        """Override the base ``list`` to enforce the **[-7d, +14d]** window
        on ``event.start`` (not ``updated_at``).

        ``since`` tightens the lower bound when supplied — useful for the
        auto-fetch scheduler's incremental tick which only wants events
        added since the last sync. ``since`` is silently floored to
        ``now - 7d`` if the caller asks for further back, because anything
        older than the window is out of policy.
        """
        self._require_connected()
        now = datetime.now(UTC)
        lower = now - WINDOW_BACK
        upper = now + WINDOW_FORWARD
        if since is not None:
            if since.tzinfo is None:
                since = since.replace(tzinfo=UTC)
            lower = max(lower, since)
        items: list[IntegrationItem] = []
        async for raw in self._bridge.walk(self.PROVIDER):
            try:
                item = self._to_item(raw)
            except (KeyError, TypeError, ValueError):
                continue
            if not (lower <= item.updated_at <= upper):
                continue
            items.append(item)
            if len(items) >= limit:
                break
        self._mark("list", "ok", items=len(items))
        return tuple(items)

    async def get(self, item_id: str) -> IntegrationItem:
        """Linear scan of ``walk()`` until we hit ``item_id``.

        Same caveat as ``GmailIntegration.get`` — Composio has no uniform
        get-by-id surface yet. The auto-fetch scheduler (#38) caches walk
        results and amortises this.
        """
        self._require_connected()
        async for raw in self._bridge.walk(self.PROVIDER):
            try:
                item = self._to_item(raw)
            except (KeyError, TypeError, ValueError):
                continue
            if item.id == item_id:
                self._mark("get", "ok", note=f"id={item_id}")
                return item
        self._mark("get", "not_found", note=f"id={item_id}")
        raise KeyError(f"calendar event {item_id!r} not found")

    async def search(
        self,
        query: str,
        *,
        limit: int = 50,
        cursor: str | None = None,
    ) -> SearchResult:
        """Substring match on ``summary + description + location`` over the
        same [-7d, +14d] window the agent expects.

        Calendar's native ``q`` parameter is left as a future enhancement
        when Composio surfaces it; substring is the responsible MVP and
        matches the search behaviour of the Gmail / GitHub integrations.
        """
        self._require_connected()
        if not query:
            raise ValueError("search query must be non-empty")
        needle = query.lower()
        now = datetime.now(UTC)
        lower = now - WINDOW_BACK
        upper = now + WINDOW_FORWARD
        matched: list[IntegrationItem] = []
        seen_shas: set[str] = set()
        async for raw in self._bridge.walk(self.PROVIDER):
            try:
                item = self._to_item(raw)
            except (KeyError, TypeError, ValueError):
                continue
            if not (lower <= item.updated_at <= upper):
                continue
            sha = item.metadata.get("sha256")
            if sha and sha in seen_shas:
                continue
            if sha:
                seen_shas.add(sha)
            haystack = f"{item.title} {item.snippet} {item.metadata.get('location', '')}".lower()
            if needle in haystack:
                matched.append(item)
                if len(matched) >= limit:
                    break
        self._mark("search", "ok", items=len(matched), note=f"q={needle[:32]}")
        return SearchResult(items=tuple(matched), total_estimated=len(matched))


__all__ = ["CalendarIntegration", "WINDOW_BACK", "WINDOW_FORWARD"]


def _parse_event_ts(value: Any) -> datetime:
    """Google Calendar event ``start`` / ``end`` is either ``{dateTime: …, timeZone: …}``
    (timed events) or ``{date: "YYYY-MM-DD"}`` (all-day events). Composio
    may surface a plain ISO string instead. Accept all three shapes; fall
    back to epoch on parse failure so a row with a weird timestamp doesn't
    poison the whole list.
    """
    if value is None:
        return datetime.fromtimestamp(0, tz=UTC)
    if isinstance(value, datetime):
        return value if value.tzinfo else value.replace(tzinfo=UTC)
    if isinstance(value, dict):
        for key in ("dateTime", "date_time", "date"):
            inner = value.get(key)
            if inner is not None:
                return _parse_event_ts(inner)
        return datetime.fromtimestamp(0, tz=UTC)
    s = str(value)
    # All-day events use a bare ``YYYY-MM-DD`` with no time component.
    if len(s) == 10 and s.count("-") == 2:
        try:
            return datetime.fromisoformat(f"{s}T00:00:00+00:00")
        except ValueError:
            return datetime.fromtimestamp(0, tz=UTC)
    try:
        return datetime.fromisoformat(s.replace("Z", "+00:00"))
    except ValueError:
        return datetime.fromtimestamp(0, tz=UTC)


def _organizer_email(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    if isinstance(value, dict):
        return str(value.get("email", ""))
    return ""
