"""
Gmail `IIntegration` impl per issue #31 / PRD-006.

Read-only — locked scope is ``gmail.readonly`` per
`wiki_core.secrets.policy_for("gmail")`. Body is fetched lazily on `get`;
`list`/`search` return headers + snippets only.

Per-thread dedup is via sha256(normalised_body) carried in `metadata.sha256`.
The MemoryStore can reject duplicates downstream without re-parsing.
"""

from __future__ import annotations

import hashlib
from datetime import UTC, datetime
from typing import Any

from wiki_core.integrations.protocol import IntegrationItem, SearchResult
from wiki_integrations.agent_tools.base import ComposioBackedIntegration


class GmailIntegration(ComposioBackedIntegration):
    """Gmail agent-tool surface backed by ComposioBridge."""

    PROVIDER = "gmail"

    def _to_item(self, raw: dict[str, Any]) -> IntegrationItem:
        message_id = str(raw.get("id") or raw["message_id"])
        sender = str(raw.get("sender") or raw.get("from", "unknown@unknown"))
        subject = str(raw.get("subject", ""))
        body = str(raw.get("body") or raw.get("snippet", ""))
        internal_date = raw.get("internal_date")
        updated_at = _parse_gmail_ts(internal_date)
        return IntegrationItem(
            id=message_id,
            title=subject or "(no subject)",
            snippet=body[:500],
            uri=f"mailto:{sender}",
            updated_at=updated_at,
            metadata={
                "thread_id": str(raw.get("thread_id", message_id)),
                "sender": sender,
                "labels": raw.get("labels") or raw.get("label_ids") or [],
                "sha256": hashlib.sha256(body.encode("utf-8")).hexdigest(),
            },
        )

    async def get(self, item_id: str) -> IntegrationItem:
        """Linear scan of `walk()` until we hit `item_id`.

        Same caveat as `GitHubIntegration.get` — Composio's bridge has no
        uniform get-by-id surface yet. The auto-fetch scheduler (#38) will
        cache results and amortise this.
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
        raise KeyError(f"gmail message {item_id!r} not found")

    async def search(
        self,
        query: str,
        *,
        limit: int = 50,
        cursor: str | None = None,
    ) -> SearchResult:
        """Match `query` substring against subject+sender+snippet.

        Gmail's native ``q`` parameter (``from:`` / ``subject:`` operators)
        is left as a future enhancement once the Composio action surfaces
        it directly. For #31 the AC says "search" without specifying Gmail
        operator syntax; substring is the responsible MVP.
        """
        self._require_connected()
        if not query:
            raise ValueError("search query must be non-empty")
        needle = query.lower()
        matched: list[IntegrationItem] = []
        seen_shas: set[str] = set()
        async for raw in self._bridge.walk(self.PROVIDER):
            try:
                item = self._to_item(raw)
            except (KeyError, TypeError, ValueError):
                continue
            sha = item.metadata.get("sha256")
            if sha and sha in seen_shas:
                continue  # dedup forwarded copies
            if sha:
                seen_shas.add(sha)
            haystack = f"{item.title} {item.uri} {item.snippet}".lower()
            if needle in haystack:
                matched.append(item)
                if len(matched) >= limit:
                    break
        self._mark("search", "ok", items=len(matched), note=f"q={needle[:32]}")
        return SearchResult(items=tuple(matched), total_estimated=len(matched))


__all__ = ["GmailIntegration"]


def _parse_gmail_ts(value: Any) -> datetime:
    """Gmail's ``internal_date`` is a millisecond-precision Unix timestamp
    string in their REST API. Composio may surface ISO-8601 instead;
    accept both.
    """
    if value is None:
        return datetime.fromtimestamp(0, tz=UTC)
    if isinstance(value, datetime):
        return value
    s = str(value)
    if s.isdigit():
        return datetime.fromtimestamp(int(s) / 1000, tz=UTC)
    try:
        return datetime.fromisoformat(s.replace("Z", "+00:00"))
    except ValueError:
        return datetime.fromtimestamp(0, tz=UTC)
