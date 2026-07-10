"""
GitHub `IIntegration` impl per issue #30 / PRD-006.

Surfaces issues + pull requests via Composio's `github` provider. The
locked scope set (``repo:status``, ``read:user``, ``read:org``) is enforced
by `wiki_core.secrets.policy_for("github")` — no write access.
"""

from __future__ import annotations

import hashlib
from datetime import datetime
from typing import Any

from wiki_core.integrations.protocol import (
    IntegrationItem,
    NotConnectedError,
    SearchResult,
)
from wiki_integrations.agent_tools.base import ComposioBackedIntegration


class GitHubIntegration(ComposioBackedIntegration):
    """GitHub agent-tool surface backed by ComposioBridge."""

    PROVIDER = "github"

    def _to_item(self, raw: dict[str, Any]) -> IntegrationItem:
        item_id = str(raw.get("id") or raw.get("item_id") or raw["number"])
        kind = str(raw.get("kind") or raw.get("type") or "issue")
        title = str(raw.get("title", ""))
        body = str(raw.get("body", ""))
        url = str(raw.get("html_url") or raw.get("url", ""))
        updated_at_raw = raw.get("updated_at")
        updated_at = _parse_iso(updated_at_raw) if updated_at_raw else datetime.fromtimestamp(0)
        return IntegrationItem(
            id=item_id,
            title=title or f"{kind} #{raw.get('number', '?')}",
            snippet=body[:500],
            uri=url,
            updated_at=updated_at,
            metadata={
                "kind": kind,
                "number": raw.get("number"),
                "state": raw.get("state"),
                "repo": raw.get("repo") or raw.get("repository"),
                "sha256": hashlib.sha256(f"{title}\n{body}".encode()).hexdigest(),
            },
        )

    async def get(self, item_id: str) -> IntegrationItem:
        """Find a single item by walking the cached stream until we hit it.

        ComposioBridge doesn't expose a uniform single-id fetch — `walk()`
        is the only iterator surface. For #30's "list/get/search" AC the
        cost of a linear scan is acceptable; #38 (auto-fetch scheduler)
        will land a proper cursor + cache.
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
        raise KeyError(f"github item {item_id!r} not found")

    async def search(
        self,
        query: str,
        *,
        limit: int = 50,
        cursor: str | None = None,
    ) -> SearchResult:
        """Substring-match `query` against titles + bodies of walked items.

        Composio's GitHub action returns the user's repos' issues/PRs in
        one stream; we filter client-side. A future PR can swap in a
        Composio search action when one exists.
        """
        self._require_connected()
        if not query:
            raise ValueError("search query must be non-empty")
        needle = query.lower()
        matched: list[IntegrationItem] = []
        async for raw in self._bridge.walk(self.PROVIDER):
            try:
                item = self._to_item(raw)
            except (KeyError, TypeError, ValueError):
                continue
            haystack = f"{item.title}\n{item.snippet}".lower()
            if needle in haystack:
                matched.append(item)
                if len(matched) >= limit:
                    break
        self._mark("search", "ok", items=len(matched), note=f"q={needle[:32]}")
        return SearchResult(items=tuple(matched), total_estimated=len(matched))


__all__ = ["GitHubIntegration", "NotConnectedError"]


def _parse_iso(value: Any) -> datetime:
    """Tolerant ISO-8601 parser. Returns epoch on garbage."""
    if isinstance(value, datetime):
        return value
    try:
        return datetime.fromisoformat(str(value).replace("Z", "+00:00"))
    except (ValueError, TypeError):
        return datetime.fromtimestamp(0)
