"""
Slack `IIntegration` impl per issue #33 / PRD-006.

Read-only — locked scopes are ``channels:history``, ``channels:read``,
``users:read`` per `wiki_core.secrets.policy_for("slack")`. DM scopes
(``im:history``, ``im:read``) are listed as ``opt_in_extra`` in the
locked scope policy but **never enabled** by this integration; a future
PR with an explicit consent UI will gate DM ingestion per ADR-017
§Per-provider OAuth scope table and the AC of issue #33.

### Channel allowlist (#33 AC)

Slack workspaces routinely have hundreds of channels; fetching every
channel's history would burn rate-limit quota and ingest noise nobody
asked for. SBW requires the user to declare an **allowlist** of channel
IDs in ``~/.sbw/config.toml``::

    [integrations.slack]
    allowed_channels = ["C01ABCDEF", "C02GHIJKL"]  # public channel IDs

If the allowlist is empty or absent, ``list``/``search`` return an empty
result and emit a structured log warning telling the user to configure
``allowed_channels``. This is the "opt-in channels" gate from the AC.

### Per-call iteration

Composio's ``SLACK_FETCH_CONVERSATION_HISTORY`` requires a ``channel``
argument, so unlike Calendar/Gmail/GitHub this integration cannot use
the bridge's default ``walk()``. It calls ``bridge.execute()`` once
per allowed channel and merges the results.

### Per-message idempotency

``metadata.sha256`` is ``sha256(channel + ts)`` so the same message
in two channels (cross-posts) gets two distinct ids. The Slack message
``ts`` is already a monotonic decimal-string timestamp that fully
identifies a message within a channel.
"""

from __future__ import annotations

import hashlib
import logging
import tomllib
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from wiki_core.integrations.protocol import IntegrationItem, SearchResult
from wiki_integrations.agent_tools.base import ComposioBackedIntegration

_log = logging.getLogger(__name__)

DEFAULT_CONFIG_PATH = Path.home() / ".sbw" / "config.toml"

_LIST_TOOL = "SLACK_FETCH_CONVERSATION_HISTORY"


class SlackIntegration(ComposioBackedIntegration):
    """Slack agent-tool surface backed by ComposioBridge.

    Override list/search to iterate over the user-configured channel
    allowlist; the base class's ``list`` / ``search`` via ``walk()`` is
    not suitable because Slack's history endpoint is per-channel.
    """

    PROVIDER = "slack"

    def __init__(
        self,
        *,
        bridge: Any,
        store: Any,
        audit_jsonl: Any,
        audit_md: Any = None,
        config_path: Path | None = None,
    ) -> None:
        super().__init__(bridge=bridge, store=store, audit_jsonl=audit_jsonl, audit_md=audit_md)
        # ``config_path`` is injectable so tests can point at a tmp file.
        self._config_path = config_path if config_path is not None else DEFAULT_CONFIG_PATH

    # ------------------------------------------------------------------ #
    # Public surface                                                     #
    # ------------------------------------------------------------------ #

    async def list(
        self,
        *,
        since: datetime | None = None,
        limit: int = 50,
    ) -> tuple[IntegrationItem, ...]:
        """Fetch recent messages from every channel on the allowlist.

        ``since`` filters by the message's ``ts``; messages older than
        ``since`` are skipped. ``limit`` is the total cap across all
        channels (not per-channel).
        """
        self._require_connected()
        channels = self._read_allowed_channels()
        if not channels:
            self._mark("list", "no_channels", note="no allowed_channels configured")
            return ()
        items = await self._collect_messages(channels, limit=limit, since=since)
        self._mark("list", "ok", items=len(items))
        return tuple(items)

    async def get(self, item_id: str) -> IntegrationItem:
        """Linear scan over the allowed channels until we hit ``item_id``.

        ``item_id`` shape is ``"<channel_id>:<ts>"``. We pull that channel's
        recent history and look for the matching ``ts``. Slack has no
        direct "get message by id" surface in the read-only scope set.
        """
        self._require_connected()
        if ":" not in item_id:
            raise KeyError(f"slack item id must be '<channel>:<ts>', got {item_id!r}")
        channel, ts = item_id.split(":", 1)
        # Constrain to a single-channel fetch.
        items = await self._collect_messages([channel], limit=100, since=None)
        for item in items:
            if item.id == item_id:
                self._mark("get", "ok", note=f"id={item_id}")
                return item
        self._mark("get", "not_found", note=f"id={item_id}")
        raise KeyError(f"slack message {item_id!r} not found")

    async def search(
        self,
        query: str,
        *,
        limit: int = 50,
        cursor: str | None = None,
    ) -> SearchResult:
        """Substring match over message text + user_id + channel across
        the allowed channels. MVP — Slack's native ``search.messages``
        endpoint requires ``search:read`` scope which is **not** in the
        locked policy."""
        self._require_connected()
        if not query:
            raise ValueError("search query must be non-empty")
        channels = self._read_allowed_channels()
        if not channels:
            self._mark("search", "no_channels", note="no allowed_channels configured")
            return SearchResult(items=(), total_estimated=0)
        needle = query.lower()
        # Pull a generous window then filter — Slack history isn't
        # text-indexed at the API level.
        all_items = await self._collect_messages(channels, limit=limit * 4, since=None)
        matched: list[IntegrationItem] = []
        seen_shas: set[str] = set()
        for item in all_items:
            sha = item.metadata.get("sha256")
            if sha and sha in seen_shas:
                continue
            if sha:
                seen_shas.add(sha)
            haystack = f"{item.title} {item.snippet} {item.metadata.get('user', '')}".lower()
            if needle in haystack:
                matched.append(item)
                if len(matched) >= limit:
                    break
        self._mark("search", "ok", items=len(matched), note=f"q={needle[:32]}")
        return SearchResult(items=tuple(matched), total_estimated=len(matched))

    def _to_item(self, raw: dict[str, Any]) -> IntegrationItem:
        """Required by the abstract base. Used by ``_messages_from_response``."""
        return _slack_message_to_item(raw)

    # ------------------------------------------------------------------ #
    # Helpers                                                            #
    # ------------------------------------------------------------------ #

    def _read_allowed_channels(self) -> list[str]:
        """Parse ``[integrations.slack] allowed_channels`` from config.toml.

        Returns ``[]`` if the file or section is missing — the integration
        treats that as "user hasn't opted into any channels yet" and
        surfaces an empty result with a warning, never a crash."""
        if not self._config_path.exists():
            return []
        try:
            raw = tomllib.loads(self._config_path.read_text(encoding="utf-8"))
        except (tomllib.TOMLDecodeError, OSError) as exc:
            _log.warning("slack: failed to read %s: %s", self._config_path, exc)
            return []
        section = (raw.get("integrations") or {}).get("slack") or {}
        allowed = section.get("allowed_channels") or []
        if not isinstance(allowed, list):
            _log.warning("slack: allowed_channels must be a list, got %s", type(allowed).__name__)
            return []
        return [str(c) for c in allowed if isinstance(c, str) and c.strip()]

    async def _collect_messages(
        self,
        channels: list[str],
        *,
        limit: int,
        since: datetime | None,
    ) -> list[IntegrationItem]:
        """Pull at most ``limit`` items across the given channels, oldest
        first per channel. Stops once the global cap is hit."""
        out: list[IntegrationItem] = []
        # Slack's ``oldest`` is a string-formatted unix timestamp. Convert.
        oldest_ts: str | None = None
        if since is not None:
            if since.tzinfo is None:
                since = since.replace(tzinfo=UTC)
            oldest_ts = f"{since.timestamp():.6f}"
        for channel in channels:
            if len(out) >= limit:
                break
            args: dict[str, Any] = {
                "channel": channel,
                "limit": min(limit - len(out), 100),
            }
            if oldest_ts is not None:
                args["oldest"] = oldest_ts
            try:
                response = await self._bridge.execute(self.PROVIDER, _LIST_TOOL, args)
            except Exception as exc:  # noqa: BLE001 -- broad to log + continue per channel
                _log.warning("slack: fetch failed for channel %s: %s", channel, exc)
                self._mark("list", "channel_error", note=f"channel={channel}")
                continue
            messages = _extract_messages(response)
            for raw in messages:
                if len(out) >= limit:
                    break
                try:
                    raw_with_channel = {**raw, "channel": channel}
                    out.append(self._to_item(raw_with_channel))
                except (KeyError, TypeError, ValueError):
                    continue
        return out


def _extract_messages(response: dict[str, Any]) -> list[dict[str, Any]]:
    """Pull the ``messages`` list out of Slack's response, accepting the
    handful of shapes Composio may surface (top-level vs nested under
    ``data`` vs flat list)."""
    if not isinstance(response, dict):
        return []
    if isinstance(response.get("messages"), list):
        return response["messages"]
    nested = response.get("data")
    if isinstance(nested, dict) and isinstance(nested.get("messages"), list):
        return nested["messages"]
    items = response.get("items")
    if isinstance(items, list):
        return items
    return []


def _slack_message_to_item(raw: dict[str, Any]) -> IntegrationItem:
    """Map one Slack message into an ``IntegrationItem``."""
    channel = str(raw.get("channel") or raw.get("channel_id", "unknown"))
    ts = str(raw.get("ts") or raw.get("timestamp", ""))
    if not ts:
        raise KeyError("slack message missing 'ts'")
    text = str(raw.get("text", ""))
    user = str(raw.get("user") or raw.get("user_id", "unknown"))
    item_id = f"{channel}:{ts}"
    updated_at = _slack_ts_to_dt(ts)
    return IntegrationItem(
        id=item_id,
        title=(text[:80] or "(no text)"),
        snippet=text[:500],
        uri=f"slack://channel/{channel}/p{ts.replace('.', '')}",
        updated_at=updated_at,
        metadata={
            "channel": channel,
            "user": user,
            "ts": ts,
            "thread_ts": raw.get("thread_ts"),
            "sha256": hashlib.sha256(f"{channel}|{ts}".encode()).hexdigest(),
        },
    )


def _slack_ts_to_dt(ts: str) -> datetime:
    """Slack's ``ts`` is a decimal string like ``1716345600.001234`` —
    seconds since epoch with microsecond precision."""
    try:
        return datetime.fromtimestamp(float(ts), tz=UTC)
    except (ValueError, TypeError):
        return datetime.fromtimestamp(0, tz=UTC)


__all__ = ["SlackIntegration"]
