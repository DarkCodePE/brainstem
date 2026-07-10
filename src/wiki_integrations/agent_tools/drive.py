"""
Google Drive `IIntegration` impl per issue #35 / PRD-006.

Read-only, **metadata-only** — locked default scope is
``drive.metadata.readonly`` per ``wiki_core.secrets.policy_for("drive")``.
Body-fetch (``drive.readonly``) is listed as ``opt_in_extra`` in the
locked scope policy but **never enabled** by this integration; a future
PR with an explicit consent UI (same #44 Settings UI gate Slack DMs use)
will offer the second click-through per ADR-017 §Per-provider OAuth
scope table.

### Two-tier scope model (#35 AC)

Drive defaults to **metadata-only**: file names, mime types, sizes,
parent folders, modified timestamps — but **not body content**. The
opt-in tier (``drive.readonly``) lets the agent fetch document/sheet
bodies; it is gated behind a second consent dialog deferred to #44.
A regression that surfaces body content without going through that gate
would fail ``test_drive_scope_is_metadata_only``.

### Mime filter (#35 AC)

Drive surfaces every file the user owns or has shared with them —
images, PDFs, archives, etc. SBW Wave 2 supports **Docs and Sheets
only**. Other mime types (PDFs, images, plain files) are filtered out
of ``list`` / ``search`` results. The contract is a module-level
constant ``_DOCS_AND_SHEETS_MIMES`` that ``test_list_excludes_pdfs``
asserts against — a future PR that wants to add PDFs has to widen this
set deliberately and update the test.

### Single-tool execute pattern (slack/notion analog)

Composio v3's ``GOOGLEDRIVE_LIST_FILES`` accepts a Drive ``q`` parameter
for search, so unlike the bridge's default ``walk()`` we call
``bridge.execute(provider, tool_slug, arguments)`` once with arguments
shaped per use case (no args = recent files; ``q=name contains 'X'``
for search). This matches the slack/notion pattern.

### Per-file idempotency

``metadata.sha256 = sha256(file_id + modified_time)``. A file edited
upstream gets a new sha (modified_time changes), so the MemoryStore can
distinguish "saw this file before" from "the file changed".
"""

from __future__ import annotations

import hashlib
import logging
from datetime import UTC, datetime
from typing import Any

from wiki_core.integrations.protocol import IntegrationItem, SearchResult
from wiki_integrations.agent_tools.base import ComposioBackedIntegration

_log = logging.getLogger(__name__)

_LIST_TOOL = "GOOGLEDRIVE_LIST_FILES"
_GET_TOOL = "GOOGLEDRIVE_FIND_FILE"

# Locked supported mime set for Wave 2 per #35 AC. PDFs / images / other
# are deferred — adding a mime type requires updating this constant AND
# the docs (``docs/integrations/google-drive.md`` MIME filter section).
_DOC_MIME = "application/vnd.google-apps.document"
_SHEET_MIME = "application/vnd.google-apps.spreadsheet"
_DOCS_AND_SHEETS_MIMES: frozenset[str] = frozenset({_DOC_MIME, _SHEET_MIME})

# Drive's q-param is a query language; the only special char that needs
# escaping for the ``name contains '...'`` shape is the single quote.
# Backslash-escape per https://developers.google.com/drive/api/guides/search-files.
_Q_SINGLE_QUOTE = "'"
_Q_ESCAPED_SINGLE_QUOTE = "\\'"


class GoogleDriveIntegration(ComposioBackedIntegration):
    """Google Drive agent-tool surface backed by ComposioBridge.

    Metadata-only by default; ``list`` / ``search`` filter to Docs +
    Sheets (PDFs deferred per AC). Uses ``bridge.execute`` with the
    ``GOOGLEDRIVE_LIST_FILES`` tool rather than ``walk()`` so we can pass
    a Drive ``q`` parameter for native search.
    """

    PROVIDER = "drive"

    # ------------------------------------------------------------------ #
    # Public surface                                                     #
    # ------------------------------------------------------------------ #

    async def list(
        self,
        *,
        since: datetime | None = None,
        limit: int = 50,
    ) -> tuple[IntegrationItem, ...]:
        """Fetch up to ``limit`` recent Docs + Sheets.

        ``since`` filters on ``modifiedTime`` — items older than ``since``
        are dropped. Mime filter (``_DOCS_AND_SHEETS_MIMES``) is applied
        before the limit, so a result page heavy on PDFs may yield fewer
        items than requested.
        """
        self._require_connected()
        raw_files = await self._fetch_files({"page_size": min(limit, 100)})
        items = self._materialise(raw_files, since=since, limit=limit)
        self._mark("list", "ok", items=len(items))
        return tuple(items)

    async def get(self, item_id: str) -> IntegrationItem:
        """Fetch a single file's metadata by Drive file id.

        Uses ``GOOGLEDRIVE_FIND_FILE`` (chosen over ``GOOGLEDRIVE_LIST_FILES``
        with a ``q=id`` filter because the dedicated tool has a simpler
        single-result argument shape — one tool call, one row).

        Mime filter is **not** applied here — ``get`` is the explicit
        single-item lookup, so the caller already knows the file id.
        Returning a known PDF id from ``get`` is fine; mass-listing them
        from ``list`` / ``search`` is what the filter prevents.
        """
        self._require_connected()
        if not item_id:
            raise KeyError("drive file id must be non-empty")
        try:
            response = await self._bridge.execute(
                self.PROVIDER,
                _GET_TOOL,
                {"file_id": item_id},
            )
        except Exception as exc:  # noqa: BLE001 -- bridge raises a soup of errors
            self._mark("get", "error", note=type(exc).__name__)
            raise KeyError(f"drive file {item_id!r} not found: {exc}") from exc
        raw = _extract_single_file(response, item_id)
        if raw is None:
            self._mark("get", "not_found", note=f"id={item_id}")
            raise KeyError(f"drive file {item_id!r} not found")
        try:
            item = self._to_item(raw)
        except (KeyError, TypeError, ValueError) as exc:
            self._mark("get", "error", note=type(exc).__name__)
            raise KeyError(f"drive file {item_id!r} could not be parsed: {exc}") from exc
        self._mark("get", "ok", note=f"id={item_id}")
        return item

    async def search(
        self,
        query: str,
        *,
        limit: int = 50,
        cursor: str | None = None,
    ) -> SearchResult:
        """Search Drive via the native ``q`` parameter.

        Uses ``name contains '<escaped>'`` — Drive's search-files syntax.
        The query is sanitised before interpolation (see
        ``_escape_q_value``) so a user-controlled string can't inject
        extra ``q`` clauses. Mime filter applies as for ``list``.
        """
        self._require_connected()
        if not query:
            raise ValueError("search query must be non-empty")
        safe = _escape_q_value(query)
        q_clause = f"name contains '{safe}'"
        raw_files = await self._fetch_files({"q": q_clause, "page_size": min(limit, 100)})
        matched = self._materialise(raw_files, since=None, limit=limit)
        self._mark("search", "ok", items=len(matched), note=f"q={query[:32]}")
        return SearchResult(items=tuple(matched), total_estimated=len(matched))

    def _to_item(self, raw: dict[str, Any]) -> IntegrationItem:
        """Map one Drive file row into a metadata-only ``IntegrationItem``.

        ``snippet`` is composed from mime type + size + path — *never*
        body content. The body-fetch path is gated behind the opt-in
        consent UI deferred to #44.
        """
        file_id = str(raw.get("id") or raw.get("file_id") or "")
        if not file_id:
            raise KeyError("drive file missing 'id'")
        name = str(raw.get("name") or raw.get("title") or "(untitled)")
        mime_type = str(raw.get("mimeType") or raw.get("mime_type") or "")
        size_bytes = _parse_size_bytes(raw.get("size"))
        modified_time = str(raw.get("modifiedTime") or raw.get("modified_time") or "")
        created_time = str(raw.get("createdTime") or raw.get("created_time") or "")
        parents = _parents_list(raw.get("parents"))
        owner_email = _owner_email(raw.get("owners") or raw.get("owner"))
        path_hint = str(raw.get("path") or "")
        snippet = _compose_metadata_snippet(
            mime_type=mime_type,
            size_bytes=size_bytes,
            path_hint=path_hint,
            parents=parents,
        )
        web_view_link = str(raw.get("webViewLink") or raw.get("web_view_link") or "")
        uri = web_view_link or f"https://drive.google.com/file/d/{file_id}"
        updated_at = _parse_iso(modified_time)
        idem = hashlib.sha256(f"{file_id}|{modified_time}".encode()).hexdigest()
        return IntegrationItem(
            id=file_id,
            title=name,
            snippet=snippet,
            uri=uri,
            updated_at=updated_at,
            metadata={
                "file_id": file_id,
                "mime_type": mime_type,
                "size_bytes": size_bytes,
                "modified_time": modified_time,
                "created_time": created_time,
                "parents": parents,
                "owner_email": owner_email,
                "sha256": idem,
            },
        )

    # ------------------------------------------------------------------ #
    # Helpers                                                            #
    # ------------------------------------------------------------------ #

    async def _fetch_files(self, arguments: dict[str, Any]) -> list[dict[str, Any]]:
        """One ``bridge.execute`` call returning the unwrapped files list.

        Defensive about response shape — Composio v3 has been known to
        nest the row list under ``data.files``, ``files``, or ``items``
        depending on the tool. Best-effort extraction with empty-list
        fallback so a weird response shape doesn't crash the agent."""
        try:
            response = await self._bridge.execute(self.PROVIDER, _LIST_TOOL, arguments)
        except Exception as exc:  # noqa: BLE001 -- bridge raises a soup of errors
            _log.warning("drive: fetch failed: %s", exc)
            self._mark("list", "error", note=type(exc).__name__)
            return []
        return _extract_files(response)

    def _materialise(
        self,
        rows: list[dict[str, Any]],
        *,
        since: datetime | None,
        limit: int,
    ) -> list[IntegrationItem]:
        """Apply mime filter + since filter + cap, return items."""
        out: list[IntegrationItem] = []
        since_aware = since
        if since_aware is not None and since_aware.tzinfo is None:
            since_aware = since_aware.replace(tzinfo=UTC)
        for raw in rows:
            mime_type = str(raw.get("mimeType") or raw.get("mime_type") or "")
            if mime_type not in _DOCS_AND_SHEETS_MIMES:
                # PDFs / images / other — explicitly deferred per #35 AC.
                continue
            try:
                item = self._to_item(raw)
            except (KeyError, TypeError, ValueError):
                continue
            if since_aware is not None and item.updated_at < since_aware:
                continue
            out.append(item)
            if len(out) >= limit:
                break
        return out


# --------------------------------------------------------------------------- #
# Response-shape helpers (module-level for unit testing)                      #
# --------------------------------------------------------------------------- #


def _extract_files(response: Any) -> list[dict[str, Any]]:
    """Pull the files list out of ``GOOGLEDRIVE_LIST_FILES``'s response.

    Accepts ``{"files": [...]}``, ``{"data": {"files": [...]}}``, or
    ``{"items": [...]}`` — Composio's v3 envelope is best-effort.
    Returns ``[]`` if the shape is unrecognisable so the integration
    fails closed rather than crashing.
    """
    if not isinstance(response, dict):
        return []
    if isinstance(response.get("files"), list):
        return [dict(f) for f in response["files"] if isinstance(f, dict)]
    nested = response.get("data")
    if isinstance(nested, dict) and isinstance(nested.get("files"), list):
        return [dict(f) for f in nested["files"] if isinstance(f, dict)]
    items = response.get("items")
    if isinstance(items, list):
        return [dict(f) for f in items if isinstance(f, dict)]
    return []


def _extract_single_file(response: Any, expected_id: str) -> dict[str, Any] | None:
    """Pull one file row out of ``GOOGLEDRIVE_FIND_FILE``'s response.

    The tool may return the file directly (``{"id": ..., "name": ...}``),
    nested under ``data`` / ``file``, or wrapped in a one-element ``files``
    list. Falls back to ``_extract_files`` and picks the matching id.
    Returns ``None`` if nothing matched.
    """
    if not isinstance(response, dict):
        return None
    # Direct shape: response IS the file row.
    if response.get("id") == expected_id or response.get("file_id") == expected_id:
        return dict(response)
    # Nested under ``data`` or ``file``.
    for key in ("file", "data"):
        inner = response.get(key)
        if isinstance(inner, dict):
            if inner.get("id") == expected_id or inner.get("file_id") == expected_id:
                return dict(inner)
    # Wrapped in a files list.
    for candidate in _extract_files(response):
        cand_id = candidate.get("id") or candidate.get("file_id")
        if cand_id == expected_id:
            return candidate
    return None


def _escape_q_value(value: str) -> str:
    """Escape single quotes for Drive's ``q`` parameter.

    Per https://developers.google.com/drive/api/guides/search-files,
    string literals in ``q`` are wrapped in single quotes and inner
    quotes are backslash-escaped. This is the only special-char SBW
    interpolates today; the caller never builds nested ``q`` clauses.
    """
    return value.replace(_Q_SINGLE_QUOTE, _Q_ESCAPED_SINGLE_QUOTE)


def _parse_iso(value: str) -> datetime:
    """Parse Drive's ISO-8601 timestamp; fall back to epoch on garbage."""
    if not value:
        return datetime.fromtimestamp(0, tz=UTC)
    try:
        return datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        return datetime.fromtimestamp(0, tz=UTC)


def _parse_size_bytes(value: Any) -> int | None:
    """Drive returns ``size`` as a string of bytes; Composio may surface
    an int. ``None`` for folders / Docs / Sheets (no canonical size).
    """
    if value is None:
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _parents_list(value: Any) -> list[str]:
    """Drive's ``parents`` is a list of folder ids. Defensive cast."""
    if isinstance(value, list):
        return [str(p) for p in value if isinstance(p, str | int)]
    if isinstance(value, str | int):
        return [str(value)]
    return []


def _owner_email(value: Any) -> str:
    """Drive's ``owners`` is a list of ``{emailAddress, ...}`` rows;
    older shapes use a bare email string."""
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    if isinstance(value, dict):
        return str(value.get("emailAddress") or value.get("email", ""))
    if isinstance(value, list) and value:
        first = value[0]
        if isinstance(first, dict):
            return str(first.get("emailAddress") or first.get("email", ""))
        if isinstance(first, str):
            return first
    return ""


def _humanise_size(size_bytes: int | None) -> str:
    """Compact 'NN.N KB' / 'N.N MB' rendering for the snippet."""
    if size_bytes is None:
        return ""
    if size_bytes < 1024:
        return f"{size_bytes} B"
    if size_bytes < 1024 * 1024:
        return f"{size_bytes / 1024:.1f} KB"
    if size_bytes < 1024 * 1024 * 1024:
        return f"{size_bytes / (1024 * 1024):.1f} MB"
    return f"{size_bytes / (1024 * 1024 * 1024):.1f} GB"


def _humanise_mime(mime_type: str) -> str:
    """Human label for the metadata snippet."""
    if mime_type == _DOC_MIME:
        return "document"
    if mime_type == _SHEET_MIME:
        return "spreadsheet"
    if mime_type == "application/pdf":
        return "pdf"
    if mime_type.startswith("application/vnd.google-apps."):
        return mime_type.removeprefix("application/vnd.google-apps.")
    return mime_type or "file"


def _compose_metadata_snippet(
    *,
    mime_type: str,
    size_bytes: int | None,
    path_hint: str,
    parents: list[str],
) -> str:
    """Compose a metadata-only snippet — explicitly NO body content.

    Shape: ``"<kind> — <size> — <path-or-parents>"``. Each piece is
    omitted if absent so a Doc (no size) still produces a meaningful
    snippet. Capped at 500 chars by ``IntegrationItem`` convention.
    """
    parts: list[str] = [_humanise_mime(mime_type)]
    size = _humanise_size(size_bytes)
    if size:
        parts.append(size)
    if path_hint:
        parts.append(path_hint)
    elif parents:
        parts.append(f"parents={','.join(parents[:3])}")
    return " — ".join(parts)[:500]


__all__ = [
    "GoogleDriveIntegration",
    "_DOCS_AND_SHEETS_MIMES",
]
