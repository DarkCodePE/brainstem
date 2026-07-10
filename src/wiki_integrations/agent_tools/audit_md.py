"""
Per-provider Markdown audit log per
[PRD-006](../../../docs/PRD-006-integrations-framework.md) AC §Audit log.

Distinct from the forensic JSONL log under `~/.sbw/logs/` (that one is in
`wiki_core.secrets.AuditLog`). This one is **human-readable** and lives
inside `knowledge-base/integrations/_log/<provider>.md` — it's the trail
a user reads when they open Obsidian and ask "what did SBW pull from Gmail
yesterday?".

Format: one Markdown list bullet per event, append-only.

```
- 2026-05-26T15:42:11Z  list  ok  items=12  next_cursor=tok_abc
- 2026-05-26T15:50:01Z  search "from:boss"  ok  items=3
```
"""

from __future__ import annotations

import threading
from datetime import UTC, datetime
from pathlib import Path


class ProviderMarkdownLog:
    """Append-only one-bullet-per-event Markdown writer.

    Thread-safe; concurrent calls from polling + agent-tool paths are
    serialised through an in-process lock. The on-disk file is touched
    with one O_APPEND open per write so writes are atomic enough for the
    single-machine local-first deployment SBW targets.
    """

    HEADER = "# Integration log\n\nAppend-only audit trail for this provider's fetches.\n\n"

    def __init__(self, *, knowledge_base: Path, provider: str) -> None:
        if not provider:
            raise ValueError("provider must be non-empty")
        self._provider = provider
        self._lock = threading.Lock()
        log_dir = Path(knowledge_base) / "integrations" / "_log"
        log_dir.mkdir(parents=True, exist_ok=True)
        self._path = log_dir / f"{provider}.md"
        if not self._path.exists():
            self._path.write_text(self.HEADER, encoding="utf-8")

    @property
    def path(self) -> Path:
        return self._path

    def append(
        self,
        *,
        op: str,
        result: str = "ok",
        items: int | None = None,
        note: str | None = None,
    ) -> None:
        """Append one bullet describing `op`.

        Parameters
        ----------
        op:
            ``connect`` / ``disconnect`` / ``list`` / ``get`` / ``search`` / ``health``.
        result:
            ``ok``, ``error``, ``not_connected``, ``rate_limited`` …
        items:
            Count for list/search; ``None`` for connect/disconnect/health.
        note:
            Short free-form annotation (search query, error class).
            **Must not contain PII or bearer tokens** — caller redacts.
        """
        ts = datetime.now(UTC).isoformat(timespec="seconds").replace("+00:00", "Z")
        parts = [ts, op, result]
        if items is not None:
            parts.append(f"items={items}")
        if note:
            parts.append(note)
        line = "- " + "  ".join(parts) + "\n"
        with self._lock, self._path.open("a", encoding="utf-8") as fh:
            fh.write(line)
