"""Launch-once lifecycle ledger for build-in-public posts (ADR-026 / PRD-019).

A ``project_launch`` is, by definition, a **one-time** event: a project is
launched once; a second "launch" of the same repo is an update, not a launch
(ADR-026 §Context, type 1). The :class:`ProjectLedger` is the tiny, durable
state that lets the engine *refuse/redirect* a second launch and track the last
time any post was drafted for a project.

It is a JSON file mapping ``repo_slug -> {"launched_at": iso|None,
"last_post_at": iso|None}``:

- ``launched_at`` — ISO-8601 timestamp of the one-time launch, or ``None`` if the
  project has never been launched.
- ``last_post_at`` — ISO-8601 timestamp of the most recent post of *any* sub-type
  (launch / feature / weekly), or ``None`` if nothing has been posted yet.

Design
------
- **Pure stdlib** (``json`` + ``pathlib``) — no network, no third-party deps. The
  ledger is local lifecycle state, mirroring ADR-026's "local git + local files,
  no new scope" posture.
- **Tolerant reads**: a missing file, an unparseable file, or a JSON document of
  the wrong shape all degrade to an *empty* ledger rather than raising — a
  corrupt ledger must never block drafting (worst case: a launch fires twice,
  which is recoverable, vs. a hard crash, which is not).
- **Self-creating writes**: ``record_*`` creates the parent directory if needed
  and atomically rewrites the whole (small) document.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

# The per-repo record shape. Both timestamps are ISO-8601 strings or ``None``.
_Record = dict[str, str | None]


class ProjectLedger:
    """Per-repo launch-once + last-post state, persisted as a small JSON file.

    Parameters
    ----------
    path:
        Filesystem path to the JSON ledger. It need not exist yet; reads of a
        missing/corrupt file yield an empty ledger, and the first write creates
        the parent directory.
    """

    def __init__(self, path: Path) -> None:
        self._path = Path(path)

    # -- queries --------------------------------------------------------------

    def was_launched(self, repo_slug: str) -> bool:
        """Return ``True`` iff ``repo_slug`` has a recorded launch timestamp.

        Used to gate the one-time ``project_launch``: a ``True`` here means a
        second launch must be refused/redirected (ADR-026)."""
        record = self._load().get(repo_slug)
        return bool(record and record.get("launched_at"))

    # -- mutations ------------------------------------------------------------

    def record_launch(self, repo_slug: str, *, when: str) -> None:
        """Record the one-time launch of ``repo_slug`` at ISO timestamp ``when``.

        Idempotent in shape: re-recording overwrites ``launched_at`` (a caller
        should gate on :meth:`was_launched` first; this method does not itself
        refuse a second launch — that policy lives at the call site)."""
        data = self._load()
        record = data.setdefault(repo_slug, self._blank())
        record["launched_at"] = when
        # A launch is also a post.
        record["last_post_at"] = when
        self._save(data)

    def record_post(self, repo_slug: str, *, when: str) -> None:
        """Update ``last_post_at`` for ``repo_slug`` to ``when``.

        Does **not** touch ``launched_at`` — a feature/weekly post is not a
        launch. If the repo has no record yet, one is created with a ``None``
        ``launched_at`` (the project has been posted but never formally launched)."""
        data = self._load()
        record = data.setdefault(repo_slug, self._blank())
        record["last_post_at"] = when
        self._save(data)

    # -- internals ------------------------------------------------------------

    @staticmethod
    def _blank() -> _Record:
        return {"launched_at": None, "last_post_at": None}

    def _load(self) -> dict[str, _Record]:
        """Read the ledger, degrading any read error / wrong shape to ``{}``."""
        try:
            raw = self._path.read_text(encoding="utf-8")
        except (FileNotFoundError, OSError):
            return {}
        try:
            parsed: Any = json.loads(raw)
        except (json.JSONDecodeError, ValueError):
            return {}
        if not isinstance(parsed, dict):
            return {}
        # Coerce each value to a record dict; drop anything malformed.
        out: dict[str, _Record] = {}
        for slug, value in parsed.items():
            if isinstance(value, dict):
                out[str(slug)] = {
                    "launched_at": value.get("launched_at"),
                    "last_post_at": value.get("last_post_at"),
                }
        return out

    def _save(self, data: dict[str, _Record]) -> None:
        """Write the whole ledger, creating the parent directory if needed."""
        self._path.parent.mkdir(parents=True, exist_ok=True)
        self._path.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n", encoding="utf-8")
