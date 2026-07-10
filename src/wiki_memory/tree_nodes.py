"""
Memory Tree node persistence per [PRD-004 FR-3](../../docs/PRD-004-memory-tree.md).

Three concentric kinds:

- `source` — one node per ingested source. Child chunks attached by
  `source_id` in `content_store`.
- `topic` — clusters of related source nodes. The seal worker populates
  this in M2 Sprint 4 (deferred).
- `global` — single-node-per-vault summary that sits at the root of the
  tree. Also seal-worker territory.

This module ships v1: schema, CRUD, tombstone, and a `score` field with a
placeholder constant value. The actual scoring (recency × reuse ×
pagerank-proxy) is a separate module that lands with PRD-008 model
routing in M3.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Literal

import aiosqlite

NodeKind = Literal["source", "topic", "global"]


@dataclass(frozen=True, slots=True)
class TreeNode:
    """A row in `tree_nodes`."""

    node_id: str
    """ULID/uuid4. Stable across re-ingests for source nodes
    (derived from source sha256); freshly generated for topic/global."""

    kind: NodeKind
    parent_id: str | None
    level: int  # 0 = source, 1 = topic, 2 = global
    summary_sha256: str | None  # sha256 of the *summary text*, not the source
    score: float
    sealed_at: str | None  # ISO; non-None once the seal worker filled in summary_sha256
    tombstoned: bool
    created_at: str
    # Temporal supersession (ADR-028 #158). Defaults keep legacy
    # constructors valid; the store populates them.
    source_key: str | None = None
    """Stable logical identity of the source (sha256 of event.source),
    shared across re-ingests of the same document. None for topic/global."""
    is_latest: bool = True
    """False once a newer ingest of the same source_key supersedes this one."""
    superseded_by: str | None = None
    """node_id of the version that superseded this one (None while latest)."""
    event_time: str | None = None
    """The source's own timestamp (when the content happened), distinct
    from `created_at` (ingest time). Mirrors supermemory's eventDate."""


_SCHEMA = """
PRAGMA journal_mode = WAL;

CREATE TABLE IF NOT EXISTS tree_nodes (
    node_id         TEXT PRIMARY KEY,
    kind            TEXT NOT NULL CHECK (kind IN ('source', 'topic', 'global')),
    parent_id       TEXT,
    level           INTEGER NOT NULL,
    summary_sha256  TEXT,
    score           REAL NOT NULL DEFAULT 0.0,
    sealed_at       TEXT,
    tombstoned      INTEGER NOT NULL DEFAULT 0,
    created_at      TEXT NOT NULL,
    source_key      TEXT,                       -- ADR-028 #158: logical source identity
    is_latest       INTEGER NOT NULL DEFAULT 1, -- 0 once superseded by a newer ingest
    superseded_by   TEXT,                       -- node_id of the superseding version
    event_time      TEXT,                       -- source's own timestamp (vs created_at)
    FOREIGN KEY (parent_id) REFERENCES tree_nodes(node_id) ON DELETE SET NULL
);

CREATE INDEX IF NOT EXISTS idx_tree_kind ON tree_nodes(kind);
CREATE INDEX IF NOT EXISTS idx_tree_parent ON tree_nodes(parent_id);
"""

#: Canonical column list for tree_nodes SELECTs feeding ``_row_to_node``.
#: Position-sensitive (row[0..12]).
_NODE_COLS = (
    "node_id, kind, parent_id, level, summary_sha256, score, sealed_at,"
    " tombstoned, created_at, source_key, is_latest, superseded_by, event_time"
)


def _utcnow_iso() -> str:
    return datetime.now(UTC).isoformat(timespec="seconds").replace("+00:00", "Z")


class TreeNodeStore:
    """CRUD + tombstone for `tree_nodes`."""

    def __init__(self, db_path: Path) -> None:
        self._db_path = Path(db_path)
        self._db: aiosqlite.Connection | None = None

    async def init(self) -> None:
        if self._db is not None:
            return
        self._db_path.parent.mkdir(parents=True, exist_ok=True)
        self._db = await aiosqlite.connect(self._db_path)
        await self._db.execute("PRAGMA foreign_keys = ON")
        await self._db.executescript(_SCHEMA)
        await self._db.commit()
        await self._maybe_add_supersession_columns()

    async def _maybe_add_supersession_columns(self) -> None:
        """ALTER TABLE adds for the supersession quartet on legacy DBs
        (ADR-028 #158). Additive; existing rows backfill to is_latest=1
        and NULL for the rest. Idempotent on every init."""
        db = self._conn()
        async with db.execute("PRAGMA table_info(tree_nodes)") as cur:
            existing = {row[1] for row in await cur.fetchall()}
        adds = (
            ("source_key", "TEXT"),
            ("is_latest", "INTEGER NOT NULL DEFAULT 1"),
            ("superseded_by", "TEXT"),
            ("event_time", "TEXT"),
        )
        changed = False
        for name, decl in adds:
            if name not in existing:
                await db.execute(f"ALTER TABLE tree_nodes ADD COLUMN {name} {decl}")
                changed = True
        if changed:
            await db.execute(
                "CREATE INDEX IF NOT EXISTS idx_tree_source_key ON tree_nodes(source_key)"
            )
            await db.commit()

    async def close(self) -> None:
        if self._db is None:
            return
        await self._db.close()
        self._db = None

    def _conn(self) -> aiosqlite.Connection:
        if self._db is None:
            raise RuntimeError("TreeNodeStore.init() must be awaited before use")
        return self._db

    async def upsert(self, node: TreeNode) -> None:
        db = self._conn()
        await db.execute(
            """
            INSERT INTO tree_nodes
                (node_id, kind, parent_id, level, summary_sha256,
                 score, sealed_at, tombstoned, created_at,
                 source_key, is_latest, superseded_by, event_time)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(node_id) DO UPDATE SET
                parent_id      = excluded.parent_id,
                level          = excluded.level,
                summary_sha256 = excluded.summary_sha256,
                score          = excluded.score,
                sealed_at      = excluded.sealed_at,
                tombstoned     = excluded.tombstoned,
                source_key     = excluded.source_key,
                is_latest      = excluded.is_latest,
                superseded_by  = excluded.superseded_by,
                event_time     = excluded.event_time
            """,
            (
                node.node_id,
                node.kind,
                node.parent_id,
                node.level,
                node.summary_sha256,
                node.score,
                node.sealed_at,
                int(node.tombstoned),
                node.created_at,
                node.source_key,
                int(node.is_latest),
                node.superseded_by,
                node.event_time,
            ),
        )
        await db.commit()

    async def create_source_node(
        self,
        *,
        node_id: str,
        parent_id: str | None = None,
        score: float = 0.0,
        source_key: str | None = None,
        event_time: str | None = None,
    ) -> TreeNode:
        """Convenience constructor for the most-common case: a fresh
        source node with no summary yet. The seal worker fills
        `summary_sha256` later.

        Pass `source_key` (ADR-028 #158) to enable temporal supersession:
        a later ingest of the same `source_key` can mark this node
        superseded via `supersede`."""
        node = TreeNode(
            node_id=node_id,
            kind="source",
            parent_id=parent_id,
            level=0,
            summary_sha256=None,
            score=score,
            sealed_at=None,
            tombstoned=False,
            created_at=_utcnow_iso(),
            source_key=source_key,
            is_latest=True,
            superseded_by=None,
            event_time=event_time,
        )
        await self.upsert(node)
        return node

    async def supersede(self, *, source_key: str, new_node_id: str) -> list[str]:
        """Mark every prior *latest* node sharing `source_key` (except
        `new_node_id` itself) as superseded by `new_node_id` (ADR-028 #158).

        Sets `is_latest = 0` and `superseded_by = new_node_id` on those
        rows. Non-destructive — superseded versions remain on disk and are
        retrievable with `include_superseded=True`. Returns the node_ids
        that were superseded (empty if this is the first ingest of the
        source). Idempotent: re-running finds nothing still-latest to flip.
        """
        db = self._conn()
        async with db.execute(
            "SELECT node_id FROM tree_nodes"
            " WHERE source_key = ? AND node_id != ? AND is_latest = 1",
            (source_key, new_node_id),
        ) as cur:
            superseded = [row[0] for row in await cur.fetchall()]
        if superseded:
            await db.execute(
                "UPDATE tree_nodes SET is_latest = 0, superseded_by = ?"
                " WHERE source_key = ? AND node_id != ? AND is_latest = 1",
                (new_node_id, source_key, new_node_id),
            )
            await db.commit()
        return superseded

    async def superseded_node_ids(self) -> list[str]:
        """All node_ids that have been superseded (is_latest = 0). Used by
        the recall path to filter stale versions out by default."""
        db = self._conn()
        async with db.execute("SELECT node_id FROM tree_nodes WHERE is_latest = 0") as cur:
            rows = await cur.fetchall()
        return [row[0] for row in rows]

    async def get(self, node_id: str) -> TreeNode | None:
        db = self._conn()
        async with db.execute(
            f"SELECT {_NODE_COLS} FROM tree_nodes WHERE node_id = ?",
            (node_id,),
        ) as cur:
            row = await cur.fetchone()
        return _row_to_node(row) if row else None

    async def list_by_kind(
        self, kind: NodeKind, *, include_tombstoned: bool = False
    ) -> list[TreeNode]:
        db = self._conn()
        sql = f"SELECT {_NODE_COLS} FROM tree_nodes WHERE kind = ?"
        if not include_tombstoned:
            sql += " AND tombstoned = 0"
        async with db.execute(sql, (kind,)) as cur:
            rows = await cur.fetchall()
        return [_row_to_node(r) for r in rows]

    async def children_of(
        self, parent_id: str, *, include_tombstoned: bool = False
    ) -> list[TreeNode]:
        db = self._conn()
        sql = f"SELECT {_NODE_COLS} FROM tree_nodes WHERE parent_id = ?"
        if not include_tombstoned:
            sql += " AND tombstoned = 0"
        async with db.execute(sql, (parent_id,)) as cur:
            rows = await cur.fetchall()
        return [_row_to_node(r) for r in rows]

    async def tombstone(self, node_id: str) -> bool:
        """Mark a node tombstoned. Returns True if a row was changed
        (i.e. the node existed and wasn't already tombstoned)."""
        db = self._conn()
        cur = await db.execute(
            "UPDATE tree_nodes SET tombstoned = 1 WHERE node_id = ? AND tombstoned = 0",
            (node_id,),
        )
        await db.commit()
        return (cur.rowcount or 0) > 0

    async def mark_sealed(
        self,
        node_id: str,
        *,
        summary_sha256: str,
        score: float | None = None,
    ) -> None:
        """Called by the seal worker once a summary has been written."""
        db = self._conn()
        if score is None:
            await db.execute(
                "UPDATE tree_nodes SET summary_sha256 = ?, sealed_at = ? WHERE node_id = ?",
                (summary_sha256, _utcnow_iso(), node_id),
            )
        else:
            await db.execute(
                "UPDATE tree_nodes SET summary_sha256 = ?, score = ?, sealed_at = ? WHERE node_id = ?",
                (summary_sha256, score, _utcnow_iso(), node_id),
            )
        await db.commit()

    async def count(self, *, include_tombstoned: bool = False) -> int:
        db = self._conn()
        sql = "SELECT COUNT(*) FROM tree_nodes"
        if not include_tombstoned:
            sql += " WHERE tombstoned = 0"
        async with db.execute(sql) as cur:
            row = await cur.fetchone()
        return int(row[0]) if row else 0


def _row_to_node(row: Sequence) -> TreeNode:
    return TreeNode(
        node_id=row[0],
        kind=row[1],
        parent_id=row[2],
        level=row[3],
        summary_sha256=row[4],
        score=float(row[5]),
        sealed_at=row[6],
        tombstoned=bool(row[7]),
        created_at=row[8],
        source_key=row[9],
        is_latest=bool(row[10]),
        superseded_by=row[11],
        event_time=row[12],
    )
