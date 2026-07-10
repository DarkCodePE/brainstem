"""
Memory Tree `content_store` — durable chunk storage per [PRD-004 FR-2](../../docs/PRD-004-memory-tree.md).

SQLite WAL-mode table keyed by chunk `sha256`. Bodies are stored verbatim
in v1; the >8KiB zstd compression promised by PRD-004 is deferred to v2
when the zstandard dependency is justified by measured bloat.

The store is **async** to keep the call shape consistent with
`wiki_core.MemoryStore` and the rest of the wiki_ingest async API.

This module is intentionally narrow: insert / get / count / list_by_source
/ delete_by_source. No scoring (that's in `tree_nodes`), no retrieval
(that's in `recall`), no chunking (that's in `chunker`). Composition lives
in the higher-level `wiki_memory.tree` module which lands with the seal
worker in M2 Sprint 4.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

import aiosqlite

from wiki_memory.chunker import Chunk


@dataclass(frozen=True, slots=True)
class StoredChunk:
    """A chunk as persisted in the content_store."""

    sha256: str
    source_id: str
    chunk_index: int
    body: str
    token_count: int
    created_at: str
    reuse_count: int = 0
    """How many recall bundles have surfaced this chunk (ADR-027 #155).
    Defaults to 0 so legacy constructors and tests stay valid; the
    persistence layer populates it from the ``reuse_count`` column."""


# Bump SCHEMA_VERSION when the chunker config or the chunks table layout
# changes such that existing rows become invalid. Tolaria-inspired
# auto-invalidation pattern (issue #127 sub-item 1): on init, if the
# stored version is lower than the constant, we truncate ``chunks`` and
# the seed script repopulates from disk. Additive column changes (like
# #119's embeddings) don't bump the version — only changes that would
# silently produce wrong results do.
SCHEMA_VERSION = 1


_SCHEMA = """
PRAGMA journal_mode = WAL;

-- ``meta`` carries singleton key/value rows for migration + cross-machine
-- validation (issue #127 sub-items 1 + 4). One row per key.
CREATE TABLE IF NOT EXISTS meta (
    key   TEXT PRIMARY KEY,
    value TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS chunks (
    sha256          TEXT PRIMARY KEY,
    source_id       TEXT NOT NULL,
    chunk_index     INTEGER NOT NULL,
    body            TEXT NOT NULL,
    token_count     INTEGER NOT NULL,
    created_at      TEXT NOT NULL,
    embedding       BLOB,           -- nullable; populated by the embedder
    embedding_model TEXT,           -- model id used for ``embedding``
    embedding_dim   INTEGER,        -- vector length, redundant for safety
    reuse_count     INTEGER NOT NULL DEFAULT 0  -- ADR-027 #155: recall surfacing count
);

CREATE INDEX IF NOT EXISTS idx_chunks_source ON chunks(source_id, chunk_index);

-- citations (ADR-027 #156): one row per (summary_sha256 -> cited chunk
-- sha256) edge, written by the seal worker. The chunk in-degree (how
-- many summaries cite a chunk) feeds the pagerank-proxy scoring signal.
-- Full shas are stored here (the vault frontmatter only keeps an 8-char
-- prefix, which is lossy and unusable for counting).
CREATE TABLE IF NOT EXISTS citations (
    summary_sha256  TEXT NOT NULL,
    chunk_sha256    TEXT NOT NULL,
    PRIMARY KEY (summary_sha256, chunk_sha256)
);

CREATE INDEX IF NOT EXISTS idx_citations_chunk ON citations(chunk_sha256);
-- idx_chunks_embedded is created in _maybe_add_embedding_columns AFTER
-- the column it references exists. SQLite's ``CREATE INDEX IF NOT
-- EXISTS`` only short-circuits on the index NAME — it still validates
-- the column list, so putting it here would crash on legacy DBs that
-- pre-date #119.

-- FTS5 virtual table (issue #118). External-content mode pointed at
-- ``chunks`` so we don't double-store the body. Tokenizer matches the
-- OpenHuman choice (``memory_store/unified/events.rs:40-46``) — porter
-- gives English stemming, unicode61 normalises diacritics so Spanish
-- queries ("observabilidad" / "observability") collapse the morphology
-- gap that plain ``LIKE`` can't.
CREATE VIRTUAL TABLE IF NOT EXISTS chunks_fts USING fts5(
    body,
    content='chunks',
    content_rowid='rowid',
    tokenize='porter unicode61'
);

-- Triggers keep ``chunks_fts`` in sync with ``chunks`` writes. External
-- content tables don't auto-sync; per FTS5 docs the canonical pattern is
-- (delete-then-insert) — that's what AFTER UPDATE does. For AFTER DELETE
-- we only need the ``'delete'`` op because the row is gone.
CREATE TRIGGER IF NOT EXISTS chunks_fts_ai AFTER INSERT ON chunks BEGIN
    INSERT INTO chunks_fts(rowid, body) VALUES (new.rowid, new.body);
END;
CREATE TRIGGER IF NOT EXISTS chunks_fts_ad AFTER DELETE ON chunks BEGIN
    INSERT INTO chunks_fts(chunks_fts, rowid, body) VALUES('delete', old.rowid, old.body);
END;
CREATE TRIGGER IF NOT EXISTS chunks_fts_au AFTER UPDATE ON chunks BEGIN
    INSERT INTO chunks_fts(chunks_fts, rowid, body) VALUES('delete', old.rowid, old.body);
    INSERT INTO chunks_fts(rowid, body) VALUES (new.rowid, new.body);
END;
"""

# FTS5 query operators we strip from user input. Leaving them in lets a
# user (or a focal_keywords list) accidentally break the parser with a
# trailing colon or stray quote. We don't try to be clever — implicit
# AND between bare tokens is the desired default, and anyone who wants
# phrase search wraps the keyword in quotes upstream.
#
# ``-`` is included because FTS5 treats ``foo -bar`` as "foo without
# bar" — passing a query like "M3-S2" silently filters out everything
# matching "S2". Stripping is safer than detecting intent (#131 scale
# scenario surfaced this).
_FTS5_DANGEROUS_CHARS = set("\"'():*-")


def _sanitize_fts5_query(needle: str) -> str:
    """Drop FTS5 special characters so user input never breaks the parser.

    Returns an empty string if nothing useful is left — callers treat
    empty as "no results" rather than running a wildcard scan."""
    cleaned = "".join(c if c not in _FTS5_DANGEROUS_CHARS else " " for c in needle)
    return " ".join(cleaned.split())


def _utcnow_iso() -> str:
    return datetime.now(UTC).isoformat(timespec="seconds").replace("+00:00", "Z")


#: Canonical column list for ``chunks`` SELECTs that build a ``StoredChunk``.
#: Position-sensitive — ``_row_to_chunk`` reads row[0..6] in this order.
_CHUNK_COLS = "sha256, source_id, chunk_index, body, token_count, created_at, reuse_count"


def _row_to_chunk(row: Sequence) -> StoredChunk:
    """Build a ``StoredChunk`` from a row whose first 7 columns are
    ``_CHUNK_COLS``. Extra trailing columns (embedding/dim in
    ``search_vector``) are ignored."""
    return StoredChunk(
        sha256=row[0],
        source_id=row[1],
        chunk_index=row[2],
        body=row[3],
        token_count=row[4],
        created_at=row[5],
        reuse_count=int(row[6]),
    )


class ContentStore:
    """Persistence layer for Memory Tree chunks.

    Construction is light; `init()` runs the schema migration and must be
    awaited before any other method. The caller owns lifecycle.
    """

    def __init__(self, db_path: Path) -> None:
        self._db_path = Path(db_path)
        self._db: aiosqlite.Connection | None = None

    async def init(self, *, vault_root: Path | str | None = None) -> None:
        """Open the DB and apply the schema. Idempotent.

        - Backfills ``chunks_fts`` when a pre-existing DB already has
          ``chunks`` rows but the FTS table is empty (issue #118).
        - Backfills the ``embedding`` columns when a pre-existing DB
          predates #119.
        - Truncates ``chunks`` if the stored SCHEMA_VERSION is older
          than the constant (issue #127 sub-item 1 — Tolaria-style).
        - When ``vault_root`` is given, stores it on first init and
          validates on subsequent inits; mismatch logs a warning so the
          caller can decide to reseed (issue #127 sub-item 4).
        """
        if self._db is not None:
            return
        self._db_path.parent.mkdir(parents=True, exist_ok=True)
        self._db = await aiosqlite.connect(self._db_path)
        await self._db.executescript(_SCHEMA)
        await self._db.commit()
        await self._maybe_add_embedding_columns()
        await self._maybe_add_reuse_count_column()
        await self._maybe_invalidate_on_schema_bump()
        await self._maybe_validate_vault_root(vault_root)
        await self._maybe_rebuild_fts()

    async def _meta_get(self, key: str) -> str | None:
        db = self._conn()
        async with db.execute("SELECT value FROM meta WHERE key = ?", (key,)) as cur:
            row = await cur.fetchone()
        return row[0] if row else None

    async def _meta_set(self, key: str, value: str) -> None:
        db = self._conn()
        await db.execute(
            "INSERT INTO meta (key, value) VALUES (?, ?)"
            " ON CONFLICT(key) DO UPDATE SET value = excluded.value",
            (key, value),
        )
        await db.commit()

    async def _maybe_invalidate_on_schema_bump(self) -> None:
        """If the on-disk SCHEMA_VERSION is older than the constant,
        truncate ``chunks`` and force the seed script to repopulate.

        We only truncate — never DROP — so the FTS5 + triggers stay
        intact and the seed flow stays simple. The version write is
        always last so a crash mid-truncate leaves the on-disk version
        unchanged and the next init retries."""
        import logging

        log = logging.getLogger(__name__)
        db = self._conn()
        stored = await self._meta_get("schema_version")
        stored_int = int(stored) if stored is not None else 0
        if stored_int >= SCHEMA_VERSION:
            return
        if stored is not None:
            # Genuine bump — invalidate.
            log.warning(
                "content_store: schema_version bumped %s → %s; truncating chunks "
                "(seed script will repopulate)",
                stored_int,
                SCHEMA_VERSION,
            )
            await db.execute("DELETE FROM chunks")
            await db.commit()
        # On first ever init (stored is None) we just stamp the version.
        await self._meta_set("schema_version", str(SCHEMA_VERSION))

    async def _maybe_validate_vault_root(self, vault_root: Path | str | None) -> None:
        """First init: stamp the vault_root absolute path. Subsequent
        inits: warn if it changed (cross-machine clones, vault moves)."""
        if vault_root is None:
            return
        import logging

        log = logging.getLogger(__name__)
        stored = await self._meta_get("vault_root")
        new_root = str(Path(vault_root).expanduser().resolve())
        if stored is None:
            await self._meta_set("vault_root", new_root)
            return
        if stored != new_root:
            log.warning(
                "content_store: vault_root changed (%s → %s). "
                "Source ids reference relative paths; consider reseeding via "
                "scripts/seed-memory-tree.py.",
                stored,
                new_root,
            )
            # Update the stored value so the warning fires once per change.
            await self._meta_set("vault_root", new_root)

    async def _maybe_add_embedding_columns(self) -> None:
        """ALTER TABLE adds for the embedding trio on legacy DBs (#119).

        SQLite ``CREATE TABLE IF NOT EXISTS`` is a no-op when the table
        already exists, so new columns from the canonical schema don't
        land. We probe ``PRAGMA table_info`` and add each column iff
        missing. Idempotent on every init."""
        db = self._conn()
        async with db.execute("PRAGMA table_info(chunks)") as cur:
            existing = {row[1] for row in await cur.fetchall()}
        adds = (
            ("embedding", "BLOB"),
            ("embedding_model", "TEXT"),
            ("embedding_dim", "INTEGER"),
        )
        for name, sql_type in adds:
            if name not in existing:
                await db.execute(f"ALTER TABLE chunks ADD COLUMN {name} {sql_type}")
        # Now safe to create the partial index on embedding_model.
        await db.execute(
            "CREATE INDEX IF NOT EXISTS idx_chunks_embedded ON chunks(embedding_model)"
            " WHERE embedding IS NOT NULL"
        )
        await db.commit()

    async def _maybe_add_reuse_count_column(self) -> None:
        """ALTER TABLE add for ``reuse_count`` on legacy DBs (ADR-027 #155).

        Additive, NOT NULL DEFAULT 0, so existing rows backfill to 0. Same
        ``PRAGMA table_info`` probe pattern as ``_maybe_add_embedding_columns``.
        Idempotent on every init."""
        db = self._conn()
        async with db.execute("PRAGMA table_info(chunks)") as cur:
            existing = {row[1] for row in await cur.fetchall()}
        if "reuse_count" not in existing:
            await db.execute("ALTER TABLE chunks ADD COLUMN reuse_count INTEGER NOT NULL DEFAULT 0")
            await db.commit()

    async def _maybe_rebuild_fts(self) -> None:
        """Rebuild ``chunks_fts`` if it's empty while ``chunks`` has rows.

        FTS5 external-content triggers only fire on writes that happen
        AFTER the table exists, so a DB that pre-dated #118 needs an
        explicit one-shot rebuild. ``SELECT COUNT(*) FROM chunks_fts``
        is misleading for external-content tables (it reports the
        backing chunks count, not the indexed count), so we probe the
        ``chunks_fts_docsize`` shadow table instead — it has one row
        per indexed document and is the authoritative emptiness signal.
        """
        db = self._conn()
        async with db.execute("SELECT COUNT(*) FROM chunks") as cur:
            chunks_count = (await cur.fetchone())[0]
        if chunks_count == 0:
            return
        async with db.execute("SELECT COUNT(*) FROM chunks_fts_docsize") as cur:
            indexed_count = (await cur.fetchone())[0]
        if indexed_count > 0:
            return
        await db.execute("INSERT INTO chunks_fts(chunks_fts) VALUES('rebuild')")
        await db.commit()

    async def close(self) -> None:
        if self._db is None:
            return
        await self._db.close()
        self._db = None

    def _conn(self) -> aiosqlite.Connection:
        if self._db is None:
            raise RuntimeError("ContentStore.init() must be awaited before use")
        return self._db

    async def insert(self, *, source_id: str, chunk: Chunk) -> bool:
        """Persist a single chunk. Returns True if inserted, False if the
        sha was already present (idempotent re-ingest path)."""
        db = self._conn()
        try:
            await db.execute(
                "INSERT INTO chunks (sha256, source_id, chunk_index, body, token_count, created_at)"
                " VALUES (?, ?, ?, ?, ?, ?)",
                (
                    chunk.sha256,
                    source_id,
                    chunk.chunk_index,
                    chunk.body,
                    chunk.token_count,
                    _utcnow_iso(),
                ),
            )
            await db.commit()
            return True
        except aiosqlite.IntegrityError:
            return False

    async def insert_many(self, *, source_id: str, chunks: Sequence[Chunk]) -> int:
        """Persist a batch. Returns the number of *new* inserts (skipping
        already-present shas). Atomic at the SQLite level."""
        db = self._conn()
        now = _utcnow_iso()
        inserted = 0
        async with db.execute("BEGIN") as _:
            pass
        try:
            for c in chunks:
                try:
                    await db.execute(
                        "INSERT INTO chunks (sha256, source_id, chunk_index, body, token_count, created_at)"
                        " VALUES (?, ?, ?, ?, ?, ?)",
                        (c.sha256, source_id, c.chunk_index, c.body, c.token_count, now),
                    )
                    inserted += 1
                except aiosqlite.IntegrityError:
                    continue
            await db.commit()
        except Exception:
            await db.rollback()
            raise
        return inserted

    async def get(self, sha256: str) -> StoredChunk | None:
        db = self._conn()
        async with db.execute(
            f"SELECT {_CHUNK_COLS} FROM chunks WHERE sha256 = ?",
            (sha256,),
        ) as cur:
            row = await cur.fetchone()
        if row is None:
            return None
        return _row_to_chunk(row)

    async def count(self) -> int:
        db = self._conn()
        async with db.execute("SELECT COUNT(*) FROM chunks") as cur:
            row = await cur.fetchone()
        return int(row[0]) if row else 0

    async def list_by_source(self, source_id: str) -> list[StoredChunk]:
        db = self._conn()
        async with db.execute(
            f"SELECT {_CHUNK_COLS} FROM chunks WHERE source_id = ? ORDER BY chunk_index",
            (source_id,),
        ) as cur:
            rows = await cur.fetchall()
        return [_row_to_chunk(row) for row in rows]

    async def delete_by_source(self, source_id: str) -> int:
        """Remove all chunks for a source. Used by the tombstone path
        (PRD-004 US-004 / ADR-029 #159). Also drops any citation edges
        whose cited chunk belonged to this source so in-degree stays
        consistent after a hard delete."""
        db = self._conn()
        cur = await db.execute("DELETE FROM chunks WHERE source_id = ?", (source_id,))
        # Citation edges pointing at now-deleted chunks would otherwise
        # leave dangling in-degree. Clean them in the same commit.
        await db.execute(
            "DELETE FROM citations WHERE chunk_sha256 NOT IN (SELECT sha256 FROM chunks)"
        )
        await db.commit()
        return cur.rowcount or 0

    # ------------------------------------------------------------------ #
    # Reuse tracking (ADR-027 #155)                                      #
    # ------------------------------------------------------------------ #

    async def increment_reuse(self, shas: Sequence[str]) -> int:
        """Increment ``reuse_count`` by 1 for each sha actually surfaced
        by a recall bundle. Single batched UPDATE. Returns rows changed.

        Unknown shas are silently ignored (the WHERE simply matches
        nothing). Empty input is a no-op."""
        unique = [s for s in dict.fromkeys(shas)]  # de-dup, preserve order
        if not unique:
            return 0
        db = self._conn()
        placeholders = ",".join("?" for _ in unique)
        cur = await db.execute(
            f"UPDATE chunks SET reuse_count = reuse_count + 1 WHERE sha256 IN ({placeholders})",
            tuple(unique),
        )
        await db.commit()
        return cur.rowcount or 0

    async def max_reuse(self) -> int:
        """Max ``reuse_count`` across all chunks (the normaliser for
        ``scoring.reuse_score``). Returns 0 on an empty store."""
        db = self._conn()
        async with db.execute("SELECT COALESCE(MAX(reuse_count), 0) FROM chunks") as cur:
            row = await cur.fetchone()
        return int(row[0]) if row else 0

    # ------------------------------------------------------------------ #
    # Citations / in-degree (ADR-027 #156)                               #
    # ------------------------------------------------------------------ #

    async def record_citations(self, *, summary_sha256: str, cited_shas: Sequence[str]) -> int:
        """Record the edges (summary -> cited chunk) for one sealed
        summary. Idempotent: re-sealing the same summary replaces its
        edges so in-degree never double-counts. Returns edges written."""
        db = self._conn()
        await db.execute("DELETE FROM citations WHERE summary_sha256 = ?", (summary_sha256,))
        written = 0
        for chunk_sha in dict.fromkeys(cited_shas):
            await db.execute(
                "INSERT OR IGNORE INTO citations (summary_sha256, chunk_sha256) VALUES (?, ?)",
                (summary_sha256, chunk_sha),
            )
            written += 1
        await db.commit()
        return written

    async def in_degrees(self, shas: Sequence[str]) -> dict[str, int]:
        """In-degree (number of distinct summaries citing it) for each
        sha in ``shas``. Shas with no citations are absent from the
        result (caller treats missing as 0). Empty input -> empty dict."""
        unique = [s for s in dict.fromkeys(shas)]
        if not unique:
            return {}
        db = self._conn()
        placeholders = ",".join("?" for _ in unique)
        async with db.execute(
            f"SELECT chunk_sha256, COUNT(DISTINCT summary_sha256) FROM citations"
            f" WHERE chunk_sha256 IN ({placeholders}) GROUP BY chunk_sha256",
            tuple(unique),
        ) as cur:
            rows = await cur.fetchall()
        return {row[0]: int(row[1]) for row in rows}

    async def max_in_degree(self) -> int:
        """Max in-degree across all cited chunks (normaliser for
        ``scoring.pagerank_proxy_score``). Returns 0 if no citations."""
        db = self._conn()
        async with db.execute(
            "SELECT COALESCE(MAX(c), 0) FROM ("
            " SELECT COUNT(DISTINCT summary_sha256) AS c FROM citations GROUP BY chunk_sha256)"
        ) as cur:
            row = await cur.fetchone()
        return int(row[0]) if row else 0

    # ------------------------------------------------------------------ #
    # Embedding API (issue #119)                                         #
    # ------------------------------------------------------------------ #

    async def set_embedding(
        self,
        sha256: str,
        *,
        vector: bytes,
        model: str,
        dim: int,
    ) -> None:
        """Attach an embedding to an existing chunk row. Used by the
        chunker's auto-embed path and by ``sbw memory reembed``."""
        db = self._conn()
        await db.execute(
            "UPDATE chunks SET embedding = ?, embedding_model = ?, embedding_dim = ?"
            " WHERE sha256 = ?",
            (vector, model, dim, sha256),
        )
        await db.commit()

    async def count_embedded(self, *, model: str | None = None) -> int:
        """Count rows with a non-null embedding. Pass ``model`` to
        narrow to a specific model id (useful when migrating)."""
        db = self._conn()
        if model is None:
            sql = "SELECT COUNT(*) FROM chunks WHERE embedding IS NOT NULL"
            args: tuple = ()
        else:
            sql = "SELECT COUNT(*) FROM chunks WHERE embedding IS NOT NULL AND embedding_model = ?"
            args = (model,)
        async with db.execute(sql, args) as cur:
            row = await cur.fetchone()
        return int(row[0]) if row else 0

    async def list_unembedded(self, *, model: str, limit: int | None = None) -> list[StoredChunk]:
        """Return chunks that have no embedding for the given model.

        A chunk with an embedding from a different model counts as
        unembedded — this is how the reembed CLI walks the delta when
        ``--model`` changes."""
        db = self._conn()
        sql = (
            f"SELECT {_CHUNK_COLS}"
            " FROM chunks"
            " WHERE embedding IS NULL OR embedding_model != ?"
            " ORDER BY source_id, chunk_index"
        )
        if limit is not None:
            sql += f" LIMIT {int(limit)}"
        async with db.execute(sql, (model,)) as cur:
            rows = await cur.fetchall()
        return [_row_to_chunk(row) for row in rows]

    async def search_vector(
        self,
        query_vector: bytes,
        query_dim: int,
        *,
        model: str,
        limit: int = 50,
    ) -> list[StoredChunk]:
        """Brute-force cosine ranker over the stored embeddings.

        Loads every row with ``embedding_model = model`` into memory,
        computes cosine vs ``query_vector``, sorts descending, returns
        top-k as ``StoredChunk``. No HNSW — adequate up to ~100k chunks
        per the OpenHuman precedent (``memory_store/vectors/store.rs:313-351``).

        Args:
            query_vector: Float32 little-endian bytes from
                ``embedder.encode_vector`` (same format as stored).
            query_dim: Vector length — must match each candidate's
                ``embedding_dim`` or the row is skipped (defensive).
            model: Restrict candidates to this embedding_model so
                cosine across models doesn't happen (different model =
                different vector space, comparison is meaningless).
            limit: Top-k cap.
        """
        from wiki_memory.embedder import cosine_similarity, decode_vector

        query_vec = decode_vector(query_vector, query_dim)

        db = self._conn()
        async with db.execute(
            f"SELECT {_CHUNK_COLS},"
            " embedding, embedding_dim FROM chunks"
            " WHERE embedding IS NOT NULL AND embedding_model = ?",
            (model,),
        ) as cur:
            rows = await cur.fetchall()

        scored: list[tuple[float, StoredChunk]] = []
        for row in rows:
            cand_blob: bytes = row[7]
            cand_dim: int = row[8]
            if cand_dim != query_dim:
                # Vector-space mismatch — skip rather than crash. The
                # reembed CLI is the right tool to migrate.
                continue
            try:
                cand_vec = decode_vector(cand_blob, cand_dim)
            except ValueError:
                continue
            score = cosine_similarity(query_vec, cand_vec)
            scored.append((score, _row_to_chunk(row)))

        scored.sort(key=lambda t: t[0], reverse=True)
        return [chunk for _, chunk in scored[:limit]]

    # ------------------------------------------------------------------ #
    # Search                                                             #
    # ------------------------------------------------------------------ #

    async def search_fts(self, needle: str, *, limit: int = 50) -> list[StoredChunk]:
        """BM25-ranked full-text search over ``chunks.body`` (issue #118).

        Uses the FTS5 virtual table with porter+unicode61 tokenizer so
        morphological variants ("agent" / "agents", "observability" /
        "observabilidad") collapse. Implicit AND between tokens is the
        default — passing "Carlos Azaustre" requires both. To force OR
        upstream, pass each token separately and union the results
        (which is what ``runner.py`` already does for focal_keywords).

        Returns chunks ordered by ascending bm25 (lower = more relevant).
        Returns empty list when the sanitised query is empty."""
        sanitised = _sanitize_fts5_query(needle)
        if not sanitised:
            return []
        db = self._conn()
        async with db.execute(
            """
            SELECT c.sha256, c.source_id, c.chunk_index, c.body, c.token_count,
                   c.created_at, c.reuse_count
            FROM chunks_fts f
            JOIN chunks c ON c.rowid = f.rowid
            WHERE chunks_fts MATCH ?
            ORDER BY bm25(chunks_fts)
            LIMIT ?
            """,
            (sanitised, limit),
        ) as cur:
            rows = await cur.fetchall()
        return [_row_to_chunk(row) for row in rows]

    async def search_substring(self, needle: str, *, limit: int = 50) -> list[StoredChunk]:
        """Case-insensitive substring scan across all chunk bodies.

        v1 retrieval primitive for the MCP recall surface (PRD-004 FR-6,
        issue #78). When PRD-005 lands a real embedding index this is
        replaced by a vector search; until then the substring scan is
        cheap enough for personal-wiki cardinalities (sub-100k chunks)
        and deterministic, which matters for the eval suite.
        """
        if not needle.strip():
            return []
        db = self._conn()
        pattern = f"%{needle}%"
        async with db.execute(
            f"SELECT {_CHUNK_COLS}"
            " FROM chunks WHERE body LIKE ? COLLATE NOCASE"
            " ORDER BY source_id, chunk_index LIMIT ?",
            (pattern, limit),
        ) as cur:
            rows = await cur.fetchall()
        return [_row_to_chunk(row) for row in rows]
