"""
Tests for the embedder + vector search path (issue #119).

The Ollama embedder is exercised via a stub so the test suite stays
hermetic (no network, no model pull). The cosine + serialisation
helpers are tested directly.
"""

from __future__ import annotations

import struct
from collections.abc import Sequence

import pytest

from wiki_memory.chunker import chunk_text
from wiki_memory.embedder import (
    EmbeddingResult,
    EmbeddingUnavailableError,
    OllamaEmbedder,
    cosine_similarity,
    decode_vector,
    encode_vector,
)


class _StubEmbedder:
    """Deterministic 4-dim embedder: each text → vector seeded by its
    first 4 ord() values. Adequate for testing the storage + retrieval
    plumbing without touching Ollama."""

    DEFAULT_MODEL = "stub-4d"

    def __init__(self, *, model: str = "stub-4d") -> None:
        self._model = model

    @property
    def model(self) -> str:
        return self._model

    async def embed_one(self, text: str) -> EmbeddingResult:
        padded = (text + "    ")[:4]
        vec = [float(ord(c) % 16) for c in padded]
        return EmbeddingResult(vector=encode_vector(vec), dim=4, model=self._model)

    async def embed_batch(self, texts: Sequence[str]) -> list[EmbeddingResult]:
        return [await self.embed_one(t) for t in texts]


class TestEncodeDecodeRoundtrip:
    def test_roundtrip_preserves_values(self) -> None:
        vec = [1.0, -2.5, 3.14, 0.0, 99.9]
        blob = encode_vector(vec)
        assert decode_vector(blob, len(vec)) == pytest.approx(vec, rel=1e-6)

    def test_decode_wrong_dim_raises(self) -> None:
        blob = encode_vector([1.0, 2.0, 3.0])
        with pytest.raises(ValueError, match="size mismatch"):
            decode_vector(blob, dim=4)


class TestCosine:
    def test_identical_vectors_cosine_1(self) -> None:
        v = [1.0, 0.5, 0.25]
        assert cosine_similarity(v, v) == pytest.approx(1.0)

    def test_orthogonal_vectors_cosine_0(self) -> None:
        assert cosine_similarity([1.0, 0.0], [0.0, 1.0]) == pytest.approx(0.0)

    def test_zero_vector_returns_0(self) -> None:
        assert cosine_similarity([0.0, 0.0], [1.0, 1.0]) == 0.0

    def test_length_mismatch_raises(self) -> None:
        with pytest.raises(ValueError):
            cosine_similarity([1.0], [1.0, 0.0])


class TestEmbeddingPersistence:
    """Round-trip embeddings through ContentStore.set_embedding +
    search_vector. Uses the stub embedder so no network."""

    @pytest.mark.asyncio
    async def test_set_embedding_and_count(self, content_store) -> None:
        chunk = chunk_text("alpha beta")[0]
        await content_store.insert(source_id="s", chunk=chunk)
        assert await content_store.count_embedded() == 0

        embedder = _StubEmbedder()
        emb = await embedder.embed_one(chunk.body)
        await content_store.set_embedding(
            chunk.sha256,
            vector=emb.vector,
            model=emb.model,
            dim=emb.dim,
        )
        assert await content_store.count_embedded() == 1
        assert await content_store.count_embedded(model="stub-4d") == 1
        assert await content_store.count_embedded(model="other-model") == 0

    @pytest.mark.asyncio
    async def test_list_unembedded_filters_by_model(self, content_store) -> None:
        c1 = chunk_text("alpha")[0]
        c2 = chunk_text("beta")[0]
        await content_store.insert(source_id="s1", chunk=c1)
        await content_store.insert(source_id="s2", chunk=c2)

        embedder = _StubEmbedder(model="m-a")
        emb1 = await embedder.embed_one(c1.body)
        await content_store.set_embedding(c1.sha256, vector=emb1.vector, model="m-a", dim=4)
        # c2 has no embedding; under model='m-a' both should be unembedded
        # if we consider c1 already done. Actually c1 IS embedded for m-a.
        unembed = await content_store.list_unembedded(model="m-a")
        unembed_shas = {c.sha256 for c in unembed}
        assert c1.sha256 not in unembed_shas
        assert c2.sha256 in unembed_shas

        # Different model — c1 counts as unembedded.
        unembed_other = await content_store.list_unembedded(model="m-b")
        unembed_other_shas = {c.sha256 for c in unembed_other}
        assert c1.sha256 in unembed_other_shas

    @pytest.mark.asyncio
    async def test_search_vector_ranks_by_cosine(self, content_store) -> None:
        # Seed 3 chunks with very different stub embeddings. Query with
        # one of them; the cosine ranker must put the exact match first.
        chunks_text = ["alpha", "betas", "gamma"]
        seeded = []
        embedder = _StubEmbedder()
        for i, text in enumerate(chunks_text):
            chunk = chunk_text(text)[0]
            await content_store.insert(source_id=f"s-{i}", chunk=chunk)
            emb = await embedder.embed_one(text)
            await content_store.set_embedding(
                chunk.sha256, vector=emb.vector, model=emb.model, dim=emb.dim
            )
            seeded.append((text, chunk))

        # Query for "alpha" — same vector as the first chunk.
        query_emb = await embedder.embed_one("alpha")
        results = await content_store.search_vector(
            query_emb.vector,
            query_emb.dim,
            model="stub-4d",
            limit=3,
        )
        assert len(results) == 3
        # Top hit is the chunk whose body is "alpha".
        assert results[0].source_id == "s-0"

    @pytest.mark.asyncio
    async def test_search_vector_skips_dim_mismatch(self, content_store) -> None:
        # Seed one chunk with dim=4, query with dim=8 — must skip,
        # not crash, and return an empty list (no compatible candidates).
        chunk = chunk_text("alpha")[0]
        await content_store.insert(source_id="s", chunk=chunk)
        emb = await _StubEmbedder().embed_one("alpha")
        await content_store.set_embedding(chunk.sha256, vector=emb.vector, model="stub-4d", dim=4)

        wrong_query = struct.pack("<8f", *([1.0] * 8))
        results = await content_store.search_vector(
            wrong_query, query_dim=8, model="stub-4d", limit=10
        )
        assert results == []

    @pytest.mark.asyncio
    async def test_search_vector_filters_by_model(self, content_store) -> None:
        # Cross-model cosine is meaningless, so search_vector excludes
        # rows whose embedding_model doesn't match.
        chunk = chunk_text("alpha")[0]
        await content_store.insert(source_id="s", chunk=chunk)
        emb = await _StubEmbedder().embed_one("alpha")
        await content_store.set_embedding(chunk.sha256, vector=emb.vector, model="stub-4d", dim=4)
        # Query with a different model name — no results.
        results = await content_store.search_vector(
            emb.vector, query_dim=4, model="other-model", limit=10
        )
        assert results == []


class TestOllamaEmbedderUnavailable:
    @pytest.mark.asyncio
    async def test_unreachable_url_raises_embedding_unavailable(self, monkeypatch) -> None:
        """Patch httpx so we don't actually wait on a real socket timeout
        in CI — the contract we care about is that a connect failure
        becomes ``EmbeddingUnavailableError``, not a raw httpx exception."""
        import httpx

        class _ExplodingClient:
            def __init__(self, *args, **kwargs):
                pass

            async def __aenter__(self):
                return self

            async def __aexit__(self, *args):
                return False

            async def post(self, *args, **kwargs):
                raise httpx.ConnectError("simulated connect refused")

        monkeypatch.setattr(httpx, "AsyncClient", _ExplodingClient)

        embedder = OllamaEmbedder(base_url="http://127.0.0.1:1", timeout=1.0)
        with pytest.raises(EmbeddingUnavailableError):
            await embedder.embed_one("hello")


class TestEmbeddingMigrationOnLegacyDb:
    """Issue #119 adds nullable embedding columns to ``chunks``. A
    DB created before the migration must get the columns added by
    ``_maybe_add_embedding_columns`` on the next ``init()``."""

    @pytest.mark.asyncio
    async def test_legacy_db_gets_embedding_columns(self, tmp_path) -> None:
        import aiosqlite

        from wiki_memory.content_store import ContentStore

        db_path = tmp_path / "legacy.db"
        # Simulate pre-#119 schema (no embedding columns).
        async with aiosqlite.connect(db_path) as legacy:
            await legacy.executescript(
                """
                CREATE TABLE chunks (
                    sha256 TEXT PRIMARY KEY,
                    source_id TEXT NOT NULL,
                    chunk_index INTEGER NOT NULL,
                    body TEXT NOT NULL,
                    token_count INTEGER NOT NULL,
                    created_at TEXT NOT NULL
                );
                """
            )
            await legacy.execute(
                "INSERT INTO chunks VALUES (?, ?, ?, ?, ?, ?)",
                ("s1", "src", 0, "legacy", 1, "2026-05-29T00:00:00Z"),
            )
            await legacy.commit()

        store = ContentStore(db_path)
        await store.init()
        try:
            # The columns landed and the row survived.
            db = store._db
            async with db.execute("PRAGMA table_info(chunks)") as cur:
                cols = {row[1] for row in await cur.fetchall()}
            assert {"embedding", "embedding_model", "embedding_dim"} <= cols

            # Pre-existing row still readable.
            existing = await store.get("s1")
            assert existing is not None
            assert existing.body == "legacy"
        finally:
            await store.close()
