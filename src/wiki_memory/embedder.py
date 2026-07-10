"""
Embedder for the Memory Tree recall surface (issue #119).

Provides a thin async wrapper around Ollama's ``/api/embed`` endpoint
plus a Protocol so tests can inject stubs.

### Why Ollama (local) and not OpenAI/Cohere

Per [[ADR-018]] single-tenant deployment posture: every SBW install
runs its own infra. Ollama runs locally, no cloud cost, no per-user
billing, and the embeddings stay on disk. OpenHuman picks the same
model + transport (``bge-m3`` via Ollama, see
``memory_store/factories.rs:586-588``) for the same reasons.

### Why bge-m3 as the default

- Multilingual (en + es) — Orlando writes mostly Spanish, sources are
  mostly English. The other Ollama-shipped option (``nomic-embed-text``)
  is English-first.
- 1024-dim sentence embeddings, fits in a 4-byte float BLOB column at
  ~4 KiB per chunk.
- Matches OpenHuman so cross-project diagnostics are easier.

Orlando must run ``ollama pull bge-m3`` before the embed pipeline
fires — until then ``EmbeddingUnavailableError`` propagates from each call
and the caller (chunker, recall) degrades to non-vector paths.

### Persistence shape

Embeddings serialise as little-endian float32 BLOBs via ``numpy.tobytes()``.
A future migration to int8 quantisation (4x smaller) is straightforward
because the ``embedding_dim`` column declares the vector length.
"""

from __future__ import annotations

import logging
import struct
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Protocol

import httpx

_log = logging.getLogger(__name__)


@dataclass(frozen=True)
class EmbeddingResult:
    """One embedding plus the metadata the content_store needs to
    validate compatibility on read."""

    vector: bytes
    """Float32 little-endian byte serialisation. Use ``decode_vector()``
    to round-trip back to a list[float] when computing cosine."""

    dim: int
    """Vector length. Pinned so a model change (which changes dim)
    triggers a re-embed instead of corrupting cosine."""

    model: str
    """The exact model string used. Stored alongside each row so the
    sbw memory reembed CLI can detect drift."""


class EmbeddingUnavailableError(RuntimeError):
    """Raised when the embedder can't reach Ollama or the model isn't
    pulled. Callers catch and degrade (chunker skips embedding,
    recall falls back to FTS5)."""


class Embedder(Protocol):
    """Protocol the chunker + content_store + recall depend on. Tests
    inject ``StubEmbedder``; production uses ``OllamaEmbedder``."""

    @property
    def model(self) -> str: ...

    async def embed_one(self, text: str) -> EmbeddingResult: ...

    async def embed_batch(self, texts: Sequence[str]) -> list[EmbeddingResult]: ...


class OllamaEmbedder:
    """Calls Ollama's ``POST /api/embed`` endpoint.

    Async via httpx. Cold-start of the model adds ~2-5s to the first
    call; subsequent calls are <50ms per text on local CPU.
    """

    DEFAULT_MODEL = "bge-m3"
    DEFAULT_BASE_URL = "http://localhost:11434"

    def __init__(
        self,
        *,
        model: str | None = None,
        base_url: str | None = None,
        timeout: float = 60.0,
    ) -> None:
        self._model = model or self.DEFAULT_MODEL
        self.base_url = (base_url or self.DEFAULT_BASE_URL).rstrip("/")
        self.timeout = timeout

    @property
    def model(self) -> str:
        return self._model

    async def embed_one(self, text: str) -> EmbeddingResult:
        results = await self.embed_batch([text])
        return results[0]

    async def embed_batch(self, texts: Sequence[str]) -> list[EmbeddingResult]:
        if not texts:
            return []
        payload = {"model": self._model, "input": list(texts)}
        try:
            async with httpx.AsyncClient(timeout=self.timeout) as client:
                response = await client.post(
                    f"{self.base_url}/api/embed",
                    json=payload,
                )
        except (httpx.ConnectError, httpx.TimeoutException) as exc:
            raise EmbeddingUnavailableError(
                f"could not reach Ollama at {self.base_url}: {exc}"
            ) from exc

        if response.status_code != 200:
            body = response.text[:200]
            if "model" in body.lower() and ("not found" in body.lower() or "pull" in body.lower()):
                raise EmbeddingUnavailableError(
                    f"Ollama model {self._model!r} not pulled. Run: ollama pull {self._model}"
                )
            raise EmbeddingUnavailableError(f"Ollama returned {response.status_code}: {body}")

        data = response.json()
        embeddings = data.get("embeddings") or [data.get("embedding")]
        if not embeddings or embeddings[0] is None:
            raise EmbeddingUnavailableError(f"Ollama returned no embeddings: {data!r}")

        return [
            EmbeddingResult(
                vector=encode_vector(vec),
                dim=len(vec),
                model=self._model,
            )
            for vec in embeddings
        ]


def encode_vector(vec: Sequence[float]) -> bytes:
    """Float32 little-endian serialisation. Symmetric with
    ``decode_vector``. Numpy-free so this module stays light."""
    return struct.pack(f"<{len(vec)}f", *vec)


def decode_vector(blob: bytes, dim: int) -> list[float]:
    """Inverse of ``encode_vector``. Pass the stored ``dim`` so
    truncation/corruption surfaces as a clear error."""
    expected_bytes = dim * 4
    if len(blob) != expected_bytes:
        raise ValueError(
            f"embedding blob size mismatch: got {len(blob)} bytes for dim={dim} "
            f"(expected {expected_bytes})"
        )
    return list(struct.unpack(f"<{dim}f", blob))


def cosine_similarity(a: Sequence[float], b: Sequence[float]) -> float:
    """Pure-Python cosine. For ~500 chunks the brute-force sort takes
    <50ms and we don't want numpy as a hard runtime dep. Switch to
    numpy if a corpus grows past ~10k chunks."""
    if len(a) != len(b):
        raise ValueError(f"vector length mismatch: {len(a)} vs {len(b)}")
    dot = 0.0
    norm_a = 0.0
    norm_b = 0.0
    for x, y in zip(a, b, strict=True):
        dot += x * y
        norm_a += x * x
        norm_b += y * y
    if norm_a == 0.0 or norm_b == 0.0:
        return 0.0
    return dot / ((norm_a**0.5) * (norm_b**0.5))


__all__ = [
    "Embedder",
    "EmbeddingResult",
    "EmbeddingUnavailableError",
    "OllamaEmbedder",
    "cosine_similarity",
    "decode_vector",
    "encode_vector",
]
