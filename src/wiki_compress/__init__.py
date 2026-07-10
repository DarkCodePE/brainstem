"""
`wiki_compress` — TokenJuice-style compression pipeline (M3 Sprint 1).

Per [PRD-007 Token Compression](../../docs/PRD-007-token-compression.md) and
[ADR-012 Token-Compression Implementation](../../docs/ADR-012-token-compression-implementation.md),
this package implements the L1 rule-based compression layer. It is the
always-on, deterministic, zero-LLM-call pre-pass that runs before any
tool result is appended to an LLM context.

Public surface
--------------

- :class:`CompressionPipeline` — composable stage list
- :class:`CompressionResult` — frozen ``(body, tokens, ratio, stages)``
- :func:`build_default_pipeline` — M3 default stage order
- :mod:`wiki_compress.stages` — individual stages (importable for tests)
- :mod:`wiki_compress.metrics` — per-stage timing + delta records

Design intent (lifted verbatim from ADR-012 Option C):

- Pure Python — no Rust extensions, no model dependency.
- Grapheme-safe — never split UTF-8 mid-character.
- Idempotent — running the pipeline twice does not over-compress.
- Lossless-ish — every reduction is auditable; URL handles preserve the
  original via the in-pipeline mapping table.
- ≤ 200 LOC per stage; trivial debugging surface.
"""

from __future__ import annotations

from wiki_compress.metrics import CompressionMetrics, StageDelta, count_tokens
from wiki_compress.pipeline import (
    CompressionPipeline,
    CompressionResult,
    Stage,
    build_default_pipeline,
    build_email_pipeline,
    build_tool_output_pipeline,
)

__all__ = [
    "CompressionMetrics",
    "CompressionPipeline",
    "CompressionResult",
    "Stage",
    "StageDelta",
    "build_default_pipeline",
    "build_email_pipeline",
    "build_tool_output_pipeline",
    "count_tokens",
]
