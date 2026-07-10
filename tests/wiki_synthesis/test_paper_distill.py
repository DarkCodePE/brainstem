"""Deterministic paper distillation (ADR-048 Fase 3 / D5).

The distiller turns a full paper extraction into an
Abstract/Contributions/Results body — grounded in the paper's own headings,
no LLM — and returns None when the text has no paper shape so the caller
keeps its existing degrade behaviour.
"""

from __future__ import annotations

import textwrap

from wiki_synthesis.paper_distill import distill_paper

PAPER = textwrap.dedent("""\
    # Attention Is Not Enough

    ## Abstract

    We study the failure modes of long-context attention and show that a
    hierarchical memory beats brute-force context windows on retrieval-heavy
    tasks. Our approach reduces token cost by 20x while keeping accuracy.

    ## 1. Introduction

    Long contexts are expensive. Prior work scales the window; we scale the
    index instead, trading recompute for retrieval.

    ## 2. Contributions

    - A hierarchical memory tree with sealed summaries.
    - A deterministic distillation pass for ingest.
    - An evaluation over 879 real pages.

    ## 5. Results

    On the vault benchmark the tree reaches 76.2% genuine rate versus 52% for
    the dump baseline, at 1/20th the token cost.

    ## References

    [1] Vaswani et al. Attention is all you need.
""")


def test_distills_abstract_contributions_results():
    body = distill_paper(PAPER, rel_path="raw/papers/attn.md")
    assert body is not None
    assert "## Abstract" in body
    assert "## Contributions" in body
    assert "## Results" in body
    assert "hierarchical memory beats brute-force" in body
    assert "76.2% genuine rate" in body
    # The raw stays in the sidecar — the note says so and names it.
    assert "raw/papers/attn.md" in body
    assert "ADR-048 D5" in body
    # References (non-required sections) are NOT dragged in.
    assert "Vaswani" not in body


def test_numbered_headings_are_normalised():
    body = distill_paper(PAPER)
    assert body is not None
    # "## 2. Contributions" and "## 5. Results" matched despite numbering.
    assert "sealed summaries" in body
    assert "76.2%" in body


def test_synonym_fallbacks_conclusion_and_introduction():
    md = textwrap.dedent("""\
        ## Abstract

        A short but real abstract describing the problem, the method and the
        headline number: 3.1x faster end to end on the public benchmark suite.

        ## 1 Introduction

        We introduce a compiler pass that fuses adjacent kernels and prove it
        preserves semantics for the full operator set of the target IR.

        ## 6 Conclusions

        Kernel fusion delivers 3.1x; limitations include dynamic shapes.
    """)
    body = distill_paper(md)
    assert body is not None
    # Contributions falls back to Introduction; Results falls back to Conclusions.
    assert "compiler pass that fuses" in body
    assert "limitations include dynamic shapes" in body


def test_missing_abstract_falls_back_to_first_prose_block():
    md = (
        "# Title Only\n\n" + "This opening paragraph is substantial prose describing the paper's "
        "problem and method in enough words to serve as an abstract stand-in. " * 3
    )
    body = distill_paper(md)
    assert body is not None
    assert "## Abstract" in body
    assert "abstract stand-in" in body


def test_no_paper_shape_returns_none():
    assert distill_paper("") is None
    assert distill_paper("# Stub\n\nTiny.") is None
    # Tables/images only — no prose to distil.
    assert distill_paper("# T\n\n| a | b |\n|---|---|\n| 1 | 2 |") is None


def test_long_sections_are_clipped():
    long_results = "word " * 3000  # ~15k chars
    md = f"## Abstract\n\n{'prose ' * 60}\n\n## Results\n\n{long_results}"
    body = distill_paper(md)
    assert body is not None
    assert len(body) < 12_000
    assert "[…]" in body


def test_distilled_body_scores_above_raw_dump():
    """The point of D5: a distilled paper page must not score raw_dump/bloat."""
    from wiki_synthesis.body_quality import score_body

    body = distill_paper(PAPER, rel_path="raw/papers/attn.md")
    page = (
        "---\n"
        'title: "Attention Is Not Enough"\n'
        "date: 2026-07-08\n"
        'sources: ["raw/papers/attn.md"]\n'
        "tags: [ingested, papers]\n"
        "origin: synthesized-deterministic\n"
        "type: Source\n"
        "---\n\n"
        f"# Attention Is Not Enough\n\n{body}\n"
    )
    result = score_body(page)
    assert result.subtype == "paper"
    assert result.verdict not in ("raw_dump", "bloat", "no_signal")
