"""Body-quality scorer contract tests (ADR-048 Fase 1, deterministic).

Fixtures mirror the SHAPE of real vault pages the 2026-06-21 scan classified, so
the scorer's verdicts stay anchored to that validated baseline.
"""

from __future__ import annotations

from wiki_synthesis.body_quality import (
    CONTRACTS,
    classify_subtype,
    parse_frontmatter,
    score_body,
)


def page(fm: str, body: str) -> str:
    return f"---\n{fm}\n---\n\n{body}"


# --------------------------------------------------------------------------- #
# Classification: OKF type wins, only Source sub-splits into repo/paper
# --------------------------------------------------------------------------- #
def test_entity_tagged_github_stays_entity() -> None:
    fm, body, raw = parse_frontmatter(
        page("type: Entity\ntags:\n  - github\n  - repo", "# X\n\nsome prose")
    )
    assert classify_subtype(fm, raw, body) == "entity"


def test_source_with_repo_tags_is_repo() -> None:
    fm, body, raw = parse_frontmatter(page("type: Source\ntags: [repo, github, codebase]", "# R"))
    assert classify_subtype(fm, raw, body) == "repo"


def test_source_with_papers_tag_is_paper() -> None:
    fm, body, raw = parse_frontmatter(page("type: Source\ntags: [ingested, papers]", "# P"))
    assert classify_subtype(fm, raw, body) == "paper"


def test_source_generic_stays_source() -> None:
    fm, body, raw = parse_frontmatter(page("type: Source\ntags: [bookmark, web]", "# S"))
    assert classify_subtype(fm, raw, body) == "source"


# --------------------------------------------------------------------------- #
# no_signal — stubs + boilerplate (the only blocking tier in ADR-048 D4)
# --------------------------------------------------------------------------- #
def test_entity_stub_is_no_signal() -> None:
    # Real shape: entities/{formbuilder,ajaxloader}.md — name-only stub.
    r = score_body(page("type: Entity\ntitle: FormBuilder", "# FormBuilder\n\nMentioned in [[x]]."))
    assert r.verdict == "no_signal"
    assert r.subtype == "entity"


def test_name_only_concept_is_no_signal() -> None:
    r = score_body(page("type: Concept\ntitle: tweenlite", "# tweenlite\n\nA thing."))
    assert r.verdict == "no_signal"


def test_zero_star_boilerplate_repo_is_no_signal() -> None:
    body = (
        "# repo\n\n## Metadata\n\n- **Stars:** 0\n\n"
        "This is a Next.js project bootstrapped with `create-next-app`.\n"
        "First, run the development server: npm run dev\n"
    )
    r = score_body(page("type: Source\ntags: [repo, github]\norigin: llm-synthesized", body))
    assert r.verdict == "no_signal"
    assert r.subtype == "repo"


# --------------------------------------------------------------------------- #
# raw_dump — body never went through synthesis
# --------------------------------------------------------------------------- #
def test_ingested_untrusted_origin_is_raw_dump() -> None:
    r = score_body(
        page(
            "type: Source\ntags: [ingested, papers]\norigin: ingested-untrusted",
            "# P\n\nfull text " * 50,
        )
    )
    assert r.verdict == "raw_dump"


def test_ingested_source_wrapper_is_raw_dump() -> None:
    body = '# P\n\n<ingested_source trust="untrusted" sha256="ab">\n# arXiv:...\nlots of text\n'
    r = score_body(page("type: Source\ntags: [papers]\norigin: llm-synthesized", body))
    assert r.verdict == "raw_dump"


def test_repo_digest_origin_is_raw_dump() -> None:
    r = score_body(page("type: Source\ntags: [repo]\norigin: repo-digest", "# r\n\n" + "x " * 300))
    assert r.verdict == "raw_dump"


def test_broken_extraction_markers_are_raw_dump() -> None:
    body = (
        "# Paper\n\n## Abstract\n\nreal abstract\n\n"
        + "prose " * 80
        + '\n"2606_images/imageFile1.png" could not be found.\n'
    )
    r = score_body(page("type: Source\ntags: [papers]\norigin: llm-synthesized", body))
    assert r.verdict == "raw_dump"


# --------------------------------------------------------------------------- #
# bloat — full-text dump as the page body
# --------------------------------------------------------------------------- #
def test_huge_body_is_bloat() -> None:
    r = score_body(
        page("type: Source\ntags: [papers]\norigin: llm-synthesized", "# P\n\n" + "word " * 12000)
    )
    assert r.verdict == "bloat"


# --------------------------------------------------------------------------- #
# weak — short, or missing required sections
# --------------------------------------------------------------------------- #
def test_repo_missing_required_sections_is_weak() -> None:
    # Has plenty of prose but no "What it is" / "Capabilities" headings.
    body = "# repo\n\n## Metadata\n\n" + "Detailed paragraph of prose about the repo. " * 20
    r = score_body(page("type: Source\ntags: [repo, github]\norigin: llm-synthesized", body))
    assert r.verdict == "weak"
    assert "required section" in " ".join(r.notes)


def test_thin_source_is_weak() -> None:
    r = score_body(
        page("type: Source\ntags: [web]\norigin: llm-synthesized", "# S\n\n" + "short. " * 20)
    )
    assert r.verdict == "weak"


# --------------------------------------------------------------------------- #
# genuine — real synthesized body with its required sections
# --------------------------------------------------------------------------- #
def test_genuine_repo_with_sections() -> None:
    body = (
        "# repo\n\n## What it is\n\n"
        + "A real synthesized description of the project. " * 8
        + "\n\n## Capabilities\n\n"
        + "It does these concrete things in detail. " * 8
    )
    r = score_body(page("type: Source\ntags: [repo, github]\norigin: llm-synthesized", body))
    assert r.verdict == "genuine"
    assert r.score >= 0.70


def test_genuine_entity_above_floor() -> None:
    body = "# Cole Medin\n\n## Overview\n\n" + "A substantive bio paragraph with real content. " * 6
    r = score_body(page("type: Entity\ntitle: Cole Medin\norigin: llm-synthesized", body))
    assert r.verdict == "genuine"


def test_score_in_unit_interval_and_notes_present() -> None:
    r = score_body(page("type: Entity\ntitle: X", "# X\n\nMentioned in [[y]]."))
    assert 0.0 <= r.score <= 1.0
    assert r.notes  # low-value verdicts always explain themselves


# --------------------------------------------------------------------------- #
# Contract table sanity
# --------------------------------------------------------------------------- #
def test_repo_and_paper_contracts_forbid_raw() -> None:
    assert CONTRACTS["repo"].forbid_raw is True
    assert CONTRACTS["paper"].forbid_raw is True
