"""Tests for ``wiki_repos.synthesize`` — deterministic wiki-page composition.

Covers the contract from PRD-012 / ADR-022: a markdown source page composed
purely from the repo digest plus optional code-graph overview and an embedded
Mermaid diagram. No network and no router are used here — the deterministic
core must stand alone, and the optional LLM refinement degrades gracefully.
"""

from __future__ import annotations

import re
from datetime import UTC, datetime

import pytest

from wiki_repos.synthesize import page_path_for, repo_slug, synthesize_page
from wiki_repos.types import Digest, DigestStats, RepoRef


def _ref() -> RepoRef:
    return RepoRef(owner="octocat", repo="Hello-World")


def _digest() -> Digest:
    content = (
        "================\n"
        "FILE: README.md\n"
        "================\n"
        "# Hello World\n"
        "\n"
        "A tiny demo that greets the world. Built in Python.\n"
        "\n"
        "================\n"
        "FILE: main.py\n"
        "================\n"
        "print('hi')\n"
    )
    return Digest(
        summary="octocat/Hello-World: a tiny Python demo that greets the world.",
        tree="repo/\n  README.md\n  main.py\n",
        content=content,
        stats=DigestStats(n_files=2, n_bytes=120, est_tokens=40, truncated=False),
    )


def _graph_overview() -> dict:
    return {
        "project": "Hello-World",
        "totals": {
            "nodes": 6,
            "edges": 5,
            "contexts": 2,
            "imports": 4,
            "intra_context_imports": 3,
            "cross_context_imports": 1,
            "encapsulation_pct": 75.0,
        },
        "contexts": [
            {"name": "core", "nodes": 4, "files": 2, "functions": 2, "classes": 0},
            {"name": "cli", "nodes": 2, "files": 1, "functions": 1, "classes": 0},
        ],
        "top_hubs": [
            {"file": "src/core/engine.py", "imported_by": 3},
            {"file": "src/core/util.py", "imported_by": 1},
        ],
        "foundation_contexts": ["core"],
    }


_DIAGRAM = "```mermaid\ngraph TD\n  core --> cli\n```"


def _fixed_clock():
    return lambda: datetime(2026, 6, 3, 12, 0, 0, tzinfo=UTC)


def _frontmatter(page: str) -> str:
    m = re.match(r"^---\n(.*?)\n---\n", page, flags=re.DOTALL)
    assert m, "page must open with a YAML frontmatter block"
    return m.group(1)


# --------------------------------------------------------------------------- #
# Convenience helpers
# --------------------------------------------------------------------------- #
def test_repo_slug_reexports_ref_slug() -> None:
    assert repo_slug(_ref()) == "octocat-hello-world"


def test_page_path_for() -> None:
    assert page_path_for(_ref()) == "wiki/sources/octocat-hello-world.md"


# --------------------------------------------------------------------------- #
# Deterministic core — with full graph
# --------------------------------------------------------------------------- #
def test_frontmatter_origin_repo_digest_when_no_router() -> None:
    page = synthesize_page(
        _ref(),
        _digest(),
        graph_overview=_graph_overview(),
        diagram=_DIAGRAM,
        clock=_fixed_clock(),
    )
    fm = _frontmatter(page)
    assert "origin: repo-digest" in fm
    assert "origin: llm-synthesized" not in fm


def test_frontmatter_required_fields_present() -> None:
    page = synthesize_page(
        _ref(),
        _digest(),
        graph_overview=_graph_overview(),
        diagram=_DIAGRAM,
        clock=_fixed_clock(),
    )
    fm = _frontmatter(page)
    assert "title:" in fm
    assert "date: 2026-06-03" in fm
    assert "created_at: 2026-06-03T12:00:00+00:00" in fm
    assert '"https://github.com/octocat/Hello-World"' in fm  # sources url
    assert "category: sources" in fm
    assert "repo: octocat/Hello-World" in fm
    assert "draft_mode: showcase" in fm


def test_frontmatter_graph_available_and_tags() -> None:
    page = synthesize_page(
        _ref(),
        _digest(),
        graph_overview=_graph_overview(),
        diagram=_DIAGRAM,
        clock=_fixed_clock(),
    )
    fm = _frontmatter(page)
    assert "graph: available" in fm
    # tags must include the structural tags.
    tag_line = next(line for line in fm.splitlines() if line.startswith("tags:"))
    assert "repo" in tag_line
    assert "code" in tag_line


def test_body_contains_diagram_block_verbatim() -> None:
    page = synthesize_page(
        _ref(),
        _digest(),
        graph_overview=_graph_overview(),
        diagram=_DIAGRAM,
        clock=_fixed_clock(),
    )
    assert "## Diagram" in page
    assert _DIAGRAM in page


def test_body_lists_a_context_name_and_hub() -> None:
    page = synthesize_page(
        _ref(),
        _digest(),
        graph_overview=_graph_overview(),
        diagram=_DIAGRAM,
        clock=_fixed_clock(),
    )
    assert "## Architecture" in page
    assert "core" in page  # context name
    assert "75.0" in page  # encapsulation %
    assert "## Key modules" in page
    assert "src/core/engine.py" in page  # top hub


def test_body_uses_readme_excerpt_from_digest() -> None:
    page = synthesize_page(
        _ref(),
        _digest(),
        graph_overview=_graph_overview(),
        diagram=_DIAGRAM,
        clock=_fixed_clock(),
    )
    # the README's first prose line should surface in the "What it is" section.
    assert "greets the world" in page


def test_no_fabrication_unknown_modules_absent() -> None:
    page = synthesize_page(
        _ref(),
        _digest(),
        graph_overview=_graph_overview(),
        diagram=_DIAGRAM,
        clock=_fixed_clock(),
    )
    # only facts from the fixtures may appear.
    assert "FooBarService" not in page
    assert "invented" not in page.lower()


# --------------------------------------------------------------------------- #
# Draft-angle note (feeds the LinkedIn drafter)
# --------------------------------------------------------------------------- #
def test_showcase_draft_angle_note() -> None:
    page = synthesize_page(_ref(), _digest(), clock=_fixed_clock())
    assert "> Draft angle: showcase" in page


def test_experiential_changes_trailing_note() -> None:
    showcase = synthesize_page(_ref(), _digest(), mode="showcase", clock=_fixed_clock())
    experiential = synthesize_page(_ref(), _digest(), mode="experiential", clock=_fixed_clock())
    assert "> Draft angle: experiential" in experiential
    assert "> Draft angle: showcase" not in experiential
    assert showcase != experiential
    # the frontmatter draft_mode must track the mode too.
    assert "draft_mode: experiential" in _frontmatter(experiential)
    # experiential should be first-person framed.
    assert "my project" in experiential.lower() or "i used" in experiential.lower()


# --------------------------------------------------------------------------- #
# Degraded path — no graph overview
# --------------------------------------------------------------------------- #
def test_graph_unavailable_when_no_overview() -> None:
    page = synthesize_page(_ref(), _digest(), graph_overview=None, clock=_fixed_clock())
    fm = _frontmatter(page)
    assert "graph: unavailable" in fm


def test_degraded_architecture_note() -> None:
    page = synthesize_page(_ref(), _digest(), graph_overview=None, clock=_fixed_clock())
    assert "## Architecture" in page
    assert "code-graph unavailable" in page


def test_empty_diagram_omits_diagram_heading() -> None:
    page = synthesize_page(_ref(), _digest(), diagram="", clock=_fixed_clock())
    assert "## Diagram" not in page


# --------------------------------------------------------------------------- #
# Optional async refinement degrades gracefully
# --------------------------------------------------------------------------- #
@pytest.mark.asyncio
async def test_refine_prose_degrades_to_input_on_failure() -> None:
    from wiki_repos.synthesize import refine_prose

    base = synthesize_page(_ref(), _digest(), clock=_fixed_clock())

    class _BoomRouter:
        async def route(self, *_a, **_k):
            raise RuntimeError("router down")

    out = await refine_prose(base, _BoomRouter())
    assert out == base  # graceful degrade — original page returned untouched


@pytest.mark.asyncio
async def test_refine_prose_none_router_returns_input() -> None:
    from wiki_repos.synthesize import refine_prose

    base = synthesize_page(_ref(), _digest(), clock=_fixed_clock())
    assert await refine_prose(base, None) == base


def test_readme_excerpt_works_on_real_digest_format():
    """`_readme_excerpt` must fire on the digest's ``===== path =====`` blocks,
    not only on the ``FILE:`` fixtures (regression for the header mismatch)."""
    from wiki_repos.synthesize import _readme_excerpt, _split_file_blocks

    content = (
        "\n===== README.md =====\n"
        "Odysseus is a self-hosted AI workspace.\n"
        "It runs entirely on your machine.\n"
        "\n===== src/index.js =====\n"
        "import {run} from './core.js';\n"
    )
    blocks = dict(_split_file_blocks(content))
    assert "README.md" in blocks
    assert "src/index.js" in blocks
    excerpt = _readme_excerpt(content)
    assert "self-hosted AI workspace" in excerpt


# --------------------------------------------------------------------------- #
# Evolution & decisions section (PRD-014 / ADR-030)
# --------------------------------------------------------------------------- #
def _history(prs=1, with_fix=True):
    from wiki_repos.types import Commit, HistoryStats, PullRequest, RepoHistory

    merged = tuple(
        PullRequest(
            number=10 + i,
            title=f"Add feature {i}",
            merged_at="2026-06-01T10:00:00Z",
            author="alice",
            body_excerpt="Improves the worker loop." if i == 0 else "",
        )
        for i in range(prs)
    )
    commits = (
        (
            Commit(sha="abcdef1", summary="fix: null deref", kind="fix"),
            Commit(sha="1111111", summary="feat: add cli", kind="feat"),
        )
        if with_fix
        else (Commit(sha="2222222", summary="docs: tidy", kind="docs"),)
    )
    return RepoHistory(
        merged_prs=merged,
        commits=commits,
        stats=HistoryStats(n_prs=len(merged), n_commits=len(commits), truncated=False),
    )


def test_evolution_section_present_when_history():
    page = synthesize_page(_ref(), _digest(), history=_history(prs=2))
    assert "## Evolution & decisions" in page
    assert "#10" in page and "#11" in page
    assert "Improves the worker loop." in page
    assert "abcdef1" in page  # the fix commit is listed


def test_no_evolution_section_when_history_absent():
    page = synthesize_page(_ref(), _digest(), history=None)
    assert "## Evolution & decisions" not in page


def test_evolution_section_kind_breakdown():
    page = synthesize_page(_ref(), _digest(), history=_history(prs=1, with_fix=True))
    # kind breakdown leads with change-shaping kinds (fix/feat present).
    assert re.search(r"Recent activity across 2 commits", page)


def test_evolution_section_renders_hostile_pr_body_inert():
    """SR-2: a mined PR body with markdown injection (heading + fake link + code
    fence) must render as inert text — no new heading, no live link, no fence —
    and produce no extra write (synthesize never writes; it returns a string)."""
    from wiki_repos.types import HistoryStats, PullRequest, RepoHistory

    hostile = (
        "# Injected Heading\n[click me](http://evil.example/x)\n```js\nfetch('http://evil')\n```"
    )
    hist = RepoHistory(
        merged_prs=(
            PullRequest(
                number=99,
                title="### sneaky title",
                merged_at="2026-06-01T10:00:00Z",
                author="mallory",
                body_excerpt=hostile,
            ),
        ),
        commits=(),
        stats=HistoryStats(n_prs=1, n_commits=0, truncated=False),
    )
    page = synthesize_page(_ref(), _digest(), history=hist, clock=_fixed_clock())

    # The only legitimate '## ' headings come from the synthesizer's own sections.
    legit = {
        "## What it is",
        "## Architecture",
        "## Evolution & decisions",
        "## Key modules",
    }
    for line in page.splitlines():
        if line.startswith("## "):
            assert line in legit, f"unexpected heading injected: {line!r}"
    # The mined heading marker and the title hashes are escaped, not live markdown.
    assert "\\# Injected Heading" in page
    assert "\\#\\#\\# sneaky title" in page
    # The fake link's brackets are escaped so it is not a live link.
    assert "\\[click me\\]" in page
    # No raw triple-backtick fence leaked from the body into the page bullet.
    assert "```js" not in page


def test_readme_excerpt_skips_ascii_banner_and_badges() -> None:
    """The 'What it is' excerpt must skip ASCII-art banners, badges and image
    lines and land on the first real prose (regression: chopratejas/headroom
    surfaced its '█' banner instead of the one-line description)."""
    from wiki_repos.synthesize import _readme_excerpt

    content = (
        "===== README.md =====\n"
        "![CI](https://img.shields.io/badge/ci-passing.svg)\n"
        "██╗  ██╗███████╗ █████╗ ██████╗\n"
        "██║  ██║██╔════╝██╔══██╗██╔══██╗\n"
        "Headroom compresses everything your AI agent reads before it reaches the LLM.\n"
    )
    excerpt = _readme_excerpt(content)
    assert "compresses everything" in excerpt
    assert "█" not in excerpt
    assert "img.shields.io" not in excerpt


def test_capabilities_captures_value_headings_not_only_install() -> None:
    """`_capabilities` must capture purpose / results / how-it-works headings —
    not only 'Installation' — so ``focus=use`` posts get the real value material
    (regression: headroom's token-reduction numbers were dropped)."""
    from wiki_repos.synthesize import _capabilities

    content = (
        "===== README.md =====\n"
        "# Tool\n"
        "## Purpose\n"
        "Reduces token consumption for AI agents.\n"
        "## Token Reduction & Cost Savings\n"
        "Code search: 17,765 to 1,408 tokens (92% reduction).\n"
        "## How It Works\n"
        "Runs as a proxy with zero code changes.\n"
    )
    digest = Digest(
        summary="",
        tree="",
        content=content,
        stats=DigestStats(n_files=1, n_bytes=10, est_tokens=5, truncated=False),
    )
    section = _capabilities(digest)
    assert "Purpose" in section
    assert "92% reduction" in section
    assert "How It Works" in section


# --------------------------------------------------------------------------- #
# GitHub value prop — description_override + topics (ADR-025, 2nd decision)
# --------------------------------------------------------------------------- #
def test_description_override_leads_what_it_is() -> None:
    """The GitHub one-line value prop must lead the ``## What it is`` section so
    the page states WHAT THE TOOL DOES, not just architecture (regression:
    chopratejas/headroom's page never said it compresses tokens 60-95%)."""
    value_prop = "Compress tool outputs and reduce LLM token usage by 60-95%."
    page = synthesize_page(
        _ref(),
        _digest(),
        graph_overview=_graph_overview(),
        diagram=_DIAGRAM,
        clock=_fixed_clock(),
        description_override=value_prop,
        topics=("compression", "llm"),
    )
    # The value prop appears...
    assert value_prop in page
    # ...and it leads the What-it-is section, before the digest summary.
    what_it_is = page.split("## What it is", 1)[1].split("##", 1)[0]
    assert value_prop in what_it_is
    lead = what_it_is.lstrip("\n").splitlines()[0].strip()
    assert lead == value_prop
    # the digest summary still survives after the lead.
    assert "tiny Python demo" in what_it_is


def test_topics_merge_into_frontmatter_tags() -> None:
    """Repo topics (its own positioning keywords) must merge into the
    frontmatter ``tags``, deduped against the detected languages."""
    page = synthesize_page(
        _ref(),
        _digest(),
        graph_overview=_graph_overview(),
        clock=_fixed_clock(),
        description_override="Compress tokens 60-95%.",
        topics=("compression", "token-optimization", "llm", "PYTHON"),
    )
    fm = _frontmatter(page)
    tag_line = next(line for line in fm.splitlines() if line.startswith("tags:"))
    assert "compression" in tag_line
    assert "token-optimization" in tag_line
    assert "llm" in tag_line
    # the structural + language tags remain.
    assert "repo" in tag_line and "code" in tag_line
    # "python" is detected as a language AND a (case-folded) topic — no dup.
    assert tag_line.count("python") == 1


def test_topics_rendered_as_body_line() -> None:
    page = synthesize_page(
        _ref(),
        _digest(),
        clock=_fixed_clock(),
        topics=("compression", "rag", "mcp"),
    )
    assert "Topics: compression, rag, mcp" in page


def test_no_description_override_unchanged_behaviour() -> None:
    """Without ``description_override``/``topics`` the page is byte-identical to
    the prior digest-only output (additive change must not regress)."""
    baseline = synthesize_page(
        _ref(),
        _digest(),
        graph_overview=_graph_overview(),
        diagram=_DIAGRAM,
        clock=_fixed_clock(),
    )
    with_defaults = synthesize_page(
        _ref(),
        _digest(),
        graph_overview=_graph_overview(),
        diagram=_DIAGRAM,
        clock=_fixed_clock(),
        description_override=None,
        topics=(),
    )
    assert baseline == with_defaults
    # the digest summary still leads What-it-is when there is no override.
    what_it_is = baseline.split("## What it is", 1)[1].split("##", 1)[0]
    assert "tiny Python demo" in what_it_is
    assert "Topics:" not in baseline


def test_empty_description_override_falls_back_to_digest() -> None:
    """An empty / whitespace override must not blank the section — it falls back
    to the digest summary exactly as before."""
    page = synthesize_page(
        _ref(),
        _digest(),
        clock=_fixed_clock(),
        description_override="   ",
    )
    what_it_is = page.split("## What it is", 1)[1].split("##", 1)[0]
    assert "tiny Python demo" in what_it_is


def test_capabilities_captures_proof_table_numbers() -> None:
    """`focus=use` posts need the headline numbers. They live under a ``## Proof``
    heading in markdown *tables* that come several sections deep — so the heading
    must match AND the char budget must reach it (regression: headroom's
    17,765→1,408 / 92% savings were cut off before capture)."""
    from wiki_repos.synthesize import _capabilities

    # Front-load enough matching sections that a 1,200-char budget would never
    # reach Proof; the numbers must still survive.
    filler = "Headroom runs as a proxy with zero code changes to your app. " * 12
    content = (
        "===== README.md =====\n"
        "# Headroom\n"
        "## What it does\n"
        f"{filler}\n"
        "## How it works\n"
        f"{filler}\n"
        "## Get started\n"
        "`pip install headroom-ai`\n"
        "## Proof\n"
        "| Workload | Before | After | Savings |\n"
        "| --- | --- | --- | --- |\n"
        "| Code search | 17,765 | 1,408 | **92%** |\n"
        "| SRE debugging | 65,694 | 5,118 | **92%** |\n"
    )
    digest = Digest(
        summary="",
        tree="",
        content=content,
        stats=DigestStats(n_files=1, n_bytes=10, est_tokens=5, truncated=False),
    )
    section = _capabilities(digest)
    assert "Proof" in section
    assert "17,765" in section
    assert "92%" in section
