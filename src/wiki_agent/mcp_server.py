"""MCP server that exposes the wiki knowledge base as tools.

Wraps the eleven LangChain wiki tools plus four ``memory.tree.*``
methods (issue #78 / PRD-004 FR-8) as MCP-compliant tools so that any
MCP client -- Claude Code, Hermes Agent, Cursor, the future Tauri shell,
or a custom LangChain agent -- can search, read, recall, and seal
against the knowledge base.

Usage (stdio, for Claude Code registration)::

    python -m wiki_agent.mcp_server

Usage (SSE, for local LLM clients like Hermes/Ollama)::

    python -m wiki_agent.mcp_server --transport sse --port 8765

Environment variables::

    WIKI_ROOT                 Path to the knowledge-base directory
                              (default: ./knowledge-base)
    WIKI_MEMORY_CONTENT_DB    SQLite path for the chunk content store
                              (default: ~/.local/state/wiki_ingest/content_store.db)
    WIKI_MEMORY_TREE_DB       SQLite path for the tree_nodes store
                              (default: ~/.local/state/wiki_ingest/tree_nodes.db)

The ``memory.tree.*`` surface lazy-opens its SQLite stores on first call
so importing this module stays free (matches the rest of the server's
deferred-init pattern).
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
from pathlib import Path
from typing import Any

from mcp.server.fastmcp import FastMCP

# Publish symbols imported at module level (cheap, no heavy deps) so the
# linkedin_publish_draft tool references them as module globals — which keeps
# them monkeypatchable in tests.
from wiki_publishing import (
    LinkedInPublisher,
    PublishError,
    extract_attachments,
    extract_bullet_style,
    extract_post_body,
    format_for_linkedin,
)

# ---------------------------------------------------------------------------
# Server instance
# ---------------------------------------------------------------------------

WIKI_ROOT = os.environ.get("WIKI_ROOT", "./knowledge-base")

mcp = FastMCP(
    "brainstem",
    instructions=(
        "Brainstem — local-first knowledge backend for AI agents. Search, "
        "read, and manage an Obsidian-compatible wiki with structured "
        "articles, entities, concepts, and cross-references."
    ),
)


# ---------------------------------------------------------------------------
# Lazy tool initialization (deferred until first call)
# ---------------------------------------------------------------------------

_tools_cache: dict | None = None


def _get_tools() -> dict:
    """Build and cache the LangChain tools bound to WIKI_ROOT."""
    global _tools_cache
    if _tools_cache is None:
        from wiki_agent.tools import create_tools

        tools = create_tools(WIKI_ROOT)
        _tools_cache = {t.name: t for t in tools}
    return _tools_cache


def _call(name: str, kwargs: dict) -> str:
    """Invoke a LangChain tool by name and return its JSON result."""
    tools = _get_tools()
    if name not in tools:
        return json.dumps({"error": f"Unknown tool: {name}"})
    return tools[name].invoke(kwargs)


# ---------------------------------------------------------------------------
# Read-only tools (safe for any caller)
# ---------------------------------------------------------------------------


@mcp.tool(annotations={"readOnlyHint": True, "idempotentHint": True, "destructiveHint": False})
def search_wiki_index(query: str) -> str:
    """Search the wiki index for pages relevant to a query.

    Returns a JSON array of matching entries with page_path, title,
    summary, and category tags.  Uses term-overlap scoring against
    the master index table.

    Args:
        query: Natural language search query.
    """
    return _call("search_wiki_index", {"query": query})


@mcp.tool(annotations={"readOnlyHint": True, "idempotentHint": True, "destructiveHint": False})
def read_wiki_file(file_path: str) -> str:
    """Read a file from the knowledge base with parsed YAML frontmatter.

    Returns JSON with content, file_path, size_bytes, and parsed
    YAML frontmatter.  Path traversal outside wiki_root is blocked.

    Args:
        file_path: Path relative to wiki_root (e.g. 'wiki/concepts/llm-wiki.md').
    """
    return _call("read_wiki_file", {"file_path": file_path})


@mcp.tool(annotations={"readOnlyHint": True, "idempotentHint": True, "destructiveHint": False})
def get_wiki_stats() -> str:
    """Get wiki statistics: page counts, entity/concept/source counts,
    last ingest date, and last lint date.  Returns JSON.
    """
    return _call("get_wiki_stats", {})


@mcp.tool(annotations={"readOnlyHint": True, "idempotentHint": True, "destructiveHint": False})
def find_cross_references(page_path: str) -> str:
    """Find all pages that link to or are linked from a given page.

    Returns JSON with inbound_links and outbound_links arrays.

    Args:
        page_path: Path to the wiki page (relative to wiki_root).
    """
    return _call("find_cross_references", {"page_path": page_path})


@mcp.tool(annotations={"readOnlyHint": True, "idempotentHint": True, "destructiveHint": False})
def detect_orphan_pages() -> str:
    """Find wiki pages that have no inbound links from other pages.
    Returns a JSON array of orphan page paths.
    """
    return _call("detect_orphan_pages", {})


@mcp.tool(annotations={"readOnlyHint": True, "idempotentHint": True, "destructiveHint": False})
def validate_frontmatter(page_path: str) -> str:
    """Check that a wiki page has valid YAML frontmatter.

    Validates required fields (title, date, sources, tags, origin)
    and checks origin trust level.  Returns JSON with valid, missing_fields,
    errors, and warnings.

    Args:
        page_path: Path to the wiki page (relative to wiki_root).
    """
    return _call("validate_frontmatter", {"page_path": page_path})


# ---------------------------------------------------------------------------
# Code knowledge-graph tools (read-only; over the Understand-Anything graph)
# ---------------------------------------------------------------------------
#
# These read the deterministic codebase graph at
# src/.understand-anything/knowledge-graph.json (generated by
# scripts/ua-codegraph.sh, auto-updated on commit). They describe how SBW's
# own source is structured — distinct from the wiki tools above. Results are a
# snapshot as of the last graph regeneration; an absent graph returns a JSON
# error telling the caller how to generate it.


def _code_graph_error(exc: Exception) -> str:
    return json.dumps({"error": str(exc)})


@mcp.tool(annotations={"readOnlyHint": True, "idempotentHint": True, "destructiveHint": False})
def code_graph_overview() -> str:
    """Whole-system architecture panorama of SBW's `src/` from the code graph.

    Returns JSON with totals (nodes/edges/contexts, intra- vs cross-context
    import counts, encapsulation %), per-context sizes, the cross-context
    coupling matrix, foundation (leaf) contexts, the most-imported files
    (blast-radius hubs), and complexity hotspots (files by symbol count).
    Use before refactors or to assess DDD isolation.
    """
    from wiki_qa.codegraph import load_code_graph, overview

    try:
        return json.dumps(overview(load_code_graph()), indent=2)
    except (FileNotFoundError, ValueError, KeyError) as exc:
        return _code_graph_error(exc)


@mcp.tool(annotations={"readOnlyHint": True, "idempotentHint": True, "destructiveHint": False})
def ask_repo(repo: str, focus: str = "", max_chars: int = 8000) -> str:
    """Structured context pack for an INGESTED repo's code graph (ADR-046 Fase 3).

    Answers "what is this repo / where is X" from the deterministic code graph
    the ingest already built — ~100-250x fewer tokens than dumping files.
    Returns JSON with `pack` (a markdown bundle: architecture overview,
    cross-context coupling, key modules, and — with `focus` — matching symbols
    with file:line). For SBW's own source use the code_graph_* tools instead.

    Args:
        repo: The ingested repo — 'owner/repo', 'owner__repo', or a bare repo
            name when unambiguous. Unknown names return the available list.
        focus: Optional symbol/term to surface matching symbols in the pack.
        max_chars: Hard budget for the pack (default 8000 chars).
    """
    from pathlib import Path

    from wiki_repos.repo_context import ask_repo_pack

    result = ask_repo_pack(
        Path(WIKI_ROOT) / "repos",
        repo,
        focus=(focus.strip() or None),
        max_chars=max(500, min(max_chars, 60_000)),
    )
    return json.dumps(result, indent=2)


@mcp.tool(annotations={"readOnlyHint": True, "idempotentHint": True, "destructiveHint": False})
def code_graph_contexts() -> str:
    """List SBW's bounded contexts (wiki_*) with node/file/function/class counts.

    Returns a JSON array, largest context first. Use to discover valid
    context names for `code_graph_subgraph`.
    """
    from wiki_qa.codegraph import list_contexts, load_code_graph

    try:
        return json.dumps(list_contexts(load_code_graph()), indent=2)
    except (FileNotFoundError, ValueError, KeyError) as exc:
        return _code_graph_error(exc)


@mcp.tool(annotations={"readOnlyHint": True, "idempotentHint": True, "destructiveHint": False})
def code_graph_subgraph(context: str) -> str:
    """Inspect one bounded context's subgraph from the code graph.

    Returns JSON with the context's counts, which contexts it `depends_on`
    and is `depended_on_by` (cross-context import edges), its internal hub
    files (`imported_by` count), and its file list. Use for impact analysis
    before changing a module. An unknown context returns the known-context
    list.

    Args:
        context: Bounded-context name, e.g. 'wiki_routing' or 'wiki_agent'
            (see code_graph_contexts).
    """
    from wiki_qa.codegraph import load_code_graph, subgraph

    try:
        return json.dumps(subgraph(load_code_graph(), context), indent=2)
    except (FileNotFoundError, ValueError, KeyError) as exc:
        return _code_graph_error(exc)


@mcp.tool(annotations={"readOnlyHint": True, "idempotentHint": True, "destructiveHint": False})
def code_graph_impact(symbol: str, max_hops: int = 4) -> str:
    """Blast-radius / impact analysis for a function via the code graph's `calls` edges.

    Given a function symbol — a bare name (`create_wiki_agent`), a
    `path/to/file.py:name`, or a full `function:...` node id — returns JSON with
    the resolved node(s), the function's direct callers and callees, and its
    transitive callers within `max_hops` (the blast radius: everything that
    could be affected if you change it, each tagged with its hop distance).
    Use before editing a function to scope the change; one query replaces many
    grep/read round-trips. An ambiguous bare name resolves to all matches; an
    unknown symbol returns near-match candidates.

    Args:
        symbol: Function symbol to analyse (name, path:name, or node id).
        max_hops: Max call-chain depth for transitive callers (default 4).
    """
    from wiki_qa.codegraph import load_code_graph
    from wiki_qa.codequery import impact

    try:
        return json.dumps(impact(load_code_graph(), symbol, max_hops=max_hops), indent=2)
    except (FileNotFoundError, ValueError, KeyError) as exc:
        return _code_graph_error(exc)


# ---------------------------------------------------------------------------
# Write tools (modify the knowledge base)
# ---------------------------------------------------------------------------


@mcp.tool(annotations={"readOnlyHint": False, "destructiveHint": True, "idempotentHint": False})
def write_page(page_path: str, content: str, overwrite: bool = False) -> str:
    """Create or update a wiki page.

    The page must be inside wiki/ directory.  Content should include
    YAML frontmatter with title, date, sources, tags, and origin.

    Creating a new page under wiki/sources/ is refused when another source
    page already references the same source (prevents duplicate slugs for the
    same bookmark). Pass overwrite=true to bypass; updating an existing page
    at the same path is always allowed.

    Args:
        page_path: Destination path (e.g. 'wiki/concepts/my-topic.md').
        content: Full markdown content including frontmatter.
        overwrite: Bypass the new-source duplicate guard (default False).
    """
    return _call(
        "write_page",
        {"page_path": page_path, "content": content, "overwrite": overwrite},
    )


@mcp.tool(annotations={"readOnlyHint": False, "destructiveHint": True, "idempotentHint": True})
def update_index_entry(
    page_path: str,
    category: str,
    summary: str,
    source_count: int,
) -> str:
    """Add or update an entry in wiki/index.md.

    Args:
        page_path: Relative path to the wiki page.
        category: Category (sources, entities, concepts, answers).
        summary: One-line summary of the page.
        source_count: Number of source documents referenced.
    """
    return _call(
        "update_index_entry",
        {
            "page_path": page_path,
            "category": category,
            "summary": summary,
            "source_count": source_count,
        },
    )


@mcp.tool(annotations={"readOnlyHint": False, "destructiveHint": False, "idempotentHint": False})
def append_to_log(entry_type: str, title: str, details: str) -> str:
    """Append a timestamped entry to wiki/log.md.

    Args:
        entry_type: One of 'ingest', 'query', 'lint'.
        title: Short title for the log entry.
        details: Details including pages affected and outcome.
    """
    return _call(
        "append_to_log",
        {"entry_type": entry_type, "title": title, "details": details},
    )


@mcp.tool(
    annotations={
        "readOnlyHint": False,
        "destructiveHint": False,
        "idempotentHint": False,
        "openWorldHint": True,
    }
)
def web_clip(url: str) -> str:
    """Fetch a web article and save as markdown in raw/bookmarks/.

    Converts HTML to markdown, adds YAML frontmatter, and saves
    with a URL-derived filename.

    Args:
        url: URL of the web article to clip.
    """
    return _call("web_clip", {"url": url})


@mcp.tool(annotations={"readOnlyHint": False, "destructiveHint": False, "idempotentHint": False})
def update_schema_lessons(lesson: str) -> str:
    """Record a lesson learned in schema/wiki-schema.md.

    Enables the wiki to self-improve by tracking patterns,
    conventions, or insights discovered during operations.

    Args:
        lesson: Concise lesson as a single sentence or short paragraph.
    """
    return _call("update_schema_lessons", {"lesson": lesson})


# ---------------------------------------------------------------------------
# Inbound / repo-as-knowledge-source — ADR-022 / PRD-012 (Phase 1, public repos)
# ---------------------------------------------------------------------------


@mcp.tool(annotations={"readOnlyHint": False, "destructiveHint": False, "idempotentHint": True})
async def ingest_github_repo(
    url: str, include_diagram: bool = True, include_history: bool = True, mode: str = "showcase"
) -> str:
    """Ingest a PUBLIC GitHub repo as one synthesised wiki source page (ADR-022).

    Turns a repo URL into knowledge: fetches the repo as a tarball (NO git
    clone, sandboxed temp dir), builds a local digest + a per-repo code-graph
    (bounded contexts, hubs, cross-context coupling), renders a Mermaid
    architecture diagram, and synthesises a single
    ``wiki/sources/<owner>-<repo>.md`` page. Re-ingesting the same URL updates
    that page, never duplicates (INV-08). The page is then searchable, sealed
    into the memory tree, and can seed a post via ``linkedin_draft_from_wiki``.

    Public repos only (private repos need an OAuth scope change — a separately
    gated Phase 2). All fetched content is treated as ingested-untrusted and
    can trigger no write/publish on its own. A pass a local directory path to
    analyse a repo you already have on disk (no fetch, no network).

    Args:
        url: Public GitHub repo URL (``https://github.com/<owner>/<repo>``) or a
            local directory path (analysed in place — experiential by default).
        include_diagram: Embed a Mermaid architecture diagram (default True).
        include_history: Mine the repo's git history (merged PRs + classified
            commits) via the GitHub API — NO clone — and add an "Evolution &
            decisions" section explaining *why/how* the code evolved (default
            True; PRD-014 / ADR-030). Degrades to snapshot-only on any failure.
        mode: LinkedIn draft angle — ``"showcase"`` (third-person, external
            tool) or ``"experiential"`` (first-person, a repo you used).

    Returns:
        JSON with the page path, draft mode, whether a graph/diagram were
        produced, the bounded-context summary, and degrade notes. On failure, a
        typed JSON ``error`` (InvalidUrl / PrivateOrUnreachable / Oversize /
        FetchFailed / DigestFailed / SynthesisFailed).
    """
    from wiki_repos import errors
    from wiki_repos import ingest_github_repo as _ingest

    def _writer(page_path: str, content: str) -> str:
        # Route through the ADR-006 write_page tool so the ingested-untrusted
        # envelope + INV-08 duplicate-source guard apply at the real boundary.
        return _call("write_page", {"page_path": page_path, "content": content, "overwrite": False})

    safe_mode = "experiential" if str(mode).lower().startswith("exp") else "showcase"
    # You control WHERE a fetched repo is extracted on your machine via
    # $SBW_REPO_WORKDIR (ADR-022). Unset => system temp. Either way the extract
    # dir is removed after the run; for a persistent local repo, pass its path.
    workdir = os.environ.get("SBW_REPO_WORKDIR")
    try:
        result = await _ingest(
            url,
            include_diagram=include_diagram,
            include_history=include_history,
            mode=safe_mode,
            wiki_root=Path(WIKI_ROOT),
            write_page=_writer,
            work_root=Path(workdir) if workdir else None,
        )
    except errors.WikiRepoError as exc:
        return json.dumps({"error": exc.kind, "detail": str(exc), "url": url, "status": "failed"})
    except Exception as exc:  # noqa: BLE001 — clean message to the agent
        return json.dumps(
            {"error": type(exc).__name__, "detail": str(exc), "url": url, "status": "failed"}
        )

    return json.dumps(
        {
            "status": "ingested",
            "page_path": result.page_path,
            "mode": result.mode,
            "graph_available": result.graph_path is not None,
            "diagram_present": result.diagram_present,
            "diagram_image": result.diagram_image_path,
            "graph_summary": result.graph_summary,
            "history_present": result.history_present,
            "notes": list(result.notes),
        },
        indent=2,
    )


async def _call_paper_fn(fn: Any, *args: Any) -> Any:
    """Call a ``wiki_papers`` function whether it shipped sync or async
    (the PRD-015 contract does not pin this)."""
    import inspect

    if inspect.iscoroutinefunction(fn):
        return await fn(*args)
    result = await asyncio.to_thread(fn, *args)
    if inspect.isawaitable(result):
        return await result
    return result


@mcp.tool(
    annotations={
        "readOnlyHint": False,
        "destructiveHint": False,
        "idempotentHint": True,
        "openWorldHint": True,
    }
)
async def ingest_arxiv_paper(url_or_id: str) -> str:
    """Fetch + extract an arXiv paper into ``raw/papers/`` (PRD-015 / ADR-032).

    The arXiv mode of the paper pipeline: resolves the ID, fetches
    authoritative metadata from the arXiv Atom API (3s rate-limit, hosts
    limited to arxiv.org/export.arxiv.org per SR-1), downloads the PDF (local
    cache keyed by versionless ID), runs the ADR-032 engine chain
    (opendataloader-pdf → Docling → pypdf → metadata-only; degrade-first,
    never fails the ingest on extraction trouble), and writes the extracted
    markdown WITH the FR-4 frontmatter (``type: paper``, arxiv_id, abstract,
    ...) to ``raw/papers/<arxiv_id>.md``.

    The normal ingest route (reactive worker, ADR-035) then picks that file
    up unchanged: this tool NEVER calls ``write_page`` itself, so the ADR-006
    untrusted envelope and the ADR-015 injection guard apply to the paper
    text exactly as to any other raw drop (SR-2). Re-running with a newer
    version supersedes the existing page per ADR-028 instead of duplicating
    (FR-5: ``source_key`` = versionless arXiv ID).

    Args:
        url_or_id: arXiv ID (``2605.23904``, optionally versioned) or URL
            (``https://arxiv.org/abs/2605.23904v2``, ``.../pdf/...``).

    Returns:
        JSON with the raw sidecar path, arxiv id/version, title, and the
        ``PaperStats`` accounting (pages, extracted_chars, sections_found,
        truncated, engine_used — FR-7). On failure, a typed JSON ``error``
        with ``status: failed``.
    """
    try:
        from wiki_papers import download_pdf, extract_paper, fetch_arxiv, parse_arxiv_id
    except Exception as exc:  # noqa: BLE001 — clean message to the agent
        return json.dumps(
            {
                "error": "PapersUnavailable",
                "detail": f"wiki_papers not importable: {type(exc).__name__}",
                "status": "failed",
            }
        )

    try:
        arxiv_id, version = await _call_paper_fn(parse_arxiv_id, url_or_id)
    except Exception as exc:  # noqa: BLE001 — invalid input, typed error out
        return json.dumps(
            {
                "error": type(exc).__name__,
                "detail": str(exc),
                "url_or_id": url_or_id,
                "status": "failed",
            }
        )

    # PDF cache lives OUTSIDE raw/ so the ingest watcher never sees it.
    cache_env = os.environ.get("SBW_PAPER_CACHE")
    cache_dir = Path(cache_env) if cache_env else Path(WIKI_ROOT) / ".paper-cache"
    try:
        meta = await _call_paper_fn(fetch_arxiv, url_or_id)
        pdf_path = await _call_paper_fn(download_pdf, meta, cache_dir)
        paper = await _call_paper_fn(extract_paper, pdf_path, meta)
    except Exception as exc:  # noqa: BLE001 — clean message to the agent
        return json.dumps(
            {
                "error": type(exc).__name__,
                "detail": str(exc),
                "arxiv_id": arxiv_id,
                "status": "failed",
            }
        )

    # Same sidecar renderer the daemon pre-pass uses (lazy import: pure
    # helpers only, keeps server startup free of wiki_ingest).
    from wiki_ingest.paper_prepass import render_paper_sidecar

    frontmatter = dict(getattr(paper, "frontmatter", None) or {})
    raw_rel = f"raw/papers/{arxiv_id}.md"
    sidecar = Path(WIKI_ROOT) / raw_rel
    sidecar.parent.mkdir(parents=True, exist_ok=True)
    sidecar.write_text(
        render_paper_sidecar(frontmatter, getattr(paper, "markdown", "") or ""),
        encoding="utf-8",
    )

    stats = getattr(paper, "stats", None)
    return json.dumps(
        {
            "status": "extracted",
            "raw_path": raw_rel,
            "arxiv_id": arxiv_id,
            "arxiv_version": frontmatter.get("arxiv_version") or version,
            "title": frontmatter.get("title") or getattr(meta, "title", None),
            "stats": {
                "pages": getattr(stats, "pages", None),
                "extracted_chars": getattr(stats, "extracted_chars", None),
                "sections_found": getattr(stats, "sections_found", None),
                "truncated": getattr(stats, "truncated", None),
                "engine_used": getattr(stats, "engine_used", None),
            },
            "note": (
                "Sidecar written to raw/papers/; the reactive ingest route "
                "(ADR-035) synthesises the wiki page on its next activation. "
                "This tool never calls write_page (SR-2)."
            ),
        },
        indent=2,
    )


# ---------------------------------------------------------------------------
# Outbound / publishing — ADR-021 Phase 1 (DRAFT-ONLY)
# ---------------------------------------------------------------------------

_linkedin_generator_cache: Any | None = None


def _get_linkedin_generator() -> Any:
    """Build and cache the LinkedIn draft generator (deferred init).

    Wires the ADR-013 model router (REASONING tier for ``intent='draft'``)
    to a read-only ``WikiContentSource`` over ``WIKI_ROOT``. No write scope,
    no LinkedIn API — Phase 1 only drafts."""
    global _linkedin_generator_cache
    if _linkedin_generator_cache is None:
        from wiki_publishing import LinkedInDraftGenerator, WikiContentSource
        from wiki_routing.factory import default_router

        _linkedin_generator_cache = LinkedInDraftGenerator(
            router=default_router(),
            content_source=WikiContentSource(wiki_root=Path(WIKI_ROOT)),
            attachment_resolver=_diagram_attachments,
        )
    return _linkedin_generator_cache


def _build_cta_modifiers(
    *,
    newsletter_url: str = "",
    newsletter_proof: str = "",
    newsletter_label: str = "",
    product_name: str = "",
    product_pitch: str = "",
    product_url: str = "",
) -> dict[str, Any]:
    """Assemble the ADR-044 CTA modifier dataclasses from plain MCP string args.

    A NewsletterCTA is built only when ``newsletter_url`` is present; a ProductPS
    only when both ``product_name`` and ``product_pitch`` are present (its required
    fields). Empty strings → no modifier (mirrors the ``(x or None)`` idiom the
    draft tools use for focus). Factual values are passed VERBATIM into the
    dataclasses — the trailer builder renders them, the model never authors them.

    Per ADR-044 ("raises a caller error" AC), a *partial* product P.S. — exactly
    one of ``product_name`` / ``product_pitch`` supplied — is a caller mistake and
    raises ``ValueError`` rather than being silently dropped (silently emitting no
    P.S. would hide the error from the caller at the MCP boundary)."""
    from wiki_publishing import NewsletterCTA, ProductPS

    kwargs: dict[str, Any] = {}
    if newsletter_url.strip():
        kwargs["newsletter_cta"] = NewsletterCTA(
            url=newsletter_url.strip(),
            proof=(newsletter_proof.strip() or None),
            label=(newsletter_label.strip() or None),
        )
    has_name, has_pitch = bool(product_name.strip()), bool(product_pitch.strip())
    if has_name != has_pitch:
        raise ValueError(
            "product_ps requires BOTH product_name and product_pitch "
            "(a product P.S. cannot be built from one alone)"
        )
    if has_name and has_pitch:
        kwargs["product_ps"] = ProductPS(
            name=product_name.strip(),
            pitch=product_pitch.strip(),
            url=(product_url.strip() or None),
        )
    return kwargs


def _diagram_attachments(snippets: Any) -> list[str]:
    """Map each source page to its rendered diagram PNG (ADR-023 Phase 1), if
    one exists, so the draft surfaces it for MANUAL attach in LinkedIn's
    composer (member-image auto-upload is deferred — see ADR-023). The repo page
    ``wiki/sources/<slug>.md`` and its diagram ``assets/diagrams/<slug>.png``
    share the same slug."""
    diagrams = Path(WIKI_ROOT) / "assets" / "diagrams"
    out: list[str] = []
    for s in snippets:
        png = diagrams / f"{Path(s.page_path).stem}.png"
        if png.is_file() and str(png) not in out:
            out.append(str(png))
    return out


@mcp.tool(annotations={"readOnlyHint": False, "destructiveHint": False, "idempotentHint": False})
async def linkedin_draft_from_wiki(
    topic: str,
    max_sources: int = 3,
    post_type: str = "repo_deep_dive",
    focus: str = "",
    newsletter_url: str = "",
    newsletter_proof: str = "",
    product_name: str = "",
    product_pitch: str = "",
    product_url: str = "",
) -> str:
    """Draft a LinkedIn post from synthesised wiki content (ADR-021 Phase 1).

    Selects the ``max_sources`` most relevant *synthesised* wiki pages for
    ``topic`` (read-only), composes a single consolidated post draft via the
    model router, and saves it to ``<wiki>/outputs/linkedin/`` for you to
    review and post **manually**.

    DRAFT-ONLY: this never publishes to LinkedIn, holds no write scope, and
    composes from your own wiki synthesis — never a verbatim re-post of a
    captured third-party article. Live publishing is a separate, gated Phase 2.

    Args:
        topic: What the post should be about (a theme, not a single article).
        max_sources: How many wiki pages to consolidate into the draft (default 3).
        post_type: The post archetype (ADR-024/ADR-044) — one of "repo_deep_dive"
            (default; repo architecture for engineers), "showcase" (tool/repo
            spotlight: what it does + why it matters + link; narrative or
            benchmark-forward), "tutorial" ("how to X" with a REAL code snippet
            only), "informativo" (a short reflective take on a theme/trend, not
            repo-centred), or "explainer" (ADR-044; a structured, evidence-backed
            explainer that TEACHES a third-party concept/paper/result — hook →
            labelled number blocks → named insight → punchy closer; uses
            friendlier ➡️ bullets). It biases which wiki content is used and how
            the post is structured. Unknown values fall back to "repo_deep_dive".
        focus: The lens (ADR-024) — "use" (user/value: what it does, capabilities,
            benchmarks as benefits, when to use it) or "code" (internals:
            architecture, modules). Empty = the post type's default
            (showcase→use, repo_deep_dive→code). Pass "use" for a turbovec-style
            "what it does + why you'd use it" post.
        newsletter_url: Optional (ADR-044). A newsletter "go-deeper" link appended
            as a deterministic closing trailer. Empty = no newsletter CTA.
        newsletter_proof: Optional social proof rendered VERBATIM (e.g. "leído por
            4000+ ingenieros"). Only used when newsletter_url is set; the model
            never invents a subscriber count.
        product_name: Optional (ADR-044). With product_pitch, adds a soft-sell
            "P.D." product trailer after the body. Both required for the P.S.
        product_pitch: Optional one-line product pitch (rendered VERBATIM).
        product_url: Optional product URL appended to the P.D.

    Returns:
        JSON with the draft body, post_type, the saved file path, the source
        pages, attachments to attach by hand, and the model used. On no matching
        content, returns a JSON ``error``.
    """
    from wiki_publishing import EmptyContentError, write_draft

    try:
        generator = _get_linkedin_generator()
        modifiers = _build_cta_modifiers(
            newsletter_url=newsletter_url,
            newsletter_proof=newsletter_proof,
            product_name=product_name,
            product_pitch=product_pitch,
            product_url=product_url,
        )
        # NB: this tool runs inside FastMCP's event loop — await directly;
        # asyncio.run() here raises "cannot be called from a running loop".
        draft = await generator.generate(
            topic,
            max_sources=max_sources,
            post_type=post_type,
            focus=(focus or None),
            **modifiers,
        )
        path = write_draft(draft, wiki_root=Path(WIKI_ROOT))
    except EmptyContentError as exc:
        return json.dumps({"error": str(exc), "topic": topic, "status": "no-content"})
    except Exception as exc:  # noqa: BLE001 — surface a clean message to the agent
        return json.dumps(
            {"error": f"{type(exc).__name__}: {exc}", "topic": topic, "status": "failed"}
        )

    return json.dumps(
        {
            "status": "draft (unpublished)",
            "topic": draft.topic,
            "post_type": draft.post_type,
            "focus": draft.focus,
            "draft_path": str(path),
            "body": draft.body,
            "model": draft.model_label,
            "attachments": list(draft.attachments),
            "attach_hint": (
                "Adjunta la(s) imagen(es) listada(s) a mano al publicar."
                if draft.attachments
                else None
            ),
            "sources": [{"title": s.title, "page_path": s.page_path} for s in draft.sources],
        }
    )


def _resolve_repo_input(raw: str) -> str:
    """Map a user input to a canonical GitHub URL (ADR-025 / #154).

    Accepts a full URL, a scheme-less ``github.com/o/r``, an ``owner/repo``
    shorthand, OR a free-text name ("headroom") — the latter is resolved to the
    best-matching repo via the GitHub Search API so the user need not paste a URL.
    """
    import re

    raw = (raw or "").strip()
    if raw.startswith("http") or "github.com" in raw:
        return raw
    if re.match(r"^[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+$", raw):  # owner/repo
        return f"https://github.com/{raw}"
    from wiki_repos.github_meta import resolve_repo

    return resolve_repo(raw)  # free-text name → Search API → canonical URL


def _get_repo_context_generator(url: str) -> Any:
    """Build a LinkedIn draft generator backed by a LIVE-fetched repo (ADR-025).

    Unlike ``_get_linkedin_generator`` (which reads the ingested KB via
    ``WikiContentSource``), this wires a ``RepoContextSource`` that fetches the
    repo's metadata + README on the fly. The attachment resolver returns the
    repo's *suggested image* (README banner/screenshot, else the GitHub OG card)
    so a MANUAL-mode post has a real image to attach — the light path produces no
    architecture diagram (that's the heavy ADR-022 ingest). No write scope, no
    heavy ingest, no wiki page."""
    from wiki_publishing import LinkedInDraftGenerator, RepoContextSource
    from wiki_routing.factory import default_router

    source = RepoContextSource.from_url(url)

    def _repo_image_attachment(_snippets: Any) -> list[str]:
        try:
            img = source.suggested_image()
        except Exception:  # noqa: BLE001 — attachments are best-effort
            img = None
        return [img] if img else []

    return LinkedInDraftGenerator(
        router=default_router(),
        content_source=source,
        attachment_resolver=_repo_image_attachment,
    )


@mcp.tool(annotations={"readOnlyHint": False, "destructiveHint": False, "idempotentHint": False})
async def linkedin_draft_from_repo(
    url: str,
    post_type: str = "showcase",
    focus: str = "use",
    newsletter_url: str = "",
    newsletter_proof: str = "",
    product_name: str = "",
    product_pitch: str = "",
    product_url: str = "",
) -> str:
    """Draft a LinkedIn post about a GitHub repo — LIVE fetch, no ingest (ADR-025).

    Use this when you want to post about a repo you have NOT ingested into the
    wiki: it fetches the repo's metadata (description, topics, stars, license,
    homepage, language) and README *live* from GitHub, then composes a draft via
    the model router. Contrast with ``linkedin_draft_from_wiki``, which composes
    from your already-ingested knowledge base — use that for themes spanning
    multiple synthesised pages; use THIS for "tell people about this one repo".

    Lightweight by design: no heavy ADR-022 ingest, no code-graph, no wiki page
    is created. Draft-only (ADR-021) and transient — the sole output is the
    ``<wiki>/outputs/linkedin/`` draft file for you to review and post manually.
    Holds no LinkedIn write scope.

    Args:
        url: The repo to post about — a full URL
            (``https://github.com/<owner>/<repo>``), a scheme-less
            ``github.com/<owner>/<repo>`` or ``<owner>/<repo>`` shorthand, OR a
            free-text name like ``"headroom"`` (resolved to the best-matching
            repo via the GitHub Search API — the chosen repo is returned as
            ``resolved_repo``). So "post about headroom" works without a URL.
        post_type: The post archetype (ADR-024). Default "showcase" (tool/repo
            spotlight: what it does + why it matters + link) — the common "tell
            people about this tool" case. Other values: "repo_deep_dive",
            "tutorial", "informativo". Unknown values fall back to the drafter
            default.
        focus: The lens (ADR-024). Default "use" (user/value: what it does,
            capabilities, benchmarks as benefits) — pairs with the showcase
            default. Pass "code" for an internals/architecture angle.
        newsletter_url: Optional (ADR-044) "go-deeper" newsletter link appended as
            a deterministic closing trailer. Empty = no newsletter CTA.
        newsletter_proof: Optional social proof rendered VERBATIM (e.g. "leído por
            4000+ ingenieros"). Only used when newsletter_url is set.
        product_name: Optional (ADR-044). With product_pitch, adds a soft-sell
            "P.D." product trailer after the body. BOTH required, or it errors.
        product_pitch: Optional one-line product pitch (rendered VERBATIM).
        product_url: Optional product URL appended to the P.D.

    Returns:
        JSON with ``status``, the repo ``url``, ``post_type``, ``focus``, the
        saved ``draft_path``, the draft ``body``, and ``sources``. If the repo
        cannot be read (private/unreachable/404), returns a JSON ``error``.
    """
    from wiki_publishing import write_draft
    from wiki_repos.github_meta import RepoMetaError

    try:
        # Accept a name/owner-repo/URL; resolve free-text to a canonical URL
        # (ADR-025 / #154) so "post about headroom" works without a pasted URL.
        repo_url = _resolve_repo_input(url)
        topic = repo_url
        try:
            from wiki_repos.fetcher import parse_github_url

            ref = parse_github_url(repo_url)
            topic = f"{ref.owner}/{ref.repo}"  # clean topic/slug
        except Exception:  # noqa: BLE001 — fall back to the URL as topic
            pass
        generator = _get_repo_context_generator(repo_url)
        modifiers = _build_cta_modifiers(
            newsletter_url=newsletter_url,
            newsletter_proof=newsletter_proof,
            product_name=product_name,
            product_pitch=product_pitch,
            product_url=product_url,
        )
        # NB: runs inside FastMCP's event loop — await directly.
        draft = await generator.generate(
            topic, post_type=post_type, focus=(focus or None), **modifiers
        )
        path = write_draft(draft, wiki_root=Path(WIKI_ROOT))
    except RepoMetaError as exc:
        return json.dumps(
            {
                "error": f"repo not found/unreachable: {exc}",
                "url": url,
                "status": "repo-unreachable",
            }
        )
    except Exception as exc:  # noqa: BLE001 — surface a clean message to the agent
        return json.dumps({"error": f"{type(exc).__name__}: {exc}", "url": url, "status": "failed"})

    return json.dumps(
        {
            "status": "draft (unpublished)",
            "query": url,
            "resolved_repo": repo_url,
            "post_type": draft.post_type,
            "focus": draft.focus,
            "draft_path": str(path),
            "body": draft.body,
            "model": draft.model_label,
            "attachments": list(draft.attachments),
            "sources": [{"title": s.title, "page_path": s.page_path} for s in draft.sources],
        }
    )


@mcp.tool(annotations={"readOnlyHint": False, "destructiveHint": False, "idempotentHint": False})
async def linkedin_draft_from_url(
    url: str,
    post_type: str = "showcase",
    focus: str = "use",
    newsletter_url: str = "",
    newsletter_proof: str = "",
    product_name: str = "",
    product_pitch: str = "",
    product_url: str = "",
) -> str:
    """Draft a LinkedIn post about ANY (non-GitHub) web page — LIVE fetch (ADR-025).

    For a tool's landing page, a launch blog, an article, a docs page, etc. It
    fetches the page, converts it to markdown, and composes a draft via the model
    router — no ingest, no wiki page, draft-only (ADR-021). For a GitHub *repo*
    prefer ``linkedin_draft_from_repo`` (the GitHub API gives richer, structured
    metadata + README); use THIS for everything else on the web.

    Args:
        url: An ``http(s)://`` web page URL.
        post_type: ADR-024/ADR-044 archetype (default "showcase"). Also
            "informativo", "tutorial", "repo_deep_dive", or "explainer" (ADR-044;
            a structured third-party-concept explainer with ➡️ bullets).
        focus: ADR-024 lens (default "use"). Or "code".
        newsletter_url: Optional (ADR-044) newsletter "go-deeper" link → a
            deterministic closing trailer. Empty = no CTA.
        newsletter_proof: Optional VERBATIM social proof (used only with
            newsletter_url; the model never invents a count).
        product_name: Optional (ADR-044); with product_pitch adds a soft-sell
            "P.D." product trailer after the body (both required).
        product_pitch: Optional one-line product pitch (rendered VERBATIM).
        product_url: Optional product URL appended to the P.D.

    Returns:
        JSON with ``status``, ``url``, ``post_type``, ``focus``, ``draft_path``,
        ``body``, ``sources``. On a fetch failure, a JSON ``error``.
    """
    from wiki_publishing import (
        LinkedInDraftGenerator,
        WebContextSource,
        WebFetchError,
        write_draft,
    )
    from wiki_routing.factory import default_router

    try:
        generator = LinkedInDraftGenerator(
            router=default_router(),
            content_source=WebContextSource.from_url(url),
        )
        modifiers = _build_cta_modifiers(
            newsletter_url=newsletter_url,
            newsletter_proof=newsletter_proof,
            product_name=product_name,
            product_pitch=product_pitch,
            product_url=product_url,
        )
        draft = await generator.generate(
            url, post_type=post_type, focus=(focus or None), **modifiers
        )
        path = write_draft(draft, wiki_root=Path(WIKI_ROOT))
    except WebFetchError as exc:
        return json.dumps(
            {"error": f"could not fetch page: {exc}", "url": url, "status": "fetch-failed"}
        )
    except Exception as exc:  # noqa: BLE001 — clean message to the agent
        return json.dumps({"error": f"{type(exc).__name__}: {exc}", "url": url, "status": "failed"})

    return json.dumps(
        {
            "status": "draft (unpublished)",
            "url": url,
            "post_type": draft.post_type,
            "focus": draft.focus,
            "draft_path": str(path),
            "body": draft.body,
            "model": draft.model_label,
            "sources": [{"title": s.title, "page_path": s.page_path} for s in draft.sources],
        }
    )


_PROJECT_SUB_TYPES = ("project_launch", "project_feature", "project_weekly")


@mcp.tool(annotations={"readOnlyHint": False, "destructiveHint": False, "idempotentHint": False})
async def linkedin_draft_project_post(
    repo_path: str = ".",
    sub_type: str = "project_feature",
    topic: str = "",
    newsletter_url: str = "",
    newsletter_proof: str = "",
    product_name: str = "",
    product_pitch: str = "",
    product_url: str = "",
) -> str:
    """Draft a build-in-public LinkedIn post about YOUR OWN project (ADR-026).

    First-person "here is what I built and why", sourced from the LOCAL repo's
    git COMMIT MESSAGES + its ADR/PRD docs (never diffs). Distinct from
    ``linkedin_draft_from_repo`` (external repos via the GitHub API). Three
    sub-types:
      - "project_launch"  — the ONE-TIME project introduction (vision + main
        ADRs). Refused if this repo was already launched (use feature/weekly).
      - "project_feature" — "how/why I built <topic>": problem → decision (the
        matched ADR) → measured numbers (only if present) → lessons. Pass the
        feature name as ``topic``.
      - "project_weekly"  — a build-log of the last ~2 weeks of real commits.

    Draft-only (ADR-021), local-only ($0, no scope). Numbers/features come ONLY
    from the commits/ADRs — never invented.

    Args:
        repo_path: Local path to your project repo (default ".", the cwd repo).
        sub_type: One of project_launch | project_feature | project_weekly.
        topic: For project_feature, the feature/ADR to write about.
        newsletter_url: Optional (ADR-044) "go-deeper" newsletter link appended as
            a deterministic closing trailer. Empty = no newsletter CTA.
        newsletter_proof: Optional social proof rendered VERBATIM. Only used when
            newsletter_url is set.
        product_name: Optional (ADR-044). With product_pitch, adds a soft-sell
            "P.D." product trailer after the body. BOTH required, or it errors.
        product_pitch: Optional one-line product pitch (rendered VERBATIM).
        product_url: Optional product URL appended to the P.D.

    Returns:
        JSON with status, sub_type, draft_path, body. A repeat launch returns a
        redirect message; a fetch/source failure returns a JSON error.
    """
    from datetime import UTC, datetime
    from pathlib import Path as _Path

    from wiki_publishing import (
        LinkedInDraftGenerator,
        ProjectContextSource,
        ProjectLedger,
        write_draft,
    )
    from wiki_routing.factory import default_router

    sub = sub_type if sub_type in _PROJECT_SUB_TYPES else "project_feature"
    repo = _Path(repo_path).expanduser().resolve()
    repo_slug = repo.name or "project"
    ledger = ProjectLedger(_Path(WIKI_ROOT) / "outputs" / "linkedin" / ".project_ledger.json")

    # Lifecycle: a project is launched ONCE (ADR-026). A repeat launch is
    # redirected to an update rather than re-announcing.
    if sub == "project_launch" and ledger.was_launched(repo_slug):
        return json.dumps(
            {
                "status": "launch-already-done",
                "repo": repo_slug,
                "hint": "Este proyecto ya fue lanzado. Usa sub_type=project_feature "
                "(una feature concreta) o project_weekly (resumen) para un update.",
            }
        )

    try:
        source = ProjectContextSource.from_repo(repo, sub, topic=(topic or None))
        generator = LinkedInDraftGenerator(router=default_router(), content_source=source)
        modifiers = _build_cta_modifiers(
            newsletter_url=newsletter_url,
            newsletter_proof=newsletter_proof,
            product_name=product_name,
            product_pitch=product_pitch,
            product_url=product_url,
        )
        draft = await generator.generate(repo_slug, post_type=sub, focus="use", **modifiers)
        path = write_draft(draft, wiki_root=_Path(WIKI_ROOT))
    except Exception as exc:  # noqa: BLE001 — clean message to the agent
        return json.dumps(
            {"error": f"{type(exc).__name__}: {exc}", "repo": repo_slug, "status": "failed"}
        )

    now = datetime.now(UTC).isoformat()
    if sub == "project_launch":
        ledger.record_launch(repo_slug, when=now)
    else:
        ledger.record_post(repo_slug, when=now)

    return json.dumps(
        {
            "status": "draft (unpublished)",
            "repo": repo_slug,
            "sub_type": sub,
            "post_type": draft.post_type,
            "draft_path": str(path),
            "body": draft.body,
            "model": draft.model_label,
        }
    )


# Typed phrase the human must echo to authorise a write. Not a one-tap button
# (ADR-021 publish-approval requirement, anti-rubber-stamp).
_LINKEDIN_CONFIRM_PHRASE = "PUBLICAR"
# Echo this instead to get the post ready to publish BY HAND (so you can attach
# the diagram image, which the LinkedIn API cannot upload). Nothing is sent.
_LINKEDIN_MANUAL_PHRASE = "MANUAL"

# Typed phrase for the post-it-yourself path: hand back the clean body + the
# image to attach by hand, and never call the API (the API can't upload images
# and a live publish would leak the local PNG path into the post). ADR-021.
_LINKEDIN_MANUAL_PHRASE = "MANUAL"


@mcp.tool(annotations={"readOnlyHint": False, "destructiveHint": True, "idempotentHint": False})
async def linkedin_publish_draft(draft_path: str, confirm: str = "") -> str:
    """PUBLISH a reviewed draft to LinkedIn — live, or hand it back to post manually.

    A LIVE publish posts TEXT ONLY: the LinkedIn API (via Composio) has no media
    upload, and an API ``lifecycleState='DRAFT'`` is orphaned/invisible, so the
    ONLY way to include the diagram image is to post by hand (the MANUAL path).

    Three outcomes, chosen by ``confirm``:
      - ``confirm=""``   → returns the EXACT body + char count + any suggested
        image, and asks how to proceed. Nothing is sent.
      - ``confirm="MANUAL"`` → returns the clean body + image path and does NOT
        publish; you paste it into LinkedIn and attach the image yourself.
      - ``confirm="PUBLICAR"`` → publishes the post LIVE and PUBLIC (text only,
        no image).

    Post-it-yourself path:
      Call with ``confirm="MANUAL"`` → returns the clean body + the image
      path to attach by hand and publishes NOTHING. Use this to post WITH an
      image (the API cannot upload media).

    Args:
        draft_path: Path to a draft ``.md`` previously written under the vault's
            ``outputs/linkedin/`` (the file ``linkedin_draft_from_wiki`` saved).
        confirm: ``"PUBLICAR"`` to publish live; ``"MANUAL"`` for the
            post-it-yourself path; anything else returns ``confirm-required``.

    Returns:
        JSON. ``status="confirm-required"`` with the full body on step 1;
        ``status="manual-ready"`` with body + ``attachments`` for the MANUAL
        path; ``status="published-on-linkedin"`` on success; ``"rejected"`` for
        a bad path/body; ``"failed"`` on a backend error (local draft preserved).
    """
    # Traversal-guard: the target must resolve to a file under outputs/linkedin.
    out_dir = (Path(WIKI_ROOT) / "outputs" / "linkedin").resolve()
    candidate = Path(draft_path)
    resolved = (candidate if candidate.is_absolute() else out_dir / candidate.name).resolve()
    try:
        resolved.relative_to(out_dir)
    except ValueError:
        return json.dumps(
            {"error": "draft_path must be a file under outputs/linkedin/", "status": "rejected"}
        )
    if not resolved.is_file():
        return json.dumps({"error": f"draft not found: {resolved.name}", "status": "rejected"})

    raw = resolved.read_text(encoding="utf-8")
    # LinkedIn renders NO markdown, so the body must be flattened to LinkedIn-ready
    # text (Unicode bold, • bullets, scheme-qualified links, no #/`/**) BEFORE it is
    # either previewed or published — otherwise literal markup leaks into the live
    # post. Applied once here so every branch (manual-ready, confirm-required, and
    # the live publish) shows and sends exactly the same flattened body.
    # ADR-044: honour the draft's recorded bullet_style (arrow for explainer);
    # absent frontmatter line → "dot" → byte-identical to pre-ADR-044 output.
    body = format_for_linkedin(extract_post_body(raw), bullet_style=extract_bullet_style(raw))
    if not body:
        return json.dumps({"error": "draft has no post body", "status": "rejected"})

    # Image(s) the user attaches by hand — never published via the API (it
    # can't upload media and would leak the local path into the post body).
    attachments = extract_attachments(raw)

    confirm_phrase = confirm.strip()

    # Post-it-yourself path: hand back the clean body + image path, never
    # touch the publisher. Lets the user paste the body into LinkedIn and
    # drag the image in manually (ADR-021).
    if confirm_phrase == _LINKEDIN_MANUAL_PHRASE:
        return json.dumps(
            {
                "status": "manual-ready",
                "draft_path": str(resolved),
                "char_count": len(body),
                "body": body,
                "attachments": attachments,
                "instruction": (
                    "Copy the body above into LinkedIn's composer"
                    + (f" and attach the image manually: {attachments[0]}." if attachments else ".")
                    + " Nothing was published — this is the post-it-yourself path."
                ),
            }
        )

    if confirm_phrase != _LINKEDIN_CONFIRM_PHRASE:
        return json.dumps(
            {
                "status": "manual-ready",
                "action": "Post this yourself in LinkedIn (nothing was published).",
                "draft_path": str(resolved),
                "char_count": len(body),
                "body": body,
                "attachments": attachments,
                "instruction": (
                    "Review the full body above. This will PUBLISH a live, public "
                    "post on your LinkedIn. To publish, call linkedin_publish_draft "
                    f'again with confirm="{_LINKEDIN_CONFIRM_PHRASE}". '
                    + (
                        "NOTE: the API cannot upload images, so the suggested image "
                        "will NOT be attached — to post WITH the image, call again "
                        f'with confirm="{_LINKEDIN_MANUAL_PHRASE}" and attach it by hand. '
                        if attachments
                        else ""
                    )
                    + "Nothing is sent until you do."
                ),
            }
        )

    try:
        from wiki_integrations.composio_bridge import ComposioBridge

        publisher = LinkedInPublisher(executor=ComposioBridge())
        result = await publisher.publish_draft(body=body, draft_path=str(resolved))
    except PublishError as exc:
        return json.dumps({"error": str(exc), "status": "failed", "draft_path": str(resolved)})
    except Exception as exc:  # noqa: BLE001 — surface a clean message; local draft is preserved
        return json.dumps(
            {
                "error": f"{type(exc).__name__}: {exc}",
                "status": "failed",
                "draft_path": str(resolved),
            }
        )

    # Audit the write (ADR-009/ADR-017). Best-effort — never fail the action on a log error.
    try:
        from wiki_core.secrets import AuditLog, policy_for

        AuditLog().write(
            event="execute_action",
            provider="linkedin",
            action="LINKEDIN_CREATE_LINKED_IN_POST",
            params={"lifecycleState": "PUBLISHED", "draft_path": str(resolved)},
            result="ok",
            scope_used=policy_for("linkedin").default,
        )
    except Exception:  # noqa: BLE001
        pass

    return json.dumps(
        {
            "status": result.status,
            "lifecycle_state": result.lifecycle_state,
            "author_urn": result.author_urn,
            "post_id": result.post_id,
            "draft_path": result.draft_path,
            "note": (
                "Published LIVE on your LinkedIn (TEXT ONLY — the API can't attach "
                "images). To include the diagram, edit the post in LinkedIn and add "
                'it by hand, or next time use confirm="MANUAL" to post with the image.'
            ),
        }
    )


# ---------------------------------------------------------------------------
# memory.tree.* surface (issue #78 / PRD-004 FR-8)
# ---------------------------------------------------------------------------
#
# Four async methods exposing the Memory Tree substrate to MCP clients.
# Stores open lazily on first call so module import stays free; the
# stack is reused across calls within a single server process. The
# server holds an asyncio lock around init to avoid double-open on
# concurrent first calls.

_DEFAULT_CONTENT_DB = "~/.local/state/wiki_ingest/content_store.db"
_DEFAULT_TREE_DB = "~/.local/state/wiki_ingest/tree_nodes.db"
_DEFAULT_VAULT_ROOT = "./knowledge-base"


def _content_db_path() -> Path:
    return Path(os.environ.get("WIKI_MEMORY_CONTENT_DB", _DEFAULT_CONTENT_DB)).expanduser()


def _tree_db_path() -> Path:
    return Path(os.environ.get("WIKI_MEMORY_TREE_DB", _DEFAULT_TREE_DB)).expanduser()


def _vault_root_path() -> Path:
    return Path(os.environ.get("WIKI_ROOT", _DEFAULT_VAULT_ROOT)).expanduser()


_memory_stack: dict[str, Any] | None = None
_memory_init_lock: asyncio.Lock | None = None


async def _get_memory_stack() -> dict[str, Any]:
    """Lazy-open the (ContentStore, TreeNodeStore, SealWorker) triple.

    Stores are opened against the env-configured DB paths and the seal
    worker uses `build_default_summariser()` so it honours
    ~/.sbw/config.toml (DeepSeek FAST + telemetry) when provider keys
    are present, NullSummariser otherwise.

    The write_sink writes summary markdown directly to disk under
    ``WIKI_ROOT/wiki/trees/`` to match the daemon's seal-on-ingest hook
    behaviour (see wiki_ingest.composition.build_seal_on_ingest_hook).
    """
    global _memory_stack, _memory_init_lock
    if _memory_stack is not None:
        return _memory_stack

    if _memory_init_lock is None:
        _memory_init_lock = asyncio.Lock()

    async with _memory_init_lock:
        if _memory_stack is not None:
            return _memory_stack

        from wiki_agent.write_sink import LocalWriteSink
        from wiki_memory import build_default_seal_worker
        from wiki_memory.content_store import ContentStore
        from wiki_memory.tree_nodes import TreeNodeStore

        content_store = ContentStore(_content_db_path())
        tree_store = TreeNodeStore(_tree_db_path())
        await content_store.init()
        await tree_store.init()

        vault_root = _vault_root_path()

        async def _file_write(page_path: str, content: str) -> str:
            full = vault_root / page_path
            full.parent.mkdir(parents=True, exist_ok=True)
            full.write_text(content, encoding="utf-8")
            return page_path

        async def _file_log(entry_type: str, title: str, details: str) -> str:
            log_path = vault_root / "wiki" / "log.md"
            log_path.parent.mkdir(parents=True, exist_ok=True)
            with log_path.open("a", encoding="utf-8") as f:
                f.write(f"\n## {entry_type} — {title}\n{details}\n")
            return "ok"

        write_sink = LocalWriteSink(_file_write, _file_log)
        seal_worker = build_default_seal_worker(
            content_store=content_store,
            tree_store=tree_store,
            write_sink=write_sink,
        )
        _memory_stack = {
            "content_store": content_store,
            "tree_store": tree_store,
            "seal_worker": seal_worker,
        }
        return _memory_stack


def _err(message: str, **extra: Any) -> str:
    """Uniform error envelope so MCP clients can dispatch on a stable
    JSON shape instead of parsing free-text."""
    payload: dict[str, Any] = {"error": message}
    payload.update(extra)
    return json.dumps(payload)


_RECALL_MODES = ("substring", "fts", "vector", "hybrid")


async def _dispatch_query_mode(content_store, query: str, mode: str, limit: int):
    """Route a query to the configured retrieval primitive.

    ``vector`` and ``hybrid`` require the Ollama embedder to be live;
    on failure they degrade to FTS5 (best-effort recall instead of
    crashing the caller). The downgrade is logged via the JSON envelope
    so eval harnesses can detect it.
    """
    from wiki_memory.embedder import EmbeddingUnavailableError

    if mode == "substring":
        return await content_store.search_substring(query, limit=limit)
    if mode == "fts":
        return await content_store.search_fts(query, limit=limit)

    embedder = _get_embedder()
    try:
        emb = await embedder.embed_one(query)
    except EmbeddingUnavailableError:
        # Graceful degradation — vector/hybrid both fall back to FTS5
        # rather than failing the recall outright. The caller sees
        # results, just not vector-ranked.
        return await content_store.search_fts(query, limit=limit)

    vector_hits = await content_store.search_vector(
        emb.vector,
        emb.dim,
        model=emb.model,
        limit=limit,
    )

    if mode == "vector":
        return vector_hits

    # mode == 'hybrid' — union vector + FTS, deduped by sha. Vector
    # results dominate (kept first) since they capture semantic intent
    # the FTS BM25 misses; FTS fills the rest of the limit budget.
    fts_hits = await content_store.search_fts(query, limit=limit)
    seen: dict[str, object] = {}
    for c in [*vector_hits, *fts_hits]:
        if c.sha256 not in seen:
            seen[c.sha256] = c
        if len(seen) >= limit:
            break
    return list(seen.values())


# Lazy embedder singleton — same lifecycle pattern as the memory stack.
# Ollama cold-start ~2-5s the first time, then warm. Reuse the client.
_embedder = None


def _get_embedder():
    global _embedder
    if _embedder is None:
        from wiki_memory.embedder import OllamaEmbedder

        model = os.environ.get("WIKI_EMBEDDING_MODEL", OllamaEmbedder.DEFAULT_MODEL)
        base_url = os.environ.get("OLLAMA_BASE_URL", OllamaEmbedder.DEFAULT_BASE_URL)
        _embedder = OllamaEmbedder(model=model, base_url=base_url)
    return _embedder


@mcp.tool(
    name="memory_tree_recall",
    annotations={"readOnlyHint": True, "idempotentHint": True, "destructiveHint": False},
)
async def memory_tree_recall(
    query: str | None = None,
    source_id: str | None = None,
    token_budget: int = 4000,
    limit: int = 50,
    mode: str = "fts",
    include_superseded: bool = False,
) -> str:
    """Recall leaf chunks from the memory tree, token-budgeted.

    Behaviour:
    - ``source_id`` only: returns that source's chunks in chunk-index order.
    - ``query`` only: ranks chunks by relevance (``mode`` selects the
      ranker — ``fts`` uses BM25 over the FTS5 index with porter+unicode61
      stemming, ``substring`` falls back to case-insensitive LIKE), then
      token-budgeted greedy pack.
    - both: scope by source_id and then substring-filter inside (the FTS
      index is global so ``mode`` is ignored when source_id is set).
    - neither: error — refuse to do an unscoped recall to protect token
      budgets and force the caller to be explicit.

    Returns JSON ``{chunks: [...], total_tokens: int, truncated: bool}``
    where each chunk is ``{sha256, source_id, chunk_index, body, token_count}``.
    Errors return ``{error: str, ...}``.

    Args:
        query: Optional query string (FTS5 tokens for ``mode='fts'``,
            substring for ``mode='substring'``).
        source_id: Optional source identifier to scope to.
        token_budget: Hard cap on the sum of returned ``token_count``.
        limit: Max chunks to consider before token packing (default 50).
        mode: ``'fts'`` (default, BM25 ranked) or ``'substring'`` (legacy
            LIKE scan, kept for parity with pre-#118 callers).
    """
    from wiki_memory.recall import build_chunk_scores, recall_leaves

    if query is None and source_id is None:
        return _err(
            "memory_tree_recall requires query or source_id",
            hint="pass at least one of query/source_id to scope the recall",
        )
    if token_budget <= 0:
        return _err("token_budget must be > 0", token_budget=token_budget)
    if mode not in _RECALL_MODES:
        return _err(
            f"mode must be one of {_RECALL_MODES}",
            received=mode,
        )

    try:
        stack = await _get_memory_stack()
    except Exception as exc:  # noqa: BLE001
        return _err(f"failed to open memory stores: {exc}", error_class=type(exc).__name__)

    content_store = stack["content_store"]
    tree_store = stack["tree_store"]
    candidates = []
    if source_id is not None:
        candidates = await content_store.list_by_source(source_id)
        if query is not None:
            needle = query.casefold()
            candidates = [c for c in candidates if needle in c.body.casefold()]
    else:
        # query is not None per the guard above
        candidates = await _dispatch_query_mode(content_store, query, mode, limit)

    # ADR-028 #158: drop chunks belonging to superseded source versions
    # unless the caller explicitly opts in. A re-ingested document's stale
    # version stays on disk but is out of default recall.
    if not include_superseded:
        try:
            superseded = set(await tree_store.superseded_node_ids())
        except Exception:  # noqa: BLE001
            superseded = set()
        if superseded:
            candidates = [c for c in candidates if c.source_id not in superseded]

    # ADR-027 #155/#156/#157: score candidates by recency × reuse ×
    # pagerank-proxy, then let recall_leaves pack the most relevant under
    # the budget (and present them in chunk_index order). When source_id
    # is set we keep document order (scores=None) — a single source reads
    # best in written order.
    scores = None
    if source_id is None and candidates:
        shas = [c.sha256 for c in candidates]
        try:
            in_degrees = await content_store.in_degrees(shas)
            max_reuse = await content_store.max_reuse()
            max_in_degree = await content_store.max_in_degree()
            scores = build_chunk_scores(
                candidates,
                max_reuse=max_reuse,
                in_degrees=in_degrees,
                max_in_degree=max_in_degree,
            )
        except Exception:  # noqa: BLE001
            scores = None  # degrade to index-order packing on any error

    bundle = recall_leaves(candidates, token_budget=token_budget, scores=scores)

    # ADR-027 #155: record that these chunks were surfaced so the reuse
    # signal accumulates. Best-effort — a failed increment must not fail
    # the recall.
    if bundle.chunks:
        try:
            await content_store.increment_reuse([c.sha256 for c in bundle.chunks])
        except Exception:  # noqa: BLE001
            pass

    return json.dumps(
        {
            "chunks": [
                {
                    "sha256": c.sha256,
                    "source_id": c.source_id,
                    "chunk_index": c.chunk_index,
                    "body": c.body,
                    "token_count": c.token_count,
                }
                for c in bundle.chunks
            ],
            "total_tokens": bundle.total_tokens,
            "truncated": bundle.truncated,
        }
    )


@mcp.tool(
    name="memory_tree_seal_now",
    annotations={
        "readOnlyHint": False,
        "destructiveHint": False,
        "idempotentHint": False,
        "openWorldHint": True,
    },
)
async def memory_tree_seal_now(source_id: str, node_id: str | None = None) -> str:
    """Force a seal for a source's chunks now.

    Builds the parent summary via the default summariser (LLM-backed
    when provider keys are present, NullSummariser otherwise per
    ``build_default_summariser``), writes the summary markdown into the
    vault under ``wiki/trees/sources/<node_id>.md``, and marks the
    source's tree_nodes row sealed.

    If ``node_id`` is omitted the source's existing node row is looked
    up by ``source_id``; the source must already have a tree_nodes row
    (one is normally created by the seal-on-ingest hook). Pass an
    explicit ``node_id`` when seeding from outside the daemon flow
    (eval suite, smoke tests).

    Args:
        source_id: Source identifier whose chunks are summarised.
        node_id: Optional tree node id (defaults to source_id).
    """
    try:
        stack = await _get_memory_stack()
    except Exception as exc:  # noqa: BLE001
        return _err(f"failed to open memory stores: {exc}", error_class=type(exc).__name__)

    target_node_id = node_id or source_id
    tree_store = stack["tree_store"]
    seal_worker = stack["seal_worker"]

    # Ensure a node row exists so mark_sealed has a row to update.
    existing = await tree_store.get(target_node_id)
    if existing is None:
        await tree_store.create_source_node(node_id=target_node_id)

    try:
        result = await seal_worker.seal_source(source_id=source_id, node_id=target_node_id)
    except Exception as exc:  # noqa: BLE001
        return _err(
            f"seal failed: {exc}",
            error_class=type(exc).__name__,
            source_id=source_id,
            node_id=target_node_id,
        )

    return json.dumps(
        {
            "node_id": result.node_id,
            "summary_sha256": result.summary_sha256,
            "parent_token_count": result.parent_token_count,
            "children_count": result.children_count,
            "page_path": result.page_path,
        }
    )


@mcp.tool(
    name="memory_tree_list_topics",
    annotations={"readOnlyHint": True, "idempotentHint": True, "destructiveHint": False},
)
async def memory_tree_list_topics(
    kind: str = "topic",
    include_tombstoned: bool = False,
) -> str:
    """List nodes in the memory tree.

    Default kind is ``topic`` per the PRD-004 FR-8 name, but ``source``
    and ``global`` are accepted because the same surface is the natural
    place to inspect any tree slice. Tombstoned rows are excluded
    unless ``include_tombstoned`` is true.

    Args:
        kind: One of ``source``, ``topic``, ``global`` (default: ``topic``).
        include_tombstoned: Surface soft-deleted rows too (default: false).
    """
    if kind not in {"source", "topic", "global"}:
        return _err(
            "kind must be one of source, topic, global",
            received=kind,
        )

    try:
        stack = await _get_memory_stack()
    except Exception as exc:  # noqa: BLE001
        return _err(f"failed to open memory stores: {exc}", error_class=type(exc).__name__)

    tree_store = stack["tree_store"]
    nodes = await tree_store.list_by_kind(kind, include_tombstoned=include_tombstoned)  # type: ignore[arg-type]
    return json.dumps(
        {
            "kind": kind,
            "include_tombstoned": include_tombstoned,
            "count": len(nodes),
            "nodes": [
                {
                    "node_id": n.node_id,
                    "kind": n.kind,
                    "parent_id": n.parent_id,
                    "level": n.level,
                    "summary_sha256": n.summary_sha256,
                    "score": n.score,
                    "sealed_at": n.sealed_at,
                    "tombstoned": n.tombstoned,
                    "created_at": n.created_at,
                }
                for n in nodes
            ],
        }
    )


@mcp.tool(
    name="memory_tree_tombstone",
    annotations={"readOnlyHint": False, "destructiveHint": True, "idempotentHint": True},
)
async def memory_tree_tombstone(node_id: str) -> str:
    """Tombstone a memory tree node AND remove its chunks from recall
    (ADR-029 #159 — a hard delete, as the name implies).

    Before ADR-029 this only flagged the node, leaving its chunks fully
    recallable via FTS/vector/substring — an explicit "forget" that
    didn't forget. Now it deletes the source's chunks from the content
    store as well, so forgotten content leaves the recall surface.

    Order: delete chunks first, then flag the node, to minimise the
    two-DB partial-failure window. Both outcomes are surfaced in the
    result (``node_tombstoned`` + ``chunks_deleted``; ``changed`` mirrors
    ``node_tombstoned`` for backwards compatibility). If the node flag
    fails after the chunks were deleted, the partial state is reported
    explicitly rather than half-completing silently. Idempotent: an
    already-tombstoned node returns ``changed: false`` and (since its
    chunks are already gone) ``chunks_deleted: 0``.

    For a *reversible* hide, use re-ingest supersession (ADR-028
    ``is_latest``) instead — tombstone is destructive.

    Args:
        node_id: The tree node id to tombstone (typically the source_id
            for source-level nodes; chunks are keyed by source_id == node_id).
    """
    try:
        stack = await _get_memory_stack()
    except Exception as exc:  # noqa: BLE001
        return _err(f"failed to open memory stores: {exc}", error_class=type(exc).__name__)

    tree_store = stack["tree_store"]
    content_store = stack["content_store"]

    # Delete chunks first (source_id == node_id for source nodes). If the
    # node flag fails afterwards we'd rather have orphaned-but-gone chunks
    # than a "tombstoned" node whose content still surfaces in recall.
    try:
        chunks_deleted = await content_store.delete_by_source(node_id)
    except Exception as exc:  # noqa: BLE001
        return _err(
            f"failed to delete chunks for {node_id}: {exc}",
            error_class=type(exc).__name__,
            node_id=node_id,
        )

    try:
        changed = await tree_store.tombstone(node_id)
    except Exception as exc:  # noqa: BLE001
        # Partial state (ADR-029): the chunks are gone from recall but the
        # node flag failed. Report it explicitly — never half-complete
        # silently — so the caller can retry the flag.
        return _err(
            f"chunks deleted but node flag failed for {node_id}: {exc}",
            error_class=type(exc).__name__,
            node_id=node_id,
            node_tombstoned=False,
            chunks_deleted=chunks_deleted,
        )
    return json.dumps(
        {
            "node_id": node_id,
            "changed": changed,
            "node_tombstoned": changed,
            "chunks_deleted": chunks_deleted,
        }
    )


# ---------------------------------------------------------------------------
# Read-only mode (ADR-034 D5)
# ---------------------------------------------------------------------------

#: Pure-read surfaces. Everything else (write_page, index/log/schema updates,
#: web_clip, repo ingest, memory-tree mutation, the LinkedIn draft/publish
#: family) is withheld when WIKI_MCP_READONLY is set, so an external pipeline
#: (e.g. auto-publish-social) can consume the vault without holding write or
#: publish capabilities.
READONLY_TOOLS: frozenset[str] = frozenset(
    {
        "search_wiki_index",
        "read_wiki_file",
        "validate_frontmatter",
        "get_wiki_stats",
        "find_cross_references",
        "detect_orphan_pages",
        "ask_repo",
        "code_graph_overview",
        "code_graph_contexts",
        "code_graph_subgraph",
        "memory_tree_list_topics",
        "memory_tree_recall",
    }
)


def apply_readonly_mode(env: dict[str, str] | None = None) -> list[str]:
    """Drop every non-read tool when WIKI_MCP_READONLY is truthy.

    Returns the list of removed tool names (empty when the flag is unset).
    Idempotent: removing an already-removed tool is skipped.
    """
    flag = (env if env is not None else os.environ).get("WIKI_MCP_READONLY", "")
    if flag.strip().lower() not in {"1", "true", "yes"}:
        return []
    registered = {tool.name for tool in mcp._tool_manager.list_tools()}
    removed = []
    for name in sorted(registered - READONLY_TOOLS):
        mcp.remove_tool(name)
        removed.append(name)
    return removed


apply_readonly_mode()


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def main() -> None:
    """Run the wiki MCP server."""
    parser = argparse.ArgumentParser(description="Wiki Knowledge Engine MCP Server")
    parser.add_argument(
        "--transport",
        choices=["stdio", "sse"],
        default="stdio",
        help="Transport protocol (default: stdio for Claude Code)",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=8765,
        help="Port for SSE transport (default: 8765)",
    )
    parser.add_argument(
        "--root",
        default=None,
        help="Override WIKI_ROOT path",
    )
    args = parser.parse_args()

    if args.root:
        global WIKI_ROOT
        WIKI_ROOT = args.root
        os.environ["WIKI_ROOT"] = args.root

    if args.transport == "sse":
        mcp._mcp_server.name = "brainstem"
        mcp.settings.port = args.port
        mcp.run(transport="sse")
    else:
        mcp.run(transport="stdio")


if __name__ == "__main__":
    main()
