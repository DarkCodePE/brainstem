"""Orchestration for repo-as-knowledge-source ingestion (PRD-012 / ADR-022).

``ingest_github_repo`` is the single entrypoint (exposed as the MCP tool of the
same name). It wires the leaf modules into the chosen pipeline:

    parse URL (host allowlist / SSRF guard)
      -> reachability probe (fail closed on private/unreachable)
      -> tarball fetch + safe extract  (NO git clone; sandboxed temp dir)
      -> local digest                  ($0 LLM)
      -> Understand-Anything code-graph ($0 LLM; degrades to None)
      -> Mermaid diagram from graph     (degrades to "")
      -> synthesize one wiki source page
      -> write_page (ADR-006 untrusted envelope + INV-08 dedup at the boundary)

Two acquisition shortcuts the architecture allows (ADR-022 reframe):
- **local-path mode**: if ``url`` points at an existing local directory (a repo
  the user already has), skip fetch+probe entirely — zero network, zero SSRF
  surface — and run the graph straight against it.
- (hosted-fetch via deepwiki/gitingest.com is a documented future option; this
  module implements the local tarball path, which keeps repo content on-machine.)

All injectable seams (``reachable_checker``, ``downloader``, ``graph_runner``,
``write_page``, ``clock``) exist so the orchestration unit-tests hermetically
(mock-first per CLAUDE.md) without network, node, or filesystem side effects.
"""

from __future__ import annotations

import logging
import os
import shutil
import tempfile
from collections.abc import Callable
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from wiki_repos import diagram as diagram_mod
from wiki_repos import history as history_mod
from wiki_repos import synthesize as synth_mod
from wiki_repos.cbm_runner import build_repo_graph_cbm
from wiki_repos.codegraph_runner import build_repo_graph
from wiki_repos.digest import build_digest
from wiki_repos.errors import WikiRepoError
from wiki_repos.fetcher import check_public_reachable, fetch_repo_tarball, parse_github_url
from wiki_repos.github_meta import RepoMeta, RepoMetaError, fetch_repo_meta
from wiki_repos.types import DraftMode, IngestResult, RepoHistory, RepoRef

logger = logging.getLogger(__name__)

WritePage = Callable[[str, str], Any]
"""(page_path, content) -> anything. The default writes to the vault filesystem;
the MCP tool injects the ADR-006 ``write_page`` tool so the untrusted envelope
and INV-08 duplicate-source guard apply at the real boundary."""

MetaFetcher = Callable[[str, str], RepoMeta]
"""(owner, repo) -> RepoMeta. The default hits the GitHub REST API best-effort
(ADR-025); tests inject a fake so the orchestration never touches the network."""


def _default_write_page(wiki_root: Path) -> WritePage:
    """Filesystem writer fallback (used outside the MCP server / in tests)."""

    def _write(page_path: str, content: str) -> str:
        dest = (wiki_root / page_path).resolve()
        wiki_dir = (wiki_root / "wiki").resolve()
        # True ancestor check — NOT a string prefix (which would accept a
        # sibling like ``wiki-evil/`` that merely shares the ``wiki`` prefix).
        if dest != wiki_dir and wiki_dir not in dest.parents:
            raise WikiRepoError(f"refusing to write outside wiki/: {page_path}")
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_text(content, encoding="utf-8")
        return str(dest)

    return _write


def _is_local_repo(url: str) -> Path | None:
    """Return the directory if ``url`` is an existing local repo path, else None."""
    if "://" in url or url.startswith("git@"):
        return None
    p = Path(url).expanduser()
    # Resolve so "." / relative paths yield a real directory name (not "").
    return p.resolve() if p.is_dir() else None


#: Env flag selecting the Stage-4 code-graph backend (ADR-046 D3).
ENV_CODEGRAPH_BACKEND = "SBW_CODEGRAPH_BACKEND"


def _select_codegraph_backend(override: str | None) -> str:
    """Resolve the code-graph backend: arg > ``$SBW_CODEGRAPH_BACKEND`` > ``ua``."""
    value = (override or os.environ.get(ENV_CODEGRAPH_BACKEND) or "ua").strip().lower()
    return value if value in ("ua", "cbm") else "ua"


def _build_graph_with_backend(
    repo_dir: Path,
    ref: RepoRef,
    *,
    out_dir: Path,
    backend: str,
    graph_runner: Callable[[list[str], float], int] | None,
    cbm_builder: Callable[..., dict | None],
    notes: list[str],
) -> dict | None:
    """Build the code-graph with the chosen backend; degrade cbm → ua (ADR-046 D3).

    When ``backend == "cbm"`` the cbm builder runs first; a ``None`` (any cbm
    degrade) falls back to the UA builder. The UA result may itself be ``None``,
    which the caller renders as digest-only — the [[ADR-022]] invariant only
    strengthens, never weakens.
    """
    if backend == "cbm":
        graph = cbm_builder(repo_dir, ref, out_dir=out_dir)
        if graph is not None:
            notes.append("graph: codebase-memory-mcp")
            return graph
        notes.append("graph: cbm degraded → ua fallback")
    return build_repo_graph(repo_dir, ref, out_dir=out_dir, runner=graph_runner)


async def ingest_github_repo(
    url: str,
    *,
    include_diagram: bool = True,
    include_history: bool = True,
    mode: DraftMode = "showcase",
    wiki_root: Path | str = "./knowledge-base",
    router: Any | None = None,
    # ---- injectable seams (tests / MCP) ----
    reachable_checker: Callable[[RepoRef], None] | None = None,
    downloader: Callable[[str, float], bytes] | None = None,
    graph_runner: Callable[[list[str], float], int] | None = None,
    codegraph_backend: str | None = None,
    cbm_graph_builder: Callable[..., dict | None] | None = None,
    history_miner: Callable[[RepoRef], RepoHistory | None] | None = None,
    write_page: WritePage | None = None,
    clock: Callable[[], datetime] | None = None,
    work_root: Path | None = None,
    png_renderer: Callable[[str, Path], Path | None] | None = None,
    refine_page_prose: bool = False,
    meta_fetcher: MetaFetcher | None = None,
) -> IngestResult:
    """Ingest a public GitHub repo (or a local repo dir) into one wiki source page.

    Raises a typed :class:`wiki_repos.errors.WikiRepoError` subclass on any
    failure (InvalidUrl / PrivateOrUnreachable / Oversize / FetchFailed /
    DigestFailed / SynthesisFailed). A missing/empty code-graph is NOT an error
    — it degrades to a digest-only page marked ``graph: unavailable``.
    """
    wiki_root = Path(wiki_root)
    clock = clock or (lambda: datetime.now(UTC))
    writer = write_page or _default_write_page(wiki_root)
    notes: list[str] = []

    local_dir = _is_local_repo(url)
    tmp_dir: Path | None = None
    meta: RepoMeta | None = None
    try:
        if local_dir is not None:
            # local-path mode: zero fetch, zero network (ADR-022 reframe).
            # No GitHub origin, so no metadata fetch — meta stays None.
            ref = RepoRef(owner=local_dir.parent.name or "local", repo=local_dir.name)
            repo_dir = local_dir
            notes.append("acquisition: local-path (no fetch)")
            if mode == "showcase":
                mode = "experiential"  # a repo you already have is your own
        else:
            ref = parse_github_url(url)
            (reachable_checker or check_public_reachable)(ref)
            if work_root is not None:
                # User-chosen extract location (ADR-022). Created if missing;
                # mkdtemp still carves a unique, cleaned subdir under it.
                Path(work_root).mkdir(parents=True, exist_ok=True)
            tmp_dir = Path(tempfile.mkdtemp(prefix="wiki_repos_", dir=work_root))
            repo_dir = fetch_repo_tarball(ref, dest_dir=tmp_dir, downloader=downloader)
            notes.append("acquisition: tarball (no git clone)")
            # ADR-025 (2nd decision): fold the GitHub value prop (description +
            # topics) into the synthesized page — the code-graph alone never says
            # WHAT the tool does. Best-effort: a failed/unreachable API DEGRADES
            # (meta stays None + a note), it must never fail the ingest.
            meta = _fetch_meta_best_effort(ref, meta_fetcher, notes)

        digest = build_digest(repo_dir, ref)
        if digest.stats.truncated:
            notes.append("digest: truncated by caps")

        graph_out = (wiki_root / "repos" / ref.graph_dirname).resolve()
        graph = _build_graph_with_backend(
            repo_dir,
            ref,
            out_dir=graph_out,
            backend=_select_codegraph_backend(codegraph_backend),
            graph_runner=graph_runner,
            cbm_builder=cbm_graph_builder or build_repo_graph_cbm,
            notes=notes,
        )
        graph_overview: dict = {}
        graph_path: str | None = None
        diagram_str = ""
        if graph:
            from wiki_repos.graphview import overview_by_layers

            # Bucket by the graph's own layers (top-level dirs) — correct for an
            # external repo; wiki_qa.codegraph.overview is tuned to SBW's source.
            graph_overview = overview_by_layers(graph)
            graph_path = str(graph_out / "knowledge-graph.json")
            if include_diagram:
                diagram_str = diagram_mod.mermaid_from_graph(graph, overview_data=graph_overview)
                if not diagram_str:
                    notes.append("diagram: empty graph, omitted")
            _prune_workdir(graph_out)
        else:
            notes.append("graph: unavailable (degraded to digest-only)")

        # ADR-023 Phase 1: render the diagram to a PNG artifact + a caption
        # sidecar (its own modality for multimodal search; NOT inlined in the
        # page). Degradable — a missing renderer never fails the ingest.
        diagram_image_path: str | None = None
        if diagram_str:
            diagram_image_path = _render_diagram_artifact(
                ref, diagram_str, graph_overview or None, wiki_root, clock, png_renderer, notes
            )

        # Git-history leg (PRD-014 / ADR-030): mine merged PRs + classified commits
        # via the GitHub API (NO clone). Degrade-first — never fails the ingest, and
        # is skipped in local-path mode (no GitHub repo to query; FR-6).
        history: RepoHistory | None = None
        if include_history:
            if local_dir is not None:
                notes.append("history: skipped (local-path mode has no GitHub API)")
            else:
                try:
                    miner = history_miner or history_mod.mine_history
                    history = miner(ref)
                except Exception as exc:  # noqa: BLE001 — history never fails the ingest
                    logger.info("history mine degrade: %s", exc)
                    history = None
                if history is None:
                    notes.append("history: unavailable (degraded to snapshot-only)")
                elif history.stats.truncated:
                    notes.append("history: truncated by caps")

        page_md = synth_mod.synthesize_page(
            ref,
            digest,
            graph_overview=graph_overview or None,
            diagram=diagram_str,
            history=history,
            mode=mode,
            router=router,
            clock=clock,
            description_override=meta.description if meta else None,
            topics=meta.topics if meta else (),
        )
        if router is not None:
            try:
                page_md = await synth_mod.refine_prose(page_md, router)
            except Exception as exc:  # degrade — never fail the ingest on LLM
                logger.warning("prose refinement failed, using deterministic page: %s", exc)
                notes.append("synthesis: LLM refine failed, deterministic page kept")

        page_path = synth_mod.page_path_for(ref)
        writer(page_path, page_md)

        return IngestResult(
            ref=ref,
            page_path=page_path,
            mode=mode,
            graph_path=graph_path,
            diagram_present=bool(diagram_str),
            diagram_image_path=diagram_image_path,
            digest_stats=digest.stats,
            graph_summary=_graph_summary(graph_overview),
            history_present=history is not None,
            notes=tuple(notes),
        )
    finally:
        if tmp_dir is not None and tmp_dir.exists():
            shutil.rmtree(tmp_dir, ignore_errors=True)


def _render_diagram_artifact(
    ref: RepoRef,
    diagram_str: str,
    graph_overview: dict | None,
    wiki_root: Path,
    clock: Callable[[], datetime],
    png_renderer: Callable[[str, Path], Path | None] | None,
    notes: list[str],
) -> str | None:
    """Render the Mermaid to a PNG under ``assets/diagrams/`` and write a caption
    sidecar (its own searchable modality). Returns the PNG path or None (degrade).
    """
    from wiki_repos import render as render_mod

    renderer = png_renderer or render_mod.mermaid_to_png
    assets_dir = (wiki_root / "assets" / "diagrams").resolve()
    png_path = assets_dir / f"{ref.slug}.png"
    try:
        rendered = renderer(diagram_str, png_path)
    except Exception as exc:  # noqa: BLE001 — render never fails the ingest
        logger.info("diagram render degrade: %s", exc)
        rendered = None
    if rendered is None:
        notes.append("diagram: PNG render unavailable (mmdc/Chrome) — page kept")
        return None

    caption = render_mod.diagram_caption(ref, graph_overview)
    sidecar = assets_dir / f"{ref.slug}.md"
    try:
        sidecar.write_text(
            render_mod.caption_markdown(
                ref, caption, image_relpath=f"{ref.slug}.png", date=clock().date().isoformat()
            ),
            encoding="utf-8",
        )
    except OSError as exc:
        logger.info("diagram caption write failed (non-fatal): %s", exc)
    notes.append("diagram: rendered to PNG + caption (multimodal artifact)")
    return str(rendered)


def _fetch_meta_best_effort(
    ref: RepoRef,
    meta_fetcher: MetaFetcher | None,
    notes: list[str],
) -> RepoMeta | None:
    """Fetch the GitHub value prop (description + topics) without ever failing.

    Returns the :class:`RepoMeta` on success, or ``None`` on any error
    (unreachable API, 404, transport failure) after appending a degrade note.
    The ingest must complete on the code-graph alone if GitHub is unavailable.
    """
    fetcher = meta_fetcher or fetch_repo_meta
    try:
        return fetcher(ref.owner, ref.repo)
    except RepoMetaError as exc:
        logger.info("github meta unavailable (degrade): %s", exc)
        notes.append("meta: github description unavailable")
        return None
    except Exception as exc:  # noqa: BLE001 — degrade on ANY fetch/transport error
        logger.info("github meta fetch error (degrade): %s", exc)
        notes.append("meta: github description unavailable")
        return None


def _prune_workdir(graph_out: Path) -> None:
    """Remove the UA scratch dir the ext runner leaves under the graph store."""
    work = graph_out / ".ua-work"
    if work.exists():
        shutil.rmtree(work, ignore_errors=True)


def _graph_summary(graph_overview: dict | None) -> dict:
    if not graph_overview:
        return {}
    totals = graph_overview.get("totals", {})
    return {
        "contexts": [c.get("name") for c in graph_overview.get("contexts", [])[:8]],
        "nodes": totals.get("nodes"),
        "edges": totals.get("edges"),
        "encapsulation_pct": totals.get("encapsulation_pct"),
    }
