"""Custom tools for the Wiki Deep Agent.

Each tool is a plain function that accepts a ``wiki_root`` path at
construction time (via closure) and returns a LangChain ``@tool``-decorated
callable.  This keeps the tools stateless while binding them to a
concrete knowledge-base directory.
"""

from __future__ import annotations

import json
import logging
import os
import re
from collections.abc import Callable
from datetime import UTC, datetime
from glob import glob
from typing import TYPE_CHECKING

import yaml
from langchain_core.tools import tool

if TYPE_CHECKING:
    from wiki_synthesis.body_quality import BodyQualityScore

_logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Semantic search infrastructure (lazy initialisation, graceful degradation)
# ---------------------------------------------------------------------------

_embedding_model_cache: dict = {"model": None, "tried": False}
_index_embeddings: dict[str, list[tuple[str, str, list[float]]]] = {}


def _cosine_sim(a: list[float], b: list[float]) -> float:
    """Pure-Python cosine similarity (no numpy required)."""
    dot = sum(x * y for x, y in zip(a, b))
    na = sum(x * x for x in a) ** 0.5
    nb = sum(x * x for x in b) ** 0.5
    return dot / (na * nb) if na and nb else 0.0


# ---------------------------------------------------------------------------
# Frontmatter validation (shared page contract)
# ---------------------------------------------------------------------------

REQUIRED_FRONTMATTER_FIELDS = frozenset({"type", "title", "date", "sources", "tags", "origin"})
# Allowed ``origin`` values. ``synthesized-deterministic`` is emitted by the
# wiki_synthesis degrade path (``wiki_synthesis/agent.py``); omitting it made
# ``validate_frontmatter`` reject EVERY deterministic-degrade page (ADR-036 D3).
VALID_ORIGINS = frozenset(
    {
        "human",
        "llm-generated",
        "llm-synthesized",
        "mcp-ingested",
        "synthesized-deterministic",
    }
)


def validate_page_frontmatter(content: str) -> dict:
    """Validate a page's YAML frontmatter against the wiki contract.

    Returns ``{valid, missing_fields, errors, warnings}``. Pure over the page
    text — the file existence/read check is the caller's responsibility. The
    constants above are the single source of truth for the required fields and
    allowed origins so callers (and the synthesis layer) cannot drift."""
    result: dict = {"valid": False, "missing_fields": [], "errors": [], "warnings": []}

    fm_match = re.match(r"^---\s*\n(.*?)\n---", content, re.DOTALL)
    if not fm_match:
        result["errors"].append("No YAML frontmatter found (expected --- delimiters)")
        return result

    try:
        frontmatter = yaml.safe_load(fm_match.group(1))
    except yaml.YAMLError as exc:
        result["errors"].append(f"Invalid YAML: {exc}")
        return result

    if not isinstance(frontmatter, dict):
        result["errors"].append("Frontmatter is not a YAML mapping")
        return result

    result["missing_fields"] = sorted(REQUIRED_FRONTMATTER_FIELDS - set(frontmatter.keys()))

    origin = frontmatter.get("origin")
    if origin and origin not in VALID_ORIGINS:
        result["errors"].append(
            f"Invalid origin '{origin}'. Must be one of: {', '.join(sorted(VALID_ORIGINS))}"
        )

    # Warn on stale generated provenance.
    if origin in ("llm-generated", "llm-synthesized"):
        page_date = frontmatter.get("date")
        if page_date is not None:
            try:
                import datetime as _dtmod

                if isinstance(page_date, _dtmod.date):
                    date_obj = page_date
                elif isinstance(page_date, str) and len(page_date) >= 10:
                    date_obj = datetime.strptime(page_date[:10], "%Y-%m-%d").date()
                else:
                    date_obj = None
                if date_obj is not None:
                    age = (datetime.now(_dtmod.UTC).date() - date_obj).days
                    if age > 90:
                        result["warnings"].append(
                            f"Stale {origin} content: {age} days old. Review or archive."
                        )
            except (ValueError, TypeError):
                pass

    result["valid"] = len(result["missing_fields"]) == 0 and len(result["errors"]) == 0
    return result


# ---------------------------------------------------------------------------
# OKF type contract (Fase 3 — OKF v0.1 §9.2): every page declares its `type`.
# Single source of truth for the folder -> type mapping; mirrored by
# scripts/migrate_to_okf.py (the one-time backfill of legacy pages). write_page
# injects the resolved type when the caller omits it, so pages cannot drift
# back out of conformance regardless of what the synthesis layer emits.
# ---------------------------------------------------------------------------

OKF_TYPE_BY_DIR = {
    "concepts": "Concept",
    "entities": "Entity",
    "sources": "Source",
    "observations": "Observation",
    "answers": "Answer",
    "synthesis": "Synthesis",
    "outputs": "Output",
}
OKF_TYPE_BY_ROOT_STEM = {"dashboards": "Dashboard"}
OKF_ROOT_DEFAULT_TYPE = "Note"


def resolve_okf_type(rel_to_wiki: str) -> str:
    """Resolve the OKF ``type`` for a path relative to the ``wiki/`` dir.

    ``rel_to_wiki`` is like ``concepts/foo.md`` or ``dashboards.md``. The first
    path segment selects the type; a file directly under ``wiki/`` maps by stem.
    Unknown folders fall back to ``Note`` (a valid, generic OKF type)."""
    parts = [p for p in rel_to_wiki.replace("\\", "/").split("/") if p and p != "."]
    if len(parts) <= 1:
        stem = os.path.splitext(parts[0])[0] if parts else ""
        return OKF_TYPE_BY_ROOT_STEM.get(stem, OKF_ROOT_DEFAULT_TYPE)
    return OKF_TYPE_BY_DIR.get(parts[0], OKF_ROOT_DEFAULT_TYPE)


def inject_okf_type(content: str, okf_type: str) -> str:
    """Insert a ``type: <okf_type>`` line into ``content``'s frontmatter when absent.

    Surgical: adds one line right after the opening ``---`` and leaves the rest
    byte-for-byte. No-op when there is no frontmatter block or a non-empty
    ``type`` is already present (so explicit caller types are preserved)."""
    lines = content.splitlines(keepends=True)
    if not lines or lines[0].strip() != "---":
        return content
    close_idx = next((i for i in range(1, len(lines)) if lines[i].strip() == "---"), None)
    if close_idx is None:
        return content
    for ln in lines[1:close_idx]:
        if ln.lstrip().lower().startswith("type:") and ln.split(":", 1)[1].strip():
            return content  # caller already set a type
    lines.insert(1, f"type: {okf_type}\n")
    return "".join(lines)


# ---------------------------------------------------------------------------
# Body-quality scoring (ADR-048): off by default, flag-don't-block. When
# SBW_QUALITY_SCORING is truthy, write_page stamps a `quality_score` /
# `quality_verdict` into frontmatter so "are we ingesting quality?" becomes a
# queryable property. SBW_QUALITY_ENFORCE (Fase 3) additionally enables the
# ONE blocking tier of the D4 policy: a `no_signal` page (name-only stub,
# 0-star boilerplate repo) is skip-written with a LOUD log — never silently
# (ADR-032 FR-7). Every other verdict still writes. Both default off.
# ---------------------------------------------------------------------------

_QUALITY_FLAG = "SBW_QUALITY_SCORING"
_QUALITY_ENFORCE_FLAG = "SBW_QUALITY_ENFORCE"
_QUALITY_TRUTHY = frozenset({"1", "true", "yes", "on"})


def quality_scoring_enabled() -> bool:
    return os.environ.get(_QUALITY_FLAG, "").strip().lower() in _QUALITY_TRUTHY


def quality_enforce_enabled() -> bool:
    """ADR-048 Fase 3: the `no_signal` skip tier (implies scoring the page)."""
    return os.environ.get(_QUALITY_ENFORCE_FLAG, "").strip().lower() in _QUALITY_TRUTHY


def score_page_quality(content: str) -> BodyQualityScore | None:
    """Score a full page, degrade-safe: returns ``None`` when scoring raises
    (quality machinery must never fail a write)."""
    try:
        from wiki_synthesis.body_quality import score_body

        return score_body(content)
    except Exception as exc:  # noqa: BLE001 — scoring never fails a write
        _logger.info("body-quality scoring degrade: %s", type(exc).__name__)
        return None


def inject_quality_frontmatter(content: str, result: BodyQualityScore | None = None) -> str:
    """Insert ``quality_score`` / ``quality_verdict`` into frontmatter.

    Surgical and idempotent: skips when the keys are already present, when there
    is no frontmatter block, or when scoring raises (degrade-first — quality
    scoring must never fail a write). ``result`` lets ``write_page`` reuse the
    score it already computed for the D4 policy instead of scoring twice."""
    lines = content.splitlines(keepends=True)
    if not lines or lines[0].strip() != "---":
        return content
    close_idx = next((i for i in range(1, len(lines)) if lines[i].strip() == "---"), None)
    if close_idx is None:
        return content
    for ln in lines[1:close_idx]:
        if ln.lstrip().lower().startswith("quality_verdict:"):
            return content  # already scored
    if result is None:
        result = score_page_quality(content)
    if result is None:
        return content
    lines.insert(close_idx, f"quality_score: {result.score}\n")
    lines.insert(close_idx + 1, f"quality_verdict: {result.verdict}\n")
    return "".join(lines)


# ---------------------------------------------------------------------------
# Source-dedup helpers (INV-08 / Causa B): stop the same bookmark producing
# multiple source pages under different LLM-chosen slugs.
# ---------------------------------------------------------------------------


def _normalize_source(value: str) -> str:
    """Canonicalize a source URL/path for comparison.

    Trims whitespace, drops a single trailing slash, and lowercases so that
    e.g. ``https://x.com/a/`` and ``https://X.com/a`` compare equal.
    """
    return str(value).strip().rstrip("/").lower()


def _extract_sources(content: str) -> set[str]:
    """Return the normalized ``sources`` set from a page's YAML frontmatter."""
    match = re.match(r"^---\s*\n(.*?)\n---", content, re.DOTALL)
    if not match:
        return set()
    try:
        frontmatter = yaml.safe_load(match.group(1)) or {}
    except yaml.YAMLError:
        return set()
    raw = frontmatter.get("sources") or []
    if isinstance(raw, str):
        raw = [raw]
    return {_normalize_source(item) for item in raw if str(item).strip()}


def _find_duplicate_source_page(
    sources_dir: str, new_sources: set[str], exclude_real_path: str
) -> str | None:
    """Return the path of an existing source page sharing a source value.

    Scans ``sources_dir`` for ``*.md`` pages and returns the first one whose
    frontmatter ``sources`` overlap ``new_sources`` (excluding the page being
    written). Returns ``None`` when there is no overlap or no sources to match.
    """
    if not new_sources or not os.path.isdir(sources_dir):
        return None
    for path in sorted(glob(os.path.join(sources_dir, "**", "*.md"), recursive=True)):
        if os.path.realpath(path) == exclude_real_path:
            continue
        try:
            with open(path, encoding="utf-8") as fh:
                existing = fh.read()
        except OSError:
            continue
        if _extract_sources(existing) & new_sources:
            return path
    return None


class _FastEmbedWrapper:
    """Wrap fastembed to match the LangChain embeddings interface."""

    def __init__(self):
        from fastembed import TextEmbedding

        self._model = TextEmbedding("sentence-transformers/all-MiniLM-L6-v2")

    def embed_query(self, text: str) -> list[float]:
        return list(next(self._model.embed([text])))

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [list(v) for v in self._model.embed(texts)]


def _get_embedding_model():
    """Try Ollama → fastembed (local ONNX) → OpenAI. Returns ``None`` if unavailable."""
    if _embedding_model_cache["tried"]:
        return _embedding_model_cache["model"]
    _embedding_model_cache["tried"] = True

    # 1. Ollama (if running)
    try:
        from langchain_ollama import OllamaEmbeddings

        m = OllamaEmbeddings(model="nomic-embed-text")
        m.embed_query("test")  # verify reachable
        _embedding_model_cache["model"] = m
        _logger.info("Semantic search: Ollama nomic-embed-text")
        return m
    except Exception:
        pass

    # 2. fastembed — local ONNX, no server, no API key (all-MiniLM-L6-v2, 22MB)
    try:
        m = _FastEmbedWrapper()
        _embedding_model_cache["model"] = m
        _logger.info("Semantic search: fastembed all-MiniLM-L6-v2 (local ONNX)")
        return m
    except Exception:
        pass

    # 3. OpenAI / OpenRouter (requires API key)
    key = os.environ.get("OPENROUTER_API_KEY") or os.environ.get("OPENAI_API_KEY")
    if key:
        try:
            from langchain_openai import OpenAIEmbeddings

            kwargs: dict = {"openai_api_key": key}
            if os.environ.get("OPENROUTER_API_KEY"):
                kwargs["openai_api_base"] = "https://openrouter.ai/api/v1"
            _embedding_model_cache["model"] = OpenAIEmbeddings(**kwargs)
            _logger.info("Semantic search: OpenAI embeddings")
            return _embedding_model_cache["model"]
        except Exception:
            pass

    _logger.info("Semantic search: no embedding model, keyword-only")
    return None


def _build_index_embeddings(
    wiki_root: str,
    index_path: str,
) -> list[tuple[str, str, list[float]]]:
    """Embed every index entry. Cached per *wiki_root*."""
    model = _get_embedding_model()
    if model is None or not os.path.exists(index_path):
        return []

    with open(index_path, encoding="utf-8") as fh:
        content = fh.read()

    entries: list[tuple[str, str]] = []
    current_section = ""
    for line in content.splitlines():
        section_match = re.match(r"^##\s+(\w+)", line)
        if section_match:
            current_section = section_match.group(1).lower()
            continue
        if not line.startswith("|") or line.startswith("| Page") or line.startswith("|--"):
            continue
        cells = [c.strip() for c in line.split("|")[1:-1]]
        if len(cells) < 4:
            continue
        page_cell = cells[0]
        summary = cells[2] if len(cells) >= 5 else cells[1]
        category = cells[1] if len(cells) >= 5 else current_section
        link_match = re.match(r"\[(.+?)\]\((.+?)\)", page_cell)
        if not link_match:
            continue
        title = link_match.group(1)
        page_path = link_match.group(2)
        entries.append((page_path, f"{title} {category} {summary}"))

    if not entries:
        return []

    try:
        texts = [text for _, text in entries]
        vectors = model.embed_documents(texts)
        result = [(path, text, vec) for (path, text), vec in zip(entries, vectors)]
        _index_embeddings[wiki_root] = result
        _logger.info("Built semantic index: %d entries", len(result))
        return result
    except Exception as exc:
        _logger.warning("Failed to build semantic index: %s", exc)
        return []


def _semantic_search(
    wiki_root: str,
    index_path: str,
    query: str,
    top_k: int = 20,
) -> dict[str, float]:
    """Return ``{page_path: cosine_score}`` for the top-*k* matches."""
    if wiki_root not in _index_embeddings:
        _build_index_embeddings(wiki_root, index_path)
    entries = _index_embeddings.get(wiki_root, [])
    if not entries:
        return {}
    model = _get_embedding_model()
    if model is None:
        return {}
    try:
        qvec = model.embed_query(query)
    except Exception:
        return {}
    scores = [(path, _cosine_sim(qvec, vec)) for path, _, vec in entries]
    scores.sort(key=lambda x: x[1], reverse=True)
    return {path: sc for path, sc in scores[:top_k]}


def _invalidate_embeddings(wiki_root: str) -> None:
    """Drop cached embeddings so the next search rebuilds them."""
    _index_embeddings.pop(wiki_root, None)


# ---------------------------------------------------------------------------
# Factory: builds all wiki tools bound to a specific wiki_root
# ---------------------------------------------------------------------------


def create_tools(wiki_root: str) -> list[Callable]:
    """Return the eleven wiki tools, each bound to *wiki_root*.

    Args:
        wiki_root: Absolute or relative path to the knowledge-base directory.

    Returns:
        List of LangChain tool-decorated callables.
    """

    wiki_dir = os.path.join(wiki_root, "wiki")
    raw_dir = os.path.join(wiki_root, "raw")
    index_path = os.path.join(wiki_dir, "index.md")
    log_path = os.path.join(wiki_dir, "log.md")

    # ------------------------------------------------------------------
    # 1. search_wiki_index
    # ------------------------------------------------------------------
    @tool
    def search_wiki_index(query: str) -> str:
        """Search wiki/index.md for pages relevant to a query.

        Uses hybrid scoring: keyword term-overlap (40%) plus semantic
        cosine similarity via embeddings (60%).  Falls back to
        keyword-only when no embedding model is available.

        Args:
            query: Natural language search query.

        Returns:
            JSON array of matching entries with page_path, title, summary, and tags.
        """
        if not os.path.exists(index_path):
            return json.dumps([])

        with open(index_path, encoding="utf-8") as fh:
            content = fh.read()

        # --- 1. Parse all entries from index ---
        entries: list[dict] = []
        current_section = ""
        for line in content.splitlines():
            section_match = re.match(r"^##\s+(\w+)", line)
            if section_match:
                current_section = section_match.group(1).lower()
                continue
            if not line.startswith("|") or line.startswith("| Page") or line.startswith("|--"):
                continue
            cells = [c.strip() for c in line.split("|")[1:-1]]
            if len(cells) < 4:
                continue
            if len(cells) >= 5:
                page_cell, category, summary, sources, updated = cells[:5]
            else:
                page_cell, summary, sources, updated = cells[:4]
                category = current_section or "unknown"
            link_match = re.match(r"\[(.+?)\]\((.+?)\)", page_cell)
            if not link_match:
                continue
            entries.append(
                {
                    "page_path": link_match.group(2),
                    "title": link_match.group(1),
                    "summary": summary,
                    "tags": category,
                }
            )

        if not entries:
            return json.dumps([])

        # --- 2. Keyword scores ---
        query_terms = [t.lower() for t in query.split() if len(t) > 2]
        max_kw = 0
        for entry in entries:
            searchable = f"{entry['title']} {entry['tags']} {entry['summary']}".lower()
            kw = sum(1 for t in query_terms if t in searchable)
            entry["_kw"] = kw
            max_kw = max(max_kw, kw)

        # --- 3. Semantic scores (if available) ---
        sem_scores = _semantic_search(wiki_root, index_path, query)

        # --- 4. Combine and filter ---
        results: list[dict] = []
        for entry in entries:
            kw_norm = entry["_kw"] / max_kw if max_kw > 0 else 0.0
            sem = sem_scores.get(entry["page_path"], 0.0)

            if sem_scores:
                combined = 0.4 * kw_norm + 0.6 * sem
            else:
                combined = kw_norm  # keyword-only fallback

            if combined > 0.1:
                results.append(
                    {
                        "page_path": entry["page_path"],
                        "title": entry["title"],
                        "summary": entry["summary"],
                        "tags": entry["tags"],
                        "_score": combined,
                    }
                )

        results.sort(key=lambda r: r["_score"], reverse=True)
        for r in results:
            del r["_score"]
        return json.dumps(results, indent=2)

    # ------------------------------------------------------------------
    # 2. append_to_log
    # ------------------------------------------------------------------
    @tool
    def append_to_log(entry_type: str, title: str, details: str) -> str:
        """Append a timestamped entry to wiki/log.md.

        Args:
            entry_type: One of 'ingest', 'query', 'lint'.
            title: Short title for the log entry.
            details: Details including pages affected and outcome.

        Returns:
            Confirmation string with timestamp.
        """
        now = datetime.now(UTC).strftime("%Y-%m-%d %H:%M:%S UTC")
        date_short = datetime.now(UTC).strftime("%Y-%m-%d")
        entry = f"\n## [{date_short}] {entry_type} | {title}\n- Timestamp: {now}\n{details}\n"

        os.makedirs(os.path.dirname(log_path), exist_ok=True)
        with open(log_path, "a", encoding="utf-8") as fh:
            fh.write(entry)

        return f"Log entry appended at {now}"

    # ------------------------------------------------------------------
    # 3. get_wiki_stats
    # ------------------------------------------------------------------
    @tool
    def get_wiki_stats() -> str:
        """Return wiki statistics.

        Returns:
            JSON object with page_count, source_count, entity_count,
            concept_count, last_ingest_date, last_lint_date, and orphan_count.
        """
        stats: dict = {
            "page_count": 0,
            "source_count": 0,
            "entity_count": 0,
            "concept_count": 0,
            "last_ingest_date": None,
            "last_lint_date": None,
            "orphan_count": 0,
        }

        if os.path.isdir(wiki_dir):
            all_pages = glob(os.path.join(wiki_dir, "**", "*.md"), recursive=True)
            stats["page_count"] = len(all_pages)

        entities_dir = os.path.join(wiki_dir, "entities")
        if os.path.isdir(entities_dir):
            stats["entity_count"] = len(glob(os.path.join(entities_dir, "*.md")))

        concepts_dir = os.path.join(wiki_dir, "concepts")
        if os.path.isdir(concepts_dir):
            stats["concept_count"] = len(glob(os.path.join(concepts_dir, "*.md")))

        sources_dir = os.path.join(wiki_dir, "sources")
        if os.path.isdir(sources_dir):
            stats["source_count"] = len(glob(os.path.join(sources_dir, "*.md")))

        # Parse log.md for last dates
        if os.path.exists(log_path):
            with open(log_path, encoding="utf-8") as fh:
                log_content = fh.read()
            for line in reversed(log_content.splitlines()):
                if "ingest" in line and stats["last_ingest_date"] is None:
                    date_match = re.search(r"\[(\d{4}-\d{2}-\d{2})\]", line)
                    if date_match:
                        stats["last_ingest_date"] = date_match.group(1)
                if "lint" in line and stats["last_lint_date"] is None:
                    date_match = re.search(r"\[(\d{4}-\d{2}-\d{2})\]", line)
                    if date_match:
                        stats["last_lint_date"] = date_match.group(1)
                if stats["last_ingest_date"] and stats["last_lint_date"]:
                    break

        return json.dumps(stats, indent=2)

    # ------------------------------------------------------------------
    # 4. find_cross_references
    # ------------------------------------------------------------------
    @tool
    def find_cross_references(page_path: str) -> str:
        """Find all pages that link to or are linked from a given page.

        Args:
            page_path: Path to the wiki page to analyse (relative to wiki_root).

        Returns:
            JSON object with inbound_links and outbound_links arrays.
        """
        abs_path = os.path.join(wiki_root, page_path) if not os.path.isabs(page_path) else page_path

        outbound: list[str] = []
        if os.path.exists(abs_path):
            with open(abs_path, encoding="utf-8") as fh:
                content = fh.read()
            # Match [[wikilink]] and [text](path.md) patterns
            outbound.extend(re.findall(r"\[\[(.+?)\]\]", content))
            outbound.extend(re.findall(r"\[.+?\]\((.+?\.md)\)", content))

        # Determine the page filename to search for inbound links
        page_name = os.path.splitext(os.path.basename(abs_path))[0]

        inbound: list[str] = []
        if os.path.isdir(wiki_dir):
            for md_file in glob(os.path.join(wiki_dir, "**", "*.md"), recursive=True):
                if os.path.abspath(md_file) == os.path.abspath(abs_path):
                    continue
                with open(md_file, encoding="utf-8") as fh:
                    file_content = fh.read()
                if f"[[{page_name}]]" in file_content or page_path in file_content:
                    rel = os.path.relpath(md_file, wiki_root)
                    inbound.append(rel)

        return json.dumps(
            {
                "inbound_links": inbound,
                "outbound_links": outbound,
            },
            indent=2,
        )

    # ------------------------------------------------------------------
    # 5. detect_orphan_pages
    # ------------------------------------------------------------------
    @tool
    def detect_orphan_pages() -> str:
        """Find wiki pages with no inbound links from other pages.

        Returns:
            JSON array of orphan page paths (relative to wiki_root).
        """
        if not os.path.isdir(wiki_dir):
            return json.dumps([])

        all_pages = glob(os.path.join(wiki_dir, "**", "*.md"), recursive=True)
        # Build set of page basenames and a mapping
        page_basenames: dict[str, str] = {}  # basename -> rel_path
        for p in all_pages:
            rel = os.path.relpath(p, wiki_root)
            basename = os.path.splitext(os.path.basename(p))[0]
            page_basenames[basename] = rel

        # Skip index.md and log.md from orphan check
        skip = {"index", "log"}

        # Build inbound link counts
        inbound_count: dict[str, int] = {name: 0 for name in page_basenames if name not in skip}

        for p in all_pages:
            with open(p, encoding="utf-8") as fh:
                content = fh.read()
            # Find all wikilinks and markdown links
            wikilinks = re.findall(r"\[\[(.+?)\]\]", content)
            md_links = re.findall(r"\[.+?\]\((.+?\.md)\)", content)
            referenced = set()
            for wl in wikilinks:
                referenced.add(wl.lower())
            for ml in md_links:
                name = os.path.splitext(os.path.basename(ml))[0]
                referenced.add(name.lower())
            for name in referenced:
                for page_name in inbound_count:
                    if page_name.lower() == name:
                        inbound_count[page_name] += 1

        orphans = [
            page_basenames[name]
            for name, count in inbound_count.items()
            if count == 0 and name not in skip
        ]
        return json.dumps(sorted(orphans), indent=2)

    # ------------------------------------------------------------------
    # 6. web_clip
    # ------------------------------------------------------------------
    @tool
    def web_clip(url: str) -> str:
        """Fetch a web article and convert to markdown for ingestion.

        Args:
            url: URL of the web article to clip.

        Returns:
            Markdown content of the article, saved to raw/bookmarks/.
        """
        try:
            import httpx
            from markdownify import markdownify as md  # type: ignore[import-untyped]
        except ImportError as exc:
            return json.dumps(
                {
                    "error": f"Missing dependency: {exc}. Install httpx and markdownify.",
                }
            )

        try:
            response = httpx.get(url, follow_redirects=True, timeout=30.0)
            response.raise_for_status()
        except httpx.HTTPError as exc:
            return json.dumps({"error": f"Failed to fetch URL: {exc}"})

        markdown_content = md(response.text, strip=["script", "style", "nav", "footer"])

        # Save to raw/bookmarks/
        bookmarks_dir = os.path.join(raw_dir, "bookmarks")
        os.makedirs(bookmarks_dir, exist_ok=True)

        # Create a safe filename from URL
        safe_name = re.sub(r"[^\w\-]", "_", url.split("//")[-1])[:80]
        filename = f"{safe_name}.md"
        filepath = os.path.join(bookmarks_dir, filename)

        now = datetime.now(UTC).strftime("%Y-%m-%d")
        frontmatter = (
            f"---\n"
            f'title: "Web clip: {url}"\n'
            f"date: {now}\n"
            f'sources: ["{url}"]\n'
            f"tags: [bookmark, web-clip]\n"
            f"---\n\n"
        )
        with open(filepath, "w", encoding="utf-8") as fh:
            fh.write(frontmatter + markdown_content)

        rel_path = os.path.relpath(filepath, wiki_root)
        return json.dumps(
            {
                "saved_path": rel_path,
                "url": url,
                "content_length": len(markdown_content),
            }
        )

    # ------------------------------------------------------------------
    # 7. validate_frontmatter
    # ------------------------------------------------------------------
    @tool
    def validate_frontmatter(page_path: str) -> str:
        """Check that a wiki page has valid YAML frontmatter.

        Args:
            page_path: Path to the wiki page to validate (relative to wiki_root).

        Returns:
            JSON object with valid (bool), missing_fields, and errors arrays.
        """
        abs_path = os.path.join(wiki_root, page_path) if not os.path.isabs(page_path) else page_path

        if not os.path.exists(abs_path):
            return json.dumps(
                {
                    "valid": False,
                    "missing_fields": [],
                    "errors": [f"File not found: {page_path}"],
                    "warnings": [],
                },
                indent=2,
            )

        with open(abs_path, encoding="utf-8") as fh:
            content = fh.read()

        return json.dumps(validate_page_frontmatter(content), indent=2)

    # ------------------------------------------------------------------
    # 8. read_wiki_file
    # ------------------------------------------------------------------
    @tool
    def read_wiki_file(file_path: str) -> str:
        """Read a file from raw/ or wiki/ and return its content with parsed frontmatter.

        Args:
            file_path: Path to the file to read (relative to wiki_root,
                or absolute).  Must be inside the wiki_root tree.

        Returns:
            JSON object with content, file_path, size_bytes, and
            frontmatter (parsed YAML if present, else null).
        """
        abs_path = os.path.join(wiki_root, file_path) if not os.path.isabs(file_path) else file_path
        # Security: ensure path is inside wiki_root
        real_root = os.path.realpath(wiki_root)
        real_path = os.path.realpath(abs_path)
        if not real_path.startswith(real_root):
            return json.dumps({"error": "Path is outside wiki_root."})

        if not os.path.exists(real_path):
            return json.dumps({"error": f"File not found: {file_path}"})

        with open(real_path, encoding="utf-8") as fh:
            content = fh.read()

        # Try to parse frontmatter
        frontmatter = None
        fm_match = re.match(r"^---\s*\n(.*?)\n---", content, re.DOTALL)
        if fm_match:
            try:
                frontmatter = yaml.safe_load(fm_match.group(1))
            except yaml.YAMLError:
                pass

        rel_path = os.path.relpath(real_path, wiki_root)
        return json.dumps(
            {
                "file_path": rel_path,
                "content": content,
                "size_bytes": len(content.encode("utf-8")),
                "frontmatter": frontmatter,
            },
            indent=2,
            default=str,
        )

    # ------------------------------------------------------------------
    # 9. write_page
    # ------------------------------------------------------------------
    @tool
    def write_page(page_path: str, content: str, overwrite: bool = False) -> str:
        """Create or update a wiki page.

        Args:
            page_path: Destination path for the page (relative to wiki_root,
                must be inside wiki/).  Example: ``wiki/sources/my-article.md``.
            content: Full markdown content of the page, including frontmatter.
            overwrite: When ``False`` (default), creating a *new* page under
                ``wiki/sources/`` is refused if another source page already
                references an overlapping ``sources`` value — this stops the
                same bookmark producing duplicate slugs (INV-08 / Causa B).
                Updating a page at an existing path is always allowed. Pass
                ``True`` to bypass the guard for intentional re-creation.

        Returns:
            JSON object with status ('created', 'updated', or 'refused'),
            page_path, and size_bytes.
        """
        abs_path = os.path.join(wiki_root, page_path) if not os.path.isabs(page_path) else page_path
        # Security: ensure path is inside wiki_root/wiki/
        real_wiki = os.path.realpath(wiki_dir)
        real_path = os.path.realpath(abs_path)
        if not real_path.startswith(real_wiki):
            return json.dumps({"error": "Path must be inside wiki/ directory."})

        rel_path = os.path.relpath(real_path, wiki_root)
        existed = os.path.exists(real_path)

        # OKF type contract (Fase 3 / §9.2): guarantee a `type` based on the
        # destination folder when the caller omitted it, so every page written
        # is born conformant. No-op if the content already declares a type.
        content = inject_okf_type(content, resolve_okf_type(os.path.relpath(real_path, real_wiki)))

        # Body-quality policy (ADR-048 D4): opt-in via SBW_QUALITY_SCORING
        # (stamp quality_score/quality_verdict) and SBW_QUALITY_ENFORCE
        # (Fase 3: skip-write `no_signal` pages — the ONLY blocking tier,
        # logged loudly, never silent). weak/raw_dump/bloat still write.
        if quality_scoring_enabled() or quality_enforce_enabled():
            quality = score_page_quality(content)
            if quality is not None:
                if quality.verdict == "no_signal" and quality_enforce_enabled():
                    _logger.warning(
                        "quality.skip_write page_path=%s score=%s notes=%s",
                        rel_path,
                        quality.score,
                        "; ".join(quality.notes),
                    )
                    return json.dumps(
                        {
                            "status": "skipped",
                            "reason": "quality-no_signal",
                            "page_path": None,
                            "target_path": rel_path,
                            "quality_score": quality.score,
                            "quality_verdict": quality.verdict,
                            "notes": list(quality.notes),
                            "hint": (
                                "ADR-048 D4: no_signal pages (name-only stubs, "
                                "boilerplate-only repos) are declined. Enrich the "
                                "body or unset SBW_QUALITY_ENFORCE to force-write."
                            ),
                        },
                        indent=2,
                    )
                content = inject_quality_frontmatter(content, quality)

        # Source-collision guard (INV-08 / Causa B): refuse a NEW source page
        # whose sources overlap an existing one, so a fresh LLM-chosen slug
        # cannot duplicate an already-ingested bookmark. Updates to an existing
        # path are unaffected; overwrite=True bypasses the guard.
        sources_dir = os.path.join(real_wiki, "sources")
        if (
            not existed
            and not overwrite
            and real_path.startswith(os.path.realpath(sources_dir) + os.sep)
        ):
            dup = _find_duplicate_source_page(sources_dir, _extract_sources(content), real_path)
            if dup is not None:
                return json.dumps(
                    {
                        "status": "refused",
                        "reason": "duplicate_source",
                        "page_path": rel_path,
                        "existing_page": os.path.relpath(os.path.realpath(dup), wiki_root),
                        "hint": (
                            "A source page already covers this source. Update the "
                            "existing page (or pass overwrite=true) instead of "
                            "creating a new slug."
                        ),
                    },
                    indent=2,
                )

        os.makedirs(os.path.dirname(real_path), exist_ok=True)

        with open(real_path, "w", encoding="utf-8") as fh:
            fh.write(content)

        _invalidate_embeddings(wiki_root)

        return json.dumps(
            {
                "status": "updated" if existed else "created",
                "page_path": rel_path,
                "size_bytes": len(content.encode("utf-8")),
            },
            indent=2,
        )

    # ------------------------------------------------------------------
    # 10. update_index_entry
    # ------------------------------------------------------------------
    @tool
    def update_index_entry(
        page_path: str,
        category: str,
        summary: str,
        source_count: int,
    ) -> str:
        """Add or update an entry in wiki/index.md.

        Args:
            page_path: Relative path to the wiki page (e.g. ``sources/my-article.md``).
            category: Category label (sources, entities, concepts, answers).
            summary: One-line summary of the page content.
            source_count: Number of source documents referenced by the page.

        Returns:
            JSON object with status ('added' or 'updated') and the
            page_path.
        """
        now = datetime.now(UTC).strftime("%Y-%m-%d")

        # Derive title from page_path slug
        basename = os.path.splitext(os.path.basename(page_path))[0]
        title = basename.replace("-", " ").title()

        # 4-column format matching sectioned index (no Category column)
        new_row = f"| [{title}]({page_path}) | {summary} | {source_count} | {now} |"

        os.makedirs(os.path.dirname(index_path), exist_ok=True)

        if not os.path.exists(index_path):
            # Create a fresh index with the entry
            header = (
                "---\n"
                f'title: "Wiki Index"\n'
                f"date: {now}\n"
                "sources: []\n"
                "tags: [index, meta]\n"
                "---\n\n"
                "# Wiki Index\n\n"
                "| Page | Summary | Sources | Updated |\n"
                "|------|---------|---------|--------|\n"
            )
            with open(index_path, "w", encoding="utf-8") as fh:
                fh.write(header + new_row + "\n")
            _invalidate_embeddings(wiki_root)
            return json.dumps({"status": "added", "page_path": page_path})

        with open(index_path, encoding="utf-8") as fh:
            lines = fh.readlines()

        # Check if entry already exists (match by page_path in link)
        updated = False
        for i, line in enumerate(lines):
            if page_path in line and line.strip().startswith("|"):
                lines[i] = new_row + "\n"
                updated = True
                break

        if not updated:
            # Append after the last table row (or header separator)
            insert_idx = len(lines)
            for i in range(len(lines) - 1, -1, -1):
                if lines[i].strip().startswith("|"):
                    insert_idx = i + 1
                    break
            lines.insert(insert_idx, new_row + "\n")

        with open(index_path, "w", encoding="utf-8") as fh:
            fh.writelines(lines)

        _invalidate_embeddings(wiki_root)
        return json.dumps(
            {
                "status": "updated" if updated else "added",
                "page_path": page_path,
            }
        )

    # ------------------------------------------------------------------
    # 11. update_schema_lessons
    # ------------------------------------------------------------------
    @tool
    def update_schema_lessons(lesson: str) -> str:
        """Append a lesson learned to schema/wiki-schema.md.

        This tool enables the wiki to self-improve by recording patterns,
        conventions, or insights discovered during ingestion and linting.

        Args:
            lesson: The lesson learned, as a single concise sentence or
                short paragraph.

        Returns:
            JSON object with status and the lesson text.
        """
        schema_path = os.path.join(wiki_root, "schema", "wiki-schema.md")

        if not os.path.exists(schema_path):
            return json.dumps(
                {
                    "error": "schema/wiki-schema.md not found. Run init first.",
                }
            )

        with open(schema_path, encoding="utf-8") as fh:
            content = fh.read()

        now = datetime.now(UTC).strftime("%Y-%m-%d")
        lesson_entry = f"- [{now}] {lesson}\n"

        # Find or create "Lessons learned" section
        section_header = "## Lessons learned"
        if section_header in content:
            # Append after the section header
            idx = content.index(section_header) + len(section_header)
            # Find end of line
            nl_idx = content.index("\n", idx)
            content = content[: nl_idx + 1] + "\n" + lesson_entry + content[nl_idx + 1 :]
        else:
            # Add the section at the end
            content = content.rstrip() + "\n\n" + section_header + "\n\n" + lesson_entry

        with open(schema_path, "w", encoding="utf-8") as fh:
            fh.write(content)

        return json.dumps(
            {
                "status": "added",
                "lesson": lesson,
            }
        )

    # ------------------------------------------------------------------
    # 12. graduate_observation
    # ------------------------------------------------------------------
    @tool
    def graduate_observation(
        observation_ids: str,
        target_type: str,
        title: str,
        content: str,
    ) -> str:
        """Graduate validated observations into durable wiki artifacts.

        Promotes observations into schema rules, concept pages, or entity
        pages.  Marks source observations as graduated.

        Args:
            observation_ids: Comma-separated OBS-IDs (e.g. ``OBS-2026-04-14-001,OBS-2026-04-15-003``).
            target_type: One of ``schema-rule``, ``concept-page``, ``entity-page``.
            title: Title for the new artifact (used as page title or rule heading).
            content: The graduated content to write.

        Returns:
            JSON object with status, target_path, and graduated_ids.
        """
        obs_ids = [oid.strip() for oid in observation_ids.split(",")]
        now = datetime.now(UTC).strftime("%Y-%m-%d")
        result: dict = {"status": "error", "graduated_ids": obs_ids}

        if target_type == "schema-rule":
            # Append to wiki-schema.md Lessons Learned section
            schema_path = os.path.join(wiki_root, "schema", "wiki-schema.md")
            if os.path.exists(schema_path):
                with open(schema_path, encoding="utf-8") as fh:
                    schema = fh.read()
                lesson_entry = f"- [{now}] (graduated from {', '.join(obs_ids)}) {content}\n"
                section_header = "## Lessons learned"
                if section_header in schema:
                    idx = schema.index(section_header) + len(section_header)
                    nl_idx = schema.index("\n", idx)
                    schema = schema[: nl_idx + 1] + "\n" + lesson_entry + schema[nl_idx + 1 :]
                else:
                    schema += f"\n\n{section_header}\n\n{lesson_entry}"
                with open(schema_path, "w", encoding="utf-8") as fh:
                    fh.write(schema)
                result["status"] = "graduated"
                result["target_path"] = "schema/wiki-schema.md"

        elif target_type in ("concept-page", "entity-page"):
            category = "concepts" if target_type == "concept-page" else "entities"
            slug = re.sub(r"[^\w\-]", "-", title.lower()).strip("-")
            slug = re.sub(r"-+", "-", slug)
            page_path = os.path.join(wiki_dir, category, f"{slug}.md")

            page_content = (
                f"---\n"
                f'title: "{title}"\n'
                f"date: {now}\n"
                f"sources: [{', '.join(obs_ids)}]\n"
                f"tags: [{category.rstrip('s')}, graduated]\n"
                f"origin: llm-synthesized\n"
                f"last_updated: {now}\n"
                f"---\n\n"
                f"# {title}\n\n"
                f"{content}\n\n"
                f"## Source observations\n\n" + "\n".join(f"- {oid}" for oid in obs_ids) + "\n"
            )

            os.makedirs(os.path.dirname(page_path), exist_ok=True)
            with open(page_path, "w", encoding="utf-8") as fh:
                fh.write(page_content)
            result["status"] = "graduated"
            result["target_path"] = os.path.relpath(page_path, wiki_root)

        else:
            result["error"] = f"Invalid target_type: {target_type}"
            return json.dumps(result, indent=2)

        # Mark source observations as graduated
        obs_dir = os.path.join(wiki_root, "observations")
        if os.path.isdir(obs_dir):
            for obs_file in glob(os.path.join(obs_dir, "*.md")):
                if os.path.basename(obs_file) == "REVIEW-LOG.md":
                    continue
                with open(obs_file, encoding="utf-8") as fh:
                    obs_content = fh.read()
                modified = False
                for oid in obs_ids:
                    if oid in obs_content:
                        obs_content = obs_content.replace(
                            "**Graduated:** false",
                            f"**Graduated:** true ({target_type}: {title})",
                            1,
                        )
                        modified = True
                if modified:
                    with open(obs_file, "w", encoding="utf-8") as fh:
                        fh.write(obs_content)

        return json.dumps(result, indent=2)

    return [
        search_wiki_index,
        append_to_log,
        get_wiki_stats,
        find_cross_references,
        detect_orphan_pages,
        web_clip,
        validate_frontmatter,
        read_wiki_file,
        write_page,
        update_index_entry,
        update_schema_lessons,
        graduate_observation,
    ]
