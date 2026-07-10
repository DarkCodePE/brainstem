"""Build the code knowledge-graph via codebase-memory-mcp (ADR-046 D1).

Degrade-first sibling of ``codegraph_runner.build_repo_graph``. Given the
already-extracted repo dir (ADR-022 Stage 2 — no clone), it:

1. runs the cbm CLI ``index_repository`` over that local dir (no network);
2. locates the SQLite graph cbm persists under its cache dir;
3. reads the ``nodes``/``edges`` tables and maps them to the UA
   ``knowledge-graph.json`` contract via :mod:`wiki_repos.cbm_adapter`;
4. writes ``knowledge-graph.json`` into ``out_dir`` and returns the dict.

Degrade philosophy mirrors the UA runner (ADR-022 / PRD-012 R-5): a missing
binary, an index failure, an absent DB, or an empty mapped graph all return
``None`` so the caller falls back (cbm → ua → digest-only, ADR-046 D3). Only a
genuinely unexpected internal error (``out_dir`` cannot be created) raises
:class:`~wiki_repos.errors.GraphFailed`.

Both I/O seams are injectable so tests never spawn the binary or touch the real
cache: ``runner`` (``(cmd, timeout) -> (exit_code, stdout)``) and ``db_reader``
(``(db_path) -> (nodes, edges)``).
"""

from __future__ import annotations

import json
import logging
import os
import sqlite3
import subprocess
from collections.abc import Callable
from pathlib import Path

from wiki_repos.cbm_adapter import to_knowledge_graph
from wiki_repos.errors import GraphFailed
from wiki_repos.types import RepoRef

logger = logging.getLogger(__name__)

#: Injectable runner: ``(cmd, timeout) -> (exit_code, stdout_text)``.
Runner = Callable[[list[str], float], "tuple[int, str]"]
#: Injectable DB reader: ``(db_path) -> (nodes, edges)``.
DbReader = Callable[[Path], "tuple[list[dict], list[dict]]"]

#: Env override for the cbm binary path (default: resolved from PATH).
ENV_BIN = "SBW_CBM_BIN"
#: Env override for the cbm cache dir (where it persists ``<project>.db``).
ENV_CACHE = "SBW_CBM_CACHE_DIR"

_DEFAULT_BIN = "codebase-memory-mcp"
_GRAPH_FILENAME = "knowledge-graph.json"


def _default_runner(cmd: list[str], timeout: float) -> tuple[int, str]:
    """Run ``cmd`` via :func:`subprocess.run`, returning (exit_code, stdout)."""
    completed = subprocess.run(  # noqa: S603
        cmd, timeout=timeout, capture_output=True, text=True, check=False
    )
    return completed.returncode, completed.stdout


def _default_cache_dir() -> Path:
    """cbm's on-disk cache: ``$SBW_CBM_CACHE_DIR`` or ``~/.cache/codebase-memory-mcp``."""
    override = os.environ.get(ENV_CACHE)
    return Path(override) if override else Path.home() / ".cache" / "codebase-memory-mcp"


def _default_db_reader(db_path: Path) -> tuple[list[dict], list[dict]]:
    """Read cbm's ``nodes``/``edges`` tables read-only (no write, no lock)."""
    con = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
    try:
        con.row_factory = sqlite3.Row
        nodes = [
            dict(r)
            for r in con.execute(
                "SELECT id, label, name, file_path, start_line, end_line FROM nodes"
            )
        ]
        edges = [dict(r) for r in con.execute("SELECT source_id, target_id, type FROM edges")]
    finally:
        con.close()
    return nodes, edges


def _parse_project(stdout: str) -> str | None:
    """Extract the indexed project name from ``index_repository`` stdout.

    cbm prints log lines then a final JSON object carrying ``"project"`` (the
    cache DB is keyed by that name). We scan from the end for the first JSON line.
    """
    for line in reversed(stdout.strip().splitlines()):
        line = line.strip()
        if not line.startswith("{"):
            continue
        try:
            obj = json.loads(line)
        except ValueError:
            continue
        project = obj.get("project")
        if isinstance(project, str) and project:
            return project
    return None


def build_repo_graph_cbm(
    repo_dir: Path,
    ref: RepoRef,
    *,
    out_dir: Path,
    runner: Runner | None = None,
    db_reader: DbReader | None = None,
    cache_dir: Path | None = None,
    timeout: float = 120.0,
) -> dict | None:
    """Build the code knowledge-graph for an extracted external repo via cbm.

    Args:
        repo_dir: Directory holding the extracted external repo source.
        ref: Validated repo reference (supplies the project / graph dir name).
        out_dir: Where ``knowledge-graph.json`` is written (created if absent).
        runner: Injectable ``(cmd, timeout) -> (exit_code, stdout)``.
        db_reader: Injectable ``(db_path) -> (nodes, edges)``.
        cache_dir: Where cbm persists ``<project>.db`` (defaults to its cache).
        timeout: Per-index wall-clock budget in seconds.

    Returns:
        The contract-shaped graph dict on success, or ``None`` on any degrade
        (binary missing, nonzero index, no project name, DB absent/unreadable,
        empty mapped graph, or unwritable output).

    Raises:
        GraphFailed: Only if ``out_dir`` cannot be created (true internal error).
    """
    run = runner or _default_runner
    read_db = db_reader or _default_db_reader
    cache = cache_dir or _default_cache_dir()
    bin_path = os.environ.get(ENV_BIN, _DEFAULT_BIN)

    try:
        out_dir.mkdir(parents=True, exist_ok=True)
    except OSError as exc:
        raise GraphFailed(f"cannot create graph out_dir {out_dir!r}: {exc}") from exc

    cmd = [bin_path, "cli", "index_repository", json.dumps({"repo_path": str(repo_dir)})]
    try:
        code, stdout = run(cmd, timeout)
    except FileNotFoundError:
        logger.info("cbm degrade: binary %r not found for %s", bin_path, ref.slug)
        return None
    except subprocess.TimeoutExpired:
        logger.info("cbm degrade: timeout after %.0fs for %s", timeout, ref.slug)
        return None
    except Exception as exc:  # noqa: BLE001 — any runner error degrades, never crashes ingest.
        logger.info("cbm degrade: runner error for %s: %s", ref.slug, exc)
        return None

    if code != 0:
        logger.info("cbm degrade: index exit %d for %s", code, ref.slug)
        return None

    project = _parse_project(stdout)
    if not project:
        logger.info("cbm degrade: no project name in index output for %s", ref.slug)
        return None

    db_path = cache / f"{project}.db"
    if not db_path.is_file():
        logger.info("cbm degrade: db not found at %s for %s", db_path, ref.slug)
        return None

    try:
        nodes, edges = read_db(db_path)
    except Exception as exc:  # noqa: BLE001 — unreadable/locked DB degrades, never crashes.
        logger.info("cbm degrade: db read error for %s: %s", ref.slug, exc)
        return None

    graph = to_knowledge_graph(nodes, edges, project_name=ref.graph_dirname)
    if not graph.get("nodes"):
        logger.info("cbm degrade: empty mapped graph for %s", ref.slug)
        return None

    try:
        (out_dir / _GRAPH_FILENAME).write_text(json.dumps(graph), encoding="utf-8")
    except OSError as exc:
        logger.info("cbm degrade: cannot write graph for %s: %s", ref.slug, exc)
        return None

    logger.info(
        "cbm graph built for %s: %d nodes, %d edges",
        ref.slug,
        len(graph["nodes"]),
        len(graph["edges"]),
    )
    return graph
