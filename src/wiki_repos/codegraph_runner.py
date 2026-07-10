"""Run the Understand-Anything code-graph builder over an extracted repo tree.

Thin, degrade-first wrapper around ``scripts/ua-codegraph-ext.sh`` (PRD-012 /
ADR-022). Given a directory containing the extracted external repo source and a
:class:`~wiki_repos.types.RepoRef`, it shells out to the generalized UA pipeline
and loads the resulting ``knowledge-graph.json``.

Degrade philosophy (PRD-012 R-5): a missing graph is the common case, not an
error. The repo may be in an unsupported language, the UA skill may be absent,
or the graph may come back empty. In every such case we return ``None`` and let
the caller fall back to digest-only synthesis — we log a one-line note and move
on. Only a genuinely unexpected internal failure (e.g. ``out_dir`` cannot be
created) raises :class:`~wiki_repos.errors.GraphFailed`.

The ``runner`` dependency is injectable so tests never spawn ``node`` or
``subprocess``: it is a callable ``(cmd: list[str], timeout: float) -> int``
returning the process exit code.
"""

from __future__ import annotations

import json
import logging
import subprocess
from collections.abc import Callable
from pathlib import Path

from wiki_repos.errors import GraphFailed
from wiki_repos.types import RepoRef

logger = logging.getLogger(__name__)

#: Injectable runner signature: ``(cmd, timeout) -> exit_code``.
Runner = Callable[[list[str], float], int]

#: Generalized UA pipeline script, relative to the repo root.
_SCRIPT_RELPATH = "scripts/ua-codegraph-ext.sh"

#: Exit code the script uses to signal "UA skill dir missing" (degrade).
_EXIT_SKILL_MISSING = 2

#: Filename the script writes the assembled graph as, inside ``out_dir``.
_GRAPH_FILENAME = "knowledge-graph.json"


def _repo_root() -> Path:
    """Locate the SBW repo root (two parents up from ``src/wiki_repos/``)."""
    return Path(__file__).resolve().parents[2]


def _default_runner(cmd: list[str], timeout: float) -> int:
    """Default runner: run ``cmd`` via :func:`subprocess.run`, return exit code.

    Output is left on the child's stderr/stdout (the script logs there); we only
    care about the exit status. Raises :class:`subprocess.TimeoutExpired` on
    timeout, which the caller catches and treats as a degrade.
    """
    completed = subprocess.run(cmd, timeout=timeout, check=False)  # noqa: S603
    return completed.returncode


def _is_empty_graph(graph: dict) -> bool:
    """A graph with no nodes is a degrade, not a usable result (PRD-012 R-5)."""
    nodes = graph.get("nodes")
    return not isinstance(nodes, list) or len(nodes) == 0


def build_repo_graph(
    repo_dir: Path,
    ref: RepoRef,
    *,
    out_dir: Path,
    runner: Runner | None = None,
    timeout: float = 120.0,
) -> dict | None:
    """Build the code knowledge-graph for an extracted external repo.

    Runs ``scripts/ua-codegraph-ext.sh <repo_dir> <out_dir> <project-name>`` via
    the (injectable) ``runner``, writing ``knowledge-graph.json`` into
    ``out_dir``. On success the graph dict is loaded and returned.

    Args:
        repo_dir: Directory holding the extracted external repo source.
        ref: Validated reference to the repo (supplies the project name).
        out_dir: Where the graph is written — caller typically passes
            ``knowledge-base/repos/<owner>__<repo>/``. Created if absent. NEVER
            SBW's own ``src/.understand-anything/``.
        runner: Injectable ``(cmd, timeout) -> exit_code`` callable. Defaults to
            a :func:`subprocess.run` wrapper.
        timeout: Per-run wall-clock budget in seconds.

    Returns:
        The parsed ``knowledge-graph.json`` dict on success, or ``None`` when the
        build degrades (missing UA skill, nonzero exit, timeout, empty/zero-node
        graph, unsupported language, or unreadable/invalid output).

    Raises:
        GraphFailed: Only for a truly unexpected internal error — currently, if
            ``out_dir`` cannot be created.
    """
    run = runner if runner is not None else _default_runner

    try:
        out_dir.mkdir(parents=True, exist_ok=True)
    except OSError as exc:
        raise GraphFailed(f"cannot create graph out_dir {out_dir!r}: {exc}") from exc

    script = _repo_root() / _SCRIPT_RELPATH
    cmd = [str(script), str(repo_dir), str(out_dir), ref.graph_dirname]

    try:
        code = run(cmd, timeout)
    except subprocess.TimeoutExpired:
        logger.info("code-graph degrade: timeout after %.0fs for %s", timeout, ref.slug)
        return None
    except Exception as exc:  # noqa: BLE001 — any runner error degrades, never crashes ingest.
        logger.info("code-graph degrade: runner error for %s: %s", ref.slug, exc)
        return None

    if code == _EXIT_SKILL_MISSING:
        logger.info("code-graph degrade: UA skill dir missing for %s", ref.slug)
        return None
    if code != 0:
        logger.info("code-graph degrade: builder exit %d for %s", code, ref.slug)
        return None

    graph_path = out_dir / _GRAPH_FILENAME
    if not graph_path.is_file():
        logger.info("code-graph degrade: no graph written for %s", ref.slug)
        return None

    try:
        graph = json.loads(graph_path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        logger.info("code-graph degrade: unreadable graph for %s: %s", ref.slug, exc)
        return None

    if not isinstance(graph, dict) or _is_empty_graph(graph):
        logger.info("code-graph degrade: empty graph for %s", ref.slug)
        return None

    logger.info(
        "code-graph built for %s: %d nodes, %d edges",
        ref.slug,
        len(graph.get("nodes", [])),
        len(graph.get("edges", [])),
    )
    return graph
