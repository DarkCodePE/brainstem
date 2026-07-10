"""Unit tests for the external-repo code-graph runner (PRD-012 / ADR-022).

These tests NEVER spawn ``node`` or a real subprocess: a fake ``runner`` is
injected to simulate the four observable outcomes of the UA pipeline —
success, nonzero exit, empty graph, and timeout — plus the degrade and
hard-error boundaries. The generic ``context_of`` in ``ua_codegraph_ext`` is
also unit-tested directly.
"""

from __future__ import annotations

import importlib.util
import json
import subprocess
import sys
from pathlib import Path

import pytest

from wiki_repos.codegraph_runner import build_repo_graph
from wiki_repos.errors import GraphFailed
from wiki_repos.types import RepoRef

# --------------------------------------------------------------------------- #
# Import the script-local module ``scripts/ua_codegraph_ext.py`` by path so the
# generic ``context_of`` can be unit-tested without packaging ``scripts/``.
# --------------------------------------------------------------------------- #
_SCRIPTS = Path(__file__).resolve().parents[2] / "scripts"
_spec = importlib.util.spec_from_file_location("ua_codegraph_ext", _SCRIPTS / "ua_codegraph_ext.py")
assert _spec and _spec.loader
ua_codegraph_ext = importlib.util.module_from_spec(_spec)
sys.modules["ua_codegraph_ext"] = ua_codegraph_ext
_spec.loader.exec_module(ua_codegraph_ext)


@pytest.fixture
def ref() -> RepoRef:
    """A validated repo reference (owner/repo) used across cases."""
    return RepoRef(owner="acme", repo="widget")


def _valid_graph() -> dict:
    """A minimal but non-empty schema-valid code graph."""
    return {
        "version": "1.0.0",
        "kind": "codebase",
        "project": {"name": "acme__widget", "languages": ["python"]},
        "nodes": [{"id": "file:pkg/a.py", "type": "file", "name": "a.py", "filePath": "pkg/a.py"}],
        "edges": [],
        "layers": [],
        "tour": [],
    }


def _writing_runner(graph: dict, exit_code: int = 0):
    """Build a fake runner that writes ``graph`` into out_dir, then returns code.

    The out_dir is parsed out of the command (``cmd[2]`` per the script's CLI
    contract: ``<script> <project-dir> <out-dir> <name>``).
    """

    def runner(cmd: list[str], timeout: float) -> int:
        out_dir = Path(cmd[2])
        (out_dir / "knowledge-graph.json").write_text(json.dumps(graph), encoding="utf-8")
        return exit_code

    return runner


# --------------------------------------------------------------------------- #
# build_repo_graph
# --------------------------------------------------------------------------- #


def test_success_returns_graph_dict(tmp_path: Path, ref: RepoRef) -> None:
    """(a) Runner writes a valid graph + returns 0 -> the dict is returned."""
    repo_dir = tmp_path / "repo"
    out_dir = tmp_path / "out"
    repo_dir.mkdir()
    graph = _valid_graph()

    result = build_repo_graph(repo_dir, ref, out_dir=out_dir, runner=_writing_runner(graph, 0))

    assert result == graph
    assert (out_dir / "knowledge-graph.json").is_file()


def test_nonzero_exit_degrades_to_none(tmp_path: Path, ref: RepoRef) -> None:
    """(b) Runner returns nonzero -> degrade to None (no raise)."""
    repo_dir = tmp_path / "repo"
    out_dir = tmp_path / "out"
    repo_dir.mkdir()

    def runner(cmd: list[str], timeout: float) -> int:
        return 1

    assert build_repo_graph(repo_dir, ref, out_dir=out_dir, runner=runner) is None


def test_skill_missing_exit_code_degrades_to_none(tmp_path: Path, ref: RepoRef) -> None:
    """UA-skill-missing exit code (2) degrades to None, never raises."""
    repo_dir = tmp_path / "repo"
    out_dir = tmp_path / "out"
    repo_dir.mkdir()

    def runner(cmd: list[str], timeout: float) -> int:
        return 2

    assert build_repo_graph(repo_dir, ref, out_dir=out_dir, runner=runner) is None


def test_zero_node_graph_degrades_to_none(tmp_path: Path, ref: RepoRef) -> None:
    """(c) Runner writes a zero-node graph -> degrade to None (PRD-012 R-5)."""
    repo_dir = tmp_path / "repo"
    out_dir = tmp_path / "out"
    repo_dir.mkdir()
    empty = _valid_graph()
    empty["nodes"] = []

    result = build_repo_graph(repo_dir, ref, out_dir=out_dir, runner=_writing_runner(empty, 0))
    assert result is None


def test_timeout_degrades_to_none(tmp_path: Path, ref: RepoRef) -> None:
    """(d) Runner raises TimeoutExpired -> degrade to None."""
    repo_dir = tmp_path / "repo"
    out_dir = tmp_path / "out"
    repo_dir.mkdir()

    def runner(cmd: list[str], timeout: float) -> int:
        raise subprocess.TimeoutExpired(cmd="ua-codegraph-ext.sh", timeout=timeout)

    assert build_repo_graph(repo_dir, ref, out_dir=out_dir, runner=runner) is None


def test_success_writes_into_out_dir_not_sbw_src(tmp_path: Path, ref: RepoRef) -> None:
    """out_dir is created and used; the command targets it (never SBW src)."""
    repo_dir = tmp_path / "repo"
    out_dir = tmp_path / "deep" / "out"  # not yet created
    repo_dir.mkdir()
    captured: dict = {}

    def runner(cmd: list[str], timeout: float) -> int:
        captured["cmd"] = cmd
        Path(cmd[2]).joinpath("knowledge-graph.json").write_text(
            json.dumps(_valid_graph()), encoding="utf-8"
        )
        return 0

    build_repo_graph(repo_dir, ref, out_dir=out_dir, runner=runner)

    assert out_dir.is_dir()
    assert captured["cmd"][2] == str(out_dir)
    assert "src/.understand-anything" not in " ".join(captured["cmd"])
    # project-name arg is the per-repo graph dir name.
    assert captured["cmd"][3] == ref.graph_dirname


def test_runner_raises_unexpected_degrades_to_none(tmp_path: Path, ref: RepoRef) -> None:
    """An arbitrary runner exception degrades to None — ingest never crashes."""
    repo_dir = tmp_path / "repo"
    out_dir = tmp_path / "out"
    repo_dir.mkdir()

    def runner(cmd: list[str], timeout: float) -> int:
        raise RuntimeError("node blew up")

    assert build_repo_graph(repo_dir, ref, out_dir=out_dir, runner=runner) is None


def test_missing_graph_file_after_success_degrades(tmp_path: Path, ref: RepoRef) -> None:
    """Runner returns 0 but writes nothing -> degrade to None."""
    repo_dir = tmp_path / "repo"
    out_dir = tmp_path / "out"
    repo_dir.mkdir()

    def runner(cmd: list[str], timeout: float) -> int:
        return 0

    assert build_repo_graph(repo_dir, ref, out_dir=out_dir, runner=runner) is None


def test_uncreatable_out_dir_raises_graphfailed(tmp_path: Path, ref: RepoRef) -> None:
    """An un-creatable out_dir is a true internal error -> GraphFailed."""
    repo_dir = tmp_path / "repo"
    repo_dir.mkdir()
    # A file occupying the out_dir's parent path makes mkdir fail with OSError.
    blocker = tmp_path / "blocker"
    blocker.write_text("not a dir", encoding="utf-8")
    bad_out = blocker / "out"

    def runner(cmd: list[str], timeout: float) -> int:  # pragma: no cover - never reached
        return 0

    with pytest.raises(GraphFailed):
        build_repo_graph(repo_dir, ref, out_dir=bad_out, runner=runner)


# --------------------------------------------------------------------------- #
# ua_codegraph_ext.context_of — generic top-level-dir bounded context
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize(
    ("file_path", "expected"),
    [
        ("pkg/sub/mod.py", "pkg"),
        ("src/main/App.java", "src"),
        ("lib/index.ts", "lib"),
        ("main.py", "root"),  # top-level file -> root
        ("", "root"),  # missing path -> root
    ],
)
def test_context_of_generic(file_path: str, expected: str) -> None:
    """context_of uses the top-level dir, with NO wiki_/search-fusion casing."""
    assert ua_codegraph_ext.context_of({"filePath": file_path}) == expected


def test_context_of_no_wiki_special_casing() -> None:
    """A ``wiki_*`` top-level dir is NOT special-cased here (generic scheme)."""
    # In the SBW-specific builder this would short-circuit; the generic one
    # simply returns the top-level dir — which happens to also be ``wiki_core``,
    # but a non-wiki dir like ``app`` must be returned verbatim, not ``root``.
    assert ua_codegraph_ext.context_of({"filePath": "app/server.go"}) == "app"
    assert ua_codegraph_ext.context_of({"filePath": "wiki_core/x.py"}) == "wiki_core"
