"""Unit tests for the cbm code-graph runner (ADR-046 D1).

These tests NEVER spawn the cbm binary nor touch the real cache: a fake
``runner`` (returning ``(exit_code, stdout)``) and a fake ``db_reader`` are
injected. They cover the degrade-first boundaries that keep the ingest alive
(binary missing, nonzero index, no project name, DB absent/unreadable, empty
mapped graph, timeout) plus the success and hard-error paths.
"""

from __future__ import annotations

import json
import subprocess
from pathlib import Path

import pytest

from wiki_repos.cbm_runner import _parse_project, build_repo_graph_cbm
from wiki_repos.errors import GraphFailed
from wiki_repos.types import RepoRef


@pytest.fixture
def ref() -> RepoRef:
    return RepoRef(owner="acme", repo="widget")


_PROJECT = "tmp-acme-widget"


def _ok_stdout() -> str:
    final = json.dumps({"project": _PROJECT, "status": "indexed", "nodes": 3})
    return f"level=info msg=pipeline.done\n{final}"


def _nodes_edges():
    nodes = [
        {
            "id": 1,
            "label": "File",
            "name": "a.py",
            "file_path": "src/a.py",
            "start_line": 1,
            "end_line": 9,
        },
        {
            "id": 2,
            "label": "Function",
            "name": "f",
            "file_path": "src/a.py",
            "start_line": 2,
            "end_line": 5,
        },
    ]
    edges = [{"source_id": 1, "target_id": 2, "type": "DEFINES"}]
    return nodes, edges


def _runner_ok(cmd, timeout):
    return 0, _ok_stdout()


def _make_db_file(cache: Path, project: str = _PROJECT) -> Path:
    cache.mkdir(parents=True, exist_ok=True)
    db = cache / f"{project}.db"
    db.write_text("", encoding="utf-8")  # presence only; db_reader is injected
    return db


# --------------------------------------------------------------------------- #
# Success
# --------------------------------------------------------------------------- #


def test_success_returns_graph_and_writes_file(tmp_path: Path, ref: RepoRef) -> None:
    out_dir = tmp_path / "out"
    cache = tmp_path / "cache"
    _make_db_file(cache)

    g = build_repo_graph_cbm(
        tmp_path / "repo",
        ref,
        out_dir=out_dir,
        runner=_runner_ok,
        db_reader=lambda _p: _nodes_edges(),
        cache_dir=cache,
    )

    assert g is not None
    assert g["project"]["name"] == ref.graph_dirname
    assert {n["type"] for n in g["nodes"]} == {"file", "function"}
    assert (out_dir / "knowledge-graph.json").is_file()
    written = json.loads((out_dir / "knowledge-graph.json").read_text())
    assert written == g


def test_repo_dir_passed_to_index_command(tmp_path: Path, ref: RepoRef) -> None:
    cache = tmp_path / "cache"
    _make_db_file(cache)
    captured: dict = {}

    def runner(cmd, timeout):
        captured["cmd"] = cmd
        return 0, _ok_stdout()

    build_repo_graph_cbm(
        tmp_path / "myrepo",
        ref,
        out_dir=tmp_path / "out",
        runner=runner,
        db_reader=lambda _p: _nodes_edges(),
        cache_dir=cache,
    )
    payload = json.loads(captured["cmd"][3])
    assert payload["repo_path"] == str(tmp_path / "myrepo")
    assert captured["cmd"][1:3] == ["cli", "index_repository"]


# --------------------------------------------------------------------------- #
# Degrade paths -> None (never raise, never crash ingest)
# --------------------------------------------------------------------------- #


def test_binary_missing_degrades(tmp_path: Path, ref: RepoRef) -> None:
    def runner(cmd, timeout):
        raise FileNotFoundError(cmd[0])

    assert (
        build_repo_graph_cbm(tmp_path / "r", ref, out_dir=tmp_path / "out", runner=runner) is None
    )


def test_nonzero_exit_degrades(tmp_path: Path, ref: RepoRef) -> None:
    assert (
        build_repo_graph_cbm(
            tmp_path / "r", ref, out_dir=tmp_path / "out", runner=lambda c, t: (1, "")
        )
        is None
    )


def test_no_project_name_degrades(tmp_path: Path, ref: RepoRef) -> None:
    assert (
        build_repo_graph_cbm(
            tmp_path / "r", ref, out_dir=tmp_path / "out", runner=lambda c, t: (0, "no json here")
        )
        is None
    )


def test_db_absent_degrades(tmp_path: Path, ref: RepoRef) -> None:
    # cache dir has no <project>.db -> degrade before db_reader is called.
    called = {"read": False}

    def reader(_p):
        called["read"] = True
        return _nodes_edges()

    g = build_repo_graph_cbm(
        tmp_path / "r",
        ref,
        out_dir=tmp_path / "out",
        runner=_runner_ok,
        db_reader=reader,
        cache_dir=tmp_path / "empty",
    )
    assert g is None and called["read"] is False


def test_db_read_error_degrades(tmp_path: Path, ref: RepoRef) -> None:
    cache = tmp_path / "cache"
    _make_db_file(cache)

    def reader(_p):
        raise sqlite_error()

    g = build_repo_graph_cbm(
        tmp_path / "r",
        ref,
        out_dir=tmp_path / "out",
        runner=_runner_ok,
        db_reader=reader,
        cache_dir=cache,
    )
    assert g is None


def test_empty_mapped_graph_degrades(tmp_path: Path, ref: RepoRef) -> None:
    cache = tmp_path / "cache"
    _make_db_file(cache)
    # only a dropped label -> adapter yields zero nodes -> degrade.
    only_module = ([{"id": 1, "label": "Module", "name": "m", "file_path": "a.py"}], [])

    g = build_repo_graph_cbm(
        tmp_path / "r",
        ref,
        out_dir=tmp_path / "out",
        runner=_runner_ok,
        db_reader=lambda _p: only_module,
        cache_dir=cache,
    )
    assert g is None


def test_timeout_degrades(tmp_path: Path, ref: RepoRef) -> None:
    def runner(cmd, timeout):
        raise subprocess.TimeoutExpired(cmd="cbm", timeout=timeout)

    assert (
        build_repo_graph_cbm(tmp_path / "r", ref, out_dir=tmp_path / "out", runner=runner) is None
    )


def test_unexpected_runner_error_degrades(tmp_path: Path, ref: RepoRef) -> None:
    def runner(cmd, timeout):
        raise RuntimeError("boom")

    assert (
        build_repo_graph_cbm(tmp_path / "r", ref, out_dir=tmp_path / "out", runner=runner) is None
    )


def test_uncreatable_out_dir_raises_graphfailed(tmp_path: Path, ref: RepoRef) -> None:
    blocker = tmp_path / "blocker"
    blocker.write_text("not a dir", encoding="utf-8")
    with pytest.raises(GraphFailed):
        build_repo_graph_cbm(tmp_path / "r", ref, out_dir=blocker / "out", runner=_runner_ok)


# --------------------------------------------------------------------------- #
# _parse_project
# --------------------------------------------------------------------------- #


def test_parse_project_reads_last_json_line() -> None:
    out = 'level=info x\nlevel=info y\n{"project":"foo-bar","status":"indexed"}'
    assert _parse_project(out) == "foo-bar"


def test_parse_project_none_when_absent() -> None:
    assert _parse_project("level=info only\nno json") is None


def test_parse_project_skips_non_project_json() -> None:
    out = '{"other":1}\n{"project":"p2"}'
    assert _parse_project(out) == "p2"


def sqlite_error() -> Exception:
    return RuntimeError("db locked")
