"""Tests for the Stage-4 code-graph backend selection + fallback (ADR-046 D3).

The fallback chain is ``cbm → ua → digest-only``. These tests exercise the pure
selector and the builder dispatcher directly (no async ingest, no real binary):
the UA builder is monkeypatched, the cbm builder is injected.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from wiki_repos import service
from wiki_repos.service import _build_graph_with_backend, _select_codegraph_backend
from wiki_repos.types import RepoRef

_REF = RepoRef(owner="acme", repo="widget")
_UA_GRAPH = {"project": {"name": "ua"}, "nodes": [{"id": "x"}], "edges": [], "layers": []}
_CBM_GRAPH = {"project": {"name": "cbm"}, "nodes": [{"id": "y"}], "edges": [], "layers": []}


# --------------------------------------------------------------------------- #
# _select_codegraph_backend
# --------------------------------------------------------------------------- #


def test_select_defaults_to_ua(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv(service.ENV_CODEGRAPH_BACKEND, raising=False)
    assert _select_codegraph_backend(None) == "ua"


def test_select_arg_overrides_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv(service.ENV_CODEGRAPH_BACKEND, "ua")
    assert _select_codegraph_backend("cbm") == "cbm"


def test_select_env_used_when_no_arg(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv(service.ENV_CODEGRAPH_BACKEND, "CBM")  # case-insensitive
    assert _select_codegraph_backend(None) == "cbm"


def test_select_unknown_falls_back_to_ua(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv(service.ENV_CODEGRAPH_BACKEND, raising=False)
    assert _select_codegraph_backend("sqlite-magic") == "ua"


# --------------------------------------------------------------------------- #
# _build_graph_with_backend — fallback chain
# --------------------------------------------------------------------------- #


def test_ua_backend_uses_ua_builder(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setattr(service, "build_repo_graph", lambda *a, **k: _UA_GRAPH)
    cbm_called = {"hit": False}

    def cbm(*a, **k):
        cbm_called["hit"] = True
        return _CBM_GRAPH

    notes: list[str] = []
    g = _build_graph_with_backend(
        tmp_path,
        _REF,
        out_dir=tmp_path / "o",
        backend="ua",
        graph_runner=None,
        cbm_builder=cbm,
        notes=notes,
    )
    assert g is _UA_GRAPH
    assert cbm_called["hit"] is False  # cbm never runs on the ua path


def test_cbm_backend_uses_cbm_when_it_succeeds(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setattr(service, "build_repo_graph", lambda *a, **k: pytest.fail("UA must not run"))
    notes: list[str] = []
    g = _build_graph_with_backend(
        tmp_path,
        _REF,
        out_dir=tmp_path / "o",
        backend="cbm",
        graph_runner=None,
        cbm_builder=lambda *a, **k: _CBM_GRAPH,
        notes=notes,
    )
    assert g is _CBM_GRAPH
    assert "graph: codebase-memory-mcp" in notes


def test_cbm_degrade_falls_back_to_ua(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setattr(service, "build_repo_graph", lambda *a, **k: _UA_GRAPH)
    notes: list[str] = []
    g = _build_graph_with_backend(
        tmp_path,
        _REF,
        out_dir=tmp_path / "o",
        backend="cbm",
        graph_runner=None,
        cbm_builder=lambda *a, **k: None,
        notes=notes,
    )
    assert g is _UA_GRAPH
    assert "graph: cbm degraded → ua fallback" in notes


def test_cbm_then_ua_both_degrade_returns_none(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setattr(service, "build_repo_graph", lambda *a, **k: None)
    notes: list[str] = []
    g = _build_graph_with_backend(
        tmp_path,
        _REF,
        out_dir=tmp_path / "o",
        backend="cbm",
        graph_runner=None,
        cbm_builder=lambda *a, **k: None,
        notes=notes,
    )
    assert g is None  # caller renders digest-only
