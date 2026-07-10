"""Tests for the token-bounded structural context pack (ADR-046 Phase 3).

Pure over a contract-shaped graph; ``repo_dir`` slices use a tmp file. Verifies
the pack carries architecture + hubs + focus hits, respects the char budget, and
degrades cleanly without line ranges or a repo dir.
"""

from __future__ import annotations

from pathlib import Path

from wiki_repos.repo_context import build_context_pack


def _graph():
    return {
        "project": {"name": "acme__widget"},
        "nodes": [
            {
                "id": "file:src/auth.py",
                "type": "file",
                "name": "auth.py",
                "filePath": "src/auth.py",
            },
            {
                "id": "function:src/auth.py:login",
                "type": "function",
                "name": "login",
                "filePath": "src/auth.py",
                "lineStart": 3,
                "lineEnd": 4,
            },
            {
                "id": "function:src/util.py:helper",
                "type": "function",
                "name": "helper",
                "filePath": "src/util.py",
            },
            {"id": "file:core/db.py", "type": "file", "name": "db.py", "filePath": "core/db.py"},
        ],
        "edges": [
            {
                "source": "function:src/auth.py:login",
                "target": "file:core/db.py",
                "type": "imports",
            },
        ],
        "layers": [
            {
                "name": "src",
                "nodeIds": [
                    "file:src/auth.py",
                    "function:src/auth.py:login",
                    "function:src/util.py:helper",
                ],
            },
            {"name": "core", "nodeIds": ["file:core/db.py"]},
        ],
    }


def test_pack_has_architecture_and_hubs() -> None:
    pack = build_context_pack(_graph())
    assert "Repo context: acme__widget" in pack
    assert "## Contexts" in pack
    assert "## Key modules" in pack
    # the cross-context import shows coupling (src -> core)
    assert "Cross-context coupling" in pack
    assert "src -> core" in pack


def test_focus_lists_matching_symbols_with_location() -> None:
    pack = build_context_pack(_graph(), focus="login")
    assert "Symbols matching 'login'" in pack
    assert "function `login` — src/auth.py:3" in pack


def test_focus_no_match_is_explicit() -> None:
    pack = build_context_pack(_graph(), focus="zzz-nope")
    assert "(none found)" in pack


def test_source_slices_when_repo_dir_given(tmp_path: Path) -> None:
    (tmp_path / "src").mkdir()
    (tmp_path / "src" / "auth.py").write_text(
        "a\nb\ndef login():\n    return 1\ne\n", encoding="utf-8"
    )
    pack = build_context_pack(_graph(), repo_dir=tmp_path, focus="login")
    assert "## Source slices" in pack
    assert "src/auth.py:3-4" in pack
    assert "def login():" in pack


def test_no_slices_without_repo_dir() -> None:
    pack = build_context_pack(_graph(), focus="login")
    assert "## Source slices" not in pack


def test_respects_char_budget() -> None:
    pack = build_context_pack(_graph(), max_chars=120)
    assert len(pack) <= 120 + len("\n\n[TRUNCATED: context pack exceeded budget]")
    assert "[TRUNCATED" in pack


def test_pack_is_compact_vs_node_count() -> None:
    # A graph with many nodes still yields a bounded pack (hubs/contexts capped).
    big = _graph()
    big["nodes"] += [
        {
            "id": f"function:src/f{i}.py:fn{i}",
            "type": "function",
            "name": f"fn{i}",
            "filePath": f"src/f{i}.py",
        }
        for i in range(200)
    ]
    pack = build_context_pack(big)
    assert len(pack) < 4000  # stays small regardless of repo size


# --------------------------------------------------------------------------- #
# ADR-046 ask_repo leg: graph-store resolution + one-call pack
# --------------------------------------------------------------------------- #
import json  # noqa: E402

from wiki_repos.repo_context import (  # noqa: E402
    ask_repo_pack,
    find_repo_graph,
    list_repo_graphs,
)

_MINI_GRAPH = {
    "project": {"name": "acme__toolkit"},
    "nodes": [
        {"id": "f1", "type": "file", "name": "core.py", "filePath": "src/core.py"},
        {
            "id": "fn1",
            "type": "function",
            "name": "distill",
            "filePath": "src/core.py",
            "lineStart": 10,
        },
    ],
    "edges": [],
    "layers": [{"name": "src", "nodeIds": ["f1", "fn1"]}],
}


def _graph_store(tmp_path: Path) -> Path:
    root = tmp_path / "repos"
    for name in ("acme__toolkit", "acme__other", "beta__toolkit"):
        d = root / name
        d.mkdir(parents=True)
        (d / "knowledge-graph.json").write_text(json.dumps(_MINI_GRAPH))
    (root / "no-graph-here").mkdir()  # dir without a graph → not listed
    return root


def test_list_repo_graphs_only_dirs_with_graph(tmp_path):
    root = _graph_store(tmp_path)
    assert list_repo_graphs(root) == ["acme__other", "acme__toolkit", "beta__toolkit"]
    assert list_repo_graphs(tmp_path / "missing") == []


def test_find_repo_graph_owner_slash_and_dirname(tmp_path):
    root = _graph_store(tmp_path)
    assert find_repo_graph(root, "acme/toolkit") == root / "acme__toolkit" / "knowledge-graph.json"
    assert find_repo_graph(root, "ACME__TOOLKIT") == root / "acme__toolkit" / "knowledge-graph.json"


def test_find_repo_graph_bare_name_requires_uniqueness(tmp_path):
    root = _graph_store(tmp_path)
    # "toolkit" matches two owners → ambiguous → None.
    assert find_repo_graph(root, "toolkit") is None
    # "other" is unique.
    assert find_repo_graph(root, "other") == root / "acme__other" / "knowledge-graph.json"
    assert find_repo_graph(root, "") is None
    assert find_repo_graph(root, "nope") is None


def test_find_repo_graph_never_traverses(tmp_path):
    root = _graph_store(tmp_path)
    (tmp_path / "outside").mkdir()
    (tmp_path / "outside" / "knowledge-graph.json").write_text("{}")
    assert find_repo_graph(root, "../outside") is None
    assert find_repo_graph(root, "..") is None


def test_ask_repo_pack_success_and_focus(tmp_path):
    root = _graph_store(tmp_path)
    res = ask_repo_pack(root, "acme/toolkit", focus="distill")
    assert res["repo"] == "acme__toolkit"
    assert "# Repo context" in res["pack"]
    assert "`distill`" in res["pack"]
    assert "src/core.py:10" in res["pack"]


def test_ask_repo_pack_unknown_lists_available(tmp_path):
    root = _graph_store(tmp_path)
    res = ask_repo_pack(root, "toolkit")  # ambiguous
    assert "error" in res
    assert res["available"] == ["acme__other", "acme__toolkit", "beta__toolkit"]


def test_ask_repo_pack_unreadable_graph_degrades(tmp_path):
    root = _graph_store(tmp_path)
    (root / "acme__other" / "knowledge-graph.json").write_text("{not json")
    res = ask_repo_pack(root, "acme/other")
    assert "error" in res
    assert "unreadable" in res["error"]
