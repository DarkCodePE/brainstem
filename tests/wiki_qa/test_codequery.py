"""Tests for `wiki_qa.codequery` — call-graph impact analysis (ADR-037).

Consumes the `calls` edges in the UA code graph (which existed but were
unconsumed) to answer "who is affected if I change this function?" via bounded,
cycle-guarded reachability — no Cypher engine, no new graph builder.
"""

from __future__ import annotations

from typing import Any

import pytest

from wiki_qa.codequery import (
    callees_of,
    callers_of,
    impact,
    reaches,
    resolve_symbol,
)

A = "function:wiki_a/m.py:a"
B = "function:wiki_a/m.py:b"
C = "function:wiki_b/c.py:c"
D = "function:wiki_b/c.py:d"
X = "function:wiki_a/x.py:x"
Y = "function:wiki_a/x.py:y"


def _fn(file: str, name: str) -> dict[str, Any]:
    return {
        "id": f"function:{file}:{name}",
        "type": "function",
        "name": name,
        "filePath": file,
        "summary": "",
        "tags": [],
        "complexity": "simple",
    }


def _call(src: str, dst: str) -> dict[str, Any]:
    return {"source": src, "target": dst, "type": "calls", "direction": "forward", "weight": 0.8}


@pytest.fixture
def g() -> dict[str, Any]:
    return {
        "version": "1.0.0",
        "kind": "codebase",
        "project": {"name": "t"},
        "nodes": [
            _fn("wiki_a/m.py", "a"),
            _fn("wiki_a/m.py", "b"),
            _fn("wiki_b/c.py", "c"),
            _fn("wiki_b/c.py", "d"),
            _fn("wiki_a/x.py", "x"),
            _fn("wiki_a/x.py", "y"),
            _fn("wiki_a/h.py", "helper"),  # ambiguous name (two files)
            _fn("wiki_b/h.py", "helper"),
        ],
        "edges": [
            _call(A, B),  # chain: a -> b -> c -> d
            _call(B, C),
            _call(C, D),
            _call(X, Y),  # cycle: x <-> y
            _call(Y, X),
            # a non-calls edge must be ignored by the call-graph queries
            {"source": A, "target": "file:wiki_b/c.py", "type": "imports", "weight": 0.7},
        ],
    }


class TestResolveSymbol:
    def test_by_bare_name(self, g: dict[str, Any]) -> None:
        assert resolve_symbol(g, "c") == [C]

    def test_by_path_and_name(self, g: dict[str, Any]) -> None:
        assert resolve_symbol(g, "wiki_b/c.py:c") == [C]

    def test_by_full_node_id(self, g: dict[str, Any]) -> None:
        assert resolve_symbol(g, C) == [C]

    def test_ambiguous_name_returns_all_sorted(self, g: dict[str, Any]) -> None:
        assert resolve_symbol(g, "helper") == [
            "function:wiki_a/h.py:helper",
            "function:wiki_b/h.py:helper",
        ]

    def test_unresolved_is_empty(self, g: dict[str, Any]) -> None:
        assert resolve_symbol(g, "ghost") == []


class TestDirectEdges:
    def test_direct_callers(self, g: dict[str, Any]) -> None:
        assert callers_of(g, "c") == [B]

    def test_direct_callees(self, g: dict[str, Any]) -> None:
        assert callees_of(g, "c") == [D]

    def test_leaf_has_no_callees(self, g: dict[str, Any]) -> None:
        assert callees_of(g, "d") == []

    def test_root_has_no_callers(self, g: dict[str, Any]) -> None:
        assert callers_of(g, "a") == []

    def test_imports_edge_ignored(self, g: dict[str, Any]) -> None:
        # `a` imports a file but the call graph must not surface that.
        assert callees_of(g, "a") == [B]


class TestReaches:
    def test_transitive_true(self, g: dict[str, Any]) -> None:
        assert reaches(g, "a", "d", max_hops=4) is True

    def test_bounded_by_max_hops(self, g: dict[str, Any]) -> None:
        # a -> b -> c -> d is 3 hops; cap at 2 cannot reach d.
        assert reaches(g, "a", "d", max_hops=2) is False

    def test_no_reverse_path(self, g: dict[str, Any]) -> None:
        assert reaches(g, "d", "a") is False

    def test_cycle_terminates_true(self, g: dict[str, Any]) -> None:
        assert reaches(g, "x", "y") is True

    def test_cycle_terminates_false_for_missing(self, g: dict[str, Any]) -> None:
        assert reaches(g, "x", "ghost") is False


class TestImpact:
    def test_blast_radius_is_transitive_callers(self, g: dict[str, Any]) -> None:
        r = impact(g, "d", max_hops=4)
        assert r["resolved"] == [D]
        assert [c["id"] for c in r["direct_callers"]] == [C]
        assert {c["id"] for c in r["transitive_callers"]} == {A, B, C}
        assert r["blast_radius_count"] == 3
        assert r["truncated"] is False

    def test_hops_recorded(self, g: dict[str, Any]) -> None:
        r = impact(g, "d", max_hops=4)
        hops = {c["id"]: c["hops"] for c in r["transitive_callers"]}
        assert hops == {C: 1, B: 2, A: 3}

    def test_bounded_hops(self, g: dict[str, Any]) -> None:
        r = impact(g, "d", max_hops=1)
        assert {c["id"] for c in r["transitive_callers"]} == {C}

    def test_node_info_enriched(self, g: dict[str, Any]) -> None:
        r = impact(g, "d")
        c = next(x for x in r["transitive_callers"] if x["id"] == C)
        assert c["name"] == "c"
        assert c["file"] == "wiki_b/c.py"

    def test_cycle_excludes_self(self, g: dict[str, Any]) -> None:
        r = impact(g, "x", max_hops=4)
        # callers(x) = {y}; callers(y) = {x} but x is the symbol → excluded.
        assert {c["id"] for c in r["transitive_callers"]} == {Y}

    def test_unresolved_returns_error_with_candidates(self, g: dict[str, Any]) -> None:
        r = impact(g, "ghost")
        assert "error" in r
        assert r["transitive_callers"] == []

    def test_truncation_flag(self, g: dict[str, Any]) -> None:
        r = impact(g, "d", max_hops=4, max_results=1)
        assert r["truncated"] is True
        assert len(r["transitive_callers"]) == 1
