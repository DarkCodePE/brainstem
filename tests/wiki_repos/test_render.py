"""Tests for ``wiki_repos.render`` — Mermaid→PNG + caption (ADR-023 Phase 1).

Hermetic: the mmdc/Chrome subprocess is injected, so no node/Chrome is spawned.
"""

from __future__ import annotations

from pathlib import Path

from wiki_repos.render import (
    _strip_fence,
    caption_markdown,
    diagram_caption,
    mermaid_to_png,
)
from wiki_repos.types import RepoRef

_BLOCK = '```mermaid\nflowchart LR\n    a["core"] --> b["routes"]\n```'


def test_strip_fence_extracts_source():
    assert _strip_fence(_BLOCK).startswith("flowchart LR")
    assert "```" not in _strip_fence(_BLOCK)
    # a bare (unfenced) string is returned as-is
    assert _strip_fence("flowchart LR\n a-->b").startswith("flowchart")


def test_render_success_writes_png(tmp_path):
    out = tmp_path / "d.png"

    def fake_runner(cmd, env, timeout):
        # mmdc would write the -o target; simulate a valid PNG.
        Path(cmd[cmd.index("-o") + 1]).write_bytes(b"\x89PNG\r\n\x1a\nfake")
        return 0

    res = mermaid_to_png(_BLOCK, out, runner=fake_runner, mmdc_path="/usr/bin/mmdc")
    assert res == out
    assert out.read_bytes().startswith(b"\x89PNG")


def test_render_degrades_when_mmdc_missing(tmp_path, monkeypatch):
    monkeypatch.setattr("wiki_repos.render._find_mmdc", lambda explicit=None: None)
    assert mermaid_to_png(_BLOCK, tmp_path / "d.png") is None


def test_render_degrades_on_nonzero_exit(tmp_path):
    res = mermaid_to_png(
        _BLOCK,
        tmp_path / "d.png",
        runner=lambda cmd, env, timeout: 1,
        mmdc_path="/usr/bin/mmdc",
    )
    assert res is None


def test_render_degrades_on_empty_block(tmp_path):
    assert (
        mermaid_to_png("```mermaid\n\n```", tmp_path / "d.png", mmdc_path="/usr/bin/mmdc") is None
    )


def test_render_degrades_when_no_output_file(tmp_path):
    # runner returns 0 but writes nothing -> degrade (no PNG produced).
    res = mermaid_to_png(
        _BLOCK, tmp_path / "d.png", runner=lambda cmd, env, timeout: 0, mmdc_path="/usr/bin/mmdc"
    )
    assert res is None


def test_caption_uses_real_graph_facts_only():
    ref = RepoRef(owner="o", repo="r")
    ov = {
        "totals": {"contexts": 3, "encapsulation_pct": 52.4},
        "contexts": [{"name": "core"}, {"name": "routes"}],
        "top_hubs": [{"file": "core/db.py"}],
    }
    cap = diagram_caption(ref, ov)
    assert "o/r" in cap and "core" in cap and "core/db.py" in cap and "52.4" in cap


def test_caption_degrades_without_graph():
    cap = diagram_caption(RepoRef(owner="o", repo="r"), None)
    assert "unavailable" in cap.lower()


def test_caption_markdown_is_standalone_artifact():
    md = caption_markdown(
        RepoRef(owner="o", repo="r"), "A diagram.", image_relpath="o-r.png", date="2026-06-03"
    )
    assert "origin: diagram-caption" in md
    assert "image: o-r.png" in md
    assert "![Architecture diagram](o-r.png)" in md
