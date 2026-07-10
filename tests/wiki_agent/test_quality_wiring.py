"""write_page × body-quality wiring (ADR-048 Fase 2).

Flag off (default) => byte-for-byte no-op (no quality_* in frontmatter).
Flag on            => quality_score/quality_verdict stamped; write NEVER blocked.
"""

from __future__ import annotations

import json
import os
import textwrap

import pytest

from wiki_agent.tools import create_tools, inject_quality_frontmatter, quality_scoring_enabled


@pytest.fixture
def tools(tmp_wiki_root):
    return {t.name: t for t in create_tools(tmp_wiki_root)}


GENUINE = textwrap.dedent("""\
    ---
    title: "Real Repo"
    date: 2026-06-21
    sources: ["https://github.com/x/y"]
    tags: [repo, github]
    origin: llm-synthesized
    ---

    # Real Repo

    ## What it is

    A genuinely synthesized description of the project, with enough prose to
    clear the floor and read like a real page rather than a stub or a dump. It
    explains the purpose, the problem the repo solves, and who would use it, in
    enough words that a future reader gets real context rather than a filetree.

    ## Capabilities

    It does several concrete things, each described with real detail so the
    body carries actual knowledge value for a future reader. The capabilities
    list names specific features, how they fit together, and what makes this
    implementation worth keeping in the knowledge base over a bare README dump.
""")

STUB = textwrap.dedent("""\
    ---
    title: "ajaxloader"
    date: 2026-06-21
    sources: []
    tags: [tool]
    origin: llm-synthesized
    ---

    # ajaxloader

    Mentioned in [[some-source]].
""")


def _write(tools, path, content):
    return json.loads(tools["write_page"].invoke({"page_path": path, "content": content}))


# --------------------------------------------------------------------------- #
# Flag OFF (default): no-op
# --------------------------------------------------------------------------- #
def test_flag_off_no_quality_fields(tools, tmp_wiki_root, monkeypatch):
    monkeypatch.delenv("SBW_QUALITY_SCORING", raising=False)
    assert quality_scoring_enabled() is False
    res = _write(tools, "wiki/sources/real-repo.md", GENUINE)
    assert res["status"] == "created"
    saved = open(os.path.join(tmp_wiki_root, "wiki", "sources", "real-repo.md")).read()
    assert "quality_verdict" not in saved
    assert "quality_score" not in saved


# --------------------------------------------------------------------------- #
# Flag ON: stamps verdict, never blocks
# --------------------------------------------------------------------------- #
def test_flag_on_stamps_genuine(tools, tmp_wiki_root, monkeypatch):
    monkeypatch.setenv("SBW_QUALITY_SCORING", "1")
    assert quality_scoring_enabled() is True
    res = _write(tools, "wiki/sources/real-repo.md", GENUINE)
    assert res["status"] == "created"
    saved = open(os.path.join(tmp_wiki_root, "wiki", "sources", "real-repo.md")).read()
    assert "quality_verdict: genuine" in saved
    assert "quality_score:" in saved


def test_flag_on_stamps_but_does_not_block_low_value(tools, tmp_wiki_root, monkeypatch):
    monkeypatch.setenv("SBW_QUALITY_SCORING", "1")
    res = _write(tools, "wiki/entities/ajaxloader.md", STUB)
    # The no_signal verdict is recorded, but Fase 2 still WRITES the page.
    assert res["status"] == "created"
    saved = open(os.path.join(tmp_wiki_root, "wiki", "entities", "ajaxloader.md")).read()
    assert "quality_verdict: no_signal" in saved


# --------------------------------------------------------------------------- #
# inject_quality_frontmatter: idempotent + degrade-safe
# --------------------------------------------------------------------------- #
def test_inject_is_idempotent():
    once = inject_quality_frontmatter(GENUINE)
    twice = inject_quality_frontmatter(once)
    assert once.count("quality_verdict:") == 1
    assert twice.count("quality_verdict:") == 1


def test_inject_noop_without_frontmatter():
    raw = "# No frontmatter\n\nbody"
    assert inject_quality_frontmatter(raw) == raw


# --------------------------------------------------------------------------- #
# ADR-048 Fase 3: SBW_QUALITY_ENFORCE — the no_signal skip tier
# --------------------------------------------------------------------------- #
def test_enforce_on_skips_no_signal_stub(tools, tmp_wiki_root, monkeypatch):
    monkeypatch.setenv("SBW_QUALITY_ENFORCE", "1")
    monkeypatch.delenv("SBW_QUALITY_SCORING", raising=False)  # enforce implies scoring
    res = _write(tools, "wiki/entities/ajaxloader.md", STUB)
    assert res["status"] == "skipped"
    assert res["reason"] == "quality-no_signal"
    assert res["page_path"] is None
    assert res["quality_verdict"] == "no_signal"
    assert res["notes"], "the skip must explain itself"
    assert not os.path.exists(os.path.join(tmp_wiki_root, "wiki", "entities", "ajaxloader.md"))


def test_enforce_on_still_writes_genuine(tools, tmp_wiki_root, monkeypatch):
    monkeypatch.setenv("SBW_QUALITY_ENFORCE", "1")
    res = _write(tools, "wiki/sources/real-repo.md", GENUINE)
    assert res["status"] == "created"
    saved = open(os.path.join(tmp_wiki_root, "wiki", "sources", "real-repo.md")).read()
    # Enforce implies scoring: the verdict is stamped even without the stamp flag.
    assert "quality_verdict: genuine" in saved


def test_enforce_on_still_writes_raw_dump_flagged(tools, tmp_wiki_root, monkeypatch):
    """no_signal is the ONLY blocking tier — raw_dump still writes (flagged)."""
    monkeypatch.setenv("SBW_QUALITY_ENFORCE", "1")
    raw_dump = GENUINE.replace("origin: llm-synthesized", "origin: ingested-untrusted")
    res = _write(tools, "wiki/sources/raw-dump.md", raw_dump)
    assert res["status"] == "created"
    saved = open(os.path.join(tmp_wiki_root, "wiki", "sources", "raw-dump.md")).read()
    assert "quality_verdict: raw_dump" in saved


def test_enforce_off_scoring_on_never_blocks(tools, tmp_wiki_root, monkeypatch):
    """Fase 2 behaviour is untouched when only the stamp flag is set."""
    monkeypatch.setenv("SBW_QUALITY_SCORING", "1")
    monkeypatch.delenv("SBW_QUALITY_ENFORCE", raising=False)
    res = _write(tools, "wiki/entities/ajaxloader.md", STUB)
    assert res["status"] == "created"


def test_enforce_skip_logs_loudly(tools, monkeypatch, caplog):
    """ADR-032 FR-7: a declined write is never silent."""
    import logging

    monkeypatch.setenv("SBW_QUALITY_ENFORCE", "1")
    with caplog.at_level(logging.WARNING):
        _write(tools, "wiki/entities/ajaxloader.md", STUB)
    assert any("quality.skip_write" in r.message for r in caplog.records)
