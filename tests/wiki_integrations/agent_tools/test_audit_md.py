"""
Tests for the per-provider Markdown audit log writer.
"""

from __future__ import annotations

import re

from wiki_integrations.agent_tools import ProviderMarkdownLog


def test_creates_log_dir_and_header(tmp_path):
    log = ProviderMarkdownLog(knowledge_base=tmp_path / "kb", provider="github")
    text = log.path.read_text(encoding="utf-8")
    assert "# Integration log" in text
    assert log.path.parent.name == "_log"
    assert log.path.parent.parent.name == "integrations"


def test_append_bullet_format(tmp_path):
    log = ProviderMarkdownLog(knowledge_base=tmp_path / "kb", provider="github")
    log.append(op="list", result="ok", items=12)
    text = log.path.read_text(encoding="utf-8")
    # One bullet was appended with op/result/items
    last = text.splitlines()[-1]
    assert last.startswith("- ")
    assert "list" in last
    assert "ok" in last
    assert "items=12" in last
    # ISO timestamp shape (rough)
    assert re.match(r"- \d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}Z", last)


def test_append_with_note(tmp_path):
    log = ProviderMarkdownLog(knowledge_base=tmp_path / "kb", provider="gmail")
    log.append(op="search", note="q=contract")
    text = log.path.read_text(encoding="utf-8")
    assert "q=contract" in text


def test_provider_required(tmp_path):
    import pytest

    with pytest.raises(ValueError):
        ProviderMarkdownLog(knowledge_base=tmp_path, provider="")
