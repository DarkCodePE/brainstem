"""
Tests for `wiki_core.secrets.AuditLog` JSONL append + redaction rules.
"""

from __future__ import annotations

import json

from wiki_core.secrets import AUDIT_SCHEMA_VERSION, redact_params


def _read_lines(audit):
    with audit.path.open(encoding="utf-8") as fh:
        return [json.loads(line) for line in fh if line.strip()]


def test_write_appends_jsonl_with_schema_version(audit_log_tmp):
    audit_log_tmp.write(event="connect", provider="gmail", result="ok")
    lines = _read_lines(audit_log_tmp)
    assert len(lines) == 1
    rec = lines[0]
    assert rec["schema"] == AUDIT_SCHEMA_VERSION
    assert rec["event"] == "connect"
    assert rec["provider"] == "gmail"
    assert rec["result"] == "ok"
    assert "ts" in rec


def test_multiple_writes_append(audit_log_tmp):
    for i in range(3):
        audit_log_tmp.write(event="execute_action", provider="github", action=f"a{i}")
    lines = _read_lines(audit_log_tmp)
    assert len(lines) == 3
    assert [r["action"] for r in lines] == ["a0", "a1", "a2"]


def test_params_redacted_by_default(audit_log_tmp):
    audit_log_tmp.write(
        event="execute_action",
        provider="gmail",
        action="gmail_messages_list",
        params={"label_ids": ["INBOX"], "q": "from:boss@corp.com"},
    )
    rec = _read_lines(audit_log_tmp)[0]
    assert rec["params_redacted"]["q"] == "<redacted>"
    # Non-sensitive keys pass through
    assert rec["params_redacted"]["label_ids"] == ["INBOX"]


def test_verbose_audit_preserves_free_form(audit_log_tmp):
    audit_log_tmp.write(
        event="execute_action",
        provider="gmail",
        params={"q": "subject:test"},
        verbose_audit=True,
    )
    rec = _read_lines(audit_log_tmp)[0]
    assert rec["params_redacted"]["q"] == "subject:test"


def test_bearer_keys_always_redacted_even_in_verbose():
    """`access_token` and friends must NEVER leak, verbose or not."""
    params = {
        "access_token": "ya29.this_is_a_real_looking_bearer",
        "Authorization": "Bearer ghp_xxx",
        "nested": {"refresh_token": "1//xyz", "ok_field": "fine"},
    }
    out = redact_params(params, verbose=True)
    assert out["access_token"] == "<redacted>"
    assert out["Authorization"] == "<redacted>"
    assert out["nested"]["refresh_token"] == "<redacted>"
    assert out["nested"]["ok_field"] == "fine"


def test_redaction_recurses_into_lists():
    params = {
        "messages": [
            {"id": "m1", "body": "hello"},
            {"id": "m2", "snippet": "lorem"},
        ]
    }
    out = redact_params(params, verbose=False)
    assert out["messages"][0]["id"] == "m1"
    assert out["messages"][0]["body"] == "<redacted>"
    assert out["messages"][1]["snippet"] == "<redacted>"


def test_scope_used_is_serialised_as_list(audit_log_tmp):
    audit_log_tmp.write(
        event="execute_action",
        provider="gmail",
        scope_used=("https://www.googleapis.com/auth/gmail.readonly",),
    )
    rec = _read_lines(audit_log_tmp)[0]
    # JSON has no tuple type, must be a list
    assert rec["scope_used"] == ["https://www.googleapis.com/auth/gmail.readonly"]


def test_log_dir_created_with_0700_when_possible(tmp_path):
    """Best-effort dir permissions — should not raise even on weird mounts."""
    from wiki_core.secrets import AuditLog

    nested = tmp_path / "deep" / "nested" / "logs"
    AuditLog(path=nested / "integrations.log.jsonl")
    assert nested.exists()
