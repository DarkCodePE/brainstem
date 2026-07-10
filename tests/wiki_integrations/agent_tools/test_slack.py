"""
Tests for `SlackIntegration`: allowlist gate, multi-channel iteration via
bridge.execute, search dedup, scope policy lock, DM-scope opt-in-only.
"""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path

import pytest

from wiki_integrations.agent_tools.slack import (
    SlackIntegration,
    _extract_messages,
    _slack_message_to_item,
    _slack_ts_to_dt,
)

from .conftest import FakeBridge


def _make(
    secret_store,
    audit_jsonl,
    audit_md_slack,
    tmp_path: Path,
    *,
    allowed=None,
    execute_responses=None,
):
    """Build a SlackIntegration wired with a FakeBridge + tmp config.toml.

    `allowed` is the channel allowlist that lands in the config; `None`
    means "don't write a config file at all" (config-absent path)."""
    cfg = tmp_path / "config.toml"
    if allowed is not None:
        lines = ["[integrations.slack]", "allowed_channels = ["]
        lines.extend(f'    "{c}",' for c in allowed)
        lines.append("]")
        cfg.write_text("\n".join(lines), encoding="utf-8")
    bridge = FakeBridge(execute_responses=execute_responses or {})
    integration = SlackIntegration(
        bridge=bridge,
        store=secret_store,
        audit_jsonl=audit_jsonl,
        audit_md=audit_md_slack,
        config_path=cfg,
    )
    return integration, bridge


# --------------------------------------------------------------------------- #
# Allowlist gate — the #33 AC headline                                        #
# --------------------------------------------------------------------------- #


@pytest.mark.asyncio
async def test_list_returns_empty_when_no_allowlist(
    secret_store, audit_jsonl, audit_md_slack, tmp_path
):
    """No config.toml → no DMs/channels fetched. The integration emits a
    warning and returns an empty tuple — never crashes."""
    slack, bridge = _make(secret_store, audit_jsonl, audit_md_slack, tmp_path, allowed=None)
    await slack.connect()
    items = await slack.list(limit=10)
    assert items == ()
    # No bridge.execute call attempted — we short-circuit at the allowlist gate.
    assert bridge.execute_calls == []


@pytest.mark.asyncio
async def test_list_returns_empty_when_allowlist_empty(
    secret_store, audit_jsonl, audit_md_slack, tmp_path
):
    """An explicit empty allowlist still returns nothing — opt-in gate."""
    slack, bridge = _make(secret_store, audit_jsonl, audit_md_slack, tmp_path, allowed=[])
    await slack.connect()
    items = await slack.list(limit=10)
    assert items == ()
    assert bridge.execute_calls == []


@pytest.mark.asyncio
async def test_list_iterates_only_allowed_channels(
    secret_store, audit_jsonl, audit_md_slack, tmp_path
):
    """With two channels on the allowlist, the bridge.execute is called
    exactly twice (one per channel) with the correct ``channel`` arg."""
    responses = {
        ("slack", "SLACK_FETCH_CONVERSATION_HISTORY"): {
            "messages": [
                {"ts": "1716345600.001", "user": "U001", "text": "hello"},
            ],
        },
    }
    slack, bridge = _make(
        secret_store,
        audit_jsonl,
        audit_md_slack,
        tmp_path,
        allowed=["C001", "C002"],
        execute_responses=responses,
    )
    await slack.connect()
    items = await slack.list(limit=20)
    # Two calls, one per channel, with the correct argument.
    assert len(bridge.execute_calls) == 2
    channels_requested = [c[2]["channel"] for c in bridge.execute_calls]
    assert channels_requested == ["C001", "C002"]
    # Each channel returned one message → two items.
    assert len(items) == 2
    assert {i.metadata["channel"] for i in items} == {"C001", "C002"}


@pytest.mark.asyncio
async def test_list_respects_global_limit_across_channels(
    secret_store, audit_jsonl, audit_md_slack, tmp_path
):
    """``limit`` is the **total** across channels, not per-channel."""
    responses = {
        ("slack", "SLACK_FETCH_CONVERSATION_HISTORY"): {
            "messages": [
                {"ts": f"171634{i:04}.000", "user": "U001", "text": f"msg {i}"} for i in range(10)
            ],
        },
    }
    slack, _ = _make(
        secret_store,
        audit_jsonl,
        audit_md_slack,
        tmp_path,
        allowed=["C001", "C002"],
        execute_responses=responses,
    )
    await slack.connect()
    items = await slack.list(limit=3)
    assert len(items) == 3


@pytest.mark.asyncio
async def test_list_channel_error_skipped(
    secret_store, audit_jsonl, audit_md_slack, tmp_path, monkeypatch
):
    """If one channel fetch raises, the other still produces items."""
    responses = {
        ("slack", "SLACK_FETCH_CONVERSATION_HISTORY"): {
            "messages": [{"ts": "1716345600.001", "user": "U", "text": "ok"}],
        },
    }
    slack, bridge = _make(
        secret_store,
        audit_jsonl,
        audit_md_slack,
        tmp_path,
        allowed=["C_BAD", "C_OK"],
        execute_responses=responses,
    )
    await slack.connect()

    real_execute = bridge.execute
    call_seq: list[str] = []

    async def _flaky_execute(provider, slug, args):
        call_seq.append(args["channel"])
        if args["channel"] == "C_BAD":
            raise RuntimeError("slack 429")
        return await real_execute(provider, slug, args)

    monkeypatch.setattr(bridge, "execute", _flaky_execute)
    items = await slack.list(limit=10)
    assert "C_BAD" in call_seq
    assert "C_OK" in call_seq
    # Only the good channel's message survived.
    assert len(items) == 1
    assert items[0].metadata["channel"] == "C_OK"


# --------------------------------------------------------------------------- #
# Search                                                                      #
# --------------------------------------------------------------------------- #


@pytest.mark.asyncio
async def test_search_substring_match(secret_store, audit_jsonl, audit_md_slack, tmp_path):
    responses = {
        ("slack", "SLACK_FETCH_CONVERSATION_HISTORY"): {
            "messages": [
                {"ts": "1716345600.001", "user": "U001", "text": "Sprint planning Friday"},
                {"ts": "1716345700.001", "user": "U002", "text": "lunch?"},
                {"ts": "1716345800.001", "user": "U003", "text": "planning doc shared"},
            ],
        },
    }
    slack, _ = _make(
        secret_store,
        audit_jsonl,
        audit_md_slack,
        tmp_path,
        allowed=["C001"],
        execute_responses=responses,
    )
    await slack.connect()
    result = await slack.search("planning")
    texts = {i.snippet for i in result.items}
    assert any("Sprint planning" in t for t in texts)
    assert any("planning doc" in t for t in texts)
    assert not any("lunch?" in t for t in texts)


@pytest.mark.asyncio
async def test_search_empty_query_rejected(secret_store, audit_jsonl, audit_md_slack, tmp_path):
    slack, _ = _make(secret_store, audit_jsonl, audit_md_slack, tmp_path, allowed=["C001"])
    await slack.connect()
    with pytest.raises(ValueError, match="non-empty"):
        await slack.search("")


@pytest.mark.asyncio
async def test_search_dedups_by_sha256(
    secret_store, audit_jsonl, audit_md_slack, tmp_path, monkeypatch
):
    """A cross-posted message in two channels has two distinct sha256s
    (channel is part of the hash), so both survive. A repeated message
    fetched twice from the **same** channel would collapse."""
    responses = {
        ("slack", "SLACK_FETCH_CONVERSATION_HISTORY"): {
            "messages": [
                {"ts": "1716345600.001", "user": "U001", "text": "shared by alice"},
                {"ts": "1716345600.001", "user": "U001", "text": "shared by alice"},  # exact dupe
            ],
        },
    }
    slack, _ = _make(
        secret_store,
        audit_jsonl,
        audit_md_slack,
        tmp_path,
        allowed=["C001"],
        execute_responses=responses,
    )
    await slack.connect()
    result = await slack.search("shared")
    # Both entries share (channel, ts) → identical sha256 → second dropped.
    assert len(result.items) == 1


# --------------------------------------------------------------------------- #
# get                                                                         #
# --------------------------------------------------------------------------- #


@pytest.mark.asyncio
async def test_get_by_channel_ts(secret_store, audit_jsonl, audit_md_slack, tmp_path):
    responses = {
        ("slack", "SLACK_FETCH_CONVERSATION_HISTORY"): {
            "messages": [
                {"ts": "1716345600.001", "user": "U001", "text": "first"},
                {"ts": "1716345700.001", "user": "U002", "text": "second"},
            ],
        },
    }
    slack, _ = _make(
        secret_store,
        audit_jsonl,
        audit_md_slack,
        tmp_path,
        allowed=["C001"],
        execute_responses=responses,
    )
    await slack.connect()
    item = await slack.get("C001:1716345700.001")
    assert item.snippet == "second"
    assert item.metadata["channel"] == "C001"


@pytest.mark.asyncio
async def test_get_malformed_id_raises(secret_store, audit_jsonl, audit_md_slack, tmp_path):
    slack, _ = _make(secret_store, audit_jsonl, audit_md_slack, tmp_path, allowed=["C001"])
    await slack.connect()
    with pytest.raises(KeyError, match="channel.*ts"):
        await slack.get("no_colon_here")


# --------------------------------------------------------------------------- #
# Scope lock                                                                  #
# --------------------------------------------------------------------------- #


def test_scopes_match_locked_policy(secret_store, audit_jsonl, audit_md_slack, tmp_path):
    slack, _ = _make(secret_store, audit_jsonl, audit_md_slack, tmp_path, allowed=["C001"])
    # ADR-017 locks Slack to read-only channel scopes; DM scopes are opt-in
    # extras and intentionally not in the default set this integration uses.
    assert "channels:history" in slack.scopes
    assert "channels:read" in slack.scopes
    assert "users:read" in slack.scopes
    # DM scopes MUST NOT be in the default set — they're gated behind a
    # separate consent UI per ADR-017 and AC of #33.
    assert "im:history" not in slack.scopes
    assert "im:read" not in slack.scopes


# --------------------------------------------------------------------------- #
# Helpers                                                                     #
# --------------------------------------------------------------------------- #


def test_extract_messages_accepts_top_level():
    assert _extract_messages({"messages": [{"ts": "1"}]}) == [{"ts": "1"}]


def test_extract_messages_accepts_nested_data():
    assert _extract_messages({"data": {"messages": [{"ts": "1"}]}}) == [{"ts": "1"}]


def test_extract_messages_accepts_items_alias():
    assert _extract_messages({"items": [{"ts": "1"}]}) == [{"ts": "1"}]


def test_extract_messages_returns_empty_on_bad_shape():
    assert _extract_messages({}) == []
    assert _extract_messages({"messages": "not a list"}) == []
    assert _extract_messages("not a dict") == []  # type: ignore[arg-type]


def test_slack_ts_to_dt_parses_decimal():
    dt = _slack_ts_to_dt("1716345600.001234")
    assert dt > datetime(2024, 1, 1, tzinfo=UTC)


def test_slack_ts_to_dt_falls_back_on_garbage():
    dt = _slack_ts_to_dt("not a ts")
    assert dt == datetime.fromtimestamp(0, tz=UTC)


def test_to_item_builds_canonical_id():
    item = _slack_message_to_item(
        {"channel": "C001", "ts": "1716345600.001", "user": "U001", "text": "x"}
    )
    assert item.id == "C001:1716345600.001"
    assert item.metadata["channel"] == "C001"


def test_to_item_raises_when_missing_ts():
    with pytest.raises(KeyError, match="ts"):
        _slack_message_to_item({"channel": "C001", "text": "no ts"})
