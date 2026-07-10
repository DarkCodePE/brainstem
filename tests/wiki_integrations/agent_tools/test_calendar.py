"""
Tests for `CalendarIntegration`: list (window), get, search, dedup,
timestamp parsing (timed + all-day + dict shapes), recurring instance
preservation.
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

import pytest

from wiki_integrations.agent_tools.calendar import (
    WINDOW_BACK,
    WINDOW_FORWARD,
    CalendarIntegration,
    _parse_event_ts,
)

from .conftest import FakeBridge


def _now() -> datetime:
    return datetime.now(UTC)


def _iso(dt: datetime) -> str:
    """Render `dt` in the ``YYYY-MM-DDTHH:MM:SS+00:00`` shape Calendar uses."""
    return dt.isoformat()


# Build payload anchored on "now" so the window filter is deterministic
# whatever the test clock says.
def _build_payload():
    now = _now()
    return [
        {
            "id": "evt_in_window",
            "summary": "Q3 planning",
            "description": "Goals and dependencies",
            "location": "Zoom",
            "htmlLink": "https://calendar.google.com/event?eid=evt_in_window",
            "start": {"dateTime": _iso(now + timedelta(days=2)), "timeZone": "UTC"},
            "end": {"dateTime": _iso(now + timedelta(days=2, hours=1))},
            "updated": _iso(now - timedelta(hours=1)),
            "organizer": {"email": "alice@example.com"},
            "calendarId": "primary",
        },
        {
            "id": "evt_yesterday",
            "summary": "1:1 with manager",
            "description": "Weekly sync",
            "location": "",
            "htmlLink": "https://calendar.google.com/event?eid=evt_yesterday",
            "start": {"dateTime": _iso(now - timedelta(days=1))},
            "end": {"dateTime": _iso(now - timedelta(days=1) + timedelta(minutes=30))},
            "updated": _iso(now - timedelta(days=1)),
            "organizer": "manager@example.com",
        },
        # Out of window — 30 days in the future.
        {
            "id": "evt_far_future",
            "summary": "Off-site (far)",
            "start": {"dateTime": _iso(now + timedelta(days=30))},
            "end": {"dateTime": _iso(now + timedelta(days=30, hours=8))},
            "updated": _iso(now - timedelta(days=10)),
        },
        # Out of window — 30 days in the past.
        {
            "id": "evt_long_past",
            "summary": "Q2 retro",
            "start": {"dateTime": _iso(now - timedelta(days=30))},
            "end": {"dateTime": _iso(now - timedelta(days=30) + timedelta(hours=1))},
            "updated": _iso(now - timedelta(days=30)),
        },
        # All-day event (no time component, just date).
        {
            "id": "evt_all_day",
            "summary": "Public holiday",
            "start": {"date": (now + timedelta(days=5)).date().isoformat()},
            "end": {"date": (now + timedelta(days=6)).date().isoformat()},
            "updated": _iso(now - timedelta(days=2)),
        },
        # Recurring event — two instances. Both must survive because they
        # have different start times even though they share recurringEventId.
        {
            "id": "evt_recur_1",
            "summary": "Standup",
            "recurringEventId": "evt_recur_master",
            "start": {"dateTime": _iso(now + timedelta(days=1, hours=9))},
            "end": {"dateTime": _iso(now + timedelta(days=1, hours=9, minutes=15))},
            "updated": _iso(now - timedelta(hours=2)),
        },
        {
            "id": "evt_recur_2",
            "summary": "Standup",
            "recurringEventId": "evt_recur_master",
            "start": {"dateTime": _iso(now + timedelta(days=2, hours=9))},
            "end": {"dateTime": _iso(now + timedelta(days=2, hours=9, minutes=15))},
            "updated": _iso(now - timedelta(hours=2)),
        },
    ]


def _make(secret_store, audit_jsonl, audit_md_calendar, payload=None):
    bridge = FakeBridge(payloads={"calendar": payload or _build_payload()})
    cal = CalendarIntegration(
        bridge=bridge,
        store=secret_store,
        audit_jsonl=audit_jsonl,
        audit_md=audit_md_calendar,
    )
    return cal, bridge


# --------------------------------------------------------------------------- #
# list — window filter is the headline AC                                     #
# --------------------------------------------------------------------------- #


@pytest.mark.asyncio
async def test_list_returns_only_events_in_window(secret_store, audit_jsonl, audit_md_calendar):
    cal, _ = _make(secret_store, audit_jsonl, audit_md_calendar)
    await cal.connect()
    items = await cal.list(limit=50)
    ids = {i.id for i in items}
    # In window: the planning event, the yesterday 1:1, the all-day, two recur instances.
    assert "evt_in_window" in ids
    assert "evt_yesterday" in ids
    assert "evt_all_day" in ids
    assert "evt_recur_1" in ids
    assert "evt_recur_2" in ids
    # Out of window: 30d future and 30d past are filtered out.
    assert "evt_far_future" not in ids
    assert "evt_long_past" not in ids


@pytest.mark.asyncio
async def test_list_since_floored_to_window_lower_bound(
    secret_store, audit_jsonl, audit_md_calendar
):
    cal, _ = _make(secret_store, audit_jsonl, audit_md_calendar)
    await cal.connect()
    # Caller asks for "everything since a year ago" — silently floored to
    # now - 7d, so evt_long_past (30d back) must still be excluded.
    items = await cal.list(since=_now() - timedelta(days=365))
    assert all(i.id != "evt_long_past" for i in items)


@pytest.mark.asyncio
async def test_list_since_tightens_lower_bound(secret_store, audit_jsonl, audit_md_calendar):
    cal, _ = _make(secret_store, audit_jsonl, audit_md_calendar)
    await cal.connect()
    # Tighter `since` than the window's lower bound — evt_yesterday should
    # disappear because it's before this since.
    items = await cal.list(since=_now() + timedelta(hours=1))
    assert all(i.id != "evt_yesterday" for i in items)


@pytest.mark.asyncio
async def test_list_window_constants_match_issue_ac(secret_store, audit_jsonl, audit_md_calendar):
    """Lock the [-7d, +14d] policy per issue #32 AC."""
    assert WINDOW_BACK == timedelta(days=7)
    assert WINDOW_FORWARD == timedelta(days=14)


# --------------------------------------------------------------------------- #
# Recurring events                                                            #
# --------------------------------------------------------------------------- #


@pytest.mark.asyncio
async def test_recurring_instances_kept_as_separate_items(
    secret_store, audit_jsonl, audit_md_calendar
):
    cal, _ = _make(secret_store, audit_jsonl, audit_md_calendar)
    await cal.connect()
    items = await cal.list(limit=50)
    recurs = [i for i in items if i.metadata.get("recurring_event_id") == "evt_recur_master"]
    assert len(recurs) == 2
    # The two instances have different sha256 idempotency keys because
    # updated is the same but id differs, OR the start differs.
    shas = {i.metadata["sha256"] for i in recurs}
    assert len(shas) == 2


# --------------------------------------------------------------------------- #
# Idempotency                                                                 #
# --------------------------------------------------------------------------- #


@pytest.mark.asyncio
async def test_per_event_sha256_idempotency(secret_store, audit_jsonl, audit_md_calendar):
    cal, _ = _make(secret_store, audit_jsonl, audit_md_calendar)
    await cal.connect()
    items = await cal.list()
    for item in items:
        sha = item.metadata.get("sha256")
        assert isinstance(sha, str) and len(sha) == 64


# --------------------------------------------------------------------------- #
# get / search                                                                #
# --------------------------------------------------------------------------- #


@pytest.mark.asyncio
async def test_get_returns_event_by_id(secret_store, audit_jsonl, audit_md_calendar):
    cal, _ = _make(secret_store, audit_jsonl, audit_md_calendar)
    await cal.connect()
    item = await cal.get("evt_in_window")
    assert item.id == "evt_in_window"
    assert item.title == "Q3 planning"
    assert item.metadata["location"] == "Zoom"


@pytest.mark.asyncio
async def test_get_unknown_raises(secret_store, audit_jsonl, audit_md_calendar):
    cal, _ = _make(secret_store, audit_jsonl, audit_md_calendar)
    await cal.connect()
    with pytest.raises(KeyError):
        await cal.get("evt_does_not_exist")


@pytest.mark.asyncio
async def test_search_matches_title_and_location(secret_store, audit_jsonl, audit_md_calendar):
    cal, _ = _make(secret_store, audit_jsonl, audit_md_calendar)
    await cal.connect()
    by_title = await cal.search("planning")
    assert any(i.id == "evt_in_window" for i in by_title.items)
    by_location = await cal.search("zoom")
    assert any(i.id == "evt_in_window" for i in by_location.items)


@pytest.mark.asyncio
async def test_search_skips_out_of_window(secret_store, audit_jsonl, audit_md_calendar):
    cal, _ = _make(secret_store, audit_jsonl, audit_md_calendar)
    await cal.connect()
    # "off-site" only matches evt_far_future which is 30d ahead — outside
    # the window, so the search must return nothing.
    result = await cal.search("off-site")
    assert result.items == ()


@pytest.mark.asyncio
async def test_search_empty_query_rejected(secret_store, audit_jsonl, audit_md_calendar):
    cal, _ = _make(secret_store, audit_jsonl, audit_md_calendar)
    await cal.connect()
    with pytest.raises(ValueError, match="non-empty"):
        await cal.search("")


# --------------------------------------------------------------------------- #
# Scope — the lock-in test                                                    #
# --------------------------------------------------------------------------- #


def test_scopes_match_locked_policy(secret_store, audit_jsonl, audit_md_calendar):
    cal, _ = _make(secret_store, audit_jsonl, audit_md_calendar)
    # ADR-017 §Per-provider OAuth scope table locks calendar to read-only.
    assert cal.scopes == ("https://www.googleapis.com/auth/calendar.readonly",)


# --------------------------------------------------------------------------- #
# Timestamp parser unit tests                                                 #
# --------------------------------------------------------------------------- #


def test_parse_event_ts_handles_datetime_dict():
    dt = _parse_event_ts({"dateTime": "2026-05-26T15:00:00+00:00"})
    assert dt == datetime(2026, 5, 26, 15, 0, 0, tzinfo=UTC)


def test_parse_event_ts_handles_all_day_date():
    dt = _parse_event_ts({"date": "2026-05-26"})
    assert dt == datetime(2026, 5, 26, 0, 0, 0, tzinfo=UTC)


def test_parse_event_ts_handles_bare_iso_string():
    dt = _parse_event_ts("2026-05-26T15:00:00Z")
    assert dt == datetime(2026, 5, 26, 15, 0, 0, tzinfo=UTC)


def test_parse_event_ts_falls_back_on_garbage():
    dt = _parse_event_ts("not a timestamp")
    assert dt == datetime.fromtimestamp(0, tz=UTC)
