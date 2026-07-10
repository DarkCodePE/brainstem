"""
Tests for `AutoFetchDLQ` — record/clear, backoff window, attempt counter.
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

import pytest

from wiki_autofetch.dlq import MAX_BACKOFF_MINUTES, AutoFetchDLQ


@pytest.fixture
def dlq(tmp_path):
    d = AutoFetchDLQ(db_path=tmp_path / "dlq.db")
    yield d
    d.close()


def test_record_failure_returns_incrementing_attempt(dlq):
    assert dlq.record_failure("gmail", "TimeoutError") == 1
    assert dlq.record_failure("gmail", "TimeoutError") == 2
    assert dlq.record_failure("gmail", "TimeoutError") == 3


def test_record_success_resets_attempt_count(dlq):
    dlq.record_failure("gmail", "TimeoutError")
    dlq.record_failure("gmail", "TimeoutError")
    assert dlq.attempt_count("gmail") == 2
    removed = dlq.record_success("gmail")
    assert removed == 2
    assert dlq.attempt_count("gmail") == 0


def test_attempt_count_isolated_per_source(dlq):
    dlq.record_failure("gmail", "X")
    dlq.record_failure("gmail", "X")
    dlq.record_failure("github", "X")
    assert dlq.attempt_count("gmail") == 2
    assert dlq.attempt_count("github") == 1


def test_record_success_on_unknown_source_returns_zero(dlq):
    assert dlq.record_success("never_failed") == 0


def test_backoff_until_grows_exponentially(dlq):
    """attempt=1 → ~2 min, attempt=2 → ~4 min, attempt=3 → ~8 min …"""
    dlq.record_failure("gmail", "X")
    bo1 = dlq.backoff_until("gmail")
    assert bo1 is not None

    dlq.record_failure("gmail", "X")
    bo2 = dlq.backoff_until("gmail")
    assert bo2 is not None
    assert bo2 > bo1


def test_backoff_capped_at_max(dlq):
    """attempt=20 would otherwise blow past 1024 minutes; should clamp."""
    for _ in range(20):
        dlq.record_failure("gmail", "X")
    bo = dlq.backoff_until("gmail")
    assert bo is not None
    # Less than 1 hour 1 minute from now
    delta = bo - datetime.now(UTC)
    assert delta <= timedelta(minutes=MAX_BACKOFF_MINUTES + 1)


def test_no_backoff_when_no_failures(dlq):
    assert dlq.backoff_until("gmail") is None
    assert dlq.is_backed_off("gmail") is False


def test_is_backed_off_respects_window(dlq):
    dlq.record_failure("gmail", "X")
    # Within the (~2 min) backoff window
    assert dlq.is_backed_off("gmail") is True
    # After the window
    future = datetime.now(UTC) + timedelta(hours=2)
    assert dlq.is_backed_off("gmail", now=future) is False


def test_list_failures_most_recent_first(dlq):
    dlq.record_failure("gmail", "A", "first")
    dlq.record_failure("gmail", "B", "second")
    dlq.record_failure("github", "C", "third")
    entries = dlq.list_failures()
    assert len(entries) == 3
    assert entries[0].error_class == "C"
    assert entries[-1].error_class == "A"


def test_list_failures_per_source(dlq):
    dlq.record_failure("gmail", "A")
    dlq.record_failure("github", "B")
    gm = dlq.list_failures("gmail")
    assert len(gm) == 1
    assert gm[0].source_name == "gmail"


def test_clear_specific_source(dlq):
    dlq.record_failure("gmail", "X")
    dlq.record_failure("github", "X")
    removed = dlq.clear("gmail")
    assert removed == 1
    assert dlq.attempt_count("gmail") == 0
    assert dlq.attempt_count("github") == 1


def test_clear_all(dlq):
    dlq.record_failure("gmail", "X")
    dlq.record_failure("github", "Y")
    removed = dlq.clear()
    assert removed == 2
    assert dlq.list_failures() == []


def test_empty_source_name_rejected(dlq):
    with pytest.raises(ValueError):
        dlq.record_failure("", "X")
    with pytest.raises(ValueError):
        dlq.record_success("")


def test_dlq_persists_across_close_and_reopen(tmp_path):
    path = tmp_path / "dlq.db"
    d1 = AutoFetchDLQ(db_path=path)
    d1.record_failure("gmail", "X")
    d1.close()

    d2 = AutoFetchDLQ(db_path=path)
    try:
        assert d2.attempt_count("gmail") == 1
    finally:
        d2.close()
