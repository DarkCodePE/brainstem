"""AutoFetchMetrics — counter shape and serialisation."""

from __future__ import annotations

from datetime import UTC, datetime

from wiki_autofetch.metrics import AutoFetchMetrics, SourceMetrics


class TestSourceMetrics:
    def test_to_dict_includes_all_fields(self) -> None:
        m = SourceMetrics(name="gmail")
        d = m.to_dict()
        for key in (
            "name",
            "events_fetched",
            "errors",
            "rate_limited",
            "last_tick_at",
            "last_tick_duration_seconds",
            "last_error_class",
        ):
            assert key in d
        assert d["name"] == "gmail"
        assert d["events_fetched"] == 0
        assert d["last_tick_at"] is None

    def test_iso_serialises_datetime(self) -> None:
        m = SourceMetrics(name="github", last_tick_at=datetime(2026, 5, 22, 12, 0, tzinfo=UTC))
        d = m.to_dict()
        assert isinstance(d["last_tick_at"], str)
        assert "2026-05-22" in d["last_tick_at"]


class TestAutoFetchMetrics:
    def test_starts_empty(self) -> None:
        m = AutoFetchMetrics()
        assert m.total_ticks == 0
        assert m.total_events_fetched == 0
        assert m.total_errors == 0
        assert m.total_rate_limited == 0
        assert m.last_tick_at is None
        assert m.sources == {}

    def test_record_source_success_creates_source(self) -> None:
        m = AutoFetchMetrics()
        m.record_source_success("gmail", events=3, duration_seconds=0.5)
        assert "gmail" in m.sources
        assert m.sources["gmail"].events_fetched == 3
        assert m.total_events_fetched == 3
        assert m.sources["gmail"].last_tick_duration_seconds == 0.5
        assert m.sources["gmail"].last_error_class is None

    def test_record_source_success_accumulates(self) -> None:
        m = AutoFetchMetrics()
        m.record_source_success("gmail", events=3, duration_seconds=0.5)
        m.record_source_success("gmail", events=5, duration_seconds=0.6)
        assert m.sources["gmail"].events_fetched == 8
        assert m.total_events_fetched == 8

    def test_record_source_error_increments(self) -> None:
        m = AutoFetchMetrics()
        m.record_source_error("slack", error_class="HTTPError", duration_seconds=1.0)
        assert m.sources["slack"].errors == 1
        assert m.sources["slack"].last_error_class == "HTTPError"
        assert m.total_errors == 1

    def test_record_source_rate_limited(self) -> None:
        m = AutoFetchMetrics()
        m.record_source_rate_limited("gmail")
        m.record_source_rate_limited("gmail")
        m.record_source_rate_limited("github")
        assert m.sources["gmail"].rate_limited == 2
        assert m.sources["github"].rate_limited == 1
        assert m.total_rate_limited == 3

    def test_record_tick_updates_timestamps(self) -> None:
        m = AutoFetchMetrics()
        m.record_tick(duration_seconds=2.0)
        assert m.total_ticks == 1
        assert m.last_tick_duration_seconds == 2.0
        assert m.last_tick_at is not None
        m.record_tick(duration_seconds=3.0)
        assert m.total_ticks == 2

    def test_to_dict_shape(self) -> None:
        m = AutoFetchMetrics()
        m.record_source_success("gmail", events=2, duration_seconds=0.4)
        m.record_source_error("slack", error_class="HTTPError", duration_seconds=1.0)
        m.record_tick(duration_seconds=1.5)
        d = m.to_dict()
        # Top-level keys
        assert set(d.keys()) == {
            "total_ticks",
            "total_events_fetched",
            "total_errors",
            "total_rate_limited",
            "last_tick_at",
            "last_tick_duration_seconds",
            "sources",
        }
        # Source map
        assert "gmail" in d["sources"]
        assert "slack" in d["sources"]
        assert d["sources"]["gmail"]["events_fetched"] == 2
        assert d["sources"]["slack"]["errors"] == 1
        # No mutable internals leak through to_dict
        d["sources"]["gmail"]["events_fetched"] = 9999
        assert m.sources["gmail"].events_fetched == 2

    def test_explicit_timestamp_honoured(self) -> None:
        m = AutoFetchMetrics()
        fixed = datetime(2026, 5, 22, 12, 0, tzinfo=UTC)
        m.record_source_success("gmail", events=1, duration_seconds=0.1, at=fixed)
        assert m.sources["gmail"].last_tick_at == fixed
        m.record_tick(duration_seconds=0.2, at=fixed)
        assert m.last_tick_at == fixed
