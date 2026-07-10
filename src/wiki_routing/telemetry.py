"""
SQLite-backed router telemetry per issue #37 AC:
"Telemetry: ``sbw doctor`` shows tier distribution + rolling cost."

The in-process `CostBudget` keeps the *active-process* day counter, but
it doesn't survive restarts and doesn't expose the tier breakdown the AC
asks for. This module is the persistent layer.

Schema::

    CREATE TABLE router_calls (
        id            INTEGER PRIMARY KEY AUTOINCREMENT,
        ts            TEXT NOT NULL,        -- ISO-8601 UTC ms
        tier          TEXT NOT NULL,        -- "reasoning" / "fast" / "vision"
        backend_label TEXT NOT NULL,
        cost_usd      REAL NOT NULL,
        success       INTEGER NOT NULL      -- 1 / 0
    );
"""

from __future__ import annotations

import sqlite3
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from pathlib import Path

DEFAULT_DB_PATH = Path.home() / ".sbw" / "state" / "router_telemetry.db"

_SCHEMA = """
CREATE TABLE IF NOT EXISTS router_calls (
    id            INTEGER PRIMARY KEY AUTOINCREMENT,
    ts            TEXT NOT NULL,
    tier          TEXT NOT NULL,
    backend_label TEXT NOT NULL,
    cost_usd      REAL NOT NULL,
    success       INTEGER NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_calls_ts ON router_calls(ts);
CREATE INDEX IF NOT EXISTS idx_calls_tier ON router_calls(tier);
"""


@dataclass(frozen=True)
class TierStats:
    tier: str
    calls: int
    cost_usd: float
    success_rate: float


class RouterTelemetry:
    """Synchronous SQLite logger. The router writes one row per call;
    `sbw doctor` reads the aggregates."""

    def __init__(self, db_path: Path | None = None) -> None:
        # Resolve at call time, not at function-definition time, so tests
        # that monkeypatch ``DEFAULT_DB_PATH`` (or HOME) take effect even
        # though this module has already been imported.
        self.db_path = Path(db_path) if db_path is not None else DEFAULT_DB_PATH
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(str(self.db_path))
        self._conn.executescript(_SCHEMA)
        self._conn.commit()

    def close(self) -> None:
        self._conn.close()

    def record(
        self,
        *,
        tier: str,
        backend_label: str,
        cost_usd: float,
        success: bool,
    ) -> None:
        self._conn.execute(
            "INSERT INTO router_calls (ts, tier, backend_label, cost_usd, success) "
            "VALUES (?, ?, ?, ?, ?)",
            (_utcnow_iso(), tier, backend_label, float(cost_usd), 1 if success else 0),
        )
        self._conn.commit()

    # ------------------------------------------------------------------ #
    # Queries — used by `sbw doctor` and `sbw router status`             #
    # ------------------------------------------------------------------ #

    def rolling_cost_usd(self, window_hours: int = 24) -> float:
        """Sum of `cost_usd` over the last `window_hours`."""
        cutoff = (
            (datetime.now(UTC) - timedelta(hours=window_hours))
            .isoformat(timespec="milliseconds")
            .replace("+00:00", "Z")
        )
        row = self._conn.execute(
            "SELECT COALESCE(SUM(cost_usd), 0) FROM router_calls WHERE ts >= ?",
            (cutoff,),
        ).fetchone()
        return float(row[0]) if row else 0.0

    def tier_distribution(self, window_hours: int = 24) -> list[TierStats]:
        """Per-tier call count + cost + success rate over the last window."""
        cutoff = (
            (datetime.now(UTC) - timedelta(hours=window_hours))
            .isoformat(timespec="milliseconds")
            .replace("+00:00", "Z")
        )
        rows = self._conn.execute(
            """
            SELECT tier,
                   COUNT(*) AS calls,
                   COALESCE(SUM(cost_usd), 0) AS cost_usd,
                   AVG(success) AS success_rate
            FROM router_calls
            WHERE ts >= ?
            GROUP BY tier
            ORDER BY calls DESC
            """,
            (cutoff,),
        ).fetchall()
        return [
            TierStats(
                tier=str(r[0]),
                calls=int(r[1]),
                cost_usd=float(r[2]),
                success_rate=float(r[3] or 0.0),
            )
            for r in rows
        ]

    def total_calls(self) -> int:
        row = self._conn.execute("SELECT COUNT(*) FROM router_calls").fetchone()
        return int(row[0]) if row else 0

    def clear(self) -> int:
        cur = self._conn.execute("DELETE FROM router_calls")
        self._conn.commit()
        return int(cur.rowcount)


def _utcnow_iso() -> str:
    return datetime.now(UTC).isoformat(timespec="milliseconds").replace("+00:00", "Z")


__all__ = ["DEFAULT_DB_PATH", "RouterTelemetry", "TierStats"]
