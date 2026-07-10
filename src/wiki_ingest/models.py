from __future__ import annotations

import time
import uuid
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from typing import Any


def uuid7() -> str:
    ts_ms = int(time.time() * 1000)
    rand = uuid.uuid4().int & ((1 << 74) - 1)
    value = (ts_ms & ((1 << 48) - 1)) << 80
    value |= 0x7 << 76
    value |= (rand >> 12) << 64 & ((1 << 12) << 64)
    value |= 0x2 << 62
    value |= rand & ((1 << 62) - 1)
    return str(uuid.UUID(int=value))


def _utcnow_iso() -> str:
    return datetime.now(UTC).isoformat(timespec="seconds").replace("+00:00", "Z")


@dataclass
class IngestEvent:
    path: str
    rel_path: str
    bucket: str
    event_type: str
    mtime: str
    size: int
    sha256: str | None = None
    mime: str | None = None
    event_id: str = field(default_factory=uuid7)
    enqueued_at: str = field(default_factory=_utcnow_iso)
    attempts: int = 0
    status: str = "pending"
    last_error: str | None = None
    started_at: str | None = None
    finished_at: str | None = None

    def to_row(self) -> tuple[Any, ...]:
        return (
            self.event_id,
            self.path,
            self.rel_path,
            self.bucket,
            self.event_type,
            self.mtime,
            self.size,
            self.sha256,
            self.mime,
            self.enqueued_at,
            self.attempts,
            self.status,
            self.last_error,
            self.started_at,
            self.finished_at,
        )

    @classmethod
    def from_row(cls, row: tuple[Any, ...]) -> IngestEvent:
        return cls(
            event_id=row[0],
            path=row[1],
            rel_path=row[2],
            bucket=row[3],
            event_type=row[4],
            mtime=row[5],
            size=row[6],
            sha256=row[7],
            mime=row[8],
            enqueued_at=row[9],
            attempts=row[10],
            status=row[11],
            last_error=row[12],
            started_at=row[13],
            finished_at=row[14],
        )

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)
