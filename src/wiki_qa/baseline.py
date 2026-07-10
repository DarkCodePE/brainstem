"""Baseline regression gate for the wiki-health linter (issue UA-3).

The wiki already carries a large pre-existing backlog of broken wikilinks
(162 at first scan). Failing CI on the *absolute* count would wedge every
build, so instead we commit a **baseline** of known issue keys and fail only on
*regressions* — issues present in the current report but absent from the
baseline. New orphans, newly broken links, and new duplicate slugs are caught;
the historical backlog is acknowledged and burned down separately.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from wiki_qa.linter import HealthReport

BASELINE_VERSION = 1


@dataclass(frozen=True)
class Regression:
    """Issue keys that are new relative to the baseline."""

    orphans: tuple[str, ...]
    broken_wikilinks: tuple[str, ...]
    duplicate_slugs: tuple[str, ...]

    @property
    def total(self) -> int:
        return len(self.orphans) + len(self.broken_wikilinks) + len(self.duplicate_slugs)

    @property
    def has_regressions(self) -> bool:
        return self.total > 0


def report_keys(report: HealthReport) -> dict[str, list[str]]:
    """Project a report into sorted, stable keys per category."""
    return {
        "orphans": sorted(report.orphans),
        "broken_wikilinks": sorted(b.key() for b in report.broken_wikilinks),
        "duplicate_slugs": sorted(d.key() for d in report.duplicate_slugs),
    }


def baseline_payload(report: HealthReport) -> dict[str, Any]:
    """Serialisable baseline document for :func:`save_baseline`."""
    return {"version": BASELINE_VERSION, "keys": report_keys(report)}


def save_baseline(path: Path | str, report: HealthReport) -> None:
    """Write ``report``'s keys to ``path`` as the new accepted baseline."""
    payload = baseline_payload(report)
    Path(path).write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def load_baseline(path: Path | str) -> dict[str, list[str]]:
    """Load baseline keys; a missing file means "no issues accepted yet"."""
    p = Path(path)
    if not p.is_file():
        return {"orphans": [], "broken_wikilinks": [], "duplicate_slugs": []}
    data = json.loads(p.read_text(encoding="utf-8"))
    keys = data.get("keys", {})
    return {
        "orphans": list(keys.get("orphans", [])),
        "broken_wikilinks": list(keys.get("broken_wikilinks", [])),
        "duplicate_slugs": list(keys.get("duplicate_slugs", [])),
    }


def compute_regressions(report: HealthReport, baseline: dict[str, list[str]]) -> Regression:
    """Return issue keys present in ``report`` but not in ``baseline``."""
    current = report_keys(report)
    return Regression(
        orphans=tuple(k for k in current["orphans"] if k not in set(baseline["orphans"])),
        broken_wikilinks=tuple(
            k for k in current["broken_wikilinks"] if k not in set(baseline["broken_wikilinks"])
        ),
        duplicate_slugs=tuple(
            k for k in current["duplicate_slugs"] if k not in set(baseline["duplicate_slugs"])
        ),
    )
