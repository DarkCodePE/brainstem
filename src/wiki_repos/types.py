"""Shared value types for repo ingestion — the contract every leaf module shares.

These dataclasses are the frozen interface between ``fetcher`` → ``digest`` →
``codegraph_runner`` → ``diagram`` → ``synthesize`` → ``service``. Changing a
field is a contract change; treat it as such.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

DraftMode = Literal["showcase", "experiential"]
"""How a downstream LinkedIn draft should be angled (ADR-021 tie-in):
- ``showcase``     — external repo: third-person, "here is this tool and why it matters".
- ``experiential`` — the user's own / a repo they used: first-person, "I used this and learned…".
"""


@dataclass(frozen=True, slots=True)
class RepoRef:
    """A validated reference to a public GitHub repository.

    Only produced by ``fetcher.parse_github_url`` after host-allowlist + shape
    validation — never constructed from raw user input elsewhere.
    """

    owner: str
    repo: str
    host: str = "github.com"
    ref: str | None = None
    """Branch/tag/commit, or None for the repo's default branch."""
    subpath: str | None = None
    """Optional ``/tree/<ref>/<subpath>`` sub-directory; None = whole repo."""

    @property
    def slug(self) -> str:
        """Filesystem-safe page slug: ``owner-repo`` lowercased."""
        return f"{self.owner}-{self.repo}".lower()

    @property
    def graph_dirname(self) -> str:
        """Per-repo graph store dir name: ``owner__repo`` (PRD-012 FR-4)."""
        return f"{self.owner}__{self.repo}"

    @property
    def canonical_url(self) -> str:
        return f"https://{self.host}/{self.owner}/{self.repo}"


@dataclass(frozen=True, slots=True)
class DigestStats:
    """Accounting for a digest run — surfaced so truncation is never silent."""

    n_files: int
    n_bytes: int
    est_tokens: int
    truncated: bool
    skipped_dirs: tuple[str, ...] = ()
    """Top-level dirs skipped wholesale (e.g. node_modules, .git)."""


@dataclass(frozen=True, slots=True)
class Digest:
    """gitingest-shaped digest produced locally (no clone): summary, file tree,
    concatenated text content, plus accounting stats."""

    summary: str
    tree: str
    content: str
    stats: DigestStats


# --------------------------------------------------------------------------- #
# Git-history mining (PRD-014 / ADR-030) — the temporal leg.
# Mined from the GitHub REST API (NO clone); all frozen, same contract
# discipline as ``Digest``.
# --------------------------------------------------------------------------- #
CommitKind = Literal[
    "fix", "feat", "refactor", "perf", "security", "chore", "docs", "test", "other"
]
"""Conventional-Commit classification of a commit's subject line."""


@dataclass(frozen=True, slots=True)
class PullRequest:
    """A merged pull request, as mined from ``/repos/{o}/{r}/pulls`` (PRD-014 FR-1)."""

    number: int
    title: str
    merged_at: str
    """ISO-8601 merge timestamp (the API's ``merged_at``)."""
    author: str = ""
    labels: tuple[str, ...] = ()
    body_excerpt: str = ""
    """Truncated PR body — design rationale, untrusted (ADR-006 envelope applies)."""


@dataclass(frozen=True, slots=True)
class Commit:
    """A recent commit, classified by Conventional-Commit type (PRD-014 FR-2)."""

    sha: str
    """Short (7-char) SHA."""
    summary: str
    """First line of the commit message."""
    kind: CommitKind
    date: str = ""
    """ISO-8601 author/commit date, or '' if unavailable."""


@dataclass(frozen=True, slots=True)
class HistoryStats:
    """Accounting for a history-mining run — truncation is never silent (FR-4)."""

    n_prs: int
    n_commits: int
    truncated: bool


@dataclass(frozen=True, slots=True)
class RepoHistory:
    """The mined temporal dimension of a repo: merged PRs + classified commits."""

    merged_prs: tuple[PullRequest, ...]
    commits: tuple[Commit, ...]
    stats: HistoryStats

    @property
    def kind_counts(self) -> dict[str, int]:
        """Commit count per Conventional-Commit kind, descending by count then name."""
        counts: dict[str, int] = {}
        for c in self.commits:
            counts[c.kind] = counts.get(c.kind, 0) + 1
        return dict(sorted(counts.items(), key=lambda kv: (-kv[1], kv[0])))


@dataclass(frozen=True, slots=True)
class IngestResult:
    """Final result of ``ingest_github_repo`` — returned to the MCP frontend."""

    ref: RepoRef
    page_path: str
    mode: DraftMode
    graph_path: str | None = None
    diagram_present: bool = False
    diagram_image_path: str | None = None
    """Path to the rendered diagram PNG (ADR-023 Phase 1), or None if not rendered."""
    digest_stats: DigestStats | None = None
    graph_summary: dict = field(default_factory=dict)
    history_present: bool = False
    """True when the git-history leg (PRD-014 / ADR-030) produced an evolution section."""
    notes: tuple[str, ...] = ()
    """Human-readable degrade/truncation notes (e.g. 'graph: unavailable')."""
