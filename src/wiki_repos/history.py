"""Mine a repo's git history (merged PRs + classified commits) via the GitHub
REST API — the temporal leg of repo ingestion (PRD-014 / ADR-030).

Why this module exists: the snapshot path (digest + code-graph + diagram) answers
*what the code IS*. It cannot answer *why it is that way* or *how it evolved* —
that lives in merged pull requests and fix/refactor commits. This module adopts
**Repo2RLEnv's history-mining technique** (mine PR/commit metadata) but NOT its
purpose (RL tasks) and NOT its acquisition (it `git clone`s full history). We use
the GitHub REST API only, reusing the :data:`wiki_repos.fetcher.Opener` seam and
the same allowlisted host — preserving the ADR-022 **no-clone** invariant.

Degrade philosophy (mirrors ``codegraph_runner``): a missing history is the common
case (rate-limit, empty repo, private/unreachable, network), NOT an error. In every
such case we return ``None`` and let the caller fall back to snapshot-only synthesis.
This module NEVER raises for an expected failure and NEVER fails the overall ingest.

Pure stdlib only: ``urllib``, ``json``, ``re``. No new runtime dependency (ADR-030
Decision 1). The ``opener`` dependency is injectable so tests never touch the
network: ``(url: str) -> (status: int, body: bytes)``.
"""

from __future__ import annotations

import json
import logging
import re
import urllib.error
import urllib.request

from wiki_repos.fetcher import Opener
from wiki_repos.types import Commit, CommitKind, HistoryStats, PullRequest, RepoHistory, RepoRef

logger = logging.getLogger(__name__)

__all__ = ["mine_history", "classify_commit"]

#: GitHub API host — hardcoded (the allowlisted host); never taken from user input.
_API_HOST = "https://api.github.com"

_USER_AGENT = "second-brain-wiki/wiki_repos (+https://github.com/DarkCodePE/second-brain-wiki)"

#: Caps (PRD-014 FR-1/FR-2). One page each → ≤2 API calls (ADR-030 AC-5).
_DEFAULT_MAX_PRS = 20
_DEFAULT_MAX_COMMITS = 50
_MAX_BODY_CHARS = 280

#: Conventional-Commit prefix → kind. Order is irrelevant (exact prefix match).
_CONVENTIONAL_RE = re.compile(
    r"^(?P<kind>fix|feat|refactor|perf|security|sec|chore|docs|doc|test|tests|style|build|ci|revert)"
    r"(?:\([^)]*\))?(?P<bang>!)?:",
    flags=re.IGNORECASE,
)
_KIND_ALIASES: dict[str, CommitKind] = {
    "fix": "fix",
    "feat": "feat",
    "refactor": "refactor",
    "perf": "perf",
    "security": "security",
    "sec": "security",
    "chore": "chore",
    "docs": "docs",
    "doc": "docs",
    "test": "test",
    "tests": "test",
    "style": "chore",
    "build": "chore",
    "ci": "chore",
    "revert": "other",
}


def _default_opener(url: str) -> tuple[int, bytes]:
    """Minimal urllib GET → ``(status, body)`` for the GitHub API.

    Mirrors ``fetcher._default_opener``: sets a User-Agent (GitHub rejects empty
    UA), requests the v3 JSON media type, and a 15s timeout. No auth header is
    sent — Phase 1 mines public repos only (PRD-014 SR-3). A token-injecting
    opener is intentionally NOT implemented here; should one be added later it
    must attach the header ONLY for ``api.github.com`` over HTTPS and never log it.
    """
    req = urllib.request.Request(  # noqa: S310 — host is the hardcoded API host
        url,
        headers={"User-Agent": _USER_AGENT, "Accept": "application/vnd.github+json"},
        method="GET",
    )
    with urllib.request.urlopen(req, timeout=15) as resp:  # noqa: S310
        return int(resp.getcode() or 0), resp.read()


def classify_commit(subject: str) -> CommitKind:
    """Classify a commit subject line by its Conventional-Commit prefix.

    Falls back to ``"other"`` when the subject has no recognised prefix.
    """
    m = _CONVENTIONAL_RE.match(subject.strip())
    if not m:
        return "other"
    return _KIND_ALIASES.get(m.group("kind").lower(), "other")


def _get_json(open_: Opener, url: str) -> list | None:
    """GET ``url`` and parse a JSON array, or ``None`` on any failure/non-200.

    Degrade-first: rate-limit (403/429), 404, 5xx, network errors, and non-list
    bodies all return ``None`` with a one-line log — never raise.
    """
    try:
        status, body = open_(url)
    except urllib.error.HTTPError as exc:  # opener may raise instead of returning
        status, body = exc.code, b""
    except Exception as exc:  # noqa: BLE001 — network/DNS/timeout → degrade
        logger.info("history: fetch failed (%s) for %s", type(exc).__name__, url)
        return None

    if status != 200:
        logger.info("history: non-200 (%s) for %s", status, url)
        return None
    try:
        data = json.loads(body.decode("utf-8") if isinstance(body, bytes) else body)
    except (ValueError, UnicodeDecodeError):
        logger.info("history: unparseable JSON for %s", url)
        return None
    return data if isinstance(data, list) else None


#: Lines we drop from a PR body before taking the prose (template boilerplate).
_BOILERPLATE_LINE_RE = re.compile(r"^\s*(?:#{1,6}\s|[-*]\s*\[[ xX]\]|<!--|-->|<!--.*-->)")
_HTML_COMMENT_RE = re.compile(r"<!--.*?-->", flags=re.DOTALL)


def _excerpt(text: str | None) -> str:
    """First prose paragraph of a PR body — boilerplate-stripped, collapsed, capped.

    Strips common PR-template noise (markdown headings, checkbox lines, HTML
    comments) so the excerpt carries rationale rather than the template, then
    caps on a word boundary with an ellipsis when truncated.
    """
    if not text:
        return ""
    # Drop HTML comments wholesale (they can span lines), then filter noise lines.
    text = _HTML_COMMENT_RE.sub(" ", text)
    prose_lines = [
        line for line in text.splitlines() if line.strip() and not _BOILERPLATE_LINE_RE.match(line)
    ]
    collapsed = " ".join(" ".join(prose_lines).split())
    if not collapsed:
        return ""
    if len(collapsed) <= _MAX_BODY_CHARS:
        return collapsed
    cut = collapsed[:_MAX_BODY_CHARS].rsplit(" ", 1)[0].rstrip(",.;:")
    return f"{cut}…"


def _parse_prs(raw: list, limit: int) -> list[PullRequest]:
    """Keep only MERGED PRs (non-null ``merged_at``), most-recently-MERGED first, capped.

    The ``/pulls`` feed is sorted by ``updated`` (a recently-commented old merge can
    outrank a newer merge), so after the merged-only filter we re-sort by ``merged_at``
    descending and *then* cap — the section's 'newest first' claim is honest (FR-1).
    """
    merged: list[PullRequest] = []
    for item in raw:
        if not isinstance(item, dict) or not item.get("merged_at"):
            continue
        labels = tuple(
            lbl.get("name", "")
            for lbl in (item.get("labels") or [])
            if isinstance(lbl, dict) and lbl.get("name")
        )
        merged.append(
            PullRequest(
                number=int(item.get("number", 0)),
                title=str(item.get("title", "")).strip(),
                merged_at=str(item.get("merged_at", "")),
                author=str((item.get("user") or {}).get("login", "")),
                labels=labels,
                body_excerpt=_excerpt(item.get("body")),
            )
        )
    # Sort by merged_at desc (ISO-8601 sorts lexically); break ties on PR number.
    merged.sort(key=lambda pr: (pr.merged_at, pr.number), reverse=True)
    return merged[:limit]


def _parse_commits(raw: list, limit: int) -> list[Commit]:
    """Classify each commit by its subject's Conventional-Commit prefix."""
    out: list[Commit] = []
    for item in raw[:limit]:
        if not isinstance(item, dict):
            continue
        commit = item.get("commit") or {}
        message = str(commit.get("message", ""))
        subject = message.splitlines()[0].strip() if message else ""
        # Chain on the DATE itself (not dict presence) so a partial author object
        # missing 'date' still falls back to committer.date.
        author = commit.get("author") or {}
        committer = commit.get("committer") or {}
        date = str(author.get("date") or committer.get("date") or "")
        out.append(
            Commit(
                sha=str(item.get("sha", ""))[:7],
                summary=subject,
                kind=classify_commit(subject),
                date=date,
            )
        )
    return out


def mine_history(
    ref: RepoRef,
    *,
    opener: Opener | None = None,
    max_prs: int = _DEFAULT_MAX_PRS,
    max_commits: int = _DEFAULT_MAX_COMMITS,
) -> RepoHistory | None:
    """Mine merged PRs + classified recent commits for ``ref`` via the GitHub API.

    Returns a :class:`RepoHistory`, or ``None`` when nothing usable could be mined
    (both legs empty/failed) — the caller degrades to snapshot-only synthesis.

    Acquisition is the GitHub REST API ONLY (ADR-030): two GET calls against the
    hardcoded ``api.github.com`` host (the allowlist), built from the already-
    validated ``ref.owner``/``ref.repo``. NO ``git clone`` is ever invoked.

    Args:
        ref: Validated repo reference (from ``fetcher.parse_github_url``).
        opener: Injectable ``(url) -> (status, body)``. Tests MUST inject a fake.
        max_prs: Cap on merged PRs to keep (one API page; FR-1).
        max_commits: Cap on recent commits to classify (one API page; FR-2).
    """
    open_ = opener or _default_opener
    base = f"{_API_HOST}/repos/{ref.owner}/{ref.repo}"

    pulls_raw = _get_json(
        open_,
        f"{base}/pulls?state=closed&sort=updated&direction=desc&per_page={max_prs}",
    )
    commits_raw = _get_json(open_, f"{base}/commits?per_page={max_commits}")

    if pulls_raw is None and commits_raw is None:
        return None  # both legs unreachable → nothing to synthesize; degrade.

    merged_prs = _parse_prs(pulls_raw, max_prs) if pulls_raw else []
    commits = _parse_commits(commits_raw, max_commits) if commits_raw else []

    if not merged_prs and not commits:
        return None  # reachable but empty (e.g. brand-new repo) → degrade.

    # 'truncated' is an honest page-full heuristic (FR-4): a returned page at the
    # cap MAY have more behind it. We never silently drop without flagging — we err
    # toward flagging. (Not a confirmed has-more signal; no Link-header read.)
    truncated = (pulls_raw is not None and len(pulls_raw) >= max_prs) or (
        commits_raw is not None and len(commits_raw) >= max_commits
    )
    return RepoHistory(
        merged_prs=tuple(merged_prs),
        commits=tuple(commits),
        stats=HistoryStats(n_prs=len(merged_prs), n_commits=len(commits), truncated=truncated),
    )
