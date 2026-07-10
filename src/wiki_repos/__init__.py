"""``wiki_repos`` — repo-as-knowledge-source bounded context (PRD-012 / ADR-022).

Turns a public GitHub URL into a synthesized wiki source page + per-repo code
graph + architecture diagram, WITHOUT running ``git clone`` on the user's
machine (tarball fetch → local digest + Understand-Anything graph). All cloned
content is treated as ``ingested-untrusted`` and flows through the existing
ADR-006 envelope on ``write_page``.

The single public entrypoint is ``service.ingest_github_repo`` (exposed as the
``ingest_github_repo`` MCP tool). Leaf modules:

- ``fetcher``          — URL validation (host allowlist, SSRF guard) + tarball fetch.
- ``digest``           — local file-tree + content digest (no clone, no GitPython).
- ``codegraph_runner`` — Understand-Anything graph over the extracted tree.
- ``diagram``          — deterministic Mermaid from the code graph.
- ``synthesize``       — compose the wiki source page.
- ``service``          — orchestration.
"""

from __future__ import annotations

from wiki_repos import errors
from wiki_repos.service import ingest_github_repo
from wiki_repos.types import (
    Commit,
    Digest,
    DigestStats,
    DraftMode,
    HistoryStats,
    IngestResult,
    PullRequest,
    RepoHistory,
    RepoRef,
)

__all__ = [
    "errors",
    "ingest_github_repo",
    "Commit",
    "Digest",
    "DigestStats",
    "DraftMode",
    "HistoryStats",
    "IngestResult",
    "PullRequest",
    "RepoHistory",
    "RepoRef",
]
