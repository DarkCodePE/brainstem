"""Typed errors for repo-as-knowledge-source ingestion (PRD-012 FR-9 / ADR-022).

Every failure surfaced to the MCP frontend is one of these — never a raw stack
trace or token. ``kind`` is the stable machine-readable tag the frontend shows.
"""
# ruff: noqa: N818 — class names ARE the PRD-012 FR-9 typed-error contract
# (InvalidUrl, PrivateOrUnreachable, Oversize, ...); the frontend and tests key
# off these exact names, so the "Error" suffix convention is waived here.

from __future__ import annotations


class WikiRepoError(RuntimeError):
    """Base for all repo-ingestion failures. Carries a stable ``kind`` tag."""

    kind = "WikiRepoError"


class InvalidUrl(WikiRepoError):
    """URL is not an accepted ``https://github.com/<owner>/<repo>`` form
    (file://, ssh, non-GitHub host, malformed path)."""

    kind = "InvalidUrl"


class PrivateOrUnreachable(WikiRepoError):
    """Repo is private, does not exist, or is unreachable on an unauthenticated
    probe. We fail closed rather than attempt an authenticated fetch on the
    public path (private repos are Phase 2, gated by an ADR-017 scope change)."""

    kind = "PrivateOrUnreachable"


class Oversize(WikiRepoError):
    """Repo (or its tarball) exceeds the configured size cap before/while fetch."""

    kind = "Oversize"


class FetchFailed(WikiRepoError):
    """Tarball download / extraction failed (network error, bad archive, timeout)."""

    kind = "FetchFailed"


class DigestFailed(WikiRepoError):
    """The local digest walker could not produce a usable digest."""

    kind = "DigestFailed"


class GraphFailed(WikiRepoError):
    """The code-graph build failed irrecoverably. NOTE: an *empty/unsupported*
    graph is NOT this error — that degrades to digest-only (PRD-012 R-5). This is
    only for a hard failure the caller must surface."""

    kind = "GraphFailed"


class SynthesisFailed(WikiRepoError):
    """Page synthesis could not produce a valid wiki page."""

    kind = "SynthesisFailed"
