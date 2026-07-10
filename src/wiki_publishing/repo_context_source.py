"""Lightweight post-context path (ADR-025) — draft a LinkedIn post about a repo
WITHOUT the heavy ADR-022 ingest.

The heavy path (ADR-022) clones/digests a repo, builds a code-graph, renders a
diagram, synthesises a wiki page, and persists it to the knowledge base — great
when you actually want the repo *in* your KB, but overkill when all you want is
"tell people about this tool I found". This module is the $0, transient
alternative: it fetches the repo's metadata + README *live* (via the shared
:func:`wiki_repos.github_meta.fetch_repo_meta`) and exposes it as a single
:class:`~wiki_publishing.linkedin_draft.WikiSnippet`, so the EXISTING ADR-024
drafter (:class:`~wiki_publishing.linkedin_draft.LinkedInDraftGenerator`) can
compose a ``post_type``/``focus`` draft from it with ZERO drafter changes.

Design
------
- :class:`RepoContextSource` implements the ``ContentSource`` protocol
  (``search(query, *, limit, categories) -> list[WikiSnippet]``). The "search"
  has already been resolved to one repo at construction time, so it *ignores*
  ``query``/``categories`` and always returns the same single snippet.
- The fetch is injectable (``fetcher``) so tests pass a fake returning a canned
  :class:`~wiki_repos.github_meta.RepoMeta` — no network in tests, ever.
- Draft-only (ADR-021) and transient: this never writes a wiki page. The only
  output is the ``outputs/linkedin/`` draft the drafter produces. No new OAuth
  scope, no heavy ingest.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence

from wiki_publishing.linkedin_draft import WikiSnippet
from wiki_repos.fetcher import parse_github_url
from wiki_repos.github_meta import RepoMeta, fetch_repo_meta, suggested_image

# (owner, repo) -> RepoMeta. Injectable so tests never hit the network.
RepoMetaFetcher = Callable[[str, str], RepoMeta]


def _compose_body(meta: RepoMeta) -> str:
    """Compose the post-relevant markdown block fed to the drafter.

    Order matters: the **description** (the value prop) leads, then the
    machine-readable facts (topics/stars/license/homepage/language) the drafter
    can weave in, then the README so ``focus="use"`` posts have real
    features/benchmarks to draw from. The canonical URL appears both as the
    snippet's ``page_path`` (so the drafter's URL-citation guarantee picks it up)
    and inline in the body for redundancy.
    """
    lines: list[str] = [f"# {meta.owner}/{meta.repo}", ""]

    if meta.description:
        lines += [meta.description, ""]

    facts: list[str] = []
    if meta.topics:
        facts.append(f"- **Topics:** {', '.join(meta.topics)}")
    if meta.stars:
        facts.append(f"- **Stars:** {meta.stars}")
    if meta.license:
        facts.append(f"- **License:** {meta.license}")
    if meta.language:
        facts.append(f"- **Language:** {meta.language}")
    if meta.homepage:
        facts.append(f"- **Homepage:** {meta.homepage}")
    facts.append(f"- **Repository:** {meta.canonical_url}")
    lines += [*facts, ""]

    if meta.readme:
        lines += ["## README", "", meta.readme]

    return "\n".join(lines).rstrip() + "\n"


class RepoContextSource:
    """A ``ContentSource`` over ONE live-fetched GitHub repo (ADR-025).

    Construct from an explicit ``owner``/``repo`` identity, or use
    :meth:`from_url` to parse a GitHub URL. The repo is resolved lazily on the
    first :meth:`search` call (so construction never touches the network), then
    cached for the lifetime of the instance.

    Parameters
    ----------
    owner:
        GitHub repo owner (user/org).
    repo:
        GitHub repo name.
    fetcher:
        Injectable ``(owner, repo) -> RepoMeta``. Defaults to the real
        :func:`wiki_repos.github_meta.fetch_repo_meta`. Tests pass a fake that
        returns a canned :class:`RepoMeta`.
    """

    def __init__(
        self,
        owner: str,
        repo: str,
        *,
        fetcher: RepoMetaFetcher = fetch_repo_meta,
    ) -> None:
        self._owner = owner
        self._repo = repo
        self._fetcher = fetcher
        self._cached: WikiSnippet | None = None
        self._meta: RepoMeta | None = None

    @classmethod
    def from_url(
        cls,
        url: str,
        *,
        fetcher: RepoMetaFetcher = fetch_repo_meta,
    ) -> RepoContextSource:
        """Build a :class:`RepoContextSource` from a GitHub repo URL.

        Uses :func:`wiki_repos.fetcher.parse_github_url` (host-allowlist + shape
        validation) to extract ``owner``/``repo``. Raises
        :class:`~wiki_repos.errors.InvalidUrl` for a non-conforming URL.
        """
        ref = parse_github_url(url)
        return cls(ref.owner, ref.repo, fetcher=fetcher)

    def _snippet(self) -> WikiSnippet:
        """Fetch (once, cached) the repo meta and build the single snippet.

        Propagates :class:`~wiki_repos.github_meta.RepoMetaError` from the
        fetcher so the MCP tool can map it to a clean "repo not found/unreachable"
        error.
        """
        if self._cached is None:
            self._meta = self._fetcher(self._owner, self._repo)
            self._cached = WikiSnippet(
                title=f"{self._meta.owner}/{self._meta.repo}",
                page_path=self._meta.canonical_url,
                body=_compose_body(self._meta),
            )
        return self._cached

    def suggested_image(self) -> str | None:
        """A relevant image URL to attach BY HAND to a post about this repo
        (ADR-025 / #154): the README's first real image, else the GitHub OG card.
        Returns ``None`` only if the repo meta can't be fetched. Fetches (cached)."""
        self._snippet()  # ensure meta is loaded/cached
        return suggested_image(self._meta) if self._meta is not None else None

    def search(
        self,
        query: str,
        *,
        limit: int = 3,
        categories: Sequence[str] | None = None,
    ) -> list[WikiSnippet]:
        """Return the single repo snippet, ignoring ``query``/``categories``.

        The "search" was already resolved to this one repo at construction time,
        so the query and category filters are no-ops — there is exactly one
        candidate. ``limit`` is honoured trivially (the list is always length 1).
        """
        return [self._snippet()]
