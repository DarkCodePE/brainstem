"""GitHub repo metadata + README fetch (ADR-025) — the $0 value-prop source.

A single, dependency-free GitHub REST fetch that returns the things a LinkedIn
post actually needs — the repo's *purpose* and *value*, which the heavy ADR-022
ingest (code-graph) does not capture: ``description``, ``topics``, ``stars``,
``homepage``, ``license``, ``language``, and the README markdown.

Shared by TWO callers:
- the lightweight post-context path (``RepoContextSource``, ADR-025) — draft a
  post from this alone, no ingest; and
- the heavy ADR-022 ingest, which now folds ``description``/``topics`` into the
  synthesized page so the KB page also states "what it does".

Public repos need no token (unauthenticated GitHub API); a ``GITHUB_TOKEN`` in
the environment is used only to lift rate limits if present. No new OAuth scope.
"""

from __future__ import annotations

import base64
import json
import os
import re
import urllib.error
import urllib.parse
import urllib.request
from collections.abc import Callable
from dataclasses import dataclass, field

_API = "https://api.github.com"
_USER_AGENT = "second-brain-wiki/wiki_repos (+https://github.com/DarkCodePE/second-brain-wiki)"

# (url) -> (status_code, body_bytes). Injectable so tests never hit the network.
Opener = Callable[[str], "tuple[int, bytes]"]


class RepoMetaError(RuntimeError):
    """The repo metadata/README could not be fetched (private/unreachable/404)."""

    kind = "RepoMetaError"


@dataclass(frozen=True, slots=True)
class RepoMeta:
    """The post-relevant facts about a public GitHub repo (ADR-025)."""

    owner: str
    repo: str
    description: str = ""
    topics: tuple[str, ...] = ()
    stars: int = 0
    homepage: str = ""
    license: str = ""
    language: str = ""
    default_branch: str = "main"
    readme: str = ""
    extra: dict = field(default_factory=dict)

    @property
    def canonical_url(self) -> str:
        return f"https://github.com/{self.owner}/{self.repo}"


# An image in a README: markdown ``![alt](url)`` OR HTML ``<img src="url">``.
# Big repos (PaddleOCR, etc.) put their banner/screenshot in an HTML <img>, not
# markdown — match both so we surface the real image, not just the OG fallback.
_MD_IMAGE_RE = re.compile(
    r"!\[[^\]]*\]\(\s*<?([^)>\s]+)>?"
    r"|<img\b[^>]*?\bsrc\s*=\s*[\"']([^\"']+)[\"']",
    re.IGNORECASE,
)
# Image URLs that make poor post images (badges/shields/CI/trendshift, not a
# real banner/screenshot).
_BADGE_HINTS = (
    "shields.io",
    "badge",
    "trendshift",
    "/actions/",
    "circleci",
    "codecov",
    "img.shields",
)


def og_card_url(owner: str, repo: str) -> str:
    """GitHub's auto-generated OpenGraph social card (name+desc+stars+language).

    Always available for a public repo; a clean fallback image for a post when the
    README has no banner/screenshot."""
    return f"https://opengraph.githubassets.com/1/{owner}/{repo}"


def first_readme_image(readme: str, owner: str, repo: str, branch: str = "main") -> str | None:
    """First non-badge image URL in the README (a banner/screenshot/demo), with
    relative paths resolved to ``raw.githubusercontent.com``. ``None`` if none."""
    for m in _MD_IMAGE_RE.finditer(readme or ""):
        url = (m.group(1) or m.group(2) or "").strip()  # markdown OR <img src>
        if not url or any(h in url.lower() for h in _BADGE_HINTS):
            continue
        if url.startswith(("http://", "https://")):
            return url
        rel = url.lstrip("./")
        return f"https://raw.githubusercontent.com/{owner}/{repo}/{branch}/{rel}"
    return None


def suggested_image(meta: RepoMeta) -> str:
    """Best image to suggest for a manual LinkedIn attach (ADR-025 / #154):
    the README's first real image if present, else the GitHub OG social card."""
    return first_readme_image(
        meta.readme, meta.owner, meta.repo, meta.default_branch or "main"
    ) or og_card_url(meta.owner, meta.repo)


def _default_opener(url: str) -> tuple[int, bytes]:
    headers = {"User-Agent": _USER_AGENT, "Accept": "application/vnd.github+json"}
    token = os.environ.get("GITHUB_TOKEN") or os.environ.get("GH_TOKEN")
    if token:
        headers["Authorization"] = f"Bearer {token}"
    req = urllib.request.Request(url, headers=headers)  # noqa: S310 — https GitHub API only
    try:
        with urllib.request.urlopen(req, timeout=15) as resp:  # noqa: S310
            return resp.status, resp.read()
    except urllib.error.HTTPError as exc:
        return exc.code, b""
    except Exception as exc:  # noqa: BLE001
        raise RepoMetaError(f"network error fetching {url}: {type(exc).__name__}") from exc


def _get_json(url: str, opener: Opener) -> dict | None:
    status, body = opener(url)
    if status != 200 or not body:
        return None
    try:
        data = json.loads(body)
    except (ValueError, TypeError):
        return None
    return data if isinstance(data, dict) else None


def resolve_repo(query: str, *, opener: Opener | None = None) -> str:
    """Resolve a free-text repo name to a canonical ``https://github.com/<owner>/<repo>``
    URL via the GitHub repository Search API (ADR-025 / #154), so a user can say
    "post about headroom" without pasting a URL.

    Selection: search ``sort=stars desc``, then prefer a result whose repo *name*
    matches the query exactly (case-insensitive); otherwise the most-starred hit.
    The chosen repo is returned so the caller can surface it for confirmation.

    Raises :class:`RepoMetaError` when no repo matches. ``opener`` is injectable.
    """
    q = (query or "").strip()
    if not q:
        raise RepoMetaError("empty repo query")
    op = opener or _default_opener
    encoded = urllib.parse.quote(q)
    url = f"{_API}/search/repositories?q={encoded}&sort=stars&order=desc&per_page=10"
    data = _get_json(url, op)
    items = (data or {}).get("items") or []
    if not items:
        raise RepoMetaError(f"no GitHub repository matched query {query!r}")

    ql = q.lower()
    exact = [it for it in items if isinstance(it, dict) and (it.get("name") or "").lower() == ql]
    pick = exact[0] if exact else items[0]
    full_name = pick.get("full_name") if isinstance(pick, dict) else None
    if not full_name:
        raise RepoMetaError(f"GitHub search result for {query!r} had no full_name")
    return f"https://github.com/{full_name}"


def fetch_repo_meta(
    owner: str,
    repo: str,
    *,
    opener: Opener | None = None,
    max_readme_chars: int = 18_000,
) -> RepoMeta:
    """Fetch a public repo's metadata + README. Raises :class:`RepoMetaError`
    when the repo can't be read (private/unreachable/404).

    ``opener`` is injectable ``(url) -> (status, bytes)`` for hermetic tests.
    """
    op = opener or _default_opener

    info = _get_json(f"{_API}/repos/{owner}/{repo}", op)
    if info is None:
        raise RepoMetaError(f"repo {owner}/{repo} not found or unreachable")

    readme = ""
    readme_json = _get_json(f"{_API}/repos/{owner}/{repo}/readme", op)
    if readme_json and isinstance(readme_json.get("content"), str):
        try:
            readme = base64.b64decode(readme_json["content"]).decode("utf-8", "replace")
        except (ValueError, TypeError):
            readme = ""
    if max_readme_chars and len(readme) > max_readme_chars:
        readme = readme[:max_readme_chars] + "\n…[README truncated]"

    topics = info.get("topics") or []
    license_obj = info.get("license") or {}
    return RepoMeta(
        owner=owner,
        repo=repo,
        description=(info.get("description") or "").strip(),
        topics=tuple(t for t in topics if isinstance(t, str)),
        stars=int(info.get("stargazers_count") or 0),
        homepage=(info.get("homepage") or "").strip(),
        license=(license_obj.get("spdx_id") or "") if isinstance(license_obj, dict) else "",
        language=(info.get("language") or "").strip(),
        default_branch=(info.get("default_branch") or "main").strip(),
        readme=readme,
    )
