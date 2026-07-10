"""GitHub URL validation + tarball fetch — no ``git clone`` (PRD-012 FR-2 / ADR-022).

This is the *front door* of the ``wiki_repos`` bounded context. Everything an
untrusted user types arrives here, so this module is deliberately paranoid:

1. :func:`parse_github_url` — host-allowlist SSRF guard + strict URL shape
   validation. Accepts ONLY ``https://github.com/<owner>/<repo>`` (optional
   ``.git`` suffix, optional ``/tree/<ref>/<subpath>``) and a scheme-less
   ``github.com/<owner>/<repo>`` shorthand. Anything else raises
   :class:`~wiki_repos.errors.InvalidUrl`.
2. :func:`check_public_reachable` — unauthenticated probe of the GitHub REST
   API. We *fail closed*: private / missing / unreachable repos raise
   :class:`~wiki_repos.errors.PrivateOrUnreachable` (private repos are Phase 2,
   gated by an ADR-017 scope change).
3. :func:`fetch_repo_tarball` — stream the repo source as a tarball from the
   GitHub API (no clone, no GitPython, no ``git`` subprocess), size-capped, then
   safely extract with the Python 3.12 ``data`` filter *plus* a belt-and-braces
   traversal guard.

All network I/O is injectable (``opener`` / ``downloader``) so the full surface
is unit-testable with zero network access (mock-first per CLAUDE.md).

Pure stdlib only: ``urllib``, ``tarfile``, ``io``, ``pathlib``, ``json``, ``re``.
"""

from __future__ import annotations

import ipaddress
import json
import re
import tarfile
import tempfile
import urllib.error
import urllib.request
from collections.abc import Callable
from pathlib import Path
from urllib.parse import unquote, urlsplit

from wiki_repos.errors import FetchFailed, InvalidUrl, Oversize, PrivateOrUnreachable
from wiki_repos.types import RepoRef

__all__ = [
    "ARXIV_ALLOWED_HOSTS",
    "DEFAULT_ALLOWED_HOSTS",
    "OUTBOUND_ALLOWED_HOSTS",
    "parse_github_url",
    "check_public_reachable",
    "fetch_repo_tarball",
    "validate_outbound_url",
]

DEFAULT_ALLOWED_HOSTS: tuple[str, ...] = ("github.com",)
"""Hosts we will fetch from. The allowlist *is* the SSRF guard — adding a host
is a security decision (ADR-022), never an implementation convenience."""

ARXIV_ALLOWED_HOSTS: tuple[str, ...] = ("arxiv.org", "export.arxiv.org")
"""ADR-032 D3 / PRD-015 SR-1: the paper pipeline's outbound hosts. Recorded
here per the ADR-017 amendment rule. No other hosts; no credentials."""

OUTBOUND_ALLOWED_HOSTS: tuple[str, ...] = DEFAULT_ALLOWED_HOSTS + ARXIV_ALLOWED_HOSTS
"""Every host any SBW outbound fetch may target. ``wiki_papers`` routes its
arXiv requests through :func:`validate_outbound_url` against this list."""

_USER_AGENT = "second-brain-wiki/wiki_repos (+https://github.com/DarkCodePE/second-brain-wiki)"

# A single GitHub repo path segment: letters, digits, ``-`` ``_`` ``.``.
# Deliberately excludes ``/`` and rejects the standalone ``.`` / ``..`` traversal
# tokens (those are caught explicitly before this is consulted, too).
_SEGMENT_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*$")

# Injectable network type aliases (documented for callers/tests).
Opener = Callable[[str], "tuple[int, bytes]"]
Downloader = Callable[[str, float], bytes]


def _reject(reason: str) -> InvalidUrl:
    """Build an :class:`InvalidUrl` without ever echoing the offending URL back
    (it may contain credentials / tokens in userinfo)."""
    return InvalidUrl(reason)


def _looks_like_ip_or_local(host: str) -> bool:
    """True if ``host`` is an IP literal (v4/v6, incl. bracketed) or a local name.

    Public repos live on a named host (``github.com``); an IP or ``localhost``
    target in the position of the host is a classic SSRF probe.
    """
    h = host.strip().lower()
    if not h:
        return True
    if h in {"localhost", "localhost.localdomain"}:
        return True
    # Bracketed IPv6, e.g. ``[::1]``.
    if h.startswith("[") and h.endswith("]"):
        h = h[1:-1]
    try:
        ipaddress.ip_address(h)
        return True
    except ValueError:
        return False


def validate_outbound_url(
    url: str,
    *,
    allowed_hosts: tuple[str, ...] = OUTBOUND_ALLOWED_HOSTS,
) -> str:
    """Generic SSRF guard for any outbound fetch (the ADR-032 D3 seam).

    Unlike :func:`parse_github_url` this validates only the *transport*
    surface — ``https`` scheme, no userinfo, no explicit port, no IP /
    ``localhost`` host, host on the allowlist — and leaves path/query
    shape to the caller (the arXiv Atom API legitimately needs query
    strings: ``/api/query?id_list=...``).

    Args:
        url: The raw, untrusted URL string.
        allowed_hosts: Host allowlist. Defaults to
            :data:`OUTBOUND_ALLOWED_HOSTS` (github + arXiv).

    Returns:
        The validated, lowercased host.

    Raises:
        InvalidUrl: For any non-conforming input.
    """
    if not isinstance(url, str):
        raise _reject("url must be a string")
    raw = url.strip()
    if not raw:
        raise _reject("empty url")

    parts = urlsplit(raw)
    if parts.scheme.lower() != "https":
        raise _reject(f"scheme must be https, got {parts.scheme!r}")
    if parts.username is not None or parts.password is not None or "@" in parts.netloc:
        raise _reject("userinfo in authority not allowed")
    if parts.port is not None:
        raise _reject("explicit port not allowed")

    host = (parts.hostname or "").lower()
    if not host:
        raise _reject("missing host")
    if _looks_like_ip_or_local(host):
        raise _reject("ip/localhost host not allowed")
    if host not in {h.lower() for h in allowed_hosts}:
        raise _reject(f"host {host!r} not in allowlist")
    return host


def parse_github_url(
    url: str,
    *,
    allowed_hosts: tuple[str, ...] = DEFAULT_ALLOWED_HOSTS,
) -> RepoRef:
    """Validate and parse a GitHub repo URL into a :class:`RepoRef`.

    Accepted shapes (case-insensitive host, case-preserving owner/repo)::

        https://github.com/<owner>/<repo>
        https://github.com/<owner>/<repo>.git
        https://github.com/<owner>/<repo>/tree/<ref>/<subpath...>
        github.com/<owner>/<repo>                 # scheme-less shorthand

    Args:
        url: The raw, untrusted URL string.
        allowed_hosts: Host allowlist (the SSRF guard). Defaults to
            :data:`DEFAULT_ALLOWED_HOSTS`.

    Returns:
        A validated :class:`RepoRef`. ``ref``/``subpath`` are populated only for
        ``/tree/<ref>/<subpath>`` URLs.

    Raises:
        InvalidUrl: For any non-conforming input — non-GitHub host, ``file://``,
            ``ssh://``, ``git@…`` SCP syntax, userinfo in the authority, IP /
            ``localhost`` hosts, non-``https`` scheme, missing owner or repo,
            any ``..`` path traversal, or query/fragment trickery.
    """
    if not isinstance(url, str):  # defensive: callers should pass str
        raise _reject("url must be a string")
    raw = url.strip()
    if not raw:
        raise _reject("empty url")

    # ── SCP-like git syntax (``git@github.com:owner/repo.git``) has no ``//`` ──
    # authority and would otherwise be misparsed. Reject explicitly.
    if "@" in raw.split("/", 1)[0] and "://" not in raw:
        raise _reject("scp-style/git@ syntax not allowed")

    # Normalise the scheme-less shorthand ``github.com/o/r`` → ``https://…``.
    # We only do this when there is no scheme at all; a bare ``ssh://`` etc.
    # keeps its scheme so it is rejected below rather than silently upgraded.
    if "://" not in raw:
        raw = "https://" + raw

    parts = urlsplit(raw)

    # ── Scheme: https only. ``http``, ``file``, ``ssh``, ``ftp`` … all out. ──
    if parts.scheme.lower() != "https":
        raise _reject(f"scheme must be https, got {parts.scheme!r}")

    # ── No query string, no fragment — they are pure attack surface here. ──
    if parts.query:
        raise _reject("query strings are not allowed")
    if parts.fragment:
        raise _reject("url fragments are not allowed")

    # ── Authority: no userinfo, no explicit port, host on the allowlist. ──
    if parts.username is not None or parts.password is not None or "@" in parts.netloc:
        raise _reject("userinfo in authority not allowed")
    if parts.port is not None:
        raise _reject("explicit port not allowed")

    host = (parts.hostname or "").lower()
    if not host:
        raise _reject("missing host")
    if _looks_like_ip_or_local(host):
        raise _reject("ip/localhost host not allowed")
    if host not in {h.lower() for h in allowed_hosts}:
        raise _reject(f"host {host!r} not in allowlist")

    # ── Path: split, percent-decode, and validate every segment. ──
    # Decode *after* splitting so an encoded ``%2F`` can't smuggle an extra
    # path separator past the segment validator.
    raw_segments = [s for s in parts.path.split("/") if s != ""]
    segments = [unquote(s) for s in raw_segments]

    # Any traversal token anywhere in the path is fatal.
    for seg in segments:
        if seg in {".", ".."} or "/" in seg or "\\" in seg:
            raise _reject("path traversal / separators not allowed")

    if len(segments) < 2:
        raise _reject("url must contain <owner>/<repo>")

    owner = segments[0]
    repo = segments[1]
    if repo.endswith(".git"):
        repo = repo[: -len(".git")]

    if not owner or not repo:
        raise _reject("missing owner or repo")
    if not _SEGMENT_RE.match(owner):
        raise _reject("invalid owner segment")
    if not _SEGMENT_RE.match(repo):
        raise _reject("invalid repo segment")

    ref: str | None = None
    subpath: str | None = None
    rest = segments[2:]
    if rest:
        # The only deeper shape we accept is ``/tree/<ref>/<subpath...>``.
        if rest[0] != "tree":
            raise _reject(f"unsupported path beyond repo: {rest[0]!r}")
        if len(rest) < 2:
            raise _reject("/tree/ requires a <ref>")
        ref_segments = rest[1:]
        # ``ref`` is the first segment after ``tree``; the remainder is subpath.
        ref = ref_segments[0]
        if not _SEGMENT_RE.match(ref):
            raise _reject("invalid ref segment")
        sub_segments = ref_segments[1:]
        if sub_segments:
            # Each subpath segment already passed the traversal screen above.
            for seg in sub_segments:
                if not _SEGMENT_RE.match(seg):
                    raise _reject("invalid subpath segment")
            subpath = "/".join(sub_segments)

    return RepoRef(owner=owner, repo=repo, host=host, ref=ref, subpath=subpath)


def _default_opener(url: str) -> tuple[int, bytes]:
    """Minimal urllib GET → ``(status, body)``. Used only when no ``opener`` is
    injected. Sets a User-Agent (GitHub rejects empty UA) and a 15s timeout."""
    req = urllib.request.Request(  # noqa: S310 — scheme is allowlisted upstream
        url,
        headers={"User-Agent": _USER_AGENT, "Accept": "application/vnd.github+json"},
        method="GET",
    )
    with urllib.request.urlopen(req, timeout=15) as resp:  # noqa: S310
        return int(resp.getcode() or 0), resp.read()


def check_public_reachable(ref: RepoRef, *, opener: Opener | None = None) -> None:
    """Verify the repo is public and reachable on an unauthenticated probe.

    Probes ``GET https://api.github.com/repos/{owner}/{repo}`` (no auth header).
    We treat anything other than a confirmed *public* repo as a closed door.

    Args:
        ref: The validated repo reference.
        opener: Injectable ``(url) -> (status, body)`` callable. Defaults to a
            urllib GET. Tests MUST inject a fake here — no real network.

    Returns:
        ``None`` if the repo is confirmed public and reachable.

    Raises:
        PrivateOrUnreachable: If the repo is private, returns 404/403/451, the
            body is not the expected JSON, or any network error occurs.
    """
    open_ = opener or _default_opener
    api_url = f"https://api.github.com/repos/{ref.owner}/{ref.repo}"

    try:
        status, body = open_(api_url)
    except urllib.error.HTTPError as exc:  # opener may raise instead of returning
        status, body = exc.code, b""
    except Exception as exc:  # network/DNS/timeout/anything → fail closed
        raise PrivateOrUnreachable(
            f"probe failed for {ref.owner}/{ref.repo}: {type(exc).__name__}"
        ) from exc

    if status in (404, 403, 451):
        raise PrivateOrUnreachable(
            f"repo {ref.owner}/{ref.repo} not publicly reachable (HTTP {status})"
        )
    if status != 200:
        raise PrivateOrUnreachable(
            f"unexpected probe status for {ref.owner}/{ref.repo}: HTTP {status}"
        )

    try:
        meta = json.loads(body.decode("utf-8") if isinstance(body, bytes) else body)
    except (ValueError, UnicodeDecodeError) as exc:
        raise PrivateOrUnreachable(
            f"unparseable probe response for {ref.owner}/{ref.repo}"
        ) from exc

    if not isinstance(meta, dict) or "private" not in meta:
        raise PrivateOrUnreachable(
            f"probe response missing 'private' flag for {ref.owner}/{ref.repo}"
        )
    if meta.get("private") is True:
        raise PrivateOrUnreachable(f"repo {ref.owner}/{ref.repo} is private")
    if meta.get("private") is not False:
        raise PrivateOrUnreachable(f"ambiguous 'private' flag for {ref.owner}/{ref.repo}")
    # private is False, status 200 → confirmed public.
    return None


def _default_downloader(url: str, timeout: float) -> bytes:
    """Stream a tarball with urllib, following redirects (GitHub 302→codeload).

    NOTE: size enforcement lives in :func:`fetch_repo_tarball`; this helper only
    reads the bytes. We read the whole stream because the body is bounded by the
    caller's ``max_bytes`` check immediately after.
    """
    req = urllib.request.Request(  # noqa: S310 — scheme allowlisted upstream
        url,
        headers={"User-Agent": _USER_AGENT},
        method="GET",
    )
    with urllib.request.urlopen(req, timeout=timeout) as resp:  # noqa: S310
        return resp.read()


def _is_within(base: Path, target: Path) -> bool:
    """True if ``target`` resolves inside ``base`` (defence-in-depth vs the
    tarfile ``data`` filter — a refused symlink/absolute member is rejected)."""
    try:
        target.resolve().relative_to(base.resolve())
        return True
    except ValueError:
        return False


def fetch_repo_tarball(
    ref: RepoRef,
    *,
    dest_dir: Path,
    max_bytes: int = 60_000_000,
    max_extracted_bytes: int = 500_000_000,
    timeout: float = 60.0,
    downloader: Downloader | None = None,
) -> Path:
    """Download the repo source as a tarball and safely extract it.

    Fetches ``GET https://api.github.com/repos/{owner}/{repo}/tarball/{ref}``
    (302-redirects to ``codeload.github.com``; an empty ``ref`` resolves to the
    repo's default branch). The archive expands to a single top-level directory
    (e.g. ``odysseus-main/``); we return the :class:`~pathlib.Path` to that root.

    Args:
        ref: The validated repo reference.
        dest_dir: Directory to extract into (created if missing). Should be empty
            / dedicated to this fetch — the single extracted child dir is the
            return value.
        max_bytes: Hard cap on the downloaded tarball size. Exceeding it raises
            :class:`Oversize`.
        timeout: Per-request socket timeout in seconds for the default
            downloader.
        downloader: Injectable ``(url, timeout) -> bytes`` callable. Defaults to
            a urllib streamed GET. Tests MUST inject a fake here.

    Returns:
        Path to the single extracted repository root directory.

    Raises:
        Oversize: If the downloaded body exceeds ``max_bytes``.
        FetchFailed: On any network/HTTP error, a corrupt/unsafe archive, or an
            unexpected post-extract layout.
    """
    download = downloader or _default_downloader
    ref_part = ref.ref or ""
    tar_url = f"https://api.github.com/repos/{ref.owner}/{ref.repo}/tarball/{ref_part}"

    # ── 1. Download (injectable) ──
    try:
        data = download(tar_url, timeout)
    except (Oversize, FetchFailed):
        raise
    except urllib.error.HTTPError as exc:
        raise FetchFailed(
            f"tarball HTTP error for {ref.owner}/{ref.repo}: HTTP {exc.code}"
        ) from exc
    except Exception as exc:
        raise FetchFailed(
            f"tarball download failed for {ref.owner}/{ref.repo}: {type(exc).__name__}"
        ) from exc

    if not isinstance(data, (bytes, bytearray)):
        raise FetchFailed("downloader did not return bytes")
    if len(data) > max_bytes:
        raise Oversize(f"tarball for {ref.owner}/{ref.repo} is {len(data)} bytes (cap {max_bytes})")
    if len(data) == 0:
        raise FetchFailed(f"empty tarball for {ref.owner}/{ref.repo}")

    dest = Path(dest_dir)
    dest.mkdir(parents=True, exist_ok=True)

    # ── 2. Persist to a temp .tar.gz, then extract safely ──
    tmp_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(suffix=".tar.gz", dir=dest, delete=False) as tmp:
            tmp.write(data)
            tmp_path = Path(tmp.name)

        try:
            with tarfile.open(tmp_path, mode="r:*") as tar:
                # Belt-and-braces traversal guard *in addition* to the data
                # filter: reject any member whose normalised name escapes dest
                # before we extract anything.
                extracted_total = 0
                for member in tar.getmembers():
                    member_path = dest / member.name
                    if not _is_within(dest, member_path):
                        raise FetchFailed(f"refused unsafe archive member: {member.name!r}")
                    # Decompression-bomb / disk-exhaustion guard: the download
                    # cap bounds *compressed* bytes; this bounds the *expanded*
                    # tree before we write any of it to disk.
                    extracted_total += max(0, member.size)
                    if extracted_total > max_extracted_bytes:
                        raise Oversize(
                            f"extracted size for {ref.owner}/{ref.repo} exceeds "
                            f"cap ({max_extracted_bytes} bytes)"
                        )
                # Python 3.12 'data' filter: blocks absolute paths, ``..``
                # traversal, and unsafe (out-of-tree / absolute) symlinks.
                tar.extractall(dest, filter="data")
        except tarfile.TarError as exc:
            raise FetchFailed(f"corrupt/unsafe archive for {ref.owner}/{ref.repo}: {exc}") from exc
    finally:
        if tmp_path is not None and tmp_path.exists():
            tmp_path.unlink()

    # ── 3. Locate the single extracted repo root ──
    children = [p for p in dest.iterdir() if p.is_dir()]
    if len(children) != 1:
        raise FetchFailed(f"expected exactly one extracted directory, found {len(children)}")
    return children[0]
