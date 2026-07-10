"""Unit tests for ``wiki_repos.fetcher`` — mock-first, zero network access.

Covers:
- :func:`parse_github_url` — accept the valid shapes, reject the SSRF / malformed
  surface (file://, ssh, scp git@, userinfo, IP/localhost, http, query/fragment,
  traversal, missing owner/repo, non-allowlisted host).
- :func:`check_public_reachable` — injected fake opener for every branch.
- :func:`fetch_repo_tarball` — injected downloader returning an in-memory
  ``.tar.gz`` fixture; asserts extraction, Oversize, and refusal of traversal
  members.
"""

from __future__ import annotations

import io
import tarfile

import pytest

from wiki_repos.errors import FetchFailed, InvalidUrl, Oversize, PrivateOrUnreachable
from wiki_repos.fetcher import (
    ARXIV_ALLOWED_HOSTS,
    DEFAULT_ALLOWED_HOSTS,
    OUTBOUND_ALLOWED_HOSTS,
    check_public_reachable,
    fetch_repo_tarball,
    parse_github_url,
    validate_outbound_url,
)
from wiki_repos.types import RepoRef

# ─────────────────────────── parse_github_url: accept ───────────────────────────


def test_default_allowed_hosts_is_github_only():
    assert DEFAULT_ALLOWED_HOSTS == ("github.com",)


def test_parse_basic_https_url():
    ref = parse_github_url("https://github.com/odysseus/odysseus")
    assert ref == RepoRef(owner="odysseus", repo="odysseus", host="github.com")
    assert ref.ref is None
    assert ref.subpath is None


def test_parse_strips_dot_git_suffix():
    ref = parse_github_url("https://github.com/octocat/Hello-World.git")
    assert ref.owner == "octocat"
    assert ref.repo == "Hello-World"


def test_parse_scheme_less_shorthand_normalised_to_https():
    ref = parse_github_url("github.com/torvalds/linux")
    assert ref.host == "github.com"
    assert ref.owner == "torvalds"
    assert ref.repo == "linux"


def test_parse_tree_ref_only():
    ref = parse_github_url("https://github.com/o/r/tree/develop")
    assert ref.ref == "develop"
    assert ref.subpath is None


def test_parse_tree_ref_and_subpath():
    ref = parse_github_url("https://github.com/o/r/tree/main/src/pkg")
    assert ref.ref == "main"
    assert ref.subpath == "src/pkg"


def test_parse_preserves_owner_repo_case():
    ref = parse_github_url("https://github.com/DarkCodePE/Second-Brain-Wiki")
    assert ref.owner == "DarkCodePE"
    assert ref.repo == "Second-Brain-Wiki"


def test_parse_host_case_insensitive():
    ref = parse_github_url("https://GitHub.com/o/r")
    assert ref.host == "github.com"


def test_parse_trailing_slash_ok():
    ref = parse_github_url("https://github.com/o/r/")
    assert ref.owner == "o"
    assert ref.repo == "r"


# ─────────────────────────── parse_github_url: reject ───────────────────────────


@pytest.mark.parametrize(
    "bad_url",
    [
        "file:///etc/passwd",
        "ssh://git@github.com/o/r.git",
        "git@github.com:o/r.git",
        "ftp://github.com/o/r",
        "http://github.com/o/r",  # non-https after no normalisation
        "https://user:pass@github.com/o/r",  # userinfo
        "https://user@github.com/o/r",  # userinfo (no pass)
        "https://gitlab.com/o/r",  # non-allowlisted host
        "https://example.com/o/r",
        "https://127.0.0.1/o/r",  # IP literal host
        "https://[::1]/o/r",  # IPv6 literal host
        "https://localhost/o/r",
        "https://github.com/o/r?token=x",  # query
        "https://github.com/o/r#frag",  # fragment
        "https://github.com/../../etc/passwd",  # traversal
        "https://github.com/o/..",  # traversal in repo slot
        "https://github.com/o",  # missing repo
        "https://github.com/",  # missing owner+repo
        "https://github.com",  # no path
        "https://github.com:22/o/r",  # explicit port
        "",  # empty
        "   ",  # whitespace only
        "github.com/o/r/blob/main/file.py",  # unsupported deep path (not /tree/)
        "https://github.com/o/r/tree",  # /tree with no ref
    ],
)
def test_parse_rejects_invalid(bad_url):
    with pytest.raises(InvalidUrl):
        parse_github_url(bad_url)


def test_parse_rejects_encoded_slash_traversal():
    # %2F decodes to '/', must not smuggle extra separators into a segment.
    with pytest.raises(InvalidUrl):
        parse_github_url("https://github.com/o/r%2F..%2Fevil")


def test_parse_rejects_encoded_dotdot():
    with pytest.raises(InvalidUrl):
        parse_github_url("https://github.com/o/r/tree/%2e%2e/x")


def test_parse_non_string_rejected():
    with pytest.raises(InvalidUrl):
        parse_github_url(None)  # type: ignore[arg-type]


def test_parse_custom_allowlist_admits_other_host():
    ref = parse_github_url(
        "https://git.internal.example/o/r",
        allowed_hosts=("git.internal.example",),
    )
    assert ref.host == "git.internal.example"


def test_parse_does_not_echo_userinfo_in_error():
    with pytest.raises(InvalidUrl) as exc:
        parse_github_url("https://secrettoken:pw@github.com/o/r")
    assert "secrettoken" not in str(exc.value)


# ─────────────────────── check_public_reachable (injected) ──────────────────────


def _opener_returning(status: int, body: bytes):
    captured = {}

    def opener(url: str):
        captured["url"] = url
        return status, body

    opener.captured = captured  # type: ignore[attr-defined]
    return opener


REF = RepoRef(owner="octocat", repo="Hello-World")


def test_reachable_public_ok():
    opener = _opener_returning(200, b'{"private": false}')
    assert check_public_reachable(REF, opener=opener) is None
    assert opener.captured["url"] == "https://api.github.com/repos/octocat/Hello-World"


def test_reachable_private_raises():
    opener = _opener_returning(200, b'{"private": true}')
    with pytest.raises(PrivateOrUnreachable):
        check_public_reachable(REF, opener=opener)


@pytest.mark.parametrize("status", [404, 403, 451])
def test_reachable_blocking_statuses(status):
    opener = _opener_returning(status, b"")
    with pytest.raises(PrivateOrUnreachable):
        check_public_reachable(REF, opener=opener)


def test_reachable_unexpected_status():
    opener = _opener_returning(500, b"")
    with pytest.raises(PrivateOrUnreachable):
        check_public_reachable(REF, opener=opener)


def test_reachable_unparseable_body():
    opener = _opener_returning(200, b"<<not json>>")
    with pytest.raises(PrivateOrUnreachable):
        check_public_reachable(REF, opener=opener)


def test_reachable_missing_private_flag():
    opener = _opener_returning(200, b'{"name": "x"}')
    with pytest.raises(PrivateOrUnreachable):
        check_public_reachable(REF, opener=opener)


def test_reachable_network_error_fails_closed():
    def opener(url: str):
        raise OSError("connection refused")

    with pytest.raises(PrivateOrUnreachable):
        check_public_reachable(REF, opener=opener)


def test_reachable_httperror_treated_as_status():
    import urllib.error

    def opener(url: str):
        raise urllib.error.HTTPError(url, 404, "Not Found", {}, None)  # type: ignore[arg-type]

    with pytest.raises(PrivateOrUnreachable):
        check_public_reachable(REF, opener=opener)


# ───────────────────────── fetch_repo_tarball (injected) ────────────────────────


def _make_tar_gz(members: dict[str, bytes], *, root: str = "odysseus-main") -> bytes:
    """Build an in-memory .tar.gz with files nested under a single ``root`` dir."""
    buf = io.BytesIO()
    with tarfile.open(fileobj=buf, mode="w:gz") as tar:
        # explicit root dir entry
        root_info = tarfile.TarInfo(name=root + "/")
        root_info.type = tarfile.DIRTYPE
        tar.addfile(root_info)
        for name, data in members.items():
            info = tarfile.TarInfo(name=f"{root}/{name}")
            info.size = len(data)
            tar.addfile(info, io.BytesIO(data))
    return buf.getvalue()


def _make_traversal_tar_gz() -> bytes:
    """Build a malicious .tar.gz with a ``../`` escaping member."""
    buf = io.BytesIO()
    with tarfile.open(fileobj=buf, mode="w:gz") as tar:
        payload = b"pwned"
        info = tarfile.TarInfo(name="../escapee.txt")
        info.size = len(payload)
        tar.addfile(info, io.BytesIO(payload))
    return buf.getvalue()


def test_fetch_extracts_and_returns_root(tmp_path):
    archive = _make_tar_gz(
        {"README.md": b"# hi", "src/main.py": b"print(1)\n"},
        root="odysseus-main",
    )
    captured = {}

    def downloader(url: str, timeout: float) -> bytes:
        captured["url"] = url
        captured["timeout"] = timeout
        return archive

    ref = RepoRef(owner="acme", repo="odysseus")
    root = fetch_repo_tarball(ref, dest_dir=tmp_path, downloader=downloader)

    assert root.name == "odysseus-main"
    assert root.parent == tmp_path
    assert (root / "README.md").read_bytes() == b"# hi"
    assert (root / "src" / "main.py").read_bytes() == b"print(1)\n"
    assert captured["url"] == ("https://api.github.com/repos/acme/odysseus/tarball/")
    # temp .tar.gz must be cleaned up — only the extracted dir remains.
    assert [p.name for p in tmp_path.iterdir()] == ["odysseus-main"]


def test_fetch_uses_ref_in_url(tmp_path):
    archive = _make_tar_gz({"x.txt": b"y"}, root="r-dev")
    captured = {}

    def downloader(url: str, timeout: float) -> bytes:
        captured["url"] = url
        return archive

    ref = RepoRef(owner="o", repo="r", ref="dev")
    fetch_repo_tarball(ref, dest_dir=tmp_path, downloader=downloader)
    assert captured["url"] == "https://api.github.com/repos/o/r/tarball/dev"


def test_fetch_oversize_raises(tmp_path):
    archive = _make_tar_gz({"big.bin": b"x" * 100}, root="r-main")

    def downloader(url: str, timeout: float) -> bytes:
        return archive

    ref = RepoRef(owner="o", repo="r")
    with pytest.raises(Oversize):
        fetch_repo_tarball(ref, dest_dir=tmp_path, max_bytes=10, downloader=downloader)


def test_fetch_refuses_traversal_member(tmp_path):
    archive = _make_traversal_tar_gz()

    def downloader(url: str, timeout: float) -> bytes:
        return archive

    ref = RepoRef(owner="o", repo="r")
    with pytest.raises(FetchFailed):
        fetch_repo_tarball(ref, dest_dir=tmp_path, downloader=downloader)
    # nothing escaped the dest dir
    assert not (tmp_path.parent / "escapee.txt").exists()


def test_fetch_empty_body_raises(tmp_path):
    def downloader(url: str, timeout: float) -> bytes:
        return b""

    ref = RepoRef(owner="o", repo="r")
    with pytest.raises(FetchFailed):
        fetch_repo_tarball(ref, dest_dir=tmp_path, downloader=downloader)


def test_fetch_network_error_raises_fetchfailed(tmp_path):
    def downloader(url: str, timeout: float) -> bytes:
        raise OSError("timeout")

    ref = RepoRef(owner="o", repo="r")
    with pytest.raises(FetchFailed):
        fetch_repo_tarball(ref, dest_dir=tmp_path, downloader=downloader)


def test_fetch_corrupt_archive_raises_fetchfailed(tmp_path):
    def downloader(url: str, timeout: float) -> bytes:
        return b"this is definitely not a gzip tarball" * 5

    ref = RepoRef(owner="o", repo="r")
    with pytest.raises(FetchFailed):
        fetch_repo_tarball(ref, dest_dir=tmp_path, downloader=downloader)


def test_fetch_non_bytes_downloader_raises(tmp_path):
    def downloader(url: str, timeout: float):
        return "not bytes"  # type: ignore[return-value]

    ref = RepoRef(owner="o", repo="r")
    with pytest.raises(FetchFailed):
        fetch_repo_tarball(ref, dest_dir=tmp_path, downloader=downloader)


def test_fetch_multiple_top_dirs_raises(tmp_path):
    """A well-formed GitHub tarball has exactly one top dir; reject otherwise."""
    buf = io.BytesIO()
    with tarfile.open(fileobj=buf, mode="w:gz") as tar:
        for root in ("a-main", "b-main"):
            info = tarfile.TarInfo(name=f"{root}/f.txt")
            info.size = 1
            tar.addfile(info, io.BytesIO(b"x"))
    archive = buf.getvalue()

    def downloader(url: str, timeout: float) -> bytes:
        return archive

    ref = RepoRef(owner="o", repo="r")
    with pytest.raises(FetchFailed):
        fetch_repo_tarball(ref, dest_dir=tmp_path, downloader=downloader)


def test_fetch_passes_timeout_through(tmp_path):
    archive = _make_tar_gz({"f": b"x"}, root="r-main")
    captured = {}

    def downloader(url: str, timeout: float) -> bytes:
        captured["timeout"] = timeout
        return archive

    ref = RepoRef(owner="o", repo="r")
    fetch_repo_tarball(ref, dest_dir=tmp_path, timeout=12.5, downloader=downloader)
    assert captured["timeout"] == 12.5


def test_fetch_rejects_decompression_bomb(tmp_path):
    """A tarball under the *compressed* cap whose members expand past
    max_extracted_bytes is refused before extraction (decompression-bomb guard)."""
    import io
    import tarfile

    import pytest

    from wiki_repos.errors import Oversize
    from wiki_repos.fetcher import fetch_repo_tarball
    from wiki_repos.types import RepoRef

    buf = io.BytesIO()
    with tarfile.open(fileobj=buf, mode="w:gz") as tar:
        big = b"A" * 50_000
        info = tarfile.TarInfo(name="repo-main/big.txt")
        info.size = len(big)
        tar.addfile(info, io.BytesIO(big))
    payload = buf.getvalue()

    ref = RepoRef(owner="o", repo="r")
    with pytest.raises(Oversize):
        fetch_repo_tarball(
            ref,
            dest_dir=tmp_path,
            max_extracted_bytes=10_000,  # 50k member > 10k extracted cap
            downloader=lambda url, timeout: payload,
        )


# ───────────── validate_outbound_url: ADR-032 D3 / PRD-015 SR-1 amendment ─────────────


def test_arxiv_allowlist_amendment_is_exact():
    """The D3 amendment adds arxiv.org + export.arxiv.org and NOTHING else."""
    assert ARXIV_ALLOWED_HOSTS == ("arxiv.org", "export.arxiv.org")
    assert set(OUTBOUND_ALLOWED_HOSTS) == {"github.com", "arxiv.org", "export.arxiv.org"}
    # The GitHub URL parser's own allowlist stays github-only.
    assert DEFAULT_ALLOWED_HOSTS == ("github.com",)


def test_outbound_accepts_arxiv_api_query_url():
    host = validate_outbound_url("https://export.arxiv.org/api/query?id_list=2605.23904")
    assert host == "export.arxiv.org"


def test_outbound_accepts_arxiv_pdf_url():
    assert validate_outbound_url("https://arxiv.org/pdf/2605.23904v2") == "arxiv.org"


def test_outbound_host_is_case_insensitive():
    assert validate_outbound_url("https://ArXiv.org/abs/2605.23904") == "arxiv.org"


def test_outbound_still_accepts_github():
    assert validate_outbound_url("https://github.com/o/r") == "github.com"


def test_outbound_rejects_unlisted_hosts():
    with pytest.raises(InvalidUrl):
        validate_outbound_url("https://evil.example.com/abs/2605.23904")
    with pytest.raises(InvalidUrl):
        validate_outbound_url("https://semanticscholar.org/paper/x")  # Phase 3, not yet


def test_outbound_rejects_lookalike_subdomain():
    with pytest.raises(InvalidUrl):
        validate_outbound_url("https://arxiv.org.evil.example/abs/1")


def test_outbound_rejects_non_https_and_ssrf_probes():
    for bad in (
        "http://arxiv.org/abs/2605.23904",
        "file:///etc/passwd",
        "https://127.0.0.1/api/query",
        "https://[::1]/api/query",
        "https://localhost/api/query",
        "https://user:pass@arxiv.org/abs/1",
        "https://arxiv.org:8443/abs/1",
        "",
    ):
        with pytest.raises(InvalidUrl):
            validate_outbound_url(bad)


def test_github_parser_does_not_inherit_arxiv_hosts():
    """parse_github_url keeps its own github-only default — an arXiv URL is
    not a repo URL even after the D3 amendment."""
    with pytest.raises(InvalidUrl):
        parse_github_url("https://arxiv.org/abs/2605.23904")
