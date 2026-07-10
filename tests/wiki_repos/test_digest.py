"""Tests for ``wiki_repos.digest.build_digest`` — local, no-network digest.

Builds temp repo trees with ``tmp_path`` and asserts that included files,
skipped dirs/binaries/oversize files, tree formatting, content blocks, caps /
truncation, and the DigestFailed guards all behave per PRD-012 / ADR-022.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from wiki_repos.digest import build_digest
from wiki_repos.errors import DigestFailed
from wiki_repos.types import RepoRef


def _ref() -> RepoRef:
    return RepoRef(owner="octocat", repo="hello-world")


def _make_repo(tmp_path: Path) -> Path:
    """Create a representative repo tree: text files, a skipped dir, a binary,
    and an oversize file."""
    repo = tmp_path / "repo"
    repo.mkdir()

    (repo / "README.md").write_text("# Hello\n\nSome docs.\n", encoding="utf-8")
    (repo / "main.py").write_text("print('hi')\n", encoding="utf-8")

    src = repo / "src"
    src.mkdir()
    (src / "app.js").write_text("const x = 1;\n", encoding="utf-8")
    (src / "util.js").write_text("export const y = 2;\n", encoding="utf-8")

    # Must be skipped wholesale.
    nm = repo / "node_modules" / "left-pad"
    nm.mkdir(parents=True)
    (nm / "index.js").write_text("module.exports = () => {};\n", encoding="utf-8")

    # Fake binary by null-byte sniff (extension not in blocklist).
    (repo / "data.dat").write_bytes(b"abc\x00\x01\x02def" + b"x" * 100)

    # Binary by extension.
    (repo / "logo.png").write_bytes(b"\x89PNG\r\n\x1a\n" + b"q" * 50)

    # Oversize text file (> default max_file_size for the small-cap test we set).
    (repo / "huge.txt").write_text("z" * 5000, encoding="utf-8")

    return repo


def test_includes_only_text_files_and_skips_dirs(tmp_path: Path) -> None:
    repo = _make_repo(tmp_path)
    digest = build_digest(repo, _ref(), max_file_size=1000)

    # Tree lists included files only.
    assert "README.md" in digest.tree
    assert "main.py" in digest.tree
    assert "app.js" in digest.tree
    assert "util.js" in digest.tree

    # Skipped artifacts must NOT appear.
    assert "node_modules" not in digest.tree
    assert "left-pad" not in digest.tree
    assert "data.dat" not in digest.tree
    assert "logo.png" not in digest.tree
    assert "huge.txt" not in digest.tree

    assert "node_modules" in digest.stats.skipped_dirs


def test_content_has_blocks_for_included_files(tmp_path: Path) -> None:
    repo = _make_repo(tmp_path)
    digest = build_digest(repo, _ref(), max_file_size=1000)

    assert "===== README.md =====" in digest.content
    assert "# Hello" in digest.content
    assert "===== src/app.js =====" in digest.content
    assert "const x = 1;" in digest.content

    # Excluded content never leaks in.
    assert "left-pad" not in digest.content
    assert "module.exports" not in digest.content


def test_stats_counts_are_accurate(tmp_path: Path) -> None:
    repo = _make_repo(tmp_path)
    digest = build_digest(repo, _ref(), max_file_size=1000)

    # 4 included text files: README.md, main.py, src/app.js, src/util.js.
    assert digest.stats.n_files == 4
    assert digest.stats.n_bytes > 0
    assert digest.stats.est_tokens == digest.stats.n_bytes // 4
    assert digest.stats.truncated is False


def test_summary_reports_repo_and_languages(tmp_path: Path) -> None:
    repo = _make_repo(tmp_path)
    digest = build_digest(repo, _ref(), max_file_size=1000)

    assert "https://github.com/octocat/hello-world" in digest.summary
    assert "4" in digest.summary  # included file count
    # Two .js files dominate the language tally.
    assert ".js" in digest.summary


def test_tiny_caps_trigger_truncation_marker(tmp_path: Path) -> None:
    repo = _make_repo(tmp_path)
    digest = build_digest(
        repo,
        _ref(),
        max_file_size=1000,
        max_total_bytes=10,
        max_total_tokens=80_000,
    )

    assert digest.stats.truncated is True
    assert "[TRUNCATED:" in digest.content


def test_token_cap_also_triggers_truncation(tmp_path: Path) -> None:
    repo = _make_repo(tmp_path)
    digest = build_digest(
        repo,
        _ref(),
        max_file_size=1000,
        max_total_bytes=10_000_000,
        max_total_tokens=2,  # 2 tokens ~ 8 bytes
    )

    assert digest.stats.truncated is True
    assert "[TRUNCATED:" in digest.content


def test_missing_dir_raises(tmp_path: Path) -> None:
    with pytest.raises(DigestFailed):
        build_digest(tmp_path / "does-not-exist", _ref())


def test_file_not_dir_raises(tmp_path: Path) -> None:
    f = tmp_path / "afile.txt"
    f.write_text("hi", encoding="utf-8")
    with pytest.raises(DigestFailed):
        build_digest(f, _ref())


def test_zero_readable_files_raises(tmp_path: Path) -> None:
    empty = tmp_path / "empty-repo"
    empty.mkdir()
    (empty / ".git").mkdir()
    (empty / ".git" / "config").write_text("[core]\n", encoding="utf-8")
    (empty / "image.png").write_bytes(b"\x89PNG" + b"\x00" * 10)
    with pytest.raises(DigestFailed):
        build_digest(empty, _ref())


def test_tree_is_deterministic_dirs_before_files_sorted(tmp_path: Path) -> None:
    repo = tmp_path / "r"
    repo.mkdir()
    (repo / "z.py").write_text("z\n", encoding="utf-8")
    (repo / "a.py").write_text("a\n", encoding="utf-8")
    pkg = repo / "pkg"
    pkg.mkdir()
    (pkg / "m.py").write_text("m\n", encoding="utf-8")

    digest = build_digest(repo, _ref())
    # Directory listed before sibling files; files alpha-sorted.
    assert digest.tree.index("pkg/") < digest.tree.index("a.py")
    assert digest.tree.index("a.py") < digest.tree.index("z.py")
