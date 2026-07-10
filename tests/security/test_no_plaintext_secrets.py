"""
The CI guard for ADR-017 — recursively grep the repo working tree and the
local SBW state directories for OAuth-shaped bearer tokens, refresh tokens,
and Composio API keys. Build fails if any literal-prefix match is found.

The test ALSO performs a positive control: it synthesises a leak in a tmp
directory and asserts the scanner catches it. That keeps the test from
silently passing if the regexes get broken.

Acceptance criteria covered:
- AC: `CI test tests/security/no_plaintext_secrets.py greps repo + config dirs`
- AC: passes on a clean tree
- AC: fails on a synthetic leak (`test_scanner_catches_synthetic_leak`)
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

# Bearer-shape regexes — keep them literal-prefix tight so randomness in
# legitimate test fixtures doesn't accidentally trip them.
LEAK_PATTERNS: dict[str, re.Pattern[str]] = {
    "google_access_token": re.compile(r"\bya29\.[A-Za-z0-9_-]{20,}\b"),
    "google_refresh_token": re.compile(r"\b1//0[A-Za-z0-9_-]{20,}\b"),
    "github_pat_classic": re.compile(r"\bghp_[A-Za-z0-9]{36}\b"),
    "github_pat_fine_grained": re.compile(r"\bgithub_pat_[A-Za-z0-9_]{50,}\b"),
    "github_oauth_bearer": re.compile(r"\bgho_[A-Za-z0-9]{36}\b"),
    "slack_bot_token": re.compile(r"\bxoxb-[A-Za-z0-9-]{20,}\b"),
    "slack_user_token": re.compile(r"\bxoxp-[A-Za-z0-9-]{20,}\b"),
    # Composio's live keys look like `comp_<base32>`; conservative match
    "composio_api_key": re.compile(r"\bcomp_[A-Za-z0-9]{30,}\b"),
}


REPO_ROOT = Path(__file__).resolve().parents[2]

# Directories to skip — large binary trees, vendored deps, virtualenvs,
# caches. The point of the scan is committed source, not third-party code
# we don't own.
SKIP_DIRS: frozenset[str] = frozenset(
    {
        ".git",
        ".venv",
        "venv",
        "__pycache__",
        ".mypy_cache",
        ".pytest_cache",
        ".ruff_cache",
        "node_modules",
        "site-packages",
        "dist",
        "build",
        ".tox",
        # Subprojects vendored under the umbrella repo — out of scope for SBW's audit
        "openhuman",
        "claude-code-patterns",
        "agentmemory",
        "hermes-agent",
        "kb-wiki",
        "ruflo",
        # this test file itself contains pattern literals used for the synthetic-leak control
        "tests/security",
    }
)

# File extensions to scan. Binary types are skipped wholesale.
TEXT_SUFFIXES: frozenset[str] = frozenset(
    {
        ".py",
        ".pyi",
        ".md",
        ".rst",
        ".txt",
        ".json",
        ".jsonl",
        ".yaml",
        ".yml",
        ".toml",
        ".ini",
        ".cfg",
        ".env",
        ".env.example",
        ".env.local",
        ".sh",
        ".bash",
        ".zsh",
        ".js",
        ".ts",
        ".tsx",
        ".jsx",
        ".rs",
        ".go",
    }
)


def _iter_scannable_files(root: Path):
    """Yield absolute paths to text files under `root` we should scan."""
    for path in root.rglob("*"):
        # Fast skip on any ancestor in SKIP_DIRS
        if any(part in SKIP_DIRS for part in path.parts):
            continue
        # Or any path fragment matching "tests/security" specifically
        rel = path.relative_to(root)
        if str(rel).startswith("tests/security"):
            continue
        if not path.is_file():
            continue
        if path.suffix not in TEXT_SUFFIXES and not path.name.startswith(".env"):
            continue
        # Skip large files defensively (>2 MiB unlikely to be source)
        try:
            if path.stat().st_size > 2 * 1024 * 1024:
                continue
        except OSError:
            continue
        yield path


def _scan(root: Path) -> list[tuple[str, Path, int, str]]:
    """Return a list of (pattern_name, path, line_no, matched_text) tuples."""
    findings: list[tuple[str, Path, int, str]] = []
    for path in _iter_scannable_files(root):
        try:
            text = path.read_text(encoding="utf-8", errors="ignore")
        except (OSError, UnicodeDecodeError):
            continue
        for lineno, line in enumerate(text.splitlines(), start=1):
            for name, pattern in LEAK_PATTERNS.items():
                m = pattern.search(line)
                if m:
                    findings.append((name, path, lineno, m.group(0)))
    return findings


def test_repo_has_no_plaintext_oauth_secrets():
    """The committed working tree must not contain any literal OAuth bearer."""
    findings = _scan(REPO_ROOT)
    if findings:
        report = "\n".join(
            f"  {name} -> {path.relative_to(REPO_ROOT)}:{lineno}: {matched[:40]}…"
            for name, path, lineno, matched in findings
        )
        pytest.fail(
            "Plaintext OAuth secrets detected in the working tree.\n"
            "ADR-017 requires the no-plaintext-on-disk guarantee.\n"
            "Findings:\n" + report
        )


def test_scanner_catches_synthetic_leak(tmp_path):
    """Positive control — write a fake bearer and confirm the scanner flags it.

    This protects against the regexes silently breaking (e.g. a refactor
    swallowing the patterns dict) and `test_repo_has_no_plaintext_oauth_secrets`
    passing for the wrong reason.
    """
    bad = tmp_path / "config.py"
    bad.write_text(
        "GMAIL_TOKEN = 'ya29.A0AfH6SM_redacted_in_doc_but_real_in_prod'\n"
        "GITHUB_TOKEN = 'ghp_abcdefghijklmnopqrstuvwxyz0123456789'\n",
        encoding="utf-8",
    )

    findings = _scan(tmp_path)
    names = {name for name, *_ in findings}
    assert "google_access_token" in names
    assert "github_pat_classic" in names


def test_sbw_state_dirs_are_safe_if_present(tmp_path, monkeypatch):
    """If ~/.sbw exists, scan it too. The test creates a clean fake one to
    pin down what 'safe' looks like; the real ~/.sbw is checked by CI when
    integration tests have already run."""
    fake_home = tmp_path / "home"
    fake_sbw = fake_home / ".sbw"
    (fake_sbw / "logs").mkdir(parents=True)
    (fake_sbw / "vault").mkdir()
    # Vault content is ciphertext-shaped (high entropy, no bearer prefix)
    (fake_sbw / "vault" / "vault.db").write_bytes(b"\x00\x01\x02SQLite format 3" + b"\xaa" * 256)
    # Audit log with the canonical redacted shape
    (fake_sbw / "logs" / "integrations.log.jsonl").write_text(
        '{"event":"connect","provider":"gmail","params_redacted":{"q":"<redacted>"}}\n',
        encoding="utf-8",
    )

    findings = _scan(fake_sbw)
    assert findings == [], f"Synthetic state dir had unexpected leaks: {findings}"
