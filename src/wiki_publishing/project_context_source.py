"""Build-in-public ``ContentSource`` over the USER'S OWN local repo (ADR-026).

Where :class:`~wiki_publishing.repo_context_source.RepoContextSource` (ADR-025)
sources an **external** repo via the GitHub API, this module sources the
**local own** repo via local ``git`` + the ``docs/ADR-*.md`` / ``docs/PRD-*.md``
decision record. It composes ONE first-person :class:`WikiSnippet` per sub-type
and feeds the EXISTING ADR-024 drafter with **zero** drafter changes — the same
"more ``post_type``s + a new ``ContentSource``" move ADR-025 proved.

Three sub-types (the ADR-026 ``PostType`` values):

- ``project_launch`` — README/vision + the ACCEPTED ADRs' titles/decisions.
- ``project_feature`` — ONE ADR (chosen by ``topic``, else most recent) — its
  Context/Problem + Decision + any measured numbers — PLUS the commit subjects
  touching that ADR's file. Feeds the problem→decision→numbers war-story.
- ``project_weekly`` — recent ``git log`` commit subjects + the ADRs in that
  window, grouped sensibly.

Security (ADR-026 hard rule)
----------------------------
The source reads commit **messages** and ADR **prose** ONLY — **never raw
diffs**. A diff can carry a leaked secret/key. :func:`_guard_git_args` rejects
any ``-p`` / ``--patch`` / ``diff`` / ``show`` argument *before* anything shells
out, and the default ``git_runner`` runs that guard on every call. Both the
``git_runner`` and the ``docs_reader`` are injectable seams so tests stay
hermetic (no real git, no subprocess, no filesystem, no network).

No fabrication
--------------
Every line of a body comes from a real commit subject or a real ADR/PRD/README.
An empty window or an empty ADR set yields a minimal *honest* snippet — the
source never pads.
"""

from __future__ import annotations

import re
import subprocess
from collections.abc import Callable, Sequence
from pathlib import Path

from wiki_publishing.linkedin_draft import WikiSnippet
from wiki_publishing.post_types import PostType

# Injectable seams.
GitRunner = Callable[[list[str]], str]
"""``(args) -> stdout`` — runs ``git -C <repo> <args>`` (message-level only)."""
DocsReader = Callable[[], "list[tuple[str, str]]"]
"""``() -> [(relpath, text)]`` for ``docs/ADR-*.md`` / ``docs/PRD-*.md`` + README."""

# The three build-in-public sub-types this source composes.
_SUB_TYPES = frozenset(
    {
        PostType.PROJECT_LAUNCH.value,
        PostType.PROJECT_FEATURE.value,
        PostType.PROJECT_WEEKLY.value,
    }
)

# A git arg is "diff-y" (leaks file content) if it is one of these subcommands
# or a patch flag. Refusing these enforces the ADR-026 "messages only" rule.
_DIFF_SUBCOMMANDS = frozenset({"diff", "show"})
_DIFF_FLAGS = ("-p", "--patch")

# Default recent window for project_weekly.
_WEEKLY_SINCE = "14 days ago"

# Normalize a git remote URL to a clean, verifiable ``https://host/org/repo``
# (drops a ``.git`` suffix and a trailing slash). Handles the scp-like
# ``git@host:org/repo.git`` and the ``https://host/org/repo(.git)`` forms — so a
# project post cites the REAL, scheme-qualified repo link instead of a URL the
# model invents from the git author name (which is scheme-less and 404s).
_SCP_REMOTE_RE = re.compile(r"^git@([^:]+):(.+?)(?:\.git)?/?$")
_HTTPS_REMOTE_RE = re.compile(r"^(https?://[^/]+/.+?)(?:\.git)?/?$", re.IGNORECASE)


def _normalize_remote_url(raw: str) -> str:
    """Return ``raw`` as ``https://host/org/repo`` or ``""`` if not a usable URL."""
    raw = raw.strip()
    if not raw:
        return ""
    m = _SCP_REMOTE_RE.match(raw)
    if m:
        return f"https://{m.group(1)}/{m.group(2)}"
    m = _HTTPS_REMOTE_RE.match(raw)
    if m:
        return m.group(1)
    return ""


# Frontmatter / heading parsing.
_FRONTMATTER_RE = re.compile(r"^---\n(.*?)\n---\n", re.DOTALL)
_STATUS_RE = re.compile(r"^status:\s*(\S+)", re.MULTILINE)
_DATE_RE = re.compile(r"^date:\s*(\S+)", re.MULTILINE)
# The `# ADR-NNN: Title` (or `# PRD-NNN: ...`) heading.
_TITLE_HEADING_RE = re.compile(r"^#\s+((?:ADR|PRD)-\d+:\s*.+)$", re.MULTILINE)
# A `## Section` heading, used to slice out Context / Decision blocks.
_SECTION_RE = re.compile(r"^##\s+(.+?)\s*$", re.MULTILINE)
# Measured numbers worth surfacing in a war-story (e.g. "26x faster", "120ms",
# "37%", "3.2x", "1,200 rows"). Captured verbatim — never rounded or invented.
_NUMBER_RE = re.compile(
    r"\b\d[\d,]*(?:\.\d+)?\s?(?:x|×|%|ms|s|GB|MB|KB|req/s|rows|lines|files)\b",
    re.IGNORECASE,
)


def _guard_git_args(args: Sequence[str]) -> None:
    """Raise ``ValueError`` if ``args`` would emit file CONTENT or a diff.

    ADR-026 hard rule: the source reads commit **messages** only — a diff can
    leak a secret. This rejects the ``diff`` / ``show`` subcommands and any
    ``-p`` / ``--patch`` flag *before* git is ever invoked.
    """
    for arg in args:
        if arg in _DIFF_SUBCOMMANDS:
            raise ValueError(
                f"refusing diff-y git arg {arg!r}: ADR-026 forbids reading file "
                "content/diffs (a diff could leak a secret); messages only"
            )
        if arg.startswith(_DIFF_FLAGS):
            raise ValueError(
                f"refusing patch flag {arg!r}: ADR-026 forbids reading diffs "
                "(a diff could leak a secret); messages only"
            )


def _default_git_runner(repo_path: Path) -> GitRunner:
    """Build the real ``git_runner``: ``git -C <repo> <args>``, message-level only.

    Every call runs :func:`_guard_git_args` first, so a diff-y arg raises before
    any subprocess is spawned. ``check=False`` — a non-repo / empty result yields
    empty stdout, which the composers handle as an honest "nothing here".
    """

    def runner(args: list[str]) -> str:
        _guard_git_args(args)
        completed = subprocess.run(  # noqa: S603 — fixed argv, no shell, guarded args
            ["git", "-C", str(repo_path), *args],
            capture_output=True,
            text=True,
            check=False,
        )
        return completed.stdout

    return runner


def _default_docs_reader(repo_path: Path) -> DocsReader:
    """Build the real ``docs_reader``: glob ``docs/ADR-*.md`` + ``docs/PRD-*.md``
    and read ``README.md`` from the repo root. Reads PROSE only (no diffs)."""

    def reader() -> list[tuple[str, str]]:
        out: list[tuple[str, str]] = []
        docs_dir = repo_path / "docs"
        if docs_dir.is_dir():
            for pattern in ("ADR-*.md", "PRD-*.md"):
                for p in sorted(docs_dir.glob(pattern)):
                    try:
                        out.append((f"docs/{p.name}", p.read_text(encoding="utf-8")))
                    except OSError:
                        continue
        readme = repo_path / "README.md"
        if readme.is_file():
            try:
                out.append(("README.md", readme.read_text(encoding="utf-8")))
            except OSError:
                pass
        return out

    return reader


class _Doc:
    """A parsed ADR/PRD doc: its relpath, frontmatter facets, title, sections."""

    __slots__ = ("relpath", "text", "status", "date", "title", "body")

    def __init__(self, relpath: str, text: str) -> None:
        self.relpath = relpath
        self.text = text
        fm_match = _FRONTMATTER_RE.match(text)
        fm = fm_match.group(1) if fm_match else ""
        status_m = _STATUS_RE.search(fm)
        self.status = status_m.group(1).strip().lower() if status_m else ""
        date_m = _DATE_RE.search(fm)
        self.date = date_m.group(1).strip() if date_m else ""
        title_m = _TITLE_HEADING_RE.search(text)
        self.title = title_m.group(1).strip() if title_m else relpath
        # Body after the frontmatter (for section slicing).
        self.body = text[fm_match.end() :] if fm_match else text

    @property
    def is_accepted(self) -> bool:
        return self.status == "accepted"

    def section(self, *needles: str) -> str:
        """Return the text of the first ``## `` section whose heading contains any
        of ``needles`` (case-insensitive), trimmed. Empty string if none."""
        matches = list(_SECTION_RE.finditer(self.body))
        for i, m in enumerate(matches):
            heading = m.group(1).lower()
            if any(n.lower() in heading for n in needles):
                start = m.end()
                end = matches[i + 1].start() if i + 1 < len(matches) else len(self.body)
                return self.body[start:end].strip()
        return ""

    def numbers(self) -> list[str]:
        """Measured numbers present in the doc body, verbatim, de-duped in order."""
        seen: dict[str, None] = {}
        for raw in _NUMBER_RE.findall(self.body):
            token = raw.strip()
            if token not in seen:
                seen[token] = None
        return list(seen)


class ProjectContextSource:
    """A ``ContentSource`` over the user's OWN local repo (ADR-026).

    Implements the ``ContentSource`` protocol
    (``search(query, *, limit, categories) -> list[WikiSnippet]``). The subject
    was already resolved to one repo + one sub-type at construction time, so
    :meth:`search` ignores ``query``/``categories`` and returns exactly one
    first-person snippet composed per the sub-type.

    Prefer :meth:`from_repo` to construct.

    Parameters
    ----------
    repo_path:
        Path to the local working tree (title = its basename; ``page_path`` = the
        path, as provenance).
    sub_type:
        One of ``"project_launch"`` / ``"project_feature"`` / ``"project_weekly"``.
    topic:
        For ``project_feature``: substring used to pick the best-matching ADR/PRD
        (case-insensitive, over filename + title). ``None`` → most recent ADR.
    git_runner:
        Injectable ``(args) -> stdout``. Defaults to the guarded subprocess
        runner. **Message-level args only** (``log --pretty=...`` etc.).
    docs_reader:
        Injectable ``() -> [(relpath, text)]``. Defaults to globbing the repo's
        ``docs/`` + ``README.md``.
    """

    def __init__(
        self,
        repo_path: str | Path,
        sub_type: str,
        *,
        topic: str | None = None,
        git_runner: GitRunner | None = None,
        docs_reader: DocsReader | None = None,
    ) -> None:
        if sub_type not in _SUB_TYPES:
            raise ValueError(
                f"sub_type {sub_type!r} is not a build-in-public type; expected "
                f"one of {sorted(_SUB_TYPES)}"
            )
        self._repo_path = Path(repo_path)
        self._sub_type = sub_type
        self._topic = topic
        self._git_runner: GitRunner = (
            git_runner if git_runner is not None else _default_git_runner(self._repo_path)
        )
        self._docs_reader: DocsReader = (
            docs_reader if docs_reader is not None else _default_docs_reader(self._repo_path)
        )
        self._cached: WikiSnippet | None = None

    # The constructor already accepts the full identity, so ``from_repo`` is a
    # thin, named alias kept for parity with ADR-025's ``RepoContextSource.from_url``
    # and to give the MCP wiring a clear, intent-revealing entry point.
    @classmethod
    def from_repo(
        cls,
        repo_path: str | Path,
        sub_type: str,
        *,
        topic: str | None = None,
        git_runner: GitRunner | None = None,
        docs_reader: DocsReader | None = None,
    ) -> ProjectContextSource:
        """Build a :class:`ProjectContextSource` for ``repo_path`` + ``sub_type``."""
        return cls(
            repo_path,
            sub_type,
            topic=topic,
            git_runner=git_runner,
            docs_reader=docs_reader,
        )

    # -- ContentSource protocol ----------------------------------------------

    def search(
        self,
        query: str,
        *,
        limit: int = 3,
        categories: Sequence[str] | None = None,
    ) -> list[WikiSnippet]:
        """Return the single composed snippet, ignoring ``query``/``categories``.

        The subject was resolved at construction; there is exactly one candidate,
        so the query and category filters are no-ops and ``limit`` is honoured
        trivially (the list is always length 1)."""
        return [self._snippet()]

    # -- composition ----------------------------------------------------------

    @property
    def _project_name(self) -> str:
        return self._repo_path.name or str(self._repo_path)

    def _snippet(self) -> WikiSnippet:
        if self._cached is None:
            body = self._compose_body()
            self._cached = WikiSnippet(
                title=self._project_name,
                page_path=str(self._repo_path),
                body=body,
            )
        return self._cached

    def _load_docs(self) -> list[_Doc]:
        return [_Doc(rel, text) for rel, text in self._docs_reader()]

    def _remote_url(self) -> str:
        """The repo's ``origin`` remote as a clean ``https://host/org/repo``.

        Read via ``git remote get-url origin`` (message-level — allowed by the
        ADR-026 diff guard). Empty string when there is no ``origin`` remote, so
        no fabricated URL is ever injected."""
        return _normalize_remote_url(self._git_runner(["remote", "get-url", "origin"]))

    def _compose_body(self) -> str:
        if self._sub_type == PostType.PROJECT_LAUNCH.value:
            body = self._compose_launch()
        elif self._sub_type == PostType.PROJECT_FEATURE.value:
            body = self._compose_feature()
        else:
            body = self._compose_weekly()
        # Surface the REAL repo URL so the drafter's deterministic source-URL
        # guarantee cites a verified, scheme-qualified link instead of one the
        # model invents from the git author name (see issue #193).
        url = self._remote_url()
        if url:
            body = f"{body}\n## Fuentes\n\n{url}\n"
        return body

    # -- project_launch -------------------------------------------------------

    def _compose_launch(self) -> str:
        docs = self._load_docs()
        readme = next((d for d in docs if d.relpath.upper().endswith("README.MD")), None)
        accepted = [d for d in docs if d.is_accepted and _is_adr(d.relpath)]

        lines: list[str] = [f"# {self._project_name} — lanzamiento", ""]
        if readme is not None:
            vision = _readme_vision(readme.text)
            if vision:
                lines += ["## Visión", "", vision, ""]
        if accepted:
            lines += ["## Decisiones de diseño aceptadas", ""]
            for d in accepted:
                lines.append(f"### {d.title}")
                decision = d.section("Decision outcome", "Decision", "Decisión")
                if decision:
                    lines += ["", _first_paragraph(decision), ""]
                else:
                    lines.append("")
        return _finish(lines, project=self._project_name)

    # -- project_feature ------------------------------------------------------

    def _compose_feature(self) -> str:
        docs = self._load_docs()
        candidates = [d for d in docs if _is_adr(d.relpath) or _is_prd(d.relpath)]
        chosen = _pick_feature_doc(candidates, self._topic)
        if chosen is None:
            return _finish(
                [
                    f"# {self._project_name} — feature",
                    "",
                    "_No hay un ADR/PRD que respalde esta feature todavía._",
                ],
                project=self._project_name,
            )

        lines: list[str] = [f"# {self._project_name} — {chosen.title}", ""]
        problem = chosen.section("Context", "Problem", "problema", "Contexto")
        if problem:
            lines += ["## Problema / contexto", "", _first_paragraph(problem), ""]
        decision = chosen.section("Decision outcome", "Decision", "Decisión")
        if decision:
            lines += ["## Decisión", "", _first_paragraph(decision), ""]
        numbers = chosen.numbers()
        if numbers:
            lines += ["## Números medidos", ""]
            lines += [f"- {n}" for n in numbers]
            lines.append("")

        # Commit subjects touching this ADR/PRD's file (messages only — guarded).
        subjects = self._commit_subjects_for_path(chosen.relpath)
        if subjects:
            lines += ["## Commits relacionados (lo que shippeé)", ""]
            lines += [f"- {s}" for s in subjects]
            lines.append("")
        return _finish(lines, project=self._project_name)

    def _commit_subjects_for_path(self, relpath: str) -> list[str]:
        out = self._git_runner(["log", "--pretty=%s", "--", relpath])
        return _nonempty_lines(out)

    # -- project_weekly -------------------------------------------------------

    def _compose_weekly(self) -> str:
        subjects = _nonempty_lines(
            self._git_runner(["log", f"--since={_WEEKLY_SINCE}", "--pretty=%s"])
        )
        docs = self._load_docs()
        # ADRs added/modified in the window: approximated by the recent ADR set —
        # we only surface those whose file shows up in the windowed git log paths.
        recent_adrs = self._adrs_in_window(docs)

        if not subjects and not recent_adrs:
            # Honest minimal snippet — never pad.
            return _finish(
                [
                    f"# {self._project_name} — build-log",
                    "",
                    "_No hay commits en la ventana reciente._",
                ],
                project=self._project_name,
            )

        lines: list[str] = [f"# {self._project_name} — build-log", ""]
        if subjects:
            grouped = _group_subjects(subjects)
            lines += ["## Lo que shippeé", ""]
            for group, items in grouped:
                if group:
                    lines.append(f"**{group}**")
                lines += [f"- {s}" for s in items]
                lines.append("")
        if recent_adrs:
            lines += ["## Decisiones registradas en la ventana", ""]
            lines += [f"- {d.title}" for d in recent_adrs]
            lines.append("")
        return _finish(lines, project=self._project_name)

    def _adrs_in_window(self, docs: list[_Doc]) -> list[_Doc]:
        """ADR docs touched in the recent window, via guarded ``git log --name-only``
        over ``docs/`` (paths only — never content)."""
        out = self._git_runner(
            ["log", f"--since={_WEEKLY_SINCE}", "--name-only", "--pretty=format:", "--", "docs"]
        )
        touched = {line.strip() for line in out.splitlines() if line.strip()}
        if not touched:
            return []
        return [d for d in docs if _is_adr(d.relpath) and d.relpath in touched]


# --------------------------------------------------------------------------- #
# module-level helpers
# --------------------------------------------------------------------------- #


def _is_adr(relpath: str) -> bool:
    return "ADR-" in relpath.upper()


def _is_prd(relpath: str) -> bool:
    return "PRD-" in relpath.upper()


def _nonempty_lines(text: str) -> list[str]:
    return [ln.strip() for ln in text.splitlines() if ln.strip()]


def _first_paragraph(text: str) -> str:
    """The first non-empty paragraph of ``text`` (blocks split on blank lines)."""
    blocks = re.split(r"\n\s*\n", text.strip())
    for block in blocks:
        cleaned = block.strip()
        if cleaned:
            return cleaned
    return text.strip()


def _readme_vision(readme: str) -> str:
    """Extract the README's lead paragraph (the vision), skipping the H1 title
    and any badge/image lines."""
    body = readme
    body = re.sub(r"^#\s+.*$", "", body, count=1, flags=re.MULTILINE)  # drop first H1
    for block in re.split(r"\n\s*\n", body.strip()):
        cleaned = block.strip()
        if not cleaned:
            continue
        if cleaned.startswith(("![", "[![", "<img", "#")):
            continue
        return cleaned
    return body.strip()


def _norm_tokens(text: str) -> list[str]:
    """Lowercase ``text`` and split on any non-alphanumeric run, so a hyphenated
    filename and a spaced topic compare on the same token basis."""
    return [t for t in re.split(r"[^a-z0-9]+", text.lower()) if t]


def _pick_feature_doc(docs: list[_Doc], topic: str | None) -> _Doc | None:
    """Pick the ADR/PRD best matching ``topic``, scored by token overlap over the
    filename + title (so a spaced topic matches a hyphenated filename, and minor
    stop-words like "as"/"a" don't break the match). Ties / no topic / no overlap
    fall back to the most recent *accepted* doc (a feature war-story is about a
    real shipped decision), else the most recent doc overall."""
    if not docs:
        return None
    if topic:
        needle = set(_norm_tokens(topic))
        if needle:
            best: _Doc | None = None
            best_score = 0
            for d in docs:
                hay = set(_norm_tokens(d.relpath)) | set(_norm_tokens(d.title))
                score = len(needle & hay)
                # Strict '>' keeps the FIRST (already date-sorted by _most_recent
                # below for ties) — but pick deterministically by date on ties.
                if score > best_score or (
                    score == best_score
                    and score > 0
                    and best is not None
                    and (d.date, d.relpath) > (best.date, best.relpath)
                ):
                    best, best_score = d, score
            if best is not None and best_score > 0:
                return best
        # Fall through when the topic overlaps nothing.
    accepted = [d for d in docs if d.is_accepted]
    return _most_recent(accepted) if accepted else _most_recent(docs)


def _most_recent(docs: list[_Doc]) -> _Doc:
    """Most recent doc by frontmatter ``date`` (lexicographic ISO sort); ties and
    missing dates fall back to relpath order so the choice is deterministic."""
    return max(docs, key=lambda d: (d.date, d.relpath))


# Conventional-commit type → human group label for the weekly build-log.
_COMMIT_GROUPS: tuple[tuple[str, str], ...] = (
    ("feat", "Nuevas features"),
    ("fix", "Arreglos"),
    ("refactor", "Refactors"),
    ("docs", "Documentación"),
    ("test", "Tests"),
    ("perf", "Performance"),
    ("chore", "Mantenimiento"),
)
_CONVENTIONAL_RE = re.compile(r"^([a-z]+)(?:\([^)]*\))?!?:")


def _group_subjects(subjects: list[str]) -> list[tuple[str, list[str]]]:
    """Group commit subjects by conventional-commit type, preserving order.

    Returns ``[(group_label, [subjects...]), ...]``. Subjects that don't parse as
    conventional commits fall into an unlabelled ("") trailing group. Never
    invents content — only re-orders the real subjects."""
    by_type: dict[str, list[str]] = {}
    other: list[str] = []
    for s in subjects:
        m = _CONVENTIONAL_RE.match(s)
        if m:
            by_type.setdefault(m.group(1), []).append(s)
        else:
            other.append(s)
    grouped: list[tuple[str, list[str]]] = []
    for ctype, label in _COMMIT_GROUPS:
        if ctype in by_type:
            grouped.append((label, by_type.pop(ctype)))
    # Any conventional type not in the known list, in first-seen order.
    for ctype, items in by_type.items():
        grouped.append((ctype, items))
    if other:
        grouped.append(("", other))
    return grouped


def _finish(lines: list[str], *, project: str) -> str:
    """Join body lines, collapse trailing blanks, ensure a single trailing NL."""
    text = "\n".join(lines).rstrip()
    return text + "\n"
