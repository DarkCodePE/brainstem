"""Lesson <-> wiki page substrate (SPEC-010 FR-3, ADR-033 D1).

A lesson is a normal wiki page: its frontmatter carries the fields the
existing ``validate_frontmatter`` contract requires (``title``, ``date``,
``sources``, ``tags``, ``origin``) plus the lesson-typed fields. The wiki —
not a side store — is the memory substrate.
"""

from __future__ import annotations

import yaml

from wiki_lessons.distill import Lesson
from wiki_lessons.verdict import Verdict

LESSON_DIR = "wiki/lessons"

#: ``origin`` value accepted by the existing frontmatter validator.
_ORIGIN = "llm-synthesized"


def lesson_page_path(lesson: Lesson) -> str:
    return f"{LESSON_DIR}/{lesson.lesson_id}.md"


def render_lesson_page(lesson: Lesson) -> str:
    """Emit the full markdown page (frontmatter + body) for a lesson."""
    front: dict[str, object] = {
        "title": lesson.title,
        "date": lesson.created_at[:10],
        "sources": [lesson.derived_from],
        "tags": ["lesson", lesson.kind, lesson.domain],
        "origin": _ORIGIN,
        "type": "lesson",
        "lesson_id": lesson.lesson_id,
        "lesson_kind": lesson.kind,
        "provenance": lesson.provenance,
        "confidence": lesson.confidence,
        "reward": lesson.verdict.reward,
        "verdict_source": lesson.verdict.source,
        "verdict_kind": lesson.verdict.kind,
        "verdict_success": lesson.verdict.success,
        "repo": lesson.repo,
        "domain": lesson.domain,
        "source_key": lesson.source_key,
        "derived_from": lesson.derived_from,
        "created_at": lesson.created_at,
    }
    if lesson.supersedes:
        front["supersedes"] = [f"[[{sid}]]" for sid in lesson.supersedes]
    if lesson.expires:
        front["expires"] = lesson.expires
    if lesson.notes:
        front["notes"] = list(lesson.notes)

    fm = yaml.safe_dump(front, sort_keys=False, allow_unicode=True).strip()

    lines = [f"# {lesson.title}", "", "## Strategy", "", lesson.strategy, ""]
    if lesson.key_learnings:
        lines += ["## Key learnings", ""]
        lines += [f"- {item}" for item in lesson.key_learnings]
        lines.append("")
    if lesson.verdict.components:
        lines += ["## Evidence", "", "| component | value |", "| --- | --- |"]
        lines += [f"| {name} | {value:.4g} |" for name, value in lesson.verdict.components]
        lines.append("")
    lines += ["## Trace", "", f"- Task: `{lesson.derived_from}`", f"- Repo: `{lesson.repo}`"]

    return f"---\n{fm}\n---\n\n" + "\n".join(lines) + "\n"


def _split_frontmatter(text: str) -> tuple[dict[str, object] | None, str]:
    if not text.startswith("---"):
        return None, text
    parts = text.split("---", 2)
    if len(parts) < 3:
        return None, text
    try:
        data = yaml.safe_load(parts[1])
    except yaml.YAMLError:
        return None, text
    if not isinstance(data, dict):
        return None, text
    return data, parts[2]


def _section(body: str, heading: str) -> str:
    marker = f"## {heading}"
    if marker not in body:
        return ""
    chunk = body.split(marker, 1)[1]
    next_idx = chunk.find("\n## ")
    if next_idx != -1:
        chunk = chunk[:next_idx]
    return chunk.strip()


def parse_lesson_page(text: str) -> Lesson | None:
    """Round-trip a page produced by :func:`render_lesson_page`.

    Returns ``None`` for pages that are not lessons or are malformed —
    callers iterate over a directory and must skip strangers gracefully.
    """
    front, body = _split_frontmatter(text)
    if front is None or front.get("type") != "lesson":
        return None
    try:
        verdict = Verdict(
            source=str(front["verdict_source"]),
            reward=float(front["reward"]),
            success=bool(front["verdict_success"]),
            kind=str(front["verdict_kind"]),
        )
        supersedes = tuple(str(item).strip("[]") for item in front.get("supersedes", []) or [])
        learnings = tuple(
            line[2:].strip()
            for line in _section(body, "Key learnings").splitlines()
            if line.startswith("- ")
        )
        return Lesson(
            lesson_id=str(front["lesson_id"]),
            source_key=str(front["source_key"]),
            title=str(front["title"]),
            kind=str(front["lesson_kind"]),
            strategy=_section(body, "Strategy"),
            key_learnings=learnings,
            domain=str(front["domain"]),
            repo=str(front["repo"]),
            provenance=str(front["provenance"]),
            confidence=float(front["confidence"]),
            verdict=verdict,
            derived_from=str(front["derived_from"]),
            created_at=str(front["created_at"]),
            supersedes=supersedes,
            expires=str(front["expires"]) if front.get("expires") else None,
            notes=tuple(str(n) for n in front.get("notes", []) or []),
        )
    except (KeyError, TypeError, ValueError):
        return None
