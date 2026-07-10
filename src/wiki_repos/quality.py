"""Narrative-quality judge for the ``## Evolution & decisions`` section (ADR-031).

PRD-014 AC-3 ("the section names a *genuine* recent theme/fix/refactor") was the one
acceptance criterion that could not be automated — it was a human spot-check. This
module turns it into a graded ``[0, 1]`` score, porting the SHAPE of Repo2RLEnv's
``pr_diff`` reward (5 deterministic components + 1 LLM judge + graceful degradation +
a catastrophic hard cap) to score *prose* instead of diffs.

Design (mirrors ``synthesize._evolution_section`` + ``refine_prose`` and ADR-030):
- **Deterministic floor.** The 5 deterministic components compute with pure stdlib from
  the mined :class:`~wiki_repos.types.RepoHistory` (the oracle). No router, no network,
  ``$0``, reproducible — the authoritative base signal.
- **Injectable judge seam.** ``judge`` is ``(QualityContext) -> float`` or ``None``; tests
  inject fakes and the real ``wiki_routing`` judge is built lazily (so the deterministic
  core keeps zero hard dependency on the routing layer).
- **Graceful degradation = weight redistribution.** No judge / judge error / no API key →
  ``llm_genuineness = None``, ``judge_status`` recorded, and its weight is redistributed
  proportionally across the weighted deterministic components so the score stays in
  ``[0, 1]`` and comparable. Never raises, never silently zeros.
- **Catastrophic hard cap.** ``signal_density < _CAP_THRESHOLD`` clamps the score to
  ``≤ _CAP_SCORE`` — a charitable judge can't inflate a section that is all
  dependabot/version-bump noise (the issue #167 failure mode).

This module is a QA/scoring tool. It is **NOT** called inside ``ingest_github_repo`` — the
AC-5 per-ingest budget (≤4 GitHub calls + ≤1 LLM call) is untouched; the judge's LLM call
happens only when the spot-check harness runs.
"""

from __future__ import annotations

import logging
import re
from collections.abc import Callable
from dataclasses import dataclass, field

from wiki_repos.history import classify_commit
from wiki_repos.types import RepoHistory

logger = logging.getLogger(__name__)

__all__ = ["QualityContext", "QualityScore", "score_history", "build_router_judge"]

#: Change-shaping kinds — the engineering signal a narrative should surface.
_CHANGE_SHAPING: frozenset[str] = frozenset({"fix", "feat", "refactor", "perf", "security"})

#: How many merged PRs the section renders (mirrors synthesize._MAX_PRS_SHOWN).
_MAX_PRS_SHOWN = 8
#: pr_coverage saturates at this many shown PRs.
_PR_COVERAGE_TARGET = 3
#: Verdict threshold: score >= this → "genuine".
_GENUINE_THRESHOLD = 0.70

#: Catastrophic-noise hard cap (mirrors pr_diff's size_sanity cap).
_CAP_THRESHOLD = 0.15
_CAP_SCORE = 0.40

#: Base weights. grounding is a guard (weight 0). Sum of all == 1.0.
_BASE_WEIGHTS: dict[str, float] = {
    "grounding": 0.00,
    "signal_density": 0.25,
    "rationale_richness": 0.25,
    "pr_coverage": 0.10,
    "diversity": 0.05,
    "llm_genuineness": 0.35,
}
_JUDGE_KEY = "llm_genuineness"

#: A PR/commit is noise (not change-shaping) when its subject matches these.
_NOISE_TITLE_RE = re.compile(
    r"^\s*(?:bump|release|chore|merge\b|revert|update|deps?\b|"
    r"build\(deps\)|version|v?\d+\.\d+)",
    flags=re.IGNORECASE,
)
_BOT_LABELS: frozenset[str] = frozenset({"dependencies", "dependabot", "automated"})
_REF_RE = re.compile(r"#(?P<num>\d+)|\b(?P<sha>[0-9a-f]{7,40})\b")


@dataclass(frozen=True, slots=True)
class QualityContext:
    """What an LLM judge sees: the rendered section + the raw mined facts.

    Passed to an injected ``judge`` so it can rate genuineness against the same
    oracle the deterministic components used (the real history), not just the prose.
    """

    section: str
    history: RepoHistory


@dataclass(frozen=True, slots=True)
class QualityScore:
    """Graded narrative-quality result (ADR-031) — mirrors pr_diff's reward.json."""

    score: float
    components: dict[str, float | None]
    weights: dict[str, float]
    judge_status: str  # "ok" | "no_api_key" | "error" | "disabled"
    capped: bool
    verdict: str  # "genuine" | "weak"
    notes: tuple[str, ...] = field(default_factory=tuple)


# --------------------------------------------------------------------------- #
# Deterministic components
# --------------------------------------------------------------------------- #
def _pr_is_signal(title: str, labels: tuple[str, ...]) -> bool:
    """A merged PR carries engineering signal (vs dependency/release/merge noise)."""
    if any(lbl.lower() in _BOT_LABELS for lbl in labels):
        return False
    if classify_commit(title) in _CHANGE_SHAPING:
        return True
    return not _NOISE_TITLE_RE.match(title or "")


def _commit_is_signal(kind: str, summary: str) -> bool:
    if summary.lower().startswith(("merge pull request", "merge branch")):
        return False
    return kind in _CHANGE_SHAPING


def _signal_density(history: RepoHistory) -> float:
    """Fraction of shown items that are change-shaping. PRs and commits are averaged
    so a flood of noisy commits can't drown a handful of strong PRs (or vice versa)."""
    fracs: list[float] = []
    shown_prs = history.merged_prs[:_MAX_PRS_SHOWN]
    if shown_prs:
        sig = sum(1 for pr in shown_prs if _pr_is_signal(pr.title, pr.labels))
        fracs.append(sig / len(shown_prs))
    if history.commits:
        sig = sum(1 for c in history.commits if _commit_is_signal(c.kind, c.summary))
        fracs.append(sig / len(history.commits))
    return (sum(fracs) / len(fracs)) if fracs else 0.0


def _rationale_richness(history: RepoHistory) -> float:
    """Fraction of shown PRs carrying a non-empty body excerpt (the *why*)."""
    shown = history.merged_prs[:_MAX_PRS_SHOWN]
    if not shown:
        return 0.0
    return sum(1 for pr in shown if pr.body_excerpt.strip()) / len(shown)


def _pr_coverage(history: RepoHistory) -> float:
    """Is there enough merged-PR material? Saturates at _PR_COVERAGE_TARGET."""
    n = min(len(history.merged_prs), _MAX_PRS_SHOWN)
    return min(n, _PR_COVERAGE_TARGET) / _PR_COVERAGE_TARGET


def _diversity(history: RepoHistory) -> float:
    """Distinct change-shaping kinds among shown items / total change-shaping kinds."""
    kinds: set[str] = set()
    for pr in history.merged_prs[:_MAX_PRS_SHOWN]:
        k = classify_commit(pr.title)
        if k in _CHANGE_SHAPING:
            kinds.add(k)
    for c in history.commits:
        if c.kind in _CHANGE_SHAPING:
            kinds.add(c.kind)
    return len(kinds) / len(_CHANGE_SHAPING)


def _grounding(section: str | None, history: RepoHistory) -> float:
    """Every ``#n`` / ``sha`` cited in the section text exists in the history.

    A hallucination guard (weight 0) that bites once ``refine_prose`` rewords the
    section. ``1.0`` when no section text is passed (nothing to contradict).
    """
    if not section:
        return 1.0
    known_nums = {str(pr.number) for pr in history.merged_prs}
    known_shas = {c.sha for c in history.commits if c.sha}
    cited = 0
    grounded = 0
    for m in _REF_RE.finditer(section):
        num, sha = m.group("num"), m.group("sha")
        if num is not None:
            cited += 1
            grounded += num in known_nums
        elif sha is not None:
            cited += 1
            # A cited sha is grounded if it prefixes (or is prefixed by) a known sha.
            grounded += any(s.startswith(sha) or sha.startswith(s) for s in known_shas)
    return (grounded / cited) if cited else 1.0


# --------------------------------------------------------------------------- #
# Scoring
# --------------------------------------------------------------------------- #
def _effective_weights(judge_active: bool) -> dict[str, float]:
    """Base weights when the judge ran; else redistribute the judge's weight
    proportionally across the other weighted components (grounding stays 0)."""
    if judge_active:
        return dict(_BASE_WEIGHTS)
    judge_w = _BASE_WEIGHTS[_JUDGE_KEY]
    redistributable = {k: w for k, w in _BASE_WEIGHTS.items() if k != _JUDGE_KEY and w > 0.0}
    total = sum(redistributable.values())
    weights = {k: 0.0 for k in _BASE_WEIGHTS}
    for k, w in redistributable.items():
        weights[k] = w + judge_w * (w / total)
    return weights


def score_history(
    history: RepoHistory | None,
    *,
    section: str | None = None,
    judge: Callable[[QualityContext], float] | None = None,
    threshold: float = _GENUINE_THRESHOLD,
) -> QualityScore:
    """Grade the narrative quality of a mined history (ADR-031).

    Args:
        history: The mined :class:`RepoHistory` (the oracle). ``None``/empty → score 0.
        section: Optional rendered section text — enables the ``grounding`` guard.
        judge: Optional ``(QualityContext) -> float in [0,1]``. ``None`` → deterministic
            only, with the judge's weight redistributed. Any exception degrades the same
            way (``judge_status="error"``) — the judge never fails the score.
        threshold: ``score >= threshold`` → ``verdict="genuine"``.
    """
    if history is None or (not history.merged_prs and not history.commits):
        return QualityScore(
            score=0.0,
            components={k: None for k in _BASE_WEIGHTS},
            weights=_effective_weights(judge_active=False),
            judge_status="disabled" if judge is None else "ok",
            capped=False,
            verdict="weak",
            notes=("no history mined — nothing to score",),
        )

    components: dict[str, float | None] = {
        "grounding": _grounding(section, history),
        "signal_density": _signal_density(history),
        "rationale_richness": _rationale_richness(history),
        "pr_coverage": _pr_coverage(history),
        "diversity": _diversity(history),
        "llm_genuineness": None,
    }

    notes: list[str] = []
    judge_status = "disabled"
    if judge is not None:
        try:
            raw = float(judge(QualityContext(section=section or "", history=history)))
            components[_JUDGE_KEY] = max(0.0, min(1.0, raw))
            judge_status = "ok"
        except Exception as exc:  # noqa: BLE001 — judge never fails the score
            logger.info("quality judge degrade: %s", type(exc).__name__)
            judge_status = "error"
            notes.append("judge failed — scored on deterministic components only")

    judge_active = components[_JUDGE_KEY] is not None
    weights = _effective_weights(judge_active)
    score = sum(v * weights[k] for k, v in components.items() if v is not None)
    score = max(0.0, min(1.0, score))

    capped = False
    sig = components["signal_density"] or 0.0
    if sig < _CAP_THRESHOLD and score > _CAP_SCORE:
        score = _CAP_SCORE
        capped = True
        notes.append(
            f"capped at {_CAP_SCORE} — signal_density {sig:.2f} < {_CAP_THRESHOLD} "
            "(section is mostly dependency/version/merge noise; see issue #167)"
        )

    return QualityScore(
        score=round(score, 6),
        components={k: (round(v, 6) if isinstance(v, float) else v) for k, v in components.items()},
        weights={k: round(w, 6) for k, w in weights.items()},
        judge_status=judge_status,
        capped=capped,
        verdict="genuine" if score >= threshold else "weak",
        notes=tuple(notes),
    )


# --------------------------------------------------------------------------- #
# Optional real judge (lazy router import — mirrors synthesize._call_router_for_refine)
# --------------------------------------------------------------------------- #
_JUDGE_SYSTEM_PROMPT = (
    "You are a strict reviewer scoring whether a repo's 'Evolution & decisions' "
    "section names a GENUINE, specific, recent theme/fix/refactor that is verifiable "
    "in the provided merged-PR and commit facts. Reward concrete engineering rationale; "
    "penalize vague filler, dependency bumps, and version-release noise. Respond with a "
    "single float in [0,1] and nothing else."
)


def build_router_judge(router: object) -> Callable[[QualityContext], float]:
    """Build a judge callable backed by a ``wiki_routing`` router (the integration seam).

    Lazily imports the routing layer so the deterministic core stays dependency-free.
    The returned callable raises on any transport/parse failure; ``score_history``
    catches it and degrades — so this never needs its own try/except here.
    """

    def _judge(ctx: QualityContext) -> float:
        from wiki_routing import Message, TaskDescriptor  # local import — optional dep

        prompt = (
            f"{_JUDGE_SYSTEM_PROMPT}\n\n--- SECTION ---\n{ctx.section}\n\n"
            f"--- FACTS ---\nmerged PRs: "
            + "; ".join(
                f"#{pr.number} {pr.title}" for pr in ctx.history.merged_prs[:_MAX_PRS_SHOWN]
            )
            + "\ncommit kinds: "
            + ", ".join(f"{k}:{n}" for k, n in ctx.history.kind_counts.items())
        )
        descriptor = TaskDescriptor(kind="judge", complexity=0.3)
        result = router.call([Message(role="user", content=prompt)], descriptor)  # type: ignore[attr-defined]
        text = getattr(result, "text", str(result)).strip()
        m = re.search(r"[0-9]*\.?[0-9]+", text)
        if not m:
            raise ValueError(f"judge returned no parseable float: {text!r}")
        return float(m.group(0))

    return _judge
