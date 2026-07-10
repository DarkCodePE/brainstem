"""Tests for the narrative-quality judge (ADR-031, wiki_repos/quality.py).

Hermetic: the LLM judge is an injectable callable, so no test touches a router or
the network. The deterministic components are pure functions of a RepoHistory.
"""

from __future__ import annotations

from wiki_repos.quality import (
    QualityContext,
    QualityScore,
    score_history,
)
from wiki_repos.types import Commit, HistoryStats, PullRequest, RepoHistory


def _pr(number: int, title: str, *, body: str = "", labels: tuple[str, ...] = ()) -> PullRequest:
    return PullRequest(
        number=number,
        title=title,
        merged_at=f"2026-05-{number:02d}T00:00:00Z",
        author="dev",
        labels=labels,
        body_excerpt=body,
    )


def _commit(sha: str, summary: str, kind: str) -> Commit:
    return Commit(sha=sha, summary=summary, kind=kind, date="2026-05-01T00:00:00Z")


def _history(prs: list[PullRequest], commits: list[Commit]) -> RepoHistory:
    return RepoHistory(
        merged_prs=tuple(prs),
        commits=tuple(commits),
        stats=HistoryStats(n_prs=len(prs), n_commits=len(commits), truncated=False),
    )


# --- rich, genuine history -------------------------------------------------- #
def _rich_history() -> RepoHistory:
    return _history(
        prs=[
            _pr(
                84,
                "fix: close x86 vs ARM recall gap via accumulator flush",
                body="Closes a precision gap…",
            ),
            _pr(
                83,
                "refactor: audit-driven Rust core fidelity work",
                body="Six waves of audit work…",
            ),
            _pr(
                80,
                "feat: add CI workflow that runs tests on every PR",
                body="Adds .github/workflows/ci.yml…",
            ),
            _pr(
                79,
                "perf: emit BLAS link directives from build.rs",
                body="Avoids cryptic linker errors…",
            ),
        ],
        commits=[
            _commit("d8d2d26", "fix: add __repr__ to TurboQuantIndex", "fix"),
            _commit("aaaaaaa", "feat: TQ+ per-coordinate calibration", "feat"),
            _commit("bbbbbbb", "refactor: split kernel module", "refactor"),
        ],
    )


# --- all-noise history (dependabot + release bumps + merges) ---------------- #
# Bodies are intentionally rich (dependabot PRs have long release-note bodies) so
# rationale_richness is high while signal_density is 0 — this forces the cap to
# actually CLAMP a score that would otherwise exceed _CAP_SCORE.
def _noise_history() -> RepoHistory:
    return _history(
        prs=[
            _pr(
                10,
                "Bump actions/checkout from 4 to 5",
                body="Release notes…",
                labels=("dependencies",),
            ),
            _pr(9, "Release: v1.2.0", body="Rolls up the changes merged this week…"),
            _pr(8, "chore: update deps", body="Updates the pinned dependency set…"),
        ],
        commits=[
            _commit("1111111", "Merge pull request #10 from dependabot/x", "other"),
            _commit("2222222", "Release: v1.2.0", "other"),
            _commit("3333333", "chore: bump version", "chore"),
        ],
    )


def test_rich_history_scores_genuine_deterministically() -> None:
    """A history full of fix/feat/refactor PRs with rationale scores high with NO judge."""
    score = score_history(_rich_history())
    assert isinstance(score, QualityScore)
    assert score.judge_status == "disabled"
    assert score.components["llm_genuineness"] is None
    assert score.components["signal_density"] == 1.0
    assert score.components["rationale_richness"] == 1.0
    assert not score.capped
    assert score.verdict == "genuine"
    assert score.score >= 0.70


def test_noise_history_is_capped() -> None:
    """An all-dependabot/release/merge history trips the signal_density hard cap."""
    score = score_history(_noise_history())
    assert score.components["signal_density"] < 0.15
    assert score.capped is True
    assert score.score <= 0.40
    assert score.verdict == "weak"
    assert any("issue #167" in n for n in score.notes)


def test_weights_redistribute_when_judge_absent() -> None:
    """With no judge, the judge weight is redistributed and effective weights sum to 1."""
    score = score_history(_rich_history())
    assert score.weights["llm_genuineness"] == 0.0
    # Effective weights sum to 1.0 (within 6-decimal display rounding).
    assert abs(sum(score.weights.values()) - 1.0) < 1e-3
    # signal_density (base 0.25) gains a proportional share of the 0.35 judge weight.
    assert score.weights["signal_density"] > 0.25


def test_judge_contributes_when_injected() -> None:
    """An injected judge sets llm_genuineness and judge_status=ok; base weights apply."""
    seen: list[QualityContext] = []

    def judge(ctx: QualityContext) -> float:
        seen.append(ctx)
        return 0.9

    score = score_history(_rich_history(), section="## Evolution & decisions\n- #84 …", judge=judge)
    assert score.judge_status == "ok"
    assert score.components["llm_genuineness"] == 0.9
    assert abs(score.weights["llm_genuineness"] - 0.35) < 1e-6
    assert len(seen) == 1 and seen[0].section.startswith("## Evolution")


def test_judge_value_is_clamped() -> None:
    score = score_history(_rich_history(), judge=lambda ctx: 5.0)
    assert score.components["llm_genuineness"] == 1.0


def test_judge_failure_degrades_to_deterministic() -> None:
    """A judge that raises never fails the score — it degrades like a missing one."""

    def boom(ctx: QualityContext) -> float:
        raise RuntimeError("router down")

    score = score_history(_rich_history(), judge=boom)
    assert score.judge_status == "error"
    assert score.components["llm_genuineness"] is None
    assert score.weights["llm_genuineness"] == 0.0
    assert score.score >= 0.70  # deterministic floor still scores it genuine
    assert any("judge failed" in n for n in score.notes)


def test_grounding_flags_hallucinated_reference() -> None:
    """A section citing a PR number not in the history drives grounding below 1.0."""
    history = _rich_history()  # PRs #84/#83/#80/#79
    section = "## Evolution & decisions\n- **#84** real\n- **#999** invented"
    score = score_history(history, section=section)
    assert score.components["grounding"] == 0.5  # 1 of 2 cited numbers grounded


def test_grounding_is_one_without_section() -> None:
    score = score_history(_rich_history())
    assert score.components["grounding"] == 1.0


def test_empty_history_scores_zero() -> None:
    for hist in (None, _history([], [])):
        score = score_history(hist)
        assert score.score == 0.0
        assert score.verdict == "weak"
        assert all(v is None for v in score.components.values())


def test_diversity_rewards_distinct_kinds() -> None:
    one_kind = _history(prs=[_pr(1, "feat: a"), _pr(2, "feat: b")], commits=[])
    many_kinds = _rich_history()  # fix + feat + refactor + perf
    assert (
        score_history(many_kinds).components["diversity"]
        > score_history(one_kind).components["diversity"]
    )
