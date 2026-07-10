"""Tests for the deterministic explainer AC-2 pre-screen (ADR-044)."""

from __future__ import annotations

from dataclasses import dataclass

from wiki_publishing.explainer_quality import score_explainer


@dataclass
class _Snip:
    body: str
    title: str = ""


def test_no_numbers_is_vacuously_grounded() -> None:
    s = score_explainer("A clean explainer with no stats.\n\n- a point\n", [_Snip("source text")])
    assert s.verdict == "grounded"
    assert s.components["numbers_grounded"] == 1.0


def test_number_present_in_source_is_grounded() -> None:
    s = score_explainer(
        "The model is 3B params and hits 95% accuracy.",
        [_Snip("It has 3B params, 95% on the bench.")],
    )
    assert s.verdict == "grounded"
    assert s.ungrounded_numbers == ()


def test_invented_number_is_flagged() -> None:
    s = score_explainer(
        "Used by 4000+ engineers and 5x faster.", [_Snip("It is faster than the baseline.")]
    )
    assert s.verdict == "ungrounded"
    assert "4000+" in s.ungrounded_numbers or "4000" in " ".join(s.ungrounded_numbers)
    assert any("5x" in u for u in s.ungrounded_numbers)


def test_cta_proof_value_is_exempt() -> None:
    # the social-proof count lives ONLY in the verbatim CTA — not fabrication.
    s = score_explainer(
        "Great concept.\n\nLa versión detallada (leído por 4000+ ingenieros): https://news.x",
        [_Snip("concept body, no numbers")],
        exempt_values=["leído por 4000+ ingenieros", "https://news.x"],
    )
    assert s.verdict == "grounded"


def test_structural_small_ints_not_flagged() -> None:
    s = score_explainer("I'll explain it in 1 minute, in 3 steps.", [_Snip("no numbers here")])
    assert s.verdict == "grounded"  # 1 and 3 are structural, not statistics


def test_digit_run_tolerates_formatting() -> None:
    s = score_explainer("Trained on 4,000 examples.", [_Snip("dataset of 4000 examples")])
    assert s.verdict == "grounded"


def test_structure_signal_rewards_skeleton() -> None:
    rich = "Hook — surprised?\n\n- ➡️ block one\n- ➡️ block two\n\nThe insight.\n"
    poor = "one flat line no structure"
    assert (
        score_explainer(rich, [_Snip("x")]).components["structure"]
        > score_explainer(poor, [_Snip("x")]).components["structure"]
    )


def test_score_weights_grounding_and_structure() -> None:
    s = score_explainer("flat, invented 999% gain", [_Snip("nothing")])
    # ungrounded number drags numbers_grounded to 0 → score is structure-only weighted
    assert s.components["numbers_grounded"] == 0.0
    assert 0.0 <= s.score <= 0.3
