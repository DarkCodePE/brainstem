"""Verdict parsing for lesson memory (SPEC-010 FR-1).

A :class:`Verdict` is the judgment over one agent trajectory. Two sources:

- ``verifier`` — Repo2RLEnv's deterministic graders (``reward.json`` /
  ``reward.txt`` written to ``/logs/verifier/``). Provenance ``EXTRACTED``.
- ``llm`` — an LLM's judgment of a trajectory with no executable verifier.
  Provenance ``INFERRED``.

Parsers degrade to ``None`` on malformed input — a missing or corrupt verdict
must never break the caller's pipeline (ADR-033 D5).
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field

#: Reward at or above which a run without an explicit ``resolved`` flag
#: counts as a success.
DEFAULT_SUCCESS_THRESHOLD: float = 0.7

PROVENANCE_EXTRACTED = "EXTRACTED"
PROVENANCE_INFERRED = "INFERRED"


def _clamp01(value: float) -> float:
    return min(max(value, 0.0), 1.0)


@dataclass(frozen=True, slots=True)
class Verdict:
    """Judgment over one trajectory.

    ``components`` is an ordered tuple of (name, value) pairs so the
    dataclass stays hashable; insertion order follows the source JSON.
    """

    source: str
    """``"verifier"`` (deterministic grader) or ``"llm"`` (judgment)."""

    reward: float
    """Scalar outcome in [0, 1]."""

    success: bool

    kind: str
    """``test_execution`` | ``diff_similarity`` | ``binary`` | ``llm_judgment``."""

    components: tuple[tuple[str, float], ...] = field(default=())

    @property
    def provenance(self) -> str:
        return PROVENANCE_EXTRACTED if self.source == "verifier" else PROVENANCE_INFERRED

    def component(self, name: str) -> float | None:
        for key, value in self.components:
            if key == name:
                return value
        return None


def _numeric_components(
    data: dict[str, object], keys: tuple[str, ...]
) -> tuple[tuple[str, float], ...]:
    out: list[tuple[str, float]] = []
    for key in keys:
        value = data.get(key)
        if isinstance(value, bool):
            out.append((key, 1.0 if value else 0.0))
        elif isinstance(value, (int, float)):
            out.append((key, float(value)))
    return tuple(out)


def parse_reward_json(
    text: str,
    *,
    success_threshold: float = DEFAULT_SUCCESS_THRESHOLD,
) -> Verdict | None:
    """Parse a Repo2RLEnv ``reward.json`` into a verifier Verdict.

    Recognizes the ``pr_runtime``/``commit_runtime`` shape (``resolved``,
    ``f2p_rate``, ``p2p_rate``), the ``pr_diff`` shape (``reward`` +
    ``components`` dict), and a minimal ``{"reward": x}`` shape. Returns
    ``None`` on anything else.
    """
    try:
        data = json.loads(text)
    except (json.JSONDecodeError, TypeError):
        return None
    if not isinstance(data, dict):
        return None

    if "f2p_rate" in data or "resolved" in data:
        f2p = data.get("f2p_rate")
        p2p = data.get("p2p_rate")
        reward = data.get("reward")
        if not isinstance(reward, (int, float)) or isinstance(reward, bool):
            if isinstance(f2p, (int, float)) and isinstance(p2p, (int, float)):
                reward = float(f2p) * float(p2p)
            else:
                return None
        reward = _clamp01(float(reward))
        resolved = data.get("resolved")
        success = bool(resolved) if isinstance(resolved, bool) else reward >= success_threshold
        components = _numeric_components(
            data,
            (
                "f2p_rate",
                "p2p_rate",
                "f2p_passed",
                "f2p_total",
                "p2p_passed",
                "p2p_total",
                "regressions",
            ),
        )
        return Verdict(
            source="verifier",
            reward=reward,
            success=success,
            kind="test_execution",
            components=components,
        )

    raw_reward = data.get("reward")
    if not isinstance(raw_reward, (int, float)) or isinstance(raw_reward, bool):
        return None
    reward = _clamp01(float(raw_reward))

    raw_components = data.get("components")
    if isinstance(raw_components, dict):
        components = tuple(
            (str(key), float(value))
            for key, value in raw_components.items()
            if isinstance(value, (int, float)) and not isinstance(value, bool)
        )
        return Verdict(
            source="verifier",
            reward=reward,
            success=reward >= success_threshold,
            kind="diff_similarity",
            components=components,
        )

    return Verdict(
        source="verifier",
        reward=reward,
        success=reward >= success_threshold,
        kind="binary",
    )


def parse_reward_txt(
    text: str,
    *,
    success_threshold: float = DEFAULT_SUCCESS_THRESHOLD,
) -> Verdict | None:
    """Parse a bare ``reward.txt`` (single float) into a verifier Verdict."""
    try:
        reward = float(text.strip())
    except (ValueError, AttributeError):
        return None
    reward = _clamp01(reward)
    return Verdict(
        source="verifier",
        reward=reward,
        success=reward >= success_threshold,
        kind="binary",
    )


def llm_verdict(*, success: bool, confidence: float) -> Verdict:
    """Construct an INFERRED verdict from an LLM judgment."""
    return Verdict(
        source="llm",
        reward=_clamp01(confidence),
        success=success,
        kind="llm_judgment",
    )
