"""Tests for verdict parsing (SPEC-010 FR-1, wiki_lessons/verdict.py).

Hermetic: parsers are pure functions over strings captured from Repo2RLEnv's
``/logs/verifier/`` output shapes.
"""

from __future__ import annotations

import json

from wiki_lessons.verdict import (
    llm_verdict,
    parse_reward_json,
    parse_reward_txt,
)

PR_RUNTIME_JSON = json.dumps(
    {
        "reward": 0.9,
        "resolved": True,
        "f2p_total": 4,
        "f2p_passed": 4,
        "f2p_rate": 1.0,
        "p2p_total": 10,
        "p2p_passed": 9,
        "p2p_rate": 0.9,
        "regressions": 1,
        "parse_status": "ok",
        "runner": "pytest",
        "exit_code": 0,
    }
)

PR_DIFF_JSON = json.dumps(
    {
        "reward": 0.62,
        "components": {
            "format_valid": 1.0,
            "size_sanity": 0.8,
            "file_targeting": 0.5,
            "region_overlap": 0.4,
            "similarity": 0.55,
            "llm_judge": None,
        },
        "weights": {"similarity": 0.3},
        "judge_status": "no_api_key",
        "capped": False,
    }
)


def test_pr_runtime_shape_is_extracted_test_execution() -> None:
    verdict = parse_reward_json(PR_RUNTIME_JSON)
    assert verdict is not None
    assert verdict.source == "verifier"
    assert verdict.provenance == "EXTRACTED"
    assert verdict.kind == "test_execution"
    assert verdict.success is True
    assert verdict.reward == 0.9
    assert verdict.component("f2p_rate") == 1.0
    assert verdict.component("p2p_rate") == 0.9


def test_pr_runtime_reward_derived_from_rates_when_missing() -> None:
    data = {"f2p_rate": 0.5, "p2p_rate": 0.8}
    verdict = parse_reward_json(json.dumps(data))
    assert verdict is not None
    assert verdict.reward == 0.4
    assert verdict.success is False  # 0.4 < default threshold 0.7


def test_pr_diff_shape_keeps_numeric_components_and_skips_nulls() -> None:
    verdict = parse_reward_json(PR_DIFF_JSON)
    assert verdict is not None
    assert verdict.kind == "diff_similarity"
    assert verdict.success is False
    names = [name for name, _ in verdict.components]
    assert "similarity" in names
    assert "llm_judge" not in names  # null skipped


def test_minimal_reward_shape_is_binary() -> None:
    verdict = parse_reward_json('{"reward": 1.0}')
    assert verdict is not None
    assert verdict.kind == "binary"
    assert verdict.success is True


def test_reward_is_clamped_to_unit_interval() -> None:
    verdict = parse_reward_json('{"reward": 3.5}')
    assert verdict is not None
    assert verdict.reward == 1.0


def test_malformed_inputs_degrade_to_none() -> None:
    assert parse_reward_json("not json") is None
    assert parse_reward_json("[1, 2]") is None
    assert parse_reward_json('{"unrelated": true}') is None
    assert parse_reward_txt("nan-ish garbage") is None


def test_reward_txt_parses_bare_float() -> None:
    verdict = parse_reward_txt(" 0.75 \n")
    assert verdict is not None
    assert verdict.reward == 0.75
    assert verdict.success is True
    assert verdict.provenance == "EXTRACTED"


def test_llm_verdict_is_inferred() -> None:
    verdict = llm_verdict(success=True, confidence=0.9)
    assert verdict.provenance == "INFERRED"
    assert verdict.kind == "llm_judgment"
    assert verdict.reward == 0.9
