"""
Tests for `wiki_routing.config.load` — TOML parsing, defaults, overrides.
"""

from __future__ import annotations

import textwrap

import pytest

from wiki_routing.config import (
    DEFAULT_FAST_MODEL,
    DEFAULT_MAX_PER_DAY_USD,
    DEFAULT_MAX_PER_TASK_USD,
    DEFAULT_REASONING_MODEL,
    load,
)
from wiki_routing.tiers import Tier


def test_missing_file_returns_defaults(tmp_path):
    cfg = load(tmp_path / "absent.toml")
    assert cfg.max_per_day_usd == DEFAULT_MAX_PER_DAY_USD
    assert cfg.max_per_task_usd == DEFAULT_MAX_PER_TASK_USD
    assert cfg.fast.model == DEFAULT_FAST_MODEL
    assert cfg.reasoning.model == DEFAULT_REASONING_MODEL
    assert cfg.overrides == {}


def test_parses_budget(tmp_path):
    cfg_path = tmp_path / "config.toml"
    cfg_path.write_text(
        textwrap.dedent("""
        [router]
        max_per_day_usd = 25.0
        max_per_task_usd = 1.0
        """)
    )
    cfg = load(cfg_path)
    assert cfg.max_per_day_usd == 25.0
    assert cfg.max_per_task_usd == 1.0


def test_parses_overrides(tmp_path):
    cfg_path = tmp_path / "config.toml"
    cfg_path.write_text(
        textwrap.dedent("""
        [router.overrides]
        seal = "REASONING"
        query = "FAST"
        ingest = "FAST"
        """)
    )
    cfg = load(cfg_path)
    assert cfg.overrides == {
        "seal": Tier.REASONING,
        "query": Tier.FAST,
        "ingest": Tier.FAST,
    }


def test_ignores_unknown_intent_in_overrides(tmp_path):
    cfg_path = tmp_path / "config.toml"
    cfg_path.write_text(
        textwrap.dedent("""
        [router.overrides]
        seal = "REASONING"
        wrong_intent = "FAST"
        """)
    )
    cfg = load(cfg_path)
    assert "wrong_intent" not in cfg.overrides
    assert cfg.overrides["seal"] == Tier.REASONING


def test_ignores_unknown_tier_in_overrides(tmp_path):
    cfg_path = tmp_path / "config.toml"
    cfg_path.write_text(
        textwrap.dedent("""
        [router.overrides]
        seal = "QUANTUM"
        """)
    )
    cfg = load(cfg_path)
    assert "seal" not in cfg.overrides


def test_parses_provider_config(tmp_path):
    cfg_path = tmp_path / "config.toml"
    cfg_path.write_text(
        textwrap.dedent("""
        [router.providers.fast]
        backend = "openrouter"
        model = "deepseek/deepseek-v4-flash"

        [router.providers.reasoning]
        backend = "anthropic"
        model = "claude-sonnet-4-5"
        """)
    )
    cfg = load(cfg_path)
    assert cfg.fast.backend == "openrouter"
    assert cfg.fast.model == "deepseek/deepseek-v4-flash"
    assert cfg.reasoning.backend == "anthropic"


def test_malformed_toml_raises(tmp_path):
    cfg_path = tmp_path / "bad.toml"
    cfg_path.write_text("not [valid toml \n")
    with pytest.raises(Exception):  # noqa: BLE001 -- tomllib raises TOMLDecodeError
        load(cfg_path)
