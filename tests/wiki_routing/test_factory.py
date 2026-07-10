"""
Tests for `wiki_routing.factory.default_router` — chain composition,
key-absent skip, keyword-only is always terminal.
"""

from __future__ import annotations

from wiki_routing.config import ProviderConfig, RouterConfig
from wiki_routing.factory import default_router
from wiki_routing.providers.keyword_only import KeywordOnlyBackend
from wiki_routing.tiers import Tier

# A FAST tier pointed at a LOCAL OpenAI-compatible server (Gemma via the
# headroom proxy): backend stays "openrouter" (OpenAI-compat client) but with a
# custom base_url + an api_key_env naming the env var that holds the local key.
_LOCAL_GEMMA_FAST = ProviderConfig(
    backend="openrouter",
    model="gemma-4-12B-it-qat",
    extra={"base_url": "http://127.0.0.1:8787/v1", "api_key_env": "GEMMA_API_KEY"},
)


def test_keyword_only_is_always_terminal_in_every_tier():
    """Even with no API keys, every tier ends in `KeywordOnlyBackend`
    so calls never fail with "no backend"."""
    router = default_router(env={})
    for tier in (Tier.FAST, Tier.REASONING, Tier.VISION):
        chain = router._tiers[tier].chain  # noqa: SLF001 -- white-box test
        terminal = chain.backends[-1]
        assert isinstance(terminal, KeywordOnlyBackend), (
            f"tier {tier} terminal is {type(terminal).__name__}, not KeywordOnlyBackend"
        )


def test_no_keys_still_builds_router():
    router = default_router(env={})
    # FAST: keyword only
    fast = router._tiers[Tier.FAST].chain.backends  # noqa: SLF001
    assert len(fast) >= 1
    # Last is always keyword
    assert isinstance(fast[-1], KeywordOnlyBackend)


def test_openrouter_key_wires_deepseek_into_fast():
    router = default_router(env={"OPENROUTER_API_KEY": "fake-key"})
    fast = router._tiers[Tier.FAST].chain.backends  # noqa: SLF001
    labels = [b.label for b in fast]
    # OpenRouter primary
    assert any("openrouter" in lbl for lbl in labels)
    # Keyword terminal
    assert any("keyword-only" in lbl for lbl in labels)


def test_anthropic_key_wires_into_reasoning():
    router = default_router(env={"ANTHROPIC_API_KEY": "fake", "OPENROUTER_API_KEY": "fake"})
    reasoning = router._tiers[Tier.REASONING].chain.backends  # noqa: SLF001
    labels = [b.label for b in reasoning]
    assert any("anthropic" in lbl for lbl in labels)
    # OpenRouter (deepseek) sits between anthropic and keyword
    assert any("openrouter" in lbl for lbl in labels)


def test_custom_config_overrides_default(tmp_path):
    cfg = RouterConfig(max_per_day_usd=42.0, max_per_task_usd=0.10)
    router = default_router(config=cfg, env={})
    assert router.budget is not None
    assert router.budget.max_per_day_usd == 42.0
    assert router.budget.max_per_task_usd == 0.10


def test_local_base_url_and_api_key_env_wired_into_fast():
    """A FAST openrouter primary with a custom base_url + api_key_env points the
    OpenAI-compatible client at the local server using the named env key."""
    cfg = RouterConfig(fast=_LOCAL_GEMMA_FAST)
    router = default_router(config=cfg, env={"GEMMA_API_KEY": "sk-local"})
    fast = router._tiers[Tier.FAST].chain.backends  # noqa: SLF001
    primary = fast[0]
    assert primary.label == "openrouter:gemma-4-12B-it-qat"
    assert primary._base_url == "http://127.0.0.1:8787/v1"  # noqa: SLF001


def test_local_fast_primary_keeps_cloud_deepseek_fallback():
    """A LOCAL FAST primary degrades to cloud deepseek (when the OpenRouter key
    is present) before ollama/keyword — local synthesis keeps a cloud net."""
    cfg = RouterConfig(fast=_LOCAL_GEMMA_FAST)
    router = default_router(
        config=cfg, env={"GEMMA_API_KEY": "sk-local", "OPENROUTER_API_KEY": "fake"}
    )
    labels = [b.label for b in router._tiers[Tier.FAST].chain.backends]  # noqa: SLF001
    assert labels[0] == "openrouter:gemma-4-12B-it-qat"  # local primary
    assert "openrouter:deepseek/deepseek-v4-flash" in labels  # cloud fallback
    assert any("keyword-only" in lbl for lbl in labels)  # terminal


def test_local_fast_primary_cascades_deepseek_then_qwen():
    """ADR-041 hybrid cascade: a LOCAL FAST primary degrades through TWO cloud
    nets in order — deepseek (cheap, 1M ctx) then qwen3.6-flash (higher quality)
    — before ollama/keyword. Both share OPENROUTER_API_KEY."""
    cfg = RouterConfig(fast=_LOCAL_GEMMA_FAST)
    router = default_router(
        config=cfg, env={"GEMMA_API_KEY": "sk-local", "OPENROUTER_API_KEY": "fake"}
    )
    labels = [b.label for b in router._tiers[Tier.FAST].chain.backends]  # noqa: SLF001
    assert labels[0] == "openrouter:gemma-4-12B-it-qat"  # local primary
    ds = labels.index("openrouter:deepseek/deepseek-v4-flash")
    qw = labels.index("openrouter:qwen/qwen3.6-flash")
    assert ds < qw, f"deepseek must precede qwen in the cascade: {labels}"
    assert isinstance(
        router._tiers[Tier.FAST].chain.backends[-1],
        KeywordOnlyBackend,  # noqa: SLF001
    )


def test_qwen_cascade_only_behind_local_primary():
    """The qwen3.6-flash cloud net is inserted ONLY behind a LOCAL primary; a
    plain cloud-OpenRouter FAST tier does NOT get it. (Explicit cloud config so
    the test is hermetic — it must not read a real ~/.sbw/config.toml.)"""
    cloud_fast = ProviderConfig(backend="openrouter", model="deepseek/deepseek-v4-flash")
    router = default_router(
        config=RouterConfig(fast=cloud_fast), env={"OPENROUTER_API_KEY": "fake"}
    )
    labels = [b.label for b in router._tiers[Tier.FAST].chain.backends]  # noqa: SLF001
    assert not any("qwen/qwen3.6-flash" in lbl for lbl in labels), labels


def test_api_key_env_unset_skips_local_primary():
    """When the named api_key_env is unset, the local primary is skipped (no
    crash) and the chain still terminates safely."""
    cfg = RouterConfig(fast=_LOCAL_GEMMA_FAST)
    router = default_router(config=cfg, env={})  # GEMMA_API_KEY absent
    labels = [b.label for b in router._tiers[Tier.FAST].chain.backends]  # noqa: SLF001
    assert not any("gemma" in lbl for lbl in labels)
    assert isinstance(router._tiers[Tier.FAST].chain.backends[-1], KeywordOnlyBackend)  # noqa: SLF001


def test_seal_override_to_fast_routes_to_local_while_draft_stays_reasoning():
    """The activation plan: override seal→FAST (local Gemma) while draft stays
    REASONING (cloud) — they share REASONING by default, so this is how synthesis
    goes local without dragging posts along."""
    from wiki_routing.policy import TaskDescriptor

    cfg = RouterConfig(fast=_LOCAL_GEMMA_FAST, overrides={"seal": Tier.FAST})
    router = default_router(config=cfg, env={"GEMMA_API_KEY": "sk-local"})
    seal = TaskDescriptor(intent="seal", estimated_input_tokens=4000)
    draft = TaskDescriptor(intent="draft", estimated_input_tokens=4000)
    assert router.tier_for(seal) == Tier.FAST  # synthesis → local Gemma
    assert router.tier_for(draft) == Tier.REASONING  # posts → cloud, untouched


def test_policy_overrides_applied(tmp_path):
    """Overrides from config map intent → tier."""
    from wiki_routing.policy import TaskDescriptor

    cfg = RouterConfig(overrides={"ingest": Tier.REASONING})
    router = default_router(config=cfg, env={})
    task = TaskDescriptor(intent="ingest", estimated_input_tokens=100)
    assert router.tier_for(task) == Tier.REASONING
