"""
Factory for the production ``ModelRouter`` per issue #37 / PRD-009 / ADR-008.

`default_router()` reads ``~/.sbw/config.toml`` (or accepts an explicit
``RouterConfig``) and wires:

- **FAST** → OpenRouter (default ``deepseek/deepseek-v4-flash``) →
  Ollama (``qwen2.5:7b``) → KeywordOnly. When the FAST primary is a LOCAL
  OpenAI-compatible server (Gemma), a cloud cascade is inserted behind it:
  ``deepseek/deepseek-v4-flash`` (cheap, 1M ctx) → ``qwen/qwen3.6-flash``
  (higher-quality, 1M ctx) before Ollama/KeywordOnly. See ADR-040 / ADR-041.
- **REASONING** → Anthropic (``claude-sonnet-4-5``) →
  OpenRouter (``deepseek-v4-flash``) → Ollama (``qwen2.5:14b``) → KeywordOnly
- **VISION** → Anthropic (``claude-sonnet-4-5`` vision) → KeywordOnly
  (no Ollama vision fallback in M3; the vision path degrades to keyword
  rather than picking a wrong-modality local model)

Missing API keys downgrade the affected tier to the next link — e.g.
no ``ANTHROPIC_API_KEY`` → REASONING starts with OpenRouter. The
``KeywordOnlyBackend`` is always the terminal link so a fully-offline
deployment still returns something rather than crashing.
"""

from __future__ import annotations

import logging
import os

from wiki_routing.config import ProviderConfig, RouterConfig
from wiki_routing.config import load as load_config
from wiki_routing.cost_ceiling import CostBudget
from wiki_routing.fallback import FallbackChain
from wiki_routing.policy import RoutingPolicy
from wiki_routing.providers.anthropic import AnthropicProvider
from wiki_routing.providers.keyword_only import KeywordOnlyBackend
from wiki_routing.providers.ollama import OllamaProvider
from wiki_routing.providers.openrouter import OpenRouterProvider
from wiki_routing.router import ModelBackend, ModelRouter
from wiki_routing.telemetry import RouterTelemetry
from wiki_routing.tiers import Tier

_log = logging.getLogger(__name__)


DEFAULT_OLLAMA_FAST_MODEL = "qwen2.5:7b"
DEFAULT_OLLAMA_REASONING_MODEL = "qwen2.5:14b"

# Cloud cascade behind a LOCAL FAST primary (Gemma). deepseek-v4-flash is the
# cheap/large-ctx first net (see _build_fast_chain); qwen3.6-flash is a
# higher-quality second cloud net before degrading to ollama/keyword. Chosen
# over qwen3.6-27b (17x output cost) per ADR-041. Both use OPENROUTER_API_KEY.
DEFAULT_QWEN_FALLBACK_MODEL = "qwen/qwen3.6-flash"


def default_router(
    *,
    config: RouterConfig | None = None,
    env: dict[str, str] | None = None,
    telemetry: RouterTelemetry | None = None,
) -> ModelRouter:
    """Build a production-shaped router.

    Parameters
    ----------
    config:
        Optional pre-loaded `RouterConfig`. When omitted, loads from
        ``~/.sbw/config.toml`` (and uses defaults if the file is absent).
    env:
        Override the environment dict (defaults to ``os.environ``).
        Tests inject a fixed mapping; production leaves this ``None``.
    """
    cfg = config if config is not None else load_config()
    env_map = env if env is not None else dict(os.environ)

    policy = RoutingPolicy(overrides=cfg.overrides)
    budget = CostBudget(
        max_per_task_usd=cfg.max_per_task_usd,
        max_per_day_usd=cfg.max_per_day_usd,
    )

    providers: dict[Tier, FallbackChain[ModelBackend]] = {
        Tier.FAST: _build_fast_chain(cfg.fast, env_map),
        Tier.REASONING: _build_reasoning_chain(cfg.reasoning, env_map),
        Tier.VISION: _build_vision_chain(cfg.vision, env_map),
    }

    tel = telemetry if telemetry is not None else RouterTelemetry()
    return ModelRouter(policy=policy, providers=providers, budget=budget, telemetry=tel)


# ----------------------------------------------------------------------- #
# Per-tier chain builders                                                 #
# ----------------------------------------------------------------------- #


def _is_local_openrouter(cfg: ProviderConfig) -> bool:
    """True iff an ``openrouter`` backend is pointed at a LOCAL OpenAI-compatible
    server (e.g. Gemma via the headroom proxy / llama-server) through a custom
    ``base_url``. Used to keep a cloud fallback behind a local primary."""
    base_url = str(cfg.extra.get("base_url", "")).lower()
    return cfg.backend == "openrouter" and ("127.0.0.1" in base_url or "localhost" in base_url)


def _build_fast_chain(primary: ProviderConfig, env: dict[str, str]) -> FallbackChain[ModelBackend]:
    chain: list[ModelBackend] = []
    primary_backend = _make_backend(primary, env)
    if primary_backend is not None:
        chain.append(primary_backend)
    # When the FAST primary is a LOCAL OpenAI-compatible server (e.g. Gemma via
    # the headroom proxy), keep a cloud deepseek link so a down / over-budget
    # local server degrades to cloud quality before falling to ollama/keyword.
    if _is_local_openrouter(primary):
        deepseek = _try_openrouter(model="deepseek/deepseek-v4-flash", env=env)
        if deepseek is not None:
            chain.append(deepseek)
        # Second cloud net: a higher-quality model for the rare case where the
        # local server AND deepseek both fail / fall short, before ollama/keyword.
        qwen = _try_openrouter(model=DEFAULT_QWEN_FALLBACK_MODEL, env=env)
        if qwen is not None:
            chain.append(qwen)
    ollama = _try_ollama(model=DEFAULT_OLLAMA_FAST_MODEL, env=env)
    if ollama is not None:
        chain.append(ollama)
    chain.append(KeywordOnlyBackend())
    return FallbackChain[ModelBackend](chain)


def _build_reasoning_chain(
    primary: ProviderConfig, env: dict[str, str]
) -> FallbackChain[ModelBackend]:
    chain: list[ModelBackend] = []
    primary_backend = _make_backend(primary, env)
    if primary_backend is not None:
        chain.append(primary_backend)
    # Second link: deepseek via OpenRouter — cheap-but-strong fallback. Added
    # when the primary is NOT cloud OpenRouter — i.e. for anthropic primaries
    # AND for a LOCAL openrouter primary (Gemma via proxy), so local synthesis
    # still has a cloud safety net before ollama/keyword.
    if primary.backend != "openrouter" or _is_local_openrouter(primary):
        deepseek = _try_openrouter(model="deepseek/deepseek-v4-flash", env=env)
        if deepseek is not None:
            chain.append(deepseek)
    ollama = _try_ollama(model=DEFAULT_OLLAMA_REASONING_MODEL, env=env)
    if ollama is not None:
        chain.append(ollama)
    chain.append(KeywordOnlyBackend())
    return FallbackChain[ModelBackend](chain)


def _build_vision_chain(
    primary: ProviderConfig, env: dict[str, str]
) -> FallbackChain[ModelBackend]:
    chain: list[ModelBackend] = []
    primary_backend = _make_backend(primary, env)
    if primary_backend is not None:
        chain.append(primary_backend)
    chain.append(KeywordOnlyBackend())
    return FallbackChain[ModelBackend](chain)


# ----------------------------------------------------------------------- #
# Backend constructors (skip on missing key)                              #
# ----------------------------------------------------------------------- #


def _make_backend(cfg: ProviderConfig, env: dict[str, str]) -> ModelBackend | None:
    if cfg.backend == "anthropic":
        return _try_anthropic(model=cfg.model, env=env)
    if cfg.backend == "openrouter":
        # ``extra`` may carry a custom ``base_url`` (a local OpenAI-compatible
        # server such as Gemma via the headroom proxy) and an ``api_key_env``
        # naming the env var that holds its key (secrets stay in the env, never
        # in config.toml). Absent both, this is unchanged cloud OpenRouter.
        return _try_openrouter(
            model=cfg.model,
            env=env,
            base_url=cfg.extra.get("base_url"),
            api_key_env=str(cfg.extra.get("api_key_env", "OPENROUTER_API_KEY")),
        )
    if cfg.backend == "ollama":
        return _try_ollama(model=cfg.model, env=env)
    if cfg.backend == "keyword":
        return KeywordOnlyBackend()
    _log.warning("router factory: unknown backend %r; skipping", cfg.backend)
    return None


def _try_anthropic(*, model: str, env: dict[str, str]) -> AnthropicProvider | None:
    key = env.get("ANTHROPIC_API_KEY")
    if not key:
        _log.info("ANTHROPIC_API_KEY unset; skipping Anthropic backend (model=%s)", model)
        return None
    try:
        return AnthropicProvider(api_key=key, model=model)
    except Exception as exc:  # noqa: BLE001
        _log.warning("AnthropicProvider construction failed: %s", exc)
        return None


def _try_openrouter(
    *,
    model: str,
    env: dict[str, str],
    base_url: str | None = None,
    api_key_env: str = "OPENROUTER_API_KEY",
) -> OpenRouterProvider | None:
    key = env.get(api_key_env)
    if not key:
        _log.info("%s unset; skipping OpenRouter backend (model=%s)", api_key_env, model)
        return None
    try:
        if base_url:
            return OpenRouterProvider(api_key=key, model=model, base_url=base_url)
        return OpenRouterProvider(api_key=key, model=model)
    except Exception as exc:  # noqa: BLE001
        _log.warning("OpenRouterProvider construction failed: %s", exc)
        return None


def _try_ollama(*, model: str, env: dict[str, str]) -> OllamaProvider | None:
    base_url = env.get("OLLAMA_BASE_URL", "http://localhost:11434")
    try:
        return OllamaProvider(model=model, base_url=base_url)
    except Exception as exc:  # noqa: BLE001
        # Ollama running is an optional convenience; the keyword fallback
        # picks up the slack if it isn't available.
        _log.info("OllamaProvider construction failed (likely not running): %s", exc)
        return None


__all__ = ["default_router"]
