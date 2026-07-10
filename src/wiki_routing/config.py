"""
Load ``~/.sbw/config.toml`` for the model router per
[ADR-008 model router policy](../../docs/ADR-008-language-strategy.md) +
issue #37 AC ("Per-task tier overrides in ``~/.sbw/config.toml``").

Schema::

    [router]
    max_per_day_usd  = 10.0
    max_per_task_usd = 0.50

    [router.overrides]
    # intent → tier (FAST | REASONING | VISION)
    seal  = "REASONING"
    query = "FAST"

    [router.providers.fast]
    backend = "openrouter"
    model   = "deepseek/deepseek-v4-flash"

    [router.providers.reasoning]
    backend = "anthropic"
    model   = "claude-sonnet-4-5"

A missing config file is **not** an error — the factory falls back to
sensible defaults ($10/day, deepseek for FAST, Sonnet for REASONING).
"""

from __future__ import annotations

import logging
import tomllib
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from wiki_routing.policy import Intent
from wiki_routing.tiers import Tier

_log = logging.getLogger(__name__)

DEFAULT_CONFIG_PATH = Path.home() / ".sbw" / "config.toml"


# Defaults per #37 AC ($10/day) and ADR-008 (Haiku for fast, Sonnet for reasoning).
DEFAULT_MAX_PER_DAY_USD = 10.0
DEFAULT_MAX_PER_TASK_USD = 0.50
DEFAULT_FAST_MODEL = "deepseek/deepseek-v4-flash"  # via OpenRouter (#37 directive)
DEFAULT_REASONING_MODEL = "claude-sonnet-4-5"  # via Anthropic
DEFAULT_VISION_MODEL = "claude-sonnet-4-5"  # via Anthropic


@dataclass(frozen=True)
class ProviderConfig:
    """One tier's backend wiring."""

    backend: str
    """One of: ``anthropic``, ``openrouter``, ``ollama``, ``keyword``."""

    model: str
    """Provider-specific model id."""

    extra: dict[str, Any] = field(default_factory=dict)
    """Backend-specific options (e.g. ``prices``, ``base_url``)."""


@dataclass(frozen=True)
class RouterConfig:
    """Parsed view of ``[router]`` from ``~/.sbw/config.toml``."""

    max_per_day_usd: float = DEFAULT_MAX_PER_DAY_USD
    max_per_task_usd: float = DEFAULT_MAX_PER_TASK_USD
    overrides: dict[Intent, Tier] = field(default_factory=dict)
    fast: ProviderConfig = field(
        default_factory=lambda: ProviderConfig(backend="openrouter", model=DEFAULT_FAST_MODEL)
    )
    reasoning: ProviderConfig = field(
        default_factory=lambda: ProviderConfig(backend="anthropic", model=DEFAULT_REASONING_MODEL)
    )
    vision: ProviderConfig = field(
        default_factory=lambda: ProviderConfig(backend="anthropic", model=DEFAULT_VISION_MODEL)
    )


def load(path: Path | None = None) -> RouterConfig:
    """Load and parse the config, or return defaults if the file is absent.

    Parse errors are NOT silent — a malformed TOML raises so the user
    notices instead of running with defaults that don't match what they
    intended.
    """
    target = Path(path) if path is not None else DEFAULT_CONFIG_PATH
    if not target.exists():
        _log.debug("router config absent at %s; using defaults", target)
        return RouterConfig()
    raw = tomllib.loads(target.read_text(encoding="utf-8"))
    router = raw.get("router", {})
    return RouterConfig(
        max_per_day_usd=float(router.get("max_per_day_usd", DEFAULT_MAX_PER_DAY_USD)),
        max_per_task_usd=float(router.get("max_per_task_usd", DEFAULT_MAX_PER_TASK_USD)),
        overrides=_parse_overrides(router.get("overrides", {})),
        fast=_parse_provider(
            router.get("providers", {}).get("fast"),
            default=ProviderConfig(backend="openrouter", model=DEFAULT_FAST_MODEL),
        ),
        reasoning=_parse_provider(
            router.get("providers", {}).get("reasoning"),
            default=ProviderConfig(backend="anthropic", model=DEFAULT_REASONING_MODEL),
        ),
        vision=_parse_provider(
            router.get("providers", {}).get("vision"),
            default=ProviderConfig(backend="anthropic", model=DEFAULT_VISION_MODEL),
        ),
    )


def _parse_overrides(raw: dict[str, Any]) -> dict[Intent, Tier]:
    """Map intent → Tier, ignoring unknown intents with a warning."""
    out: dict[Intent, Tier] = {}
    valid_intents = {"seal", "ingest", "query", "lint", "vision"}
    for intent_name, tier_name in raw.items():
        if intent_name not in valid_intents:
            _log.warning("router config: unknown intent %r in overrides; ignoring", intent_name)
            continue
        try:
            tier = Tier(str(tier_name).lower())
        except ValueError:
            _log.warning(
                "router config: unknown tier %r for intent %s; ignoring",
                tier_name,
                intent_name,
            )
            continue
        out[intent_name] = tier  # type: ignore[assignment] -- validated above
    return out


def _parse_provider(raw: dict[str, Any] | None, *, default: ProviderConfig) -> ProviderConfig:
    if raw is None:
        return default
    backend = str(raw.get("backend", default.backend))
    model = str(raw.get("model", default.model))
    extra = {k: v for k, v in raw.items() if k not in {"backend", "model"}}
    return ProviderConfig(backend=backend, model=model, extra=extra)


__all__ = [
    "DEFAULT_CONFIG_PATH",
    "DEFAULT_FAST_MODEL",
    "DEFAULT_MAX_PER_DAY_USD",
    "DEFAULT_MAX_PER_TASK_USD",
    "DEFAULT_REASONING_MODEL",
    "DEFAULT_VISION_MODEL",
    "ProviderConfig",
    "RouterConfig",
    "load",
]
