"""
CLI handler for ``sbw router {status,dry-run}`` per issue #37 AC.
"""

from __future__ import annotations

import argparse
import json
import sys

from wiki_routing.telemetry import RouterTelemetry


def run_router_cli(args: argparse.Namespace) -> int:
    action = getattr(args, "router_action", None)
    if action == "status":
        return _cmd_status(as_json=getattr(args, "json", False))
    if action == "dry-run":
        return _cmd_dry_run(
            intent=args.intent,
            caller_priority=args.caller_priority,
        )
    print(f"Unknown router action: {action!r}", file=sys.stderr)
    return 1


def _cmd_status(*, as_json: bool = False) -> int:
    tel = RouterTelemetry()
    try:
        rolling = tel.rolling_cost_usd(window_hours=24)
        tiers = tel.tier_distribution(window_hours=24)
        total = tel.total_calls()
    finally:
        tel.close()

    if as_json:
        print(
            json.dumps(
                {
                    "total_calls_lifetime": total,
                    "rolling_24h_cost_usd": round(rolling, 4),
                    "tiers_24h": [
                        {
                            "tier": s.tier,
                            "calls": s.calls,
                            "cost_usd": round(s.cost_usd, 4),
                            "success_rate": round(s.success_rate, 3),
                        }
                        for s in tiers
                    ],
                }
            )
        )
        return 0

    print(f"router: {total} lifetime calls, ${rolling:.4f} spent in last 24h")
    if not tiers:
        print("  (no calls in the last 24h)")
        return 0
    print(f"  {'TIER':<10} {'CALLS':>6}  {'COST':>10}  SUCCESS")
    for s in tiers:
        print(f"  {s.tier:<10} {s.calls:>6}  ${s.cost_usd:>8.4f}  {s.success_rate * 100:5.1f}%")
    return 0


def _cmd_dry_run(*, intent: str, caller_priority: str) -> int:
    """Show which tier the policy would pick for a given task — useful
    for verifying an override in ``~/.sbw/config.toml`` landed."""
    from wiki_routing.config import load
    from wiki_routing.policy import RoutingPolicy, TaskDescriptor

    cfg = load()
    policy = RoutingPolicy(overrides=cfg.overrides)
    task = TaskDescriptor(
        intent=intent,  # type: ignore[arg-type]
        estimated_input_tokens=0,
        caller_priority=caller_priority,  # type: ignore[arg-type]
    )
    tier = policy.route(task)
    print(f"intent={intent} priority={caller_priority} → tier={tier}")
    if intent in cfg.overrides:
        print("  (override applied from config.toml)")
    return 0


__all__ = ["run_router_cli"]
