"""
``sbw doctor`` — local environment health check.

Print a one-screen status of the components the wiki harness depends on so
contributors can answer "is my install healthy?" without piecing it together
from logs. Tracks the M1 sprint 1 acceptance criterion: every developer can
run a single command and see green/yellow/red for each subsystem.

The check is read-only. It never mutates state, never touches `knowledge-base/`,
and never makes paid network calls. LLM API keys are only *probed* (HEAD
request equivalent) when ``--probe-network`` is set explicitly.

Per [PRD-008 model routing](../../docs/PRD-008-model-routing.md) and
[ADR-013](../../docs/ADR-013-model-router-policy.md), M3 will extend this
into a cost+latency dashboard surface; today it is intentionally a
skeleton.
"""

from __future__ import annotations

import argparse
import importlib.metadata
import json
import os
import platform
import shutil
import sys
from collections.abc import Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Literal

Status = Literal["ok", "warn", "fail", "skip"]

REPO_ROOT = Path(__file__).resolve().parents[2]
KB_ROOT = REPO_ROOT / "knowledge-base"


@dataclass
class Check:
    name: str
    status: Status
    detail: str


# --------------------------------------------------------------------------- #
# Individual checks                                                           #
# --------------------------------------------------------------------------- #


def check_python_version() -> Check:
    major, minor = sys.version_info[:2]
    if (major, minor) < (3, 11):
        return Check("python", "fail", f"need ≥3.11 (running {major}.{minor})")
    return Check(
        "python", "ok", f"{major}.{minor}.{sys.version_info.micro} on {platform.platform()}"
    )


def check_required_packages() -> Check:
    required = ("deepagents", "langgraph", "langchain", "fastembed", "mcp", "watchdog")
    missing: list[str] = []
    versions: list[str] = []
    for pkg in required:
        try:
            versions.append(f"{pkg}={importlib.metadata.version(pkg)}")
        except importlib.metadata.PackageNotFoundError:
            missing.append(pkg)
    if missing:
        return Check("packages", "fail", f"missing: {', '.join(missing)}")
    return Check("packages", "ok", ", ".join(versions))


def check_wiki_root() -> Check:
    if not KB_ROOT.exists():
        return Check("knowledge-base/", "fail", f"missing: {KB_ROOT}")
    raw = KB_ROOT / "raw"
    wiki = KB_ROOT / "wiki"
    if not raw.exists() or not wiki.exists():
        return Check("knowledge-base/", "warn", "missing raw/ or wiki/ subdir; run `sbw init`")
    page_count = sum(1 for _ in wiki.rglob("*.md"))
    raw_count = sum(1 for _ in raw.rglob("*") if _.is_file())
    return Check("knowledge-base/", "ok", f"{page_count} wiki pages, {raw_count} raw items")


def check_data_stores() -> Check:
    stores = [
        ("session-db", REPO_ROOT / ".wiki-agent.db"),
        ("ingest-db", REPO_ROOT / "wiki_ingest.db"),
        ("ruvector", REPO_ROOT / "ruvector.db"),
    ]
    rows: list[str] = []
    for label, path in stores:
        if not path.exists():
            rows.append(f"{label}=absent (rebuilds on demand)")
            continue
        size_mb = path.stat().st_size / 1_048_576
        rows.append(f"{label}={size_mb:.1f}MB")
    return Check("data-stores", "ok", "; ".join(rows))


def check_mcp_config() -> Check:
    cfg = REPO_ROOT / ".mcp.json"
    if not cfg.exists():
        return Check(
            "mcp", "warn", ".mcp.json missing — Claude Code won't discover the wiki engine"
        )
    try:
        data = json.loads(cfg.read_text())
    except json.JSONDecodeError as e:
        return Check("mcp", "fail", f"invalid JSON: {e}")
    servers = list(data.get("mcpServers", {}).keys())
    if not servers:
        return Check("mcp", "warn", "no mcpServers defined")
    return Check("mcp", "ok", f"servers: {', '.join(servers)}")


def check_llm_keys(probe_network: bool) -> Check:
    keys = {
        "ANTHROPIC_API_KEY": os.environ.get("ANTHROPIC_API_KEY"),
        "OPENROUTER_API_KEY": os.environ.get("OPENROUTER_API_KEY"),
        "OPENAI_API_KEY": os.environ.get("OPENAI_API_KEY"),
    }
    present = [name for name, val in keys.items() if val]
    if not present:
        return Check("llm-keys", "warn", "no LLM key in env; only --offline workflows will work")
    detail = "present: " + ", ".join(name.split("_")[0].lower() for name in present)
    if probe_network:
        # Deliberately not making a real network call here — would belong in
        # an opt-in connectivity check separate from this skeleton.
        detail += " (probe-network: not yet implemented)"
    return Check("llm-keys", "ok", detail)


def check_binaries() -> Check:
    needed = ("git", "python3")
    optional = ("rg", "fd", "obsidian")
    missing = [b for b in needed if shutil.which(b) is None]
    if missing:
        return Check("binaries", "fail", f"missing required: {', '.join(missing)}")
    present_opt = [b for b in optional if shutil.which(b) is not None]
    return Check(
        "binaries", "ok", f"required ok; optional present: {', '.join(present_opt) or 'none'}"
    )


def check_systemd_unit() -> Check:
    """ADR-006 systemd user unit. Skip on non-Linux."""
    if platform.system() != "Linux":
        return Check("systemd:wiki-ingest", "skip", f"platform={platform.system()}")
    rc = os.system("systemctl --user is-active wiki-ingest.service >/dev/null 2>&1") >> 8  # noqa: S605
    if rc == 0:
        return Check("systemd:wiki-ingest", "ok", "active (running)")
    if rc == 3:
        return Check(
            "systemd:wiki-ingest", "warn", "inactive — daemon not running (issue #18 deploys it)"
        )
    return Check("systemd:wiki-ingest", "warn", f"unknown state (rc={rc})")


def check_autofetch_timer() -> Check:
    """sbw-auto-fetch.timer (issue #38). Skip on non-Linux."""
    if platform.system() != "Linux":
        return Check("systemd:auto-fetch", "skip", f"platform={platform.system()}")
    rc = os.system("systemctl --user is-active sbw-auto-fetch.timer >/dev/null 2>&1") >> 8  # noqa: S605
    if rc == 0:
        return Check("systemd:auto-fetch", "ok", "timer active (20-min cadence)")
    if rc == 3:
        return Check(
            "systemd:auto-fetch",
            "warn",
            "timer inactive — run scripts/install-auto-fetch.sh to enable",
        )
    return Check("systemd:auto-fetch", "warn", f"unknown state (rc={rc})")


def check_router_budget() -> Check:
    """Show 24h rolling cost + tier distribution per #37 AC."""
    try:
        from wiki_routing.config import load as load_config
        from wiki_routing.telemetry import RouterTelemetry
    except ImportError:
        return Check("router:budget", "skip", "wiki_routing not importable")
    try:
        cfg = load_config()
        tel = RouterTelemetry()
        try:
            rolling = tel.rolling_cost_usd(window_hours=24)
            tiers = tel.tier_distribution(window_hours=24)
        finally:
            tel.close()
    except Exception as exc:  # noqa: BLE001
        return Check("router:budget", "warn", f"telemetry open failed: {exc}")
    if not tiers:
        return Check(
            "router:budget",
            "ok",
            f"no calls in 24h; budget ${cfg.max_per_day_usd:.2f}/day",
        )
    dist = ", ".join(f"{s.tier}={s.calls}" for s in tiers)
    pct = (rolling / cfg.max_per_day_usd * 100) if cfg.max_per_day_usd else 0
    status: Status = "warn" if pct >= 80 else "ok"
    return Check(
        "router:budget",
        status,
        f"24h: ${rolling:.4f} of ${cfg.max_per_day_usd:.2f} ({pct:.0f}%); {dist}",
    )


def check_autofetch_dlq() -> Check:
    """Dead-letter queue health for the auto-fetch tick (#38)."""
    try:
        from wiki_autofetch.dlq import AutoFetchDLQ
    except ImportError:
        return Check("auto-fetch:dlq", "skip", "wiki_autofetch package unavailable")
    try:
        dlq = AutoFetchDLQ()
    except Exception as exc:  # noqa: BLE001
        return Check("auto-fetch:dlq", "warn", f"DLQ open failed: {exc}")
    try:
        failures = dlq.list_failures()
    finally:
        dlq.close()
    if not failures:
        return Check("auto-fetch:dlq", "ok", "no failures recorded")
    by_source: dict[str, int] = {}
    for entry in failures:
        by_source[entry.source_name] = by_source.get(entry.source_name, 0) + 1
    summary = ", ".join(f"{s}={n}" for s, n in sorted(by_source.items()))
    detail = f"{len(failures)} entries ({summary}); run `sbw fetch status` for details"
    return Check("auto-fetch:dlq", "warn", detail)


# --------------------------------------------------------------------------- #
# Renderer                                                                    #
# --------------------------------------------------------------------------- #


def _render_text(checks: Sequence[Check]) -> str:
    glyphs: dict[Status, str] = {"ok": "[ok]", "warn": "[!!]", "fail": "[xx]", "skip": "[--]"}
    width = max(len(c.name) for c in checks)
    lines = [f"  {glyphs[c.status]}  {c.name.ljust(width)}  {c.detail}" for c in checks]
    header = "sbw doctor — local environment health"
    return header + "\n" + ("─" * len(header)) + "\n" + "\n".join(lines)


def _render_json(checks: Sequence[Check]) -> str:
    return json.dumps([asdict(c) for c in checks], indent=2)


# --------------------------------------------------------------------------- #
# Entry point                                                                 #
# --------------------------------------------------------------------------- #


def run(*, probe_network: bool = False, output_format: Literal["text", "json"] = "text") -> int:
    checks = [
        check_python_version(),
        check_required_packages(),
        check_binaries(),
        check_wiki_root(),
        check_data_stores(),
        check_mcp_config(),
        check_llm_keys(probe_network),
        check_systemd_unit(),
        check_autofetch_timer(),
        check_autofetch_dlq(),
        check_router_budget(),
    ]
    output = _render_json(checks) if output_format == "json" else _render_text(checks)
    print(output)
    return 1 if any(c.status == "fail" for c in checks) else 0


def main() -> int:
    parser = argparse.ArgumentParser(
        prog="sbw doctor", description="Local environment health for Second Brain Wiki"
    )
    parser.add_argument("--json", action="store_true", help="emit machine-readable JSON")
    parser.add_argument(
        "--probe-network", action="store_true", help="probe LLM endpoints (reserved for follow-up)"
    )
    args = parser.parse_args()
    return run(probe_network=args.probe_network, output_format="json" if args.json else "text")


if __name__ == "__main__":
    raise SystemExit(main())
