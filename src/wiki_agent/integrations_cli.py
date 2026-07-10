"""
CLI handler for `sbw integrations {list,revoke,audit}` per ADR-017 §Revocation
contract and issue #39 AC.

Kept in a separate module so the rest of `wiki_agent.cli` doesn't have to
import `keyring` (and trip the no-backend error) on every invocation.
"""

from __future__ import annotations

import argparse
import json
import sys
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from wiki_core.secrets import SecretStore


def run_integrations_cli(args: argparse.Namespace) -> int:
    """Dispatch the `integrations` subcommand. Returns a process exit code."""
    action = getattr(args, "integrations_action", None)
    if action == "list":
        return _cmd_list()
    if action == "revoke":
        return _cmd_revoke(args.provider)
    if action == "audit":
        return _cmd_audit()
    if action == "connect":
        return _cmd_connect(args.provider)
    if action == "search":
        return _cmd_search(args.provider, args.query, args.limit)
    print(f"Unknown integrations action: {action!r}", file=sys.stderr)
    return 1


def _open_store() -> SecretStore | None:
    """Open the default keyring-backed `SecretStore`, or print a friendly
    error if no keyring backend is available."""
    from wiki_core.secrets import KeyringSecretStore, SecretStoreUnavailable

    try:
        return KeyringSecretStore()
    except SecretStoreUnavailable as exc:
        print(f"Secret store unavailable: {exc}", file=sys.stderr)
        print(
            "Hint: install libsecret (Linux) or unlock the Login Keychain (macOS), "
            "or run with $SBW_VAULT_PASSWORD for headless mode.",
            file=sys.stderr,
        )
        return None


def _cmd_list() -> int:
    """List integrations with their real upstream Composio status (#107).

    Resolution strategy:
    1. Open the keyring as before (gates whether SBW thinks the provider
       has been connected locally at all).
    2. Best-effort query Composio for the live status of each connection.
       Any failure (network, stub mode, missing API key) falls back to
       the legacy "yes/no" keyring-presence column.
    3. Render a CONNECTED column that reflects the real upstream state:
       ``connected``, ``initializing``, ``expired``, or ``no`` when the
       provider has no connection at all.
    """
    import asyncio

    from wiki_core.secrets import PROVIDER_SCOPES

    store = _open_store()
    if store is None:
        return 2

    # Try to fetch live status from Composio. ``None`` means the query
    # couldn't run (no API key, network blip, bridge import error); in
    # that case we fall back to the legacy keyring-only display so the
    # command stays useful offline.
    upstream_status: dict[str, str] | None = None
    try:
        from wiki_integrations.composio_bridge import ComposioBridge

        bridge = ComposioBridge()
        connections = asyncio.run(bridge.list_connections())
        upstream_status = {}
        for c in connections:
            # Prefer ``connected`` rows when Composio returns multiple
            # entries for the same provider (the API is server-ordered
            # but we don't want a stale row to mask a fresh active one).
            if c.status == "connected" or c.provider not in upstream_status:
                upstream_status[c.provider] = c.status
    except Exception:  # noqa: BLE001
        upstream_status = None

    print(f"{'PROVIDER':<10} {'CONNECTED':<13} SCOPES")
    print("-" * 70)
    for provider, policy in sorted(PROVIDER_SCOPES.items()):
        connection_name = f"composio.connection.{provider}"
        keyring_has = store.get(connection_name) is not None
        if upstream_status is None:
            # Bridge unavailable — legacy keyring-only display.
            connected_label = "yes" if keyring_has else "no"
        elif provider in upstream_status:
            connected_label = upstream_status[provider]
        else:
            # Bridge query succeeded; provider absent ⇒ no live connection.
            connected_label = "no"
        scopes = ", ".join(policy.default)
        print(f"{provider:<10} {connected_label:<13} {scopes}")
        if policy.opt_in_extra:
            print(f"{'':<10} {'':<13} (opt-in extra: {', '.join(policy.opt_in_extra)})")
    return 0


def _cmd_revoke(provider: str) -> int:
    from wiki_core.secrets import (
        SUPPORTED_PROVIDERS,
        AuditLog,
        disconnect,
    )

    if provider not in SUPPORTED_PROVIDERS:
        print(
            f"Unknown provider {provider!r}. Supported: {', '.join(SUPPORTED_PROVIDERS)}",
            file=sys.stderr,
        )
        return 1

    store = _open_store()
    if store is None:
        return 2

    audit = AuditLog()
    # No upstream revoker wired here — Composio HTTP client lands in #1
    # (PRD-006 framework). Local-only revocation still clears the vault
    # rows + audit-logs the event, which is the AC for #39.
    result = disconnect(provider, store=store, audit=audit, upstream_revoker=None)

    if result.local_cleared:
        print(f"OK: cleared all local secrets for {provider}.")
        print(f"Audit log appended: {audit.path}")
        return 0
    print(
        f"PARTIAL: {len(result.remaining_names)} entries could not be removed.",
        file=sys.stderr,
    )
    for name in result.remaining_names:
        print(f"  - {name}", file=sys.stderr)
    return 3


def _build_integration(provider: str, store):
    """Instantiate the right `IIntegration` for `provider`, wired with the
    default ComposioBridge and audit sinks. Returns ``None`` if unknown."""
    from pathlib import Path

    from wiki_core.secrets import AuditLog
    from wiki_integrations.agent_tools import (
        CalendarIntegration,
        GitHubIntegration,
        GmailIntegration,
        GoogleDriveIntegration,
        ProviderMarkdownLog,
        SlackIntegration,
    )
    from wiki_integrations.composio_bridge import ComposioBridge

    cls_map = {
        "calendar": CalendarIntegration,
        "drive": GoogleDriveIntegration,
        "github": GitHubIntegration,
        "gmail": GmailIntegration,
        "slack": SlackIntegration,
    }
    cls = cls_map.get(provider)
    if cls is None:
        return None
    bridge = ComposioBridge()
    kb = Path.cwd() / "knowledge-base"
    audit_md = ProviderMarkdownLog(knowledge_base=kb, provider=provider)
    return cls(bridge=bridge, store=store, audit_jsonl=AuditLog(), audit_md=audit_md)


def _cmd_connect(provider: str) -> int:
    import asyncio

    from wiki_core.secrets import SUPPORTED_PROVIDERS

    if provider not in SUPPORTED_PROVIDERS:
        print(
            f"Unknown provider {provider!r}. Supported: {', '.join(SUPPORTED_PROVIDERS)}",
            file=sys.stderr,
        )
        return 1
    store = _open_store()
    if store is None:
        return 2
    integration = _build_integration(provider, store)
    if integration is None:
        print(
            f"Provider {provider!r} has no agent-tool implementation yet "
            f"(M3 wired: calendar + drive + github + gmail + slack).",
            file=sys.stderr,
        )
        return 1
    result = asyncio.run(integration.connect())
    print(f"OK: {provider} status={result.status} connection_id={result.connection_id}")
    if result.redirect_url:
        print(f"Open this URL to complete the flow: {result.redirect_url}")
    return 0


def _cmd_search(provider: str, query: str, limit: int) -> int:
    """Search a provider for ``query`` (#108).

    Empty / whitespace-only ``query`` falls back to ``IIntegration.list()``
    so the user can browse the "latest window" without first knowing
    what to search for. This matches the CLI's intent — "show me what's
    in my Calendar" should not error.
    """
    import asyncio

    from wiki_core.integrations.protocol import NotConnectedError
    from wiki_core.secrets import SUPPORTED_PROVIDERS

    if provider not in SUPPORTED_PROVIDERS:
        print(
            f"Unknown provider {provider!r}. Supported: {', '.join(SUPPORTED_PROVIDERS)}",
            file=sys.stderr,
        )
        return 1
    store = _open_store()
    if store is None:
        return 2
    integration = _build_integration(provider, store)
    if integration is None:
        print(
            f"Provider {provider!r} has no agent-tool implementation yet.",
            file=sys.stderr,
        )
        return 1

    normalised_query = query.strip()
    try:
        if not normalised_query:
            items = asyncio.run(integration.list(limit=limit))
        else:
            result = asyncio.run(integration.search(normalised_query, limit=limit))
            items = result.items
    except NotConnectedError:
        print(
            f"Not connected to {provider}. Run: sbw integrations connect --provider {provider}",
            file=sys.stderr,
        )
        return 4
    if not items:
        if normalised_query:
            print(f"No matches for {normalised_query!r}.")
        else:
            print(f"No recent items for {provider}.")
        return 0
    if not normalised_query:
        print(f"# recent items for {provider} (showing up to {limit})")
    for item in items:
        print(f"  {item.id}  {item.title[:60]:<60}  {item.uri}")
    return 0


def _cmd_audit() -> int:
    from wiki_core.secrets import AuditLog

    audit = AuditLog()
    if not audit.path.exists():
        print(f"No audit log at {audit.path} (no integrations have run yet).")
        return 0
    lines = audit.path.read_text(encoding="utf-8").splitlines()
    tail = lines[-20:]
    for raw in tail:
        try:
            rec = json.loads(raw)
        except json.JSONDecodeError:
            print(raw)
            continue
        print(
            f"{rec.get('ts', '?')}  {rec.get('event', '?'):<22}  "
            f"{rec.get('provider', '?'):<10}  result={rec.get('result', '?')}"
        )
    return 0
