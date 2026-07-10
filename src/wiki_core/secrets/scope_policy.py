"""
Frozen per-provider OAuth scope table per [ADR-017](../../../docs/ADR-017-oauth-token-storage-and-scope-policy.md) §Per-provider OAuth scope table.

This module is the *source of truth* for which OAuth scopes SBW requests.
Mutating it without a new ADR amendment is forbidden — `test_scope_policy_locked`
asserts the literal strings match the ADR.

When a provider requires expanded scope (e.g. Google deprecates a scope, or
a new feature needs `gmail.modify`), the integration code must:

1. Open a new ADR amending this table.
2. Update the values here.
3. Update the corresponding integration code.
4. Re-run the locked test, which now passes against the new values.

The integration code MUST read from `PROVIDER_SCOPES[provider]` rather than
hardcoding scope strings inline — this is what makes the lock enforceable.
"""

from __future__ import annotations

from dataclasses import dataclass
from types import MappingProxyType


@dataclass(frozen=True)
class ScopePolicy:
    """OAuth scopes for a single provider.

    `default` is what SBW requests on connect. `opt_in_extra` is documented
    additional scope requested via a separate consent screen (e.g. Drive
    body-fetch). Empty tuple means no opt-in path.
    """

    provider: str
    default: tuple[str, ...]
    opt_in_extra: tuple[str, ...] = ()


# Locked per ADR-017. Mutating these strings without a new ADR amendment
# triggers a CI failure via test_scope_policy_locked.
_PROVIDER_SCOPES: dict[str, ScopePolicy] = {
    "gmail": ScopePolicy(
        provider="gmail",
        default=("https://www.googleapis.com/auth/gmail.readonly",),
    ),
    "calendar": ScopePolicy(
        provider="calendar",
        default=("https://www.googleapis.com/auth/calendar.readonly",),
    ),
    "drive": ScopePolicy(
        provider="drive",
        default=("https://www.googleapis.com/auth/drive.metadata.readonly",),
        opt_in_extra=("https://www.googleapis.com/auth/drive.readonly",),
    ),
    "github": ScopePolicy(
        provider="github",
        default=("repo:status", "read:user", "read:org"),
    ),
    "notion": ScopePolicy(
        provider="notion",
        # Notion uses workspace-scoped tokens; SBW asks for read capability only.
        default=("read_content",),
    ),
    "slack": ScopePolicy(
        provider="slack",
        default=("channels:history", "channels:read", "users:read"),
        # DM ingest requires a separate consent dialog per ADR-017.
        opt_in_extra=("im:history", "im:read"),
    ),
    # ── First write-capable provider — ADR-017 amended by ADR-021 Phase 2a. ──
    # This deliberately punctures the read-only invariant. `w_member_social`
    # is required to create a post on the member's own feed; the OIDC scopes
    # (openid/profile/email) resolve the author URN via LINKEDIN_GET_MY_INFO.
    # Phase 2a only ever writes lifecycleState='DRAFT' (a native LinkedIn
    # draft the user publishes manually) — gated by a HITL typed confirm.
    "linkedin": ScopePolicy(
        provider="linkedin",
        default=("w_member_social", "openid", "profile", "email"),
    ),
}

PROVIDER_SCOPES: MappingProxyType[str, ScopePolicy] = MappingProxyType(_PROVIDER_SCOPES)
"""Read-only view of the scope policy. Import this, not the underlying dict."""


SUPPORTED_PROVIDERS: tuple[str, ...] = tuple(PROVIDER_SCOPES.keys())


def policy_for(provider: str) -> ScopePolicy:
    """Return the locked policy or raise `KeyError` with a helpful message."""
    try:
        return PROVIDER_SCOPES[provider]
    except KeyError:
        raise KeyError(
            f"No scope policy registered for provider {provider!r}. "
            f"Supported: {', '.join(SUPPORTED_PROVIDERS)}. Adding a provider "
            f"requires an ADR amendment per ADR-017."
        ) from None
