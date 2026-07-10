"""
Lock test for `wiki_core.secrets.PROVIDER_SCOPES` — asserts the literal scope
strings match ADR-017 §Per-provider OAuth scope table.

Mutating these without a new ADR amendment is a deliberate failure: the
amendment process must update this test alongside the policy.
"""

from __future__ import annotations

import pytest

from wiki_core.secrets import PROVIDER_SCOPES, SUPPORTED_PROVIDERS, policy_for

# Locked values per docs/ADR-017-oauth-token-storage-and-scope-policy.md.
# DO NOT CHANGE without an ADR amendment.
EXPECTED = {
    "gmail": {
        "default": ("https://www.googleapis.com/auth/gmail.readonly",),
        "opt_in_extra": (),
    },
    "calendar": {
        "default": ("https://www.googleapis.com/auth/calendar.readonly",),
        "opt_in_extra": (),
    },
    "drive": {
        "default": ("https://www.googleapis.com/auth/drive.metadata.readonly",),
        "opt_in_extra": ("https://www.googleapis.com/auth/drive.readonly",),
    },
    "github": {
        "default": ("repo:status", "read:user", "read:org"),
        "opt_in_extra": (),
    },
    "notion": {
        "default": ("read_content",),
        "opt_in_extra": (),
    },
    "slack": {
        "default": ("channels:history", "channels:read", "users:read"),
        "opt_in_extra": ("im:history", "im:read"),
    },
    # First write-capable provider — added by the ADR-021 Phase 2a amendment.
    "linkedin": {
        "default": ("w_member_social", "openid", "profile", "email"),
        "opt_in_extra": (),
    },
}


def test_supported_providers_match_adr():
    """The set of providers matches the locked ADR-017 table (incl. the
    ADR-021 Phase 2a linkedin amendment)."""
    assert set(SUPPORTED_PROVIDERS) == set(EXPECTED.keys())


def test_linkedin_is_the_only_write_scope():
    """LinkedIn is the single deliberate write puncture (ADR-021 Phase 2a).
    Every other provider stays read-only — guards against an accidental
    second write scope slipping in without its own ADR."""
    li = policy_for("linkedin")
    assert "w_member_social" in li.default  # the intended write scope
    for provider in SUPPORTED_PROVIDERS:
        if provider == "linkedin":
            continue
        scopes = (*policy_for(provider).default, *policy_for(provider).opt_in_extra)
        assert not any("w_member_social" in s for s in scopes)


@pytest.mark.parametrize("provider", sorted(EXPECTED.keys()))
def test_scope_strings_locked(provider):
    expected = EXPECTED[provider]
    actual = policy_for(provider)
    assert actual.default == expected["default"], (
        f"ADR-017 lock violated for {provider}: default scope mutated. "
        f"Expected {expected['default']!r}, got {actual.default!r}. "
        f"This change requires an ADR amendment."
    )
    assert actual.opt_in_extra == expected["opt_in_extra"], (
        f"ADR-017 lock violated for {provider}: opt-in extra scope mutated."
    )


def test_no_gmail_write_scope():
    """Gmail must never request modify/send — locked anti-feature."""
    g = policy_for("gmail")
    forbidden = ("gmail.modify", "gmail.send", "gmail.compose", "https://mail.google.com/")
    all_scopes = (*g.default, *g.opt_in_extra)
    for bad in forbidden:
        assert not any(bad in s for s in all_scopes), (
            f"Gmail scope policy includes forbidden write scope matching {bad!r}"
        )


def test_no_github_write_scope():
    """GitHub must not request `repo` (full write); only fine-grained read scopes."""
    g = policy_for("github")
    all_scopes = (*g.default, *g.opt_in_extra)
    assert "repo" not in all_scopes
    assert "delete_repo" not in all_scopes
    assert "admin:org" not in all_scopes


def test_slack_dm_only_via_opt_in():
    """Slack DM scopes MUST be opt-in-extra, never default."""
    s = policy_for("slack")
    for scope in s.default:
        assert "im:" not in scope, (
            f"Slack default scope {scope!r} includes DM access — must be opt-in only"
        )


def test_policy_immutable_at_runtime():
    """PROVIDER_SCOPES is a MappingProxyType — mutations must raise."""
    with pytest.raises(TypeError):
        PROVIDER_SCOPES["gmail"] = None  # type: ignore[index]


def test_policy_for_unknown_provider_raises():
    with pytest.raises(KeyError, match="No scope policy registered"):
        policy_for("twitter")
