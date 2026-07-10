"""
Concrete `OAuthIntegrationSource` subclasses, one per provider.

Phase 1 ships the two highest-value providers (Gmail, GitHub) per PRD-005.
Slack/Notion/Drive/Calendar follow the same shape and land in a follow-up
PR each — that's the registry pattern PRD-005 calls out.
"""

from __future__ import annotations

from wiki_integrations.providers.github import GitHubIntegrationSource
from wiki_integrations.providers.gmail import GmailIntegrationSource

__all__ = ["GitHubIntegrationSource", "GmailIntegrationSource"]
