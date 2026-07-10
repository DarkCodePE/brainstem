"""
`wiki_integrations.agent_tools` — concrete `IIntegration` impls per
[PRD-006](../../../docs/PRD-006-integrations-framework.md) Wave 1.

These are the surfaces the Deep Agents agent calls when the user says
"search my Gmail for X" or "list my open PRs". The polling/ingest counterpart
lives in `wiki_integrations.providers.{github,gmail}`.
"""

from __future__ import annotations

from wiki_integrations.agent_tools.audit_md import ProviderMarkdownLog
from wiki_integrations.agent_tools.base import ComposioBackedIntegration
from wiki_integrations.agent_tools.calendar import CalendarIntegration
from wiki_integrations.agent_tools.drive import GoogleDriveIntegration
from wiki_integrations.agent_tools.github import GitHubIntegration
from wiki_integrations.agent_tools.gmail import GmailIntegration
from wiki_integrations.agent_tools.slack import SlackIntegration

__all__ = [
    "CalendarIntegration",
    "ComposioBackedIntegration",
    "GitHubIntegration",
    "GmailIntegration",
    "GoogleDriveIntegration",
    "ProviderMarkdownLog",
    "SlackIntegration",
]
