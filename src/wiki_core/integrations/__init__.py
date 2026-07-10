"""
`wiki_core.integrations` — typed `IIntegration` Protocol per PRD-006.

The Protocol is the agent-tool surface (synchronous CRUD-ish). The polling-
side `IngestSource` Protocol in `wiki_core.protocols` is a separate concern;
a single provider can implement both.

PRD-006 + ADR-017 together require:

- Token storage via `wiki_core.secrets.SecretStore` (OS keychain).
- Revocation via `wiki_core.secrets.disconnect()`.
- Audit log: structured JSONL (forensic, ADR-017) + per-provider Markdown
  (human-readable, PRD-006).

Concrete implementations live in `src/wiki_integrations/agent_tools/`.
"""

from __future__ import annotations

from wiki_core.integrations.protocol import (
    ConnectResult,
    IIntegration,
    IntegrationError,
    IntegrationItem,
    NotConnectedError,
    SearchResult,
)

__all__ = [
    "ConnectResult",
    "IIntegration",
    "IntegrationError",
    "IntegrationItem",
    "NotConnectedError",
    "SearchResult",
]
