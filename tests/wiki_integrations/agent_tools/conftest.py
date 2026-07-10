"""Shared fixtures for the new `IIntegration` agent-tool tests.

Reuses the `FakeWalker` from `tests/wiki_integrations/conftest.py` and
adds a fake `ComposioBridge`-shaped object that also satisfies the
`connect` / `list_active` surface the `ComposioBackedIntegration` base
needs.
"""

from __future__ import annotations

from collections.abc import AsyncIterator
from dataclasses import dataclass
from typing import Any

import keyring
import keyring.backend
import keyring.errors
import pytest


@dataclass
class FakeConnection:
    provider: str
    connection_id: str
    status: str = "connected"
    redirect_url: str | None = None
    metadata: dict[str, Any] | None = None


class FakeBridge:
    """Minimal `ComposioBridge`-shaped fake.

    Exposes the four surfaces `ComposioBackedIntegration` subclasses
    touch: `connect()`, `list_active()`, `walk(provider)`, and
    `execute(provider, tool_slug, arguments)`. Pre-seed walk payloads
    and execute responses via the constructor; track invocations for
    assertions.
    """

    def __init__(
        self,
        *,
        payloads: dict[str, list[dict[str, Any]]] | None = None,
        execute_responses: dict[tuple[str, str], dict[str, Any]] | None = None,
        already_connected: list[str] | None = None,
    ) -> None:
        self._payloads = {k: list(v) for k, v in (payloads or {}).items()}
        self._execute_responses = dict(execute_responses or {})
        self._connections: dict[str, FakeConnection] = {
            p: FakeConnection(provider=p, connection_id=f"existing-{p}")
            for p in (already_connected or [])
        }
        self.connect_calls: list[str] = []
        self.list_active_calls: int = 0
        self.walk_calls: list[str] = []
        self.execute_calls: list[tuple[str, str, dict[str, Any]]] = []

    async def connect(self, provider: str) -> FakeConnection:
        self.connect_calls.append(provider)
        existing = self._connections.get(provider)
        if existing is not None:
            return existing
        conn = FakeConnection(provider=provider, connection_id=f"new-{provider}")
        self._connections[provider] = conn
        return conn

    async def list_active(self) -> list[FakeConnection]:
        self.list_active_calls += 1
        return [c for c in self._connections.values() if c.status == "connected"]

    async def walk(self, provider: str) -> AsyncIterator[dict[str, Any]]:
        self.walk_calls.append(provider)
        for item in self._payloads.get(provider, []):
            yield dict(item)

    async def execute(
        self, provider: str, tool_slug: str, arguments: dict[str, Any]
    ) -> dict[str, Any]:
        self.execute_calls.append((provider, tool_slug, dict(arguments)))
        return dict(self._execute_responses.get((provider, tool_slug), {}))


class _InMemoryBackend(keyring.backend.KeyringBackend):
    priority = 100

    def __init__(self) -> None:
        super().__init__()
        self._store: dict[tuple[str, str], str] = {}

    def get_password(self, service: str, username: str) -> str | None:
        return self._store.get((service, username))

    def set_password(self, service: str, username: str, password: str) -> None:
        self._store[(service, username)] = password

    def delete_password(self, service: str, username: str) -> None:
        try:
            del self._store[(service, username)]
        except KeyError:
            raise keyring.errors.PasswordDeleteError(
                f"no entry for {service!r}/{username!r}"
            ) from None


@pytest.fixture
def secret_store():
    from wiki_core.secrets import KeyringSecretStore

    backend = _InMemoryBackend()
    original = keyring.get_keyring()
    keyring.set_keyring(backend)
    try:
        yield KeyringSecretStore(service="test.agent_tools")
    finally:
        keyring.set_keyring(original)


@pytest.fixture
def audit_jsonl(tmp_path):
    from wiki_core.secrets import AuditLog

    return AuditLog(path=tmp_path / "integrations.log.jsonl")


@pytest.fixture
def audit_md(tmp_path):
    from wiki_integrations.agent_tools import ProviderMarkdownLog

    return ProviderMarkdownLog(knowledge_base=tmp_path / "kb", provider="github")


@pytest.fixture
def audit_md_gmail(tmp_path):
    from wiki_integrations.agent_tools import ProviderMarkdownLog

    return ProviderMarkdownLog(knowledge_base=tmp_path / "kb", provider="gmail")


@pytest.fixture
def audit_md_calendar(tmp_path):
    from wiki_integrations.agent_tools import ProviderMarkdownLog

    return ProviderMarkdownLog(knowledge_base=tmp_path / "kb", provider="calendar")


@pytest.fixture
def audit_md_slack(tmp_path):
    from wiki_integrations.agent_tools import ProviderMarkdownLog

    return ProviderMarkdownLog(knowledge_base=tmp_path / "kb", provider="slack")


@pytest.fixture
def audit_md_drive(tmp_path):
    from wiki_integrations.agent_tools import ProviderMarkdownLog

    return ProviderMarkdownLog(knowledge_base=tmp_path / "kb", provider="drive")
