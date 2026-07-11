"""ADR-051 — Brainstem product CLI (init / mcp / doctor).

The MCP server itself is covered elsewhere (``test_mcp_e2e_stdio`` and friends);
here we cover the thin product front door: vault bootstrap and its printed
connection one-liner, root-resolution precedence, argv/env translation onto
``wiki_agent.mcp_server.main``, and the doctor delegation.
"""

from __future__ import annotations

import sys
from types import SimpleNamespace

import pytest

from wiki_agent import brainstem_cli

# ---------------------------------------------------------------------------
# _resolve_root precedence: --root > $WIKI_ROOT > default
# ---------------------------------------------------------------------------


def test_resolve_root_prefers_explicit_flag(tmp_path, monkeypatch):
    monkeypatch.setenv("WIKI_ROOT", str(tmp_path / "from-env"))
    explicit = tmp_path / "explicit"
    assert brainstem_cli._resolve_root(str(explicit)) == explicit.resolve()


def test_resolve_root_falls_back_to_env(tmp_path, monkeypatch):
    env_root = tmp_path / "from-env"
    monkeypatch.setenv("WIKI_ROOT", str(env_root))
    assert brainstem_cli._resolve_root(None) == env_root.resolve()


def test_resolve_root_defaults_to_home_vault(monkeypatch):
    monkeypatch.delenv("WIKI_ROOT", raising=False)
    assert brainstem_cli._resolve_root(None) == brainstem_cli.DEFAULT_VAULT


# ---------------------------------------------------------------------------
# brainstem init
# ---------------------------------------------------------------------------


def test_init_bootstraps_vault_and_prints_one_liner(tmp_path, capsys):
    vault = tmp_path / "vault"
    rc = brainstem_cli.main(["init", "--root", str(vault)])
    assert rc == 0

    for sub in ("wiki", "wiki/sources", "wiki/concepts", "wiki/entities", "raw"):
        assert (vault / sub).is_dir()
    assert (vault / "wiki" / "index.md").read_text().startswith("# Wiki Index")

    out = capsys.readouterr().out
    assert "Vault created" in out
    assert f'claude mcp add brainstem -- uvx brainstem-mcp mcp --root "{vault.resolve()}"' in out
    assert "--readonly" in out


def test_init_is_idempotent_and_keeps_existing_index(tmp_path, capsys):
    vault = tmp_path / "vault"
    assert brainstem_cli.main(["init", "--root", str(vault)]) == 0
    index = vault / "wiki" / "index.md"
    index.write_text("# Custom index — do not clobber\n")

    assert brainstem_cli.main(["init", "--root", str(vault)]) == 0
    assert index.read_text() == "# Custom index — do not clobber\n"
    assert "already exists" in capsys.readouterr().out


# ---------------------------------------------------------------------------
# brainstem mcp — argv/env translation onto mcp_server.main
# ---------------------------------------------------------------------------


@pytest.fixture
def fake_mcp_server(monkeypatch):
    """Intercept the lazy ``from wiki_agent import mcp_server`` import."""
    captured = SimpleNamespace(argv=None, calls=0)

    def fake_main():
        captured.argv = list(sys.argv)
        captured.calls += 1

    import wiki_agent.mcp_server as real

    monkeypatch.setattr(real, "main", fake_main)
    return captured


def test_mcp_forwards_root_and_transport(tmp_path, monkeypatch, fake_mcp_server):
    monkeypatch.delenv("WIKI_MCP_READONLY", raising=False)
    vault = tmp_path / "vault"
    (vault / "wiki").mkdir(parents=True)

    rc = brainstem_cli.main(["mcp", "--root", str(vault)])
    assert rc == 0
    assert fake_mcp_server.calls == 1
    assert fake_mcp_server.argv == [
        "brainstem-mcp",
        "--root",
        str(vault.resolve()),
        "--transport",
        "stdio",
    ]
    assert "WIKI_MCP_READONLY" not in __import__("os").environ


def test_mcp_readonly_sets_env_before_server_import(tmp_path, monkeypatch, fake_mcp_server):
    monkeypatch.delenv("WIKI_MCP_READONLY", raising=False)
    vault = tmp_path / "vault"
    (vault / "wiki").mkdir(parents=True)

    brainstem_cli.main(["mcp", "--readonly", "--root", str(vault)])
    assert __import__("os").environ.get("WIKI_MCP_READONLY") == "1"


def test_mcp_sse_forwards_port(tmp_path, monkeypatch, fake_mcp_server):
    monkeypatch.delenv("WIKI_MCP_READONLY", raising=False)
    vault = tmp_path / "vault"
    (vault / "wiki").mkdir(parents=True)

    brainstem_cli.main(["mcp", "--root", str(vault), "--transport", "sse", "--port", "9999"])
    assert fake_mcp_server.argv[-4:] == ["--transport", "sse", "--port", "9999"]


def test_mcp_zero_config_bootstraps_default_vault(tmp_path, monkeypatch, fake_mcp_server, capsys):
    """First run with no --root, no $WIKI_ROOT, no vault → init instead of failing."""
    monkeypatch.delenv("WIKI_ROOT", raising=False)
    monkeypatch.delenv("WIKI_MCP_READONLY", raising=False)
    default_vault = tmp_path / "home" / ".brainstem" / "vault"
    monkeypatch.setattr(brainstem_cli, "DEFAULT_VAULT", default_vault)

    rc = brainstem_cli.main(["mcp"])
    assert rc == 0
    assert (default_vault / "wiki" / "index.md").exists()
    assert "Vault created" in capsys.readouterr().out
    assert fake_mcp_server.argv[1:3] == ["--root", str(default_vault)]


# ---------------------------------------------------------------------------
# brainstem doctor / --version
# ---------------------------------------------------------------------------


def test_doctor_delegates_and_returns_exit_code(monkeypatch):
    import wiki_agent.doctor as doctor_mod

    monkeypatch.setattr(doctor_mod, "main", lambda: 3)
    assert brainstem_cli.main(["doctor"]) == 3


def test_version_flag_prints_dist_version(capsys):
    with pytest.raises(SystemExit) as exc:
        brainstem_cli.main(["--version"])
    assert exc.value.code == 0
    assert "brainstem-mcp" in capsys.readouterr().out
