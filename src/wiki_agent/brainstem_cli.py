"""Brainstem product CLI — thin front door over the existing entry points.

Subcommands:
    brainstem init    Bootstrap a vault and print the MCP connection one-liner.
    brainstem mcp     Serve the knowledge-backend MCP server (stdio/sse).
    brainstem doctor  Local environment health checks.

The heavy lifting stays in wiki_agent.mcp_server and wiki_agent.doctor; this
module only translates the product-facing UX (flags, defaults, first-run
bootstrap) onto those entry points.
"""

from __future__ import annotations

import argparse
import os
import sys
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path

DEFAULT_VAULT = Path.home() / ".brainstem" / "vault"

_INDEX_TEMPLATE = """\
# Wiki Index

| Page | Category | Summary | Sources |
|------|----------|---------|---------|
"""


def _dist_version() -> str:
    for dist in ("brainstem-mcp", "second-brain-wiki"):
        try:
            return version(dist)
        except PackageNotFoundError:
            continue
    return "unknown"


def _resolve_root(root: str | None) -> Path:
    if root:
        return Path(root).expanduser().resolve()
    env_root = os.environ.get("WIKI_ROOT")
    if env_root:
        return Path(env_root).expanduser().resolve()
    return DEFAULT_VAULT


def cmd_init(args: argparse.Namespace) -> int:
    vault = _resolve_root(args.root)
    created = not vault.exists()
    for sub in ("wiki", "wiki/sources", "wiki/concepts", "wiki/entities", "raw"):
        (vault / sub).mkdir(parents=True, exist_ok=True)
    index = vault / "wiki" / "index.md"
    if not index.exists():
        index.write_text(_INDEX_TEMPLATE, encoding="utf-8")

    verb = "created" if created else "already exists — verified structure"
    print(f"Vault {verb}: {vault}")
    print()
    print("Connect your agent (Claude Code):")
    print(f'  claude mcp add brainstem -- uvx brainstem-mcp mcp --root "{vault}"')
    print()
    print("Or run the server directly:")
    print(f'  brainstem mcp --root "{vault}"')
    print()
    print("Read-only profile (12 safe tools):")
    print(f'  brainstem mcp --readonly --root "{vault}"')
    return 0


def cmd_mcp(args: argparse.Namespace) -> int:
    if args.readonly:
        os.environ["WIKI_MCP_READONLY"] = "1"

    vault = _resolve_root(args.root)
    if not (vault / "wiki").is_dir() and args.root is None and "WIKI_ROOT" not in os.environ:
        # Zero-config first run: bootstrap the default vault instead of failing.
        init_args = argparse.Namespace(root=str(vault))
        cmd_init(init_args)

    forwarded = ["brainstem-mcp", "--root", str(vault), "--transport", args.transport]
    if args.transport == "sse":
        forwarded += ["--port", str(args.port)]

    # mcp_server.main() parses sys.argv and WIKI_MCP_READONLY at import time,
    # so the env var must be set before the import.
    sys.argv = forwarded
    from wiki_agent import mcp_server

    mcp_server.main()
    return 0


def cmd_doctor(_args: argparse.Namespace) -> int:
    from wiki_agent import doctor

    sys.argv = ["brainstem-doctor"]
    return doctor.main()


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="brainstem",
        description="Brainstem — local-first knowledge backend for AI agents (MCP).",
    )
    parser.add_argument("--version", action="version", version=f"brainstem-mcp {_dist_version()}")
    sub = parser.add_subparsers(dest="command", required=True)

    p_init = sub.add_parser("init", help="bootstrap a vault and print the connection one-liner")
    p_init.add_argument("--root", default=None, help=f"vault path (default: {DEFAULT_VAULT})")
    p_init.set_defaults(func=cmd_init)

    p_mcp = sub.add_parser("mcp", help="serve the MCP server")
    p_mcp.add_argument(
        "--root", default=None, help="vault path (default: $WIKI_ROOT or ~/.brainstem/vault)"
    )
    p_mcp.add_argument("--readonly", action="store_true", help="expose only the 12 read-only tools")
    p_mcp.add_argument("--transport", choices=["stdio", "sse"], default="stdio")
    p_mcp.add_argument("--port", type=int, default=8765, help="port for sse transport")
    p_mcp.set_defaults(func=cmd_mcp)

    p_doc = sub.add_parser("doctor", help="local environment health checks")
    p_doc.set_defaults(func=cmd_doctor)

    return parser


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
