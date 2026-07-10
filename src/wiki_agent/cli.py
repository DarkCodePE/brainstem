"""CLI entry point for the Wiki Deep Agent.

Usage:
    python -m wiki_agent ingest path/to/source.md
    python -m wiki_agent query "What is X?"
    python -m wiki_agent lint
    python -m wiki_agent init [--root ./knowledge-base]
    python -m wiki_agent stats
"""

from __future__ import annotations

import argparse
import json
import sys
import uuid

from wiki_agent.agent import create_wiki_agent


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="wiki_agent",
        description="LLM Wiki Deep Agent -- maintain an Obsidian-compatible knowledge base",
    )
    parser.add_argument(
        "--root",
        default="./knowledge-base",
        help="Path to the knowledge-base directory (default: ./knowledge-base)",
    )
    parser.add_argument(
        "--model",
        default="claude-sonnet-4-5-20250929",
        help="Anthropic model identifier (default: claude-sonnet-4-5-20250929)",
    )
    parser.add_argument(
        "--supervised",
        action="store_true",
        help="Enable human-in-the-loop approval before wiki writes",
    )
    parser.add_argument(
        "--session",
        default=None,
        help="Session ID for conversation continuity across invocations",
    )

    subparsers = parser.add_subparsers(dest="command", required=True)

    # ingest
    ingest_parser = subparsers.add_parser("ingest", help="Process a source into the wiki")
    ingest_parser.add_argument("source", help="Path to the source document")
    ingest_parser.add_argument(
        "--force", action="store_true", help="Re-ingest even if already processed"
    )

    # query
    query_parser = subparsers.add_parser("query", help="Ask a question about wiki contents")
    query_parser.add_argument("question", help="Natural language question")
    query_parser.add_argument(
        "--file-answer", action="store_true", help="File the answer as a wiki page"
    )

    # lint
    subparsers.add_parser("lint", help="Health-check the wiki")

    # recall (ADR-034 D4: deterministic, no-LLM page recall for external pipelines)
    recall_parser = subparsers.add_parser(
        "recall", help="Resolve a query to page metadata + token-budgeted body (JSON, no LLM)"
    )
    recall_parser.add_argument("query", help="Natural language topic query")
    recall_parser.add_argument(
        "--token-budget", type=int, default=1500, help="Max body tokens to emit (default: 1500)"
    )
    recall_parser.add_argument(
        "--limit", type=int, default=1, help="Max pages to return (default: 1)"
    )
    recall_parser.add_argument("--pretty", action="store_true", help="Indent the JSON output")

    # capture
    capture_parser = subparsers.add_parser("capture", help="Capture a TIL observation")
    capture_parser.add_argument(
        "observation", help="The observation text (e.g. 'TIL: Hermes uses MCP')"
    )

    # review
    review_parser = subparsers.add_parser(
        "review", help="Review unreviewed observations and propose graduations"
    )
    review_parser.add_argument(
        "--since", default=None, help="Review observations since this date (YYYY-MM-DD)"
    )

    # init
    subparsers.add_parser("init", help="Initialise the knowledge-base directory structure")

    # serve
    serve_parser = subparsers.add_parser(
        "serve", help="Start the wiki MCP server for agent consumption"
    )
    serve_parser.add_argument(
        "--transport",
        choices=["stdio", "sse"],
        default="stdio",
        help="Transport protocol: stdio (Claude Code) or sse (HTTP for Hermes/Ollama)",
    )
    serve_parser.add_argument("--port", type=int, default=8765, help="Port for SSE transport")

    # setup-obsidian
    subparsers.add_parser("setup-obsidian", help="Show Obsidian MCP integration setup instructions")

    # stats
    subparsers.add_parser("stats", help="Show wiki statistics")

    # integrations — OAuth provider management per ADR-017
    integrations_parser = subparsers.add_parser(
        "integrations", help="Manage OAuth integrations (list / revoke)"
    )
    integrations_sub = integrations_parser.add_subparsers(dest="integrations_action", required=True)
    integrations_sub.add_parser("list", help="List supported providers and connection status")
    revoke_parser = integrations_sub.add_parser(
        "revoke", help="Revoke an integration: clear tokens + record audit event"
    )
    revoke_parser.add_argument(
        "--provider",
        required=True,
        help="Provider id (gmail, calendar, drive, github, notion, slack)",
    )
    integrations_sub.add_parser("audit", help="Print the last 20 entries of the audit log")
    connect_parser = integrations_sub.add_parser(
        "connect", help="Start the OAuth flow for a provider via Composio"
    )
    connect_parser.add_argument("--provider", required=True, help="Provider id")
    search_parser = integrations_sub.add_parser(
        "search", help="Search an integration (substring match)"
    )
    search_parser.add_argument("--provider", required=True, help="Provider id")
    search_parser.add_argument("--query", required=True, help="Search query")
    search_parser.add_argument(
        "--limit", type=int, default=20, help="Max results to return (default 20)"
    )

    # router — model router status + dry-run per #37
    router_parser = subparsers.add_parser("router", help="Model router: status + policy dry-run")
    router_sub = router_parser.add_subparsers(dest="router_action", required=True)
    router_status = router_sub.add_parser("status", help="Show tier distribution + 24h cost")
    router_status.add_argument("--json", action="store_true", help="Emit machine-readable JSON")
    router_dry = router_sub.add_parser(
        "dry-run", help="Show which tier policy would pick for a task"
    )
    router_dry.add_argument(
        "--intent", required=True, choices=["seal", "ingest", "query", "lint", "vision"]
    )
    router_dry.add_argument(
        "--caller-priority", default="background", choices=["foreground", "background"]
    )

    # fetch — auto-fetch tick control per #38
    fetch_parser = subparsers.add_parser("fetch", help="Auto-fetch tick: status / one-shot run")
    fetch_sub = fetch_parser.add_subparsers(dest="fetch_action", required=True)
    fetch_sub.add_parser("status", help="Show DLQ state + last-tick info")
    fetch_run = fetch_sub.add_parser(
        "run", help="Run one auto-fetch tick under the lockfile (systemd-equivalent)"
    )
    fetch_run.add_argument(
        "--provider", help="Filter the report to one provider (the tick still runs all)"
    )
    fetch_run.add_argument("--json", action="store_true", help="Emit a machine-readable report")
    fetch_clear = fetch_sub.add_parser(
        "clear", help="Clear DLQ entries (operator action after fixing root cause)"
    )
    fetch_clear.add_argument("--provider", help="Restrict to one provider; default clears all")

    return parser


def _run_agent(agent, message: str, session: str | None = None) -> None:
    """Invoke the agent and print the response."""
    thread_id = session or str(uuid.uuid4())
    config = {"configurable": {"thread_id": thread_id}}

    result = agent.invoke(
        {"messages": [{"role": "user", "content": message}]},
        config=config,
    )

    # Extract the last assistant message
    messages = result.get("messages", [])
    for msg in reversed(messages):
        if isinstance(msg, dict):
            content = msg.get("content", "")
            role = msg.get("role", "")
        else:
            content = getattr(msg, "content", "") or ""
            role = getattr(msg, "type", "") or ""
        if role in ("ai", "assistant") and content:
            print(content)
            return

    print("No response generated.")


def main(argv: list[str] | None = None) -> int:
    """Main CLI entry point.

    Args:
        argv: Command-line arguments (defaults to sys.argv[1:]).

    Returns:
        Exit code (0 for success, 1 for error).
    """
    parser = _build_parser()
    args = parser.parse_args(argv)

    if args.command == "init":
        from wiki_agent.setup.init_wiki import init_knowledge_base

        init_knowledge_base(args.root)
        print(f"Knowledge base initialised at: {args.root}")
        return 0

    if args.command == "serve":
        import os

        os.environ["WIKI_ROOT"] = os.path.abspath(args.root)
        from wiki_agent.mcp_server import mcp

        if args.transport == "sse":
            mcp.settings.port = args.port
            print(f"Starting wiki MCP server (SSE) on port {args.port}...", file=sys.stderr)
            print(f"  Wiki root: {os.path.abspath(args.root)}", file=sys.stderr)
            print(f"  Connect: http://localhost:{args.port}/sse", file=sys.stderr)
            mcp.run(transport="sse")
        else:
            mcp.run(transport="stdio")
        return 0

    if args.command == "setup-obsidian":
        from wiki_agent.setup.obsidian_mcp import print_setup_instructions

        print_setup_instructions(args.root)
        return 0

    if args.command == "integrations":
        from wiki_agent.integrations_cli import run_integrations_cli

        return run_integrations_cli(args)

    if args.command == "fetch":
        from wiki_agent.fetch_cli import run_fetch_cli

        return run_fetch_cli(args)

    if args.command == "recall":
        from wiki_agent.recall_cli import run_recall_cli

        return run_recall_cli(args)

    if args.command == "router":
        from wiki_agent.router_cli import run_router_cli

        return run_router_cli(args)

    if args.command == "stats":
        from wiki_agent.tools import create_tools

        tools = create_tools(args.root)
        stats_tool = next(t for t in tools if t.name == "get_wiki_stats")
        result = stats_tool.invoke({})
        stats = json.loads(result)
        print(json.dumps(stats, indent=2))
        return 0

    # Create the agent for ingest/query/lint commands
    agent = create_wiki_agent(
        wiki_root=args.root,
        model=args.model,
        supervised=args.supervised,
        db_path=None,  # uses default <wiki_root>/.wiki-agent.db
    )

    session = getattr(args, "session", None)

    if args.command == "ingest":
        force_note = " (force re-ingest)" if args.force else ""
        message = f"Ingest this source: {args.source}{force_note}"
        _run_agent(agent, message, session=session)
    elif args.command == "query":
        file_note = " Please file the answer as a wiki page." if args.file_answer else ""
        message = f"{args.question}{file_note}"
        _run_agent(agent, message, session=session)
    elif args.command == "lint":
        _run_agent(agent, "Run a lint health check on the wiki.", session=session)
    elif args.command == "capture":
        obs_text = args.observation
        if not obs_text.upper().startswith("TIL:"):
            obs_text = f"TIL: {obs_text}"
        _run_agent(agent, f"Capture this observation: {obs_text}", session=session)
    elif args.command == "review":
        since_note = f" since {args.since}" if args.since else ""
        _run_agent(
            agent,
            f"Review unreviewed observations{since_note} and propose graduations.",
            session=session,
        )
    else:
        parser.print_help()
        return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())
