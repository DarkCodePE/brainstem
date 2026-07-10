"""Obsidian MCP server configuration generator.

Generates the JSON snippet needed to register the Obsidian MCP server
with Claude Code, enabling wiki search, backlinks, and graph navigation
directly from the agent.

Usage::

    python -m wiki_agent.setup.obsidian_mcp /path/to/vault
    # or programmatically:
    from wiki_agent.setup.obsidian_mcp import generate_obsidian_mcp_config
    config = generate_obsidian_mcp_config("/path/to/vault")
"""

from __future__ import annotations

import json
import os
import sys


def generate_obsidian_mcp_config(vault_path: str) -> dict:
    """Generate Obsidian MCP server configuration for Claude Code.

    Args:
        vault_path: Absolute path to the Obsidian vault (knowledge-base directory).

    Returns:
        Dict with the MCP server configuration ready for ~/.claude.json.
    """
    abs_vault = os.path.abspath(vault_path)
    return {
        "obsidian": {
            "command": "npx",
            "args": ["-y", "obsidian-mcp"],
            "env": {
                "OBSIDIAN_VAULT_PATH": abs_vault,
            },
        }
    }


def print_setup_instructions(vault_path: str) -> None:
    """Print setup instructions for Obsidian MCP integration."""
    config = generate_obsidian_mcp_config(vault_path)
    config_json = json.dumps({"mcpServers": config}, indent=2)

    print("=" * 60)
    print("Obsidian MCP Integration Setup")
    print("=" * 60)
    print()
    print(f"Vault path: {os.path.abspath(vault_path)}")
    print()
    print("Option 1 -- CLI registration (recommended):")
    print()
    print("  claude mcp add obsidian -- npx -y obsidian-mcp")
    print(f"  # Then set OBSIDIAN_VAULT_PATH={os.path.abspath(vault_path)}")
    print()
    print("Option 2 -- Add to ~/.claude.json manually:")
    print()
    print(config_json)
    print()
    print("Available tools after registration:")
    print("  - obsidian_search: Search vault by keyword")
    print("  - obsidian_read: Read a specific note")
    print("  - obsidian_create: Create a new note")
    print("  - obsidian_append: Append content to a note")
    print("  - obsidian_backlinks: Find pages linking to a note")
    print("  - obsidian_tags: List all tags in the vault")
    print("  - obsidian_orphans: Find notes with no links")
    print()
    print("After setup, restart Claude Code for changes to take effect.")
    print("=" * 60)


if __name__ == "__main__":
    vault = sys.argv[1] if len(sys.argv) > 1 else "./knowledge-base"
    print_setup_instructions(vault)
