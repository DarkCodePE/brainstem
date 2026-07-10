"""Initialise the knowledge-base directory structure.

Creates the full directory tree, initial ``index.md``, ``log.md``,
and the wiki schema file used by the orchestrator prompt.

Usage::

    python -m wiki_agent init --root ./knowledge-base
    # or programmatically:
    from wiki_agent.setup.init_wiki import init_knowledge_base
    init_knowledge_base("./knowledge-base")
"""

from __future__ import annotations

import os
from datetime import UTC, datetime


def init_knowledge_base(root: str = "./knowledge-base") -> list[str]:
    """Create the knowledge-base directory structure and seed files.

    Args:
        root: Path where the knowledge base will be created.

    Returns:
        List of files and directories created.
    """
    created: list[str] = []
    now = datetime.now(UTC).strftime("%Y-%m-%d")

    # Directory tree
    dirs = [
        os.path.join(root, "raw", "articles"),
        os.path.join(root, "raw", "papers"),
        os.path.join(root, "raw", "bookmarks"),
        os.path.join(root, "raw", "voice-notes"),
        os.path.join(root, "raw", "repos"),
        os.path.join(root, "raw", "datasets"),
        os.path.join(root, "raw", "images"),
        os.path.join(root, "raw", "assets"),
        os.path.join(root, "wiki", "sources"),
        os.path.join(root, "wiki", "entities"),
        os.path.join(root, "wiki", "concepts"),
        os.path.join(root, "wiki", "answers"),
        os.path.join(root, "wiki", "synthesis"),
        os.path.join(root, "wiki", "outputs", "slides"),
        os.path.join(root, "wiki", "outputs", "charts"),
        os.path.join(root, "schema"),
        os.path.join(root, "schema", "templates"),
        os.path.join(root, "schema", "workflows"),
        os.path.join(root, "skills"),
    ]

    for d in dirs:
        os.makedirs(d, exist_ok=True)
        created.append(d)

    # wiki/index.md
    index_path = os.path.join(root, "wiki", "index.md")
    if not os.path.exists(index_path):
        index_content = (
            "---\n"
            f'title: "Wiki Index"\n'
            f"date: {now}\n"
            "sources: []\n"
            "tags: [index, meta]\n"
            "---\n\n"
            "# Wiki Index\n\n"
            "| Page | Category | Summary | Sources | Updated |\n"
            "|------|----------|---------|---------|--------|\n"
        )
        with open(index_path, "w", encoding="utf-8") as fh:
            fh.write(index_content)
        created.append(index_path)

    # wiki/log.md
    log_path = os.path.join(root, "wiki", "log.md")
    if not os.path.exists(log_path):
        log_content = (
            "---\n"
            f'title: "Operation Log"\n'
            f"date: {now}\n"
            "sources: []\n"
            "tags: [log, meta]\n"
            "---\n\n"
            "# Operation Log\n\n"
            f"## [{now}] init | Knowledge base initialised\n"
            "- Directory structure created\n"
            "- Index and log files seeded\n"
        )
        with open(log_path, "w", encoding="utf-8") as fh:
            fh.write(log_content)
        created.append(log_path)

    # schema/wiki-schema.md
    schema_path = os.path.join(root, "schema", "wiki-schema.md")
    if not os.path.exists(schema_path):
        schema_content = (
            "---\n"
            f'title: "Wiki Schema"\n'
            f"date: {now}\n"
            "sources: []\n"
            "tags: [schema, meta]\n"
            "---\n\n"
            "# Wiki Schema\n\n"
            "## Required frontmatter fields\n\n"
            "Every wiki page MUST include:\n\n"
            "```yaml\n"
            "---\n"
            'title: "Human-readable title"\n'
            "date: YYYY-MM-DD\n"
            'sources: ["path/to/source.md"]\n'
            "tags: [tag1, tag2]\n"
            "origin: human | llm-generated | llm-synthesized | mcp-ingested\n"
            "---\n"
            "```\n\n"
            "## Source provenance (origin field)\n\n"
            "The ``origin`` field tracks how each page was produced:\n\n"
            "| Origin | Meaning | Trust level |\n"
            "|--------|---------|-------------|\n"
            "| ``human`` | Written by a person | Highest -- ground truth |\n"
            "| ``llm-generated`` | Creative LLM output (drafts, memos) | Review before citing as fact |\n"
            "| ``llm-synthesized`` | Assembled from multiple sources by the LLM | Depends on source quality |\n"
            "| ``mcp-ingested`` | Pulled from an MCP server | Accurate at ingest time, may go stale |\n\n"
            "**Rules:**\n"
            "- Set ``origin`` when creating a page. Never leave it blank.\n"
            "- Never overwrite ``origin: human`` with ``llm-synthesized``.\n"
            "- Treat ``human`` and ``mcp-ingested`` as factual when citing.\n"
            "- Treat ``llm-generated`` and ``llm-synthesized`` as provisional.\n"
            "- Flag ``llm-generated`` pages older than 90 days as stale.\n"
            "- Flag ``mcp-ingested`` pages older than 30 days for re-ingestion.\n\n"
            "## Page categories\n\n"
            "- **sources/** -- One summary page per ingested source document\n"
            "- **entities/** -- One page per person, project, or organisation\n"
            "- **concepts/** -- One page per idea, framework, or methodology\n"
            "- **answers/** -- Filed answers from query operations\n"
            "- **synthesis/** -- Cross-source synthesis pages\n\n"
            "## Cross-reference conventions\n\n"
            "Use ``[[Page Name]]`` wikilink syntax to reference other pages.\n"
            "Every entity or concept mentioned in a page should be cross-referenced.\n\n"
            "## Slug format\n\n"
            "Page filenames use kebab-case: ``entity-name.md``, ``concept-name.md``.\n"
            "No special characters, no spaces, all lowercase.\n\n"
            "## Context hygiene\n\n"
            "- Target 3-5 files per operation. Do not bulk-load the wiki.\n"
            "- Delegate exploration to subagents. Keep orchestrator context clean.\n"
            "- 2-3 iteration max per failing approach. Then reassess.\n"
            "- Chain, don't re-read. Trust subagent summaries.\n\n"
            "## Lessons learned\n\n"
        )
        with open(schema_path, "w", encoding="utf-8") as fh:
            fh.write(schema_content)
        created.append(schema_path)

    return created


if __name__ == "__main__":
    import sys

    root_path = sys.argv[1] if len(sys.argv) > 1 else "./knowledge-base"
    files = init_knowledge_base(root_path)
    print(f"Created {len(files)} items at {root_path}")
    for f in files:
        print(f"  {f}")
