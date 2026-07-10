# Brainstem

**Local-first knowledge backend for AI agents.**

Connect any MCP host — Claude Code, Cursor, or your own agent harness — to a
personal knowledge engine: an Obsidian-compatible wiki with indexed retrieval,
token-budgeted memory recall, and secure ingestion. Your agents get quality
context; you keep your knowledge on your machine.

```
┌────────────┐   MCP (stdio/sse)   ┌───────────────────────────────┐
│ Your agent │ ◄─────────────────► │ Brainstem                     │
│ (Claude    │   27 tools          │  · search + read + write wiki │
│  Code,     │   (12 read-only)    │  · memory tree recall (BM25 + │
│  Cursor…)  │                     │    vectors, token-budgeted)   │
└────────────┘                     │  · code graph over your repos │
                                   │  · vault = plain Markdown     │
                                   └───────────────────────────────┘
```

## Install

```sh
# Run without installing (recommended)
uvx brainstem-mcp init

# Or install
pip install brainstem-mcp
brainstem init
```

`brainstem init` creates a vault at `~/.brainstem/vault` (override with
`--root`) and prints the exact one-liner to connect your agent:

```sh
claude mcp add brainstem -- uvx brainstem-mcp mcp --root ~/.brainstem/vault
```

That's it. No API key is required to serve, search, and read — your agent
brings its own model.

## Docker

```sh
docker build -t brainstem .
docker run -i -v ~/.brainstem/vault:/vault brainstem            # stdio
docker run -p 8765:8765 -v ~/.brainstem/vault:/vault brainstem \
  mcp --root /vault --transport sse                             # sse
```

## What your agent can do

- **Search & read** — `search_wiki_index`, `read_wiki_file`, cross-references,
  orphan detection over a plain-Markdown Obsidian-compatible vault.
- **Remember & recall** — a hierarchical memory tree with BM25 + vector search
  and token-budgeted `memory_tree_recall`, built for agent context windows.
- **Write knowledge** — structured pages with YAML frontmatter, index upkeep,
  supersession instead of duplication.
- **Understand code** — ingest a repository and query its architecture
  (`ask_repo`, code-graph overview/impact) as part of your agent's context.

## Read-only profile

Expose only the 12 safe read tools (no writes, no publishing):

```sh
brainstem mcp --readonly
```

## Security posture

Brainstem treats everything it ingests as **untrusted input**: ingested content
cannot trigger writes or publishing on its own, and outbound actions are
draft-only by design. The vault is plain Markdown on your disk — no cloud, no
telemetry, single-tenant by architecture.

## Brainstem vs. OpenWiki

OpenWiki generates documentation that agents read as files. Brainstem is a
**live backend**: your agent queries an indexed knowledge base through MCP
tools at run time — retrieval, memory, and ingestion as part of the harness,
not a batch-generated artifact.

## License

MIT
