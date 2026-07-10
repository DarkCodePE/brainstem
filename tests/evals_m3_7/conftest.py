"""Fixtures for M3.7 property-based scenarios.

Each scenario builds its own populated content_store + tree_store via
``fresh_stack`` so they never share state — independence is critical
for deterministic assertions about retrieval invariants.
"""

from __future__ import annotations

from collections.abc import AsyncIterator
from pathlib import Path

import pytest_asyncio

from wiki_memory.content_store import ContentStore
from wiki_memory.tree_nodes import TreeNodeStore


@pytest_asyncio.fixture
async def fresh_stack(tmp_path: Path) -> AsyncIterator[tuple[ContentStore, TreeNodeStore]]:
    """A freshly initialised (content_store, tree_store) pair under
    tmp_path. Closed cleanly after the test so pytest can swap aiosqlite
    threads without leaking handles."""
    content = ContentStore(tmp_path / "content.db")
    tree = TreeNodeStore(tmp_path / "tree.db")
    await content.init()
    await tree.init()
    try:
        yield content, tree
    finally:
        await content.close()
        await tree.close()
