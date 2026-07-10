"""Fixtures for the Memory Tree v1 tests."""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path

import pytest_asyncio

from wiki_memory.content_store import ContentStore
from wiki_memory.tree_nodes import TreeNodeStore


@pytest_asyncio.fixture
async def content_store(tmp_path: Path) -> Iterator[ContentStore]:
    s = ContentStore(tmp_path / "content_store.db")
    await s.init()
    try:
        yield s
    finally:
        await s.close()


@pytest_asyncio.fixture
async def tree_store(tmp_path: Path) -> Iterator[TreeNodeStore]:
    s = TreeNodeStore(tmp_path / "tree_nodes.db")
    await s.init()
    try:
        yield s
    finally:
        await s.close()
