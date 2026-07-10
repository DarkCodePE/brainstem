"""
Tests for the seal-on-ingest wire-in inside `wiki_ingest.composition`.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from wiki_ingest.composition import build_daemon, build_seal_on_ingest_hook
from wiki_ingest.config import Config


def _make_cfg(tmp_path: Path, *, seal_enabled: bool = False) -> Config:
    wiki_root = tmp_path / "knowledge-base"
    raw = wiki_root / "raw"
    ingested = raw / "_ingested"
    db = tmp_path / "ingest.db"
    for d in (wiki_root, raw, ingested, wiki_root / "wiki" / "trees"):
        d.mkdir(parents=True, exist_ok=True)
    return Config(
        wiki_root=wiki_root,
        raw_dir=raw,
        ingested_dir=ingested,
        db_path=db,
        debounce_seconds=3.0,
        worker_pool_size=2,
        rate_limit_per_minute=10,
        max_file_size_mb=25,
        mcp_command=("python", "-m", "wiki_agent.mcp_server"),
        seal_on_ingest_enabled=seal_enabled,
    )


class TestSealHookFactory:
    @pytest.mark.asyncio
    async def test_build_seal_on_ingest_hook_returns_hook_and_resources(
        self, tmp_path: Path
    ) -> None:
        cfg = _make_cfg(tmp_path)
        hook, resources = await build_seal_on_ingest_hook(cfg)
        try:
            assert hook is not None
            assert len(resources) == 2  # content_store + tree_store
            assert all(hasattr(r, "close") for r in resources)
        finally:
            for r in resources:
                await r.close()

    @pytest.mark.asyncio
    async def test_default_paths_are_siblings_of_db_path(self, tmp_path: Path) -> None:
        cfg = _make_cfg(tmp_path)
        assert cfg.memory_content_store_path == cfg.db_path.parent / "content_store.db"
        assert cfg.memory_tree_nodes_path == cfg.db_path.parent / "tree_nodes.db"

    @pytest.mark.asyncio
    async def test_storage_files_materialise(self, tmp_path: Path) -> None:
        cfg = _make_cfg(tmp_path)
        hook, resources = await build_seal_on_ingest_hook(cfg)
        try:
            assert cfg.memory_content_store_path.exists()
            assert cfg.memory_tree_nodes_path.exists()
        finally:
            for r in resources:
                await r.close()


class TestBuildDaemonWithSealHook:
    @pytest.mark.asyncio
    async def test_daemon_without_seal_hook_when_disabled(self, tmp_path: Path) -> None:
        cfg = _make_cfg(tmp_path, seal_enabled=False)
        daemon = await build_daemon(cfg)
        assert daemon.pool._post_write_hook is None
        await daemon.queue.close()

    @pytest.mark.asyncio
    async def test_daemon_with_seal_hook_when_enabled(self, tmp_path: Path) -> None:
        cfg = _make_cfg(tmp_path, seal_enabled=True)
        daemon = await build_daemon(cfg)
        try:
            assert daemon.pool._post_write_hook is not None
        finally:
            await daemon.queue.close()

    @pytest.mark.asyncio
    async def test_enable_seal_hook_kwarg_overrides_config(self, tmp_path: Path) -> None:
        cfg = _make_cfg(tmp_path, seal_enabled=False)
        daemon = await build_daemon(cfg, enable_seal_hook=True)
        try:
            assert daemon.pool._post_write_hook is not None
        finally:
            await daemon.queue.close()

    @pytest.mark.asyncio
    async def test_explicit_disable_kwarg_overrides_config(self, tmp_path: Path) -> None:
        cfg = _make_cfg(tmp_path, seal_enabled=True)
        daemon = await build_daemon(cfg, enable_seal_hook=False)
        try:
            assert daemon.pool._post_write_hook is None
        finally:
            await daemon.queue.close()


class TestConfigEnv:
    def test_env_var_flips_seal_enabled(self, monkeypatch, tmp_path: Path) -> None:
        monkeypatch.setenv("WIKI_ROOT", str(tmp_path / "knowledge-base"))
        monkeypatch.setenv("WIKI_RAW_DIR", str(tmp_path / "knowledge-base" / "raw"))
        monkeypatch.setenv("INGEST_DB", str(tmp_path / "ingest.db"))
        monkeypatch.setenv("WIKI_SEAL_ON_INGEST_ENABLED", "1")
        (tmp_path / "knowledge-base" / "raw").mkdir(parents=True, exist_ok=True)
        cfg = Config.from_env()
        assert cfg.seal_on_ingest_enabled is True

    def test_env_var_default_off(self, monkeypatch, tmp_path: Path) -> None:
        monkeypatch.setenv("WIKI_ROOT", str(tmp_path / "knowledge-base"))
        monkeypatch.setenv("WIKI_RAW_DIR", str(tmp_path / "knowledge-base" / "raw"))
        monkeypatch.setenv("INGEST_DB", str(tmp_path / "ingest.db"))
        monkeypatch.delenv("WIKI_SEAL_ON_INGEST_ENABLED", raising=False)
        (tmp_path / "knowledge-base" / "raw").mkdir(parents=True, exist_ok=True)
        cfg = Config.from_env()
        assert cfg.seal_on_ingest_enabled is False

    def test_memory_paths_override_via_env(self, monkeypatch, tmp_path: Path) -> None:
        custom_content = tmp_path / "custom" / "content.db"
        custom_tree = tmp_path / "custom" / "tree.db"
        monkeypatch.setenv("WIKI_ROOT", str(tmp_path / "knowledge-base"))
        monkeypatch.setenv("WIKI_RAW_DIR", str(tmp_path / "knowledge-base" / "raw"))
        monkeypatch.setenv("INGEST_DB", str(tmp_path / "ingest.db"))
        monkeypatch.setenv("WIKI_MEMORY_CONTENT_STORE_PATH", str(custom_content))
        monkeypatch.setenv("WIKI_MEMORY_TREE_NODES_PATH", str(custom_tree))
        (tmp_path / "knowledge-base" / "raw").mkdir(parents=True, exist_ok=True)
        cfg = Config.from_env()
        assert cfg.memory_content_store_path == custom_content
        assert cfg.memory_tree_nodes_path == custom_tree
