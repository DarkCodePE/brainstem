from __future__ import annotations

import os
import sys
from dataclasses import dataclass, field
from pathlib import Path


def _env_path(name: str, default: str | None = None) -> Path | None:
    raw = os.environ.get(name, default)
    return Path(raw).expanduser().resolve() if raw else None


def _env_float(name: str, default: float) -> float:
    raw = os.environ.get(name)
    return float(raw) if raw else default


def _env_int(name: str, default: int) -> int:
    raw = os.environ.get(name)
    return int(raw) if raw else default


def _env_bool(name: str, default: bool) -> bool:
    raw = os.environ.get(name)
    if raw is None:
        return default
    return raw.strip().lower() in ("1", "true", "yes", "on")


@dataclass
class Config:
    wiki_root: Path
    raw_dir: Path
    ingested_dir: Path
    db_path: Path
    debounce_seconds: float = 3.0
    worker_pool_size: int = 2
    rate_limit_per_minute: int = 10
    max_file_size_mb: int = 25
    allowed_mime_prefixes: tuple[str, ...] = (
        "text/",
        "application/pdf",
        "application/json",
        "application/x-yaml",
    )
    mcp_command: tuple[str, ...] = field(default_factory=tuple)
    metrics_path: Path | None = None
    # M3 Sprint 2: AutoFetch worker wiring (PRD-006). Off by default until
    # M3 deploys it; the daemon stays a pure filesystem-watcher when these
    # are unset.
    autofetch_enabled: bool = False
    autofetch_interval_seconds: int = 1200
    autofetch_rate_limit_per_minute: int = 60

    # M3 Sprint 4: seal-on-ingest hook (PRD-004 wire-in). Off by default;
    # flip to True (or set WIKI_SEAL_ON_INGEST_ENABLED=1) so every
    # ingested page auto-chunks into content_store, creates a tree_source
    # node, and schedules a background seal via the LLM-backed
    # Summariser. Storage paths default to siblings of `db_path`.
    seal_on_ingest_enabled: bool = False
    memory_content_store_path: Path | None = None
    memory_tree_nodes_path: Path | None = None

    # ADR-035 D3: synthesis-on-ingest hook (the in-repo port of the
    # Hermes batch synthesis). Off by default; flip via
    # WIKI_SYNTHESIS_ENABLED=1. The router tier for the reword pass is
    # WIKI_SYNTHESIS_ROUTER_TIER (REASONING per the ADR; FAST for
    # cheap bulk backfills).
    synthesis_enabled: bool = False
    synthesis_router_tier: str = "REASONING"

    @classmethod
    def from_env(cls) -> Config:
        wiki_root = _env_path("WIKI_ROOT") or (Path.cwd() / "knowledge-base")
        raw_dir = _env_path("WIKI_RAW_DIR") or (wiki_root / "raw")
        ingested_dir = _env_path("WIKI_INGESTED_DIR") or (raw_dir / "_ingested")
        db_path = _env_path("INGEST_DB") or (wiki_root / ".wiki-ingest.db")
        metrics_path_raw = os.environ.get("INGEST_METRICS_PATH", "/tmp/wiki_ingest.prom")
        metrics_path = Path(metrics_path_raw) if metrics_path_raw else None

        mcp_cmd_raw = os.environ.get("WIKI_MCP_COMMAND")
        if mcp_cmd_raw:
            mcp_command = tuple(mcp_cmd_raw.split())
        else:
            mcp_command = (sys.executable, "-m", "wiki_agent.mcp_server")

        return cls(
            wiki_root=wiki_root,
            raw_dir=raw_dir,
            ingested_dir=ingested_dir,
            db_path=db_path,
            debounce_seconds=_env_float("INGEST_DEBOUNCE_SECONDS", 3.0),
            worker_pool_size=_env_int("INGEST_WORKER_POOL_SIZE", 2),
            rate_limit_per_minute=_env_int("INGEST_RATE_LIMIT_PER_MIN", 10),
            max_file_size_mb=_env_int("INGEST_MAX_FILE_SIZE_MB", 25),
            mcp_command=mcp_command,
            metrics_path=metrics_path,
            autofetch_enabled=_env_bool("WIKI_AUTOFETCH_ENABLED", False),
            autofetch_interval_seconds=_env_int("WIKI_AUTOFETCH_INTERVAL_SECONDS", 1200),
            autofetch_rate_limit_per_minute=_env_int("WIKI_AUTOFETCH_RATE_LIMIT_PER_MINUTE", 60),
            seal_on_ingest_enabled=_env_bool("WIKI_SEAL_ON_INGEST_ENABLED", False),
            memory_content_store_path=_env_path("WIKI_MEMORY_CONTENT_STORE_PATH"),
            memory_tree_nodes_path=_env_path("WIKI_MEMORY_TREE_NODES_PATH"),
            synthesis_enabled=_env_bool("WIKI_SYNTHESIS_ENABLED", False),
            synthesis_router_tier=os.environ.get("WIKI_SYNTHESIS_ROUTER_TIER", "REASONING"),
        )

    def __post_init__(self) -> None:
        if not isinstance(self.wiki_root, Path):
            self.wiki_root = Path(self.wiki_root)
        if not isinstance(self.raw_dir, Path):
            self.raw_dir = Path(self.raw_dir)
        if not isinstance(self.ingested_dir, Path):
            self.ingested_dir = Path(self.ingested_dir)
        if not isinstance(self.db_path, Path):
            self.db_path = Path(self.db_path)

        if self.debounce_seconds <= 0:
            raise ValueError("debounce_seconds must be > 0")
        if self.worker_pool_size < 1:
            raise ValueError("worker_pool_size must be >= 1")
        if self.rate_limit_per_minute < 1:
            raise ValueError("rate_limit_per_minute must be >= 1")
        if self.max_file_size_mb <= 0:
            raise ValueError("max_file_size_mb must be > 0")
        if not self.allowed_mime_prefixes:
            raise ValueError("allowed_mime_prefixes must not be empty")
        if not self.raw_dir.is_absolute():
            raise ValueError(f"raw_dir must be absolute: {self.raw_dir}")
        if self.autofetch_interval_seconds < 1:
            raise ValueError("autofetch_interval_seconds must be >= 1")
        if self.autofetch_rate_limit_per_minute < 1:
            raise ValueError("autofetch_rate_limit_per_minute must be >= 1")
        self.synthesis_router_tier = self.synthesis_router_tier.strip().upper()
        if self.synthesis_router_tier not in ("FAST", "REASONING", "VISION"):
            raise ValueError(
                f"synthesis_router_tier must be FAST|REASONING|VISION, "
                f"got {self.synthesis_router_tier!r}"
            )
        # M3-S4: default the memory store paths to siblings of db_path
        # if not explicitly set. Keeps the deployment surface clean —
        # turning on WIKI_SEAL_ON_INGEST_ENABLED is the only required
        # knob; storage paths follow automatically.
        if self.memory_content_store_path is None:
            self.memory_content_store_path = self.db_path.parent / "content_store.db"
        elif not isinstance(self.memory_content_store_path, Path):
            self.memory_content_store_path = Path(self.memory_content_store_path)
        if self.memory_tree_nodes_path is None:
            self.memory_tree_nodes_path = self.db_path.parent / "tree_nodes.db"
        elif not isinstance(self.memory_tree_nodes_path, Path):
            self.memory_tree_nodes_path = Path(self.memory_tree_nodes_path)
        if self.ingested_dir != self.raw_dir / "_ingested" and not str(
            self.ingested_dir
        ).startswith(str(self.raw_dir)):
            pass

    @property
    def max_file_size_bytes(self) -> int:
        return self.max_file_size_mb * 1024 * 1024
