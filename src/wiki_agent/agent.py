"""Main agent factory for the Wiki Deep Agent.

Provides ``create_wiki_agent()`` which assembles a Deep Agents
orchestrator with six specialised subagents and returns a compiled
LangGraph graph ready to invoke.

The orchestrator delegates via the built-in ``task`` tool provided by
SubAgentMiddleware.  Each subagent receives a restricted set of
domain-specific wiki tools.
"""

from __future__ import annotations

import logging
import os
from typing import TYPE_CHECKING

from deepagents import create_deep_agent
from deepagents.backends import FilesystemBackend
from langchain_core.language_models.chat_models import BaseChatModel
from langgraph.checkpoint.memory import MemorySaver
from langgraph.store.memory import InMemoryStore

from wiki_agent.middleware import default_hook_middleware
from wiki_agent.prompts import (
    CAPTURE_AGENT_PROMPT,
    INDEX_AGENT_PROMPT,
    INGEST_AGENT_PROMPT,
    LINT_AGENT_PROMPT,
    QUERY_AGENT_PROMPT,
    REVIEW_AGENT_PROMPT,
    WIKI_ORCHESTRATOR_PROMPT,
)
from wiki_agent.state import (
    CaptureResult,
    IndexResult,
    IngestResult,
    LintResult,
    QueryResult,
    ReviewResult,
)
from wiki_agent.tools import create_tools
from wiki_memory.summariser import Summariser
from wiki_memory.summariser_factory import build_default_summariser

# Optional: SQLite checkpointer for persistent sessions
try:
    from langgraph.checkpoint.sqlite import SqliteSaver

    _SQLITE_AVAILABLE = True
except ImportError:
    _SQLITE_AVAILABLE = False

if TYPE_CHECKING:
    from langgraph.graph.state import CompiledStateGraph

logger = logging.getLogger(__name__)


def _build_llm(model: str, temperature: float = 0.3, max_tokens: int = 8192) -> BaseChatModel:
    """Build an LLM instance with provider auto-detection and fallback.

    Provider priority:
      1. OpenRouter (if OPENROUTER_API_KEY is set)
      2. Ollama (if running locally)
      3. Anthropic (if ANTHROPIC_API_KEY is set)

    The ``model`` parameter can include a provider prefix to force a
    specific backend:  ``openrouter:model/name``, ``ollama:model-name``,
    or ``anthropic:model-name``.
    """
    # Explicit provider prefix
    if ":" in model and model.split(":")[0] in ("openrouter", "ollama", "anthropic", "local"):
        provider, model_name = model.split(":", 1)
    else:
        provider = None
        model_name = model

    # --- Local llama-server / any OpenAI-compatible endpoint ---
    if provider == "local":
        try:
            from langchain_openai import ChatOpenAI

            base_url = os.environ.get("LOCAL_LLM_BASE_URL", "http://localhost:8001/v1")
            local_model = os.environ.get("LOCAL_LLM_MODEL", model_name)
            logger.info("Using local LLM at %s with model: %s", base_url, local_model)
            return ChatOpenAI(
                model=local_model,
                openai_api_key="sk-local",
                openai_api_base=base_url,
                temperature=temperature,
                max_tokens=max_tokens,
            )
        except ImportError:
            logger.warning("langchain-openai not installed, cannot use local provider")
        except Exception as exc:
            logger.warning("Local LLM init failed: %s", exc)

    # --- Try OpenRouter first ---
    openrouter_key = os.environ.get("OPENROUTER_API_KEY")
    if provider in (None, "openrouter") and openrouter_key:
        try:
            from langchain_openai import ChatOpenAI

            openrouter_model = os.environ.get("OPENROUTER_MODEL", model_name)
            logger.info("Using OpenRouter with model: %s", openrouter_model)
            return ChatOpenAI(
                model=openrouter_model,
                openai_api_key=openrouter_key,
                openai_api_base="https://openrouter.ai/api/v1",
                temperature=temperature,
                max_tokens=max_tokens,
            )
        except ImportError:
            logger.warning("langchain-openai not installed, skipping OpenRouter")
        except Exception as exc:
            logger.warning("OpenRouter init failed: %s", exc)

    # --- Try Ollama as fallback ---
    if provider in (None, "ollama"):
        try:
            from langchain_ollama import ChatOllama

            if provider == "ollama":
                ollama_model = model_name
            else:
                ollama_model = os.environ.get("OLLAMA_MODEL", "qwen3.5")
            logger.info("Using Ollama with model: %s", ollama_model)
            return ChatOllama(
                model=ollama_model,
                temperature=temperature,
                num_predict=max_tokens,
            )
        except ImportError:
            logger.warning("langchain-ollama not installed, skipping Ollama")
        except Exception as exc:
            logger.warning("Ollama init failed: %s", exc)

    # --- Anthropic as final fallback ---
    if provider in (None, "anthropic"):
        from langchain_anthropic import ChatAnthropic

        logger.info("Using Anthropic with model: %s", model_name)
        return ChatAnthropic(
            model=model_name,
            temperature=temperature,
            max_tokens=max_tokens,
        )

    raise RuntimeError(
        f"No LLM provider available for model={model}. "
        "Set OPENROUTER_API_KEY, start Ollama, or set ANTHROPIC_API_KEY."
    )


def create_wiki_agent(
    wiki_root: str = "./knowledge-base",
    model: str = "claude-sonnet-4-5-20250929",
    supervised: bool = False,
    db_path: str | None = None,
    summariser: Summariser | None = None,
) -> CompiledStateGraph:
    """Create and return the compiled Wiki Deep Agent.

    Uses the Deep Agents framework (``create_deep_agent``) with six
    specialised subagents registered via SubAgentMiddleware.  The
    orchestrator delegates using the built-in ``task`` tool.

    Args:
        wiki_root: Path to the knowledge-base directory on disk.
        model: Model identifier (supports provider prefixes).
        supervised: If ``True``, enable human-in-the-loop interrupts
            before wiki write operations.
        db_path: Path to SQLite database for session persistence.
            Defaults to ``<wiki_root>/.wiki-agent.db``.
        summariser: Optional ``Summariser`` to plumb into downstream
            seal workers. When ``None`` (default), the factory at
            ``wiki_memory.summariser_factory.build_default_summariser``
            picks the right Summariser from the environment: an
            LLM-backed ``RouterSummariser`` wrapped in a
            ``CompositeSummariser`` with ``NullSummariser`` as the
            deterministic fallback when provider keys are configured,
            or ``NullSummariser`` directly when they aren't. The
            current orchestrator does not construct a ``SealWorker``
            itself — that lives in the daemon and the ingest pipeline
            — but the resolved Summariser is stashed in the resulting
            graph metadata for callers that build a seal worker on
            top (``wiki_memory.build_default_seal_worker``).

    Returns:
        A compiled LangGraph graph ready for ``.invoke()`` or
        ``.stream()`` calls.
    """
    # Resolve the Summariser early so any construction error surfaces
    # before we burn the heavier LLM/checkpointer init below.
    # Real LLM dispatch does not happen here — the factory only
    # constructs objects (see ``build_default_summariser`` docstring).
    resolved_summariser = summariser if summariser is not None else build_default_summariser()
    logger.info(
        "Seal-path summariser resolved to %s",
        type(resolved_summariser).__name__,
    )

    llm = _build_llm(model=model, temperature=0.3, max_tokens=8192)

    # Build domain-specific wiki tools
    all_tools = create_tools(wiki_root)
    tool_by_name = {t.name: t for t in all_tools}

    # Tool subsets per subagent (restricted access)
    ingest_tools = [
        tool_by_name["search_wiki_index"],
        tool_by_name["append_to_log"],
        tool_by_name["validate_frontmatter"],
        tool_by_name["read_wiki_file"],
        tool_by_name["write_page"],
        tool_by_name["update_index_entry"],
        tool_by_name["update_schema_lessons"],
    ]
    query_tools = [
        tool_by_name["search_wiki_index"],
        tool_by_name["read_wiki_file"],
        tool_by_name["write_page"],
    ]
    lint_tools = [
        tool_by_name["detect_orphan_pages"],
        tool_by_name["find_cross_references"],
        tool_by_name["validate_frontmatter"],
        tool_by_name["get_wiki_stats"],
        tool_by_name["read_wiki_file"],
        tool_by_name["write_page"],
    ]
    index_tools = [
        tool_by_name["read_wiki_file"],
        tool_by_name["write_page"],
        tool_by_name["update_index_entry"],
        tool_by_name["find_cross_references"],
        tool_by_name["get_wiki_stats"],
    ]
    capture_tools = [
        tool_by_name["read_wiki_file"],
        tool_by_name["write_page"],
        tool_by_name["append_to_log"],
    ]
    review_tools = [
        tool_by_name["read_wiki_file"],
        tool_by_name["write_page"],
        tool_by_name["search_wiki_index"],
        tool_by_name["append_to_log"],
        tool_by_name["graduate_observation"],
    ]

    # Checkpointer: SQLite for persistence, MemorySaver as fallback
    if _SQLITE_AVAILABLE:
        import sqlite3

        _db = db_path or os.path.join(wiki_root, ".wiki-agent.db")
        checkpointer = SqliteSaver(sqlite3.connect(_db, check_same_thread=False))
        logger.info("Session persistence: SQLite at %s", _db)
    else:
        checkpointer = MemorySaver()
        logger.info(
            "Session persistence: in-memory (install langgraph-checkpoint-sqlite for durable sessions)"
        )

    # HITL configuration
    interrupt_config: dict = {}
    if supervised:
        interrupt_config["interrupt_on"] = {"write_page": True}

    # Skills directory
    skills_dir = os.path.join(os.path.dirname(__file__), "skills")

    # Build the Deep Agent orchestrator
    #
    # `middleware=default_hook_middleware()` ports the legacy Claude Code
    # hooks (hooks/safety-gate.sh, hooks/context-updater.sh) into the
    # harness so the same observation/safety behaviour fires when the
    # agent runs outside Claude Code (CLI, MCP SSE, batch). Set
    # WIKI_DISABLE_HOOKS=1 to opt out (see wiki_agent.middleware). Issue #25.
    orchestrator = create_deep_agent(
        name="wiki-orchestrator",
        model=llm,
        tools=all_tools,
        middleware=default_hook_middleware(),
        system_prompt=WIKI_ORCHESTRATOR_PROMPT,
        subagents=[
            {
                "name": "ingest-agent",
                "description": "Process source documents into wiki pages with summaries, entities, and concepts",
                "system_prompt": INGEST_AGENT_PROMPT,
                "tools": ingest_tools,
                "response_format": IngestResult,
            },
            {
                "name": "query-agent",
                "description": "Answer questions using wiki knowledge base with citations",
                "system_prompt": QUERY_AGENT_PROMPT,
                "tools": query_tools,
                "response_format": QueryResult,
            },
            {
                "name": "lint-agent",
                "description": "Health-check the wiki for orphans, broken links, stale content, contradictions",
                "system_prompt": LINT_AGENT_PROMPT,
                "tools": lint_tools,
                "response_format": LintResult,
            },
            {
                "name": "index-agent",
                "description": "Maintain wiki index, backlinks, and structural integrity",
                "system_prompt": INDEX_AGENT_PROMPT,
                "tools": index_tools,
                "response_format": IndexResult,
            },
            {
                "name": "capture-agent",
                "description": "Record TIL observations into the wiki learning loop",
                "system_prompt": CAPTURE_AGENT_PROMPT,
                "tools": capture_tools,
                "response_format": CaptureResult,
            },
            {
                "name": "review-agent",
                "description": "Review observations, cluster by theme, propose graduations",
                "system_prompt": REVIEW_AGENT_PROMPT,
                "tools": review_tools,
                "response_format": ReviewResult,
            },
        ],
        backend=FilesystemBackend(root_dir=wiki_root, virtual_mode=True),
        skills=[skills_dir] if os.path.isdir(skills_dir) else [],
        checkpointer=checkpointer,
        store=InMemoryStore(),
        **interrupt_config,
    )

    logger.info(
        "Wiki Deep Agent created: wiki_root=%s, model=%s, supervised=%s, sqlite=%s",
        wiki_root,
        model,
        supervised,
        _SQLITE_AVAILABLE,
    )

    return orchestrator
