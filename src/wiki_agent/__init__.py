"""Wiki Deep Agent -- an LLM-powered Obsidian-compatible knowledge base.

Usage::

    from wiki_agent import create_wiki_agent

    agent = create_wiki_agent(wiki_root="./knowledge-base")
    result = agent.invoke(
        {"messages": [{"role": "user", "content": "Ingest raw/articles/sample.md"}]},
        config={"configurable": {"thread_id": "demo"}},
    )
"""


def create_wiki_agent(
    wiki_root: str = "./knowledge-base",
    model: str = "claude-sonnet-4-5-20250929",
    supervised: bool = False,
    db_path: str | None = None,
):
    """Create the compiled Wiki Deep Agent graph.

    This is a lazy wrapper that defers the heavy imports (LangChain,
    LangGraph, Deep Agents) until the function is actually called,
    so lightweight operations like ``init`` and ``stats`` do not require
    external dependencies.
    """
    from wiki_agent.agent import create_wiki_agent as _create

    return _create(wiki_root=wiki_root, model=model, supervised=supervised, db_path=db_path)


__all__ = ["create_wiki_agent"]
__version__ = "0.1.0"
