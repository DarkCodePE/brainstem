"""Tests for system prompt definitions."""

from wiki_agent.prompts import (
    INGEST_AGENT_PROMPT,
    LINT_AGENT_PROMPT,
    QUERY_AGENT_PROMPT,
    WIKI_ORCHESTRATOR_PROMPT,
)


class TestPrompts:
    def test_orchestrator_prompt_not_empty(self):
        assert len(WIKI_ORCHESTRATOR_PROMPT) > 100

    def test_ingest_prompt_not_empty(self):
        assert len(INGEST_AGENT_PROMPT) > 100

    def test_query_prompt_not_empty(self):
        assert len(QUERY_AGENT_PROMPT) > 100

    def test_lint_prompt_not_empty(self):
        assert len(LINT_AGENT_PROMPT) > 100

    def test_orchestrator_mentions_routing(self):
        assert "ingest" in WIKI_ORCHESTRATOR_PROMPT.lower()
        assert "query" in WIKI_ORCHESTRATOR_PROMPT.lower()
        assert "lint" in WIKI_ORCHESTRATOR_PROMPT.lower()

    def test_orchestrator_mentions_tools(self):
        assert "search_wiki_index" in WIKI_ORCHESTRATOR_PROMPT
        assert "append_to_log" in WIKI_ORCHESTRATOR_PROMPT

    def test_ingest_mentions_workflow_steps(self):
        prompt = INGEST_AGENT_PROMPT.lower()
        assert "summary" in prompt
        assert "entity" in prompt or "entities" in prompt
        assert "concept" in prompt
        assert "index" in prompt
        assert "log" in prompt

    def test_query_mentions_citations(self):
        assert "cit" in QUERY_AGENT_PROMPT.lower()

    def test_lint_mentions_orphan(self):
        assert "orphan" in LINT_AGENT_PROMPT.lower()

    def test_lint_mentions_frontmatter(self):
        assert "frontmatter" in LINT_AGENT_PROMPT.lower()
