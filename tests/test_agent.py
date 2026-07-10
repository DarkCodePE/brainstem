"""Tests for the wiki agent factory.

These tests verify the agent creation without making real LLM calls.
"""

from unittest.mock import MagicMock, patch


class TestCreateWikiAgent:
    @patch("wiki_agent.agent._build_llm")
    def test_returns_compiled_graph(self, mock_build_llm, tmp_wiki_root):
        mock_llm = MagicMock()
        mock_llm.bind_tools = MagicMock(return_value=mock_llm)
        mock_build_llm.return_value = mock_llm

        from wiki_agent.agent import create_wiki_agent

        agent = create_wiki_agent(wiki_root=tmp_wiki_root)
        # Should return a compiled graph with invoke method
        assert hasattr(agent, "invoke")
        assert hasattr(agent, "stream")
        assert hasattr(agent, "get_state")

    @patch("wiki_agent.agent._build_llm")
    def test_supervised_mode(self, mock_build_llm, tmp_wiki_root):
        mock_llm = MagicMock()
        mock_llm.bind_tools = MagicMock(return_value=mock_llm)
        mock_build_llm.return_value = mock_llm

        from wiki_agent.agent import create_wiki_agent

        agent = create_wiki_agent(wiki_root=tmp_wiki_root, supervised=True)
        assert hasattr(agent, "invoke")

    @patch("wiki_agent.agent._build_llm")
    def test_custom_model(self, mock_build_llm, tmp_wiki_root):
        mock_llm = MagicMock()
        mock_llm.bind_tools = MagicMock(return_value=mock_llm)
        mock_build_llm.return_value = mock_llm

        from wiki_agent.agent import create_wiki_agent

        create_wiki_agent(wiki_root=tmp_wiki_root, model="claude-haiku-4-5-20251001")
        mock_build_llm.assert_called_with(
            model="claude-haiku-4-5-20251001",
            temperature=0.3,
            max_tokens=8192,
        )
