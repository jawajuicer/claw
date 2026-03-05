"""Tests for claw.memory_engine.retriever — MemoryRetriever."""

from __future__ import annotations

from unittest.mock import AsyncMock



class TestRetrieveContext:
    """Test context retrieval from memory collections."""

    def test_empty_memory_returns_empty_string(self, settings, mock_retriever):
        result = mock_retriever.retrieve_context("hello")
        assert result == ""

    def test_facts_included_in_context(self, settings, mock_retriever):
        mock_retriever.store.query_facts.return_value = [
            {"document": "User's name is Chuck", "distance": 0.1},
        ]
        result = mock_retriever.retrieve_context("What is my name?")
        assert "Chuck" in result
        assert "Known facts" in result

    def test_conversations_included_in_context(self, settings, mock_retriever):
        mock_retriever.store.query_conversations.return_value = [
            {"document": "[user] I like pizza", "distance": 0.2},
        ]
        result = mock_retriever.retrieve_context("What do I like?")
        assert "pizza" in result
        assert "past conversations" in result.lower()

    def test_context_truncated_to_max_chars(self, settings, mock_retriever):
        mock_retriever.store.query_facts.return_value = [
            {"document": "x" * 1000, "distance": 0.1},
        ]
        result = mock_retriever.retrieve_context("query", max_chars=200)
        assert len(result) <= 200 + len("\n--- End Memory ---")

    def test_both_facts_and_conversations(self, settings, mock_retriever):
        mock_retriever.store.query_facts.return_value = [
            {"document": "Fact one", "distance": 0.1},
        ]
        mock_retriever.store.query_conversations.return_value = [
            {"document": "Convo one", "distance": 0.2},
        ]
        result = mock_retriever.retrieve_context("test")
        assert "Fact one" in result
        assert "Convo one" in result


class TestStoreConversationTurn:
    """Test storing conversation turns in memory."""

    def test_stores_user_turn(self, settings, mock_retriever):
        mock_retriever.store_conversation_turn("user", "Hello world", "session123")
        mock_retriever.store.add_conversation.assert_called_once()
        call_args = mock_retriever.store.add_conversation.call_args
        assert "[user] Hello world" in call_args.kwargs.get("text", call_args[1].get("text", ""))

    def test_stores_assistant_turn(self, settings, mock_retriever):
        mock_retriever.store_conversation_turn("assistant", "Hi there!", "session123")
        call_args = mock_retriever.store.add_conversation.call_args
        text = call_args.kwargs.get("text", call_args[1].get("text", ""))
        assert "[assistant] Hi there!" in text

    def test_metadata_includes_session_id(self, settings, mock_retriever):
        mock_retriever.store_conversation_turn("user", "test", "sess42")
        call_args = mock_retriever.store.add_conversation.call_args
        metadata = call_args.kwargs.get("metadata", call_args[1].get("metadata", {}))
        assert metadata["session_id"] == "sess42"
        assert metadata["role"] == "user"
        assert "timestamp" in metadata


class TestExtractAndStoreFacts:
    """Test LLM-based fact extraction."""

    async def test_no_facts_when_llm_returns_none(self, settings, mock_retriever, mock_llm):
        mock_llm.chat_simple = AsyncMock(return_value="NONE")
        facts = await mock_retriever.extract_and_store_facts("user: Hello\nassistant: Hi", mock_llm)
        assert facts == []

    async def test_extracts_and_stores_facts(self, settings, mock_retriever, mock_llm):
        mock_llm.chat_simple = AsyncMock(return_value="- User's name is Chuck\n- User likes pizza")
        facts = await mock_retriever.extract_and_store_facts(
            "user: My name is Chuck and I like pizza\nassistant: Nice!",
            mock_llm,
        )
        assert len(facts) == 2
        assert "Chuck" in facts[0]
        assert "pizza" in facts[1]
        assert mock_retriever.store.add_fact.call_count == 2

    async def test_llm_error_returns_empty_list(self, settings, mock_retriever, mock_llm):
        mock_llm.chat_simple = AsyncMock(side_effect=RuntimeError("LLM down"))
        facts = await mock_retriever.extract_and_store_facts("conversation text", mock_llm)
        assert facts == []

    async def test_empty_response_returns_empty(self, settings, mock_retriever, mock_llm):
        mock_llm.chat_simple = AsyncMock(return_value="")
        facts = await mock_retriever.extract_and_store_facts("conversation text", mock_llm)
        assert facts == []

    async def test_strips_dash_prefix_from_facts(self, settings, mock_retriever, mock_llm):
        mock_llm.chat_simple = AsyncMock(return_value="- Fact one\n- Fact two")
        facts = await mock_retriever.extract_and_store_facts("conversation", mock_llm)
        assert facts[0] == "Fact one"
        assert facts[1] == "Fact two"


class TestGetStats:
    """Test stats delegation."""

    def test_get_stats_delegates_to_store(self, settings, mock_retriever):
        mock_retriever.store.stats.return_value = {"conversations": 10, "facts": 5, "categories": 2}
        stats = mock_retriever.get_stats()
        assert stats == {"conversations": 10, "facts": 5, "categories": 2}
