"""Tests for claw.agent_core.agent — Agent loop, routing, direct dispatch."""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock



class TestUsageStats:
    """Test UsageStats dataclass."""

    def test_initial_values(self):
        from claw.agent_core.agent import UsageStats

        stats = UsageStats()
        assert stats.prompt_tokens == 0
        assert stats.completion_tokens == 0
        assert stats.llm_calls == 0
        assert stats.tokens_per_sec == 0.0

    def test_tokens_per_sec_calculation(self):
        from claw.agent_core.agent import UsageStats

        stats = UsageStats(completion_tokens=100, elapsed_s=2.0)
        assert stats.tokens_per_sec == 50.0

    def test_accumulate_adds_usage(self):
        from claw.agent_core.agent import UsageStats

        stats = UsageStats()
        usage = SimpleNamespace(prompt_tokens=10, completion_tokens=5, total_tokens=15)
        response = SimpleNamespace(usage=usage)
        stats.accumulate(response)
        assert stats.prompt_tokens == 10
        assert stats.completion_tokens == 5
        assert stats.llm_calls == 1

    def test_accumulate_handles_none_usage(self):
        from claw.agent_core.agent import UsageStats

        stats = UsageStats()
        response = SimpleNamespace(usage=None)
        stats.accumulate(response)
        assert stats.llm_calls == 1
        assert stats.prompt_tokens == 0

    def test_to_dict(self):
        from claw.agent_core.agent import UsageStats

        stats = UsageStats(prompt_tokens=100, completion_tokens=50, total_tokens=150,
                           llm_calls=2, elapsed_s=1.5)
        d = stats.to_dict()
        assert d["prompt_tokens"] == 100
        assert d["llm_calls"] == 2
        assert d["elapsed_s"] == 1.5
        assert d["tokens_per_sec"] == round(50 / 1.5, 1)


class TestAgentProcessUtterance:
    """Test the main agent loop."""

    async def test_simple_response_no_tools(self, settings, mock_llm, mock_retriever):
        from claw.agent_core.agent import Agent

        agent = Agent(llm=mock_llm, retriever=mock_retriever)
        result = await agent.process_utterance("Hello there")
        assert result == "Test response"
        assert agent.last_usage is not None
        assert agent.last_usage.llm_calls == 1

    async def test_timeout_returns_error_message(self, settings, mock_llm, mock_retriever):
        import asyncio
        from claw.agent_core.agent import Agent

        mock_llm.chat = AsyncMock(side_effect=asyncio.TimeoutError("timed out"))
        agent = Agent(llm=mock_llm, retriever=mock_retriever)
        result = await agent.process_utterance("test")
        assert "too long" in result.lower()
        assert agent.last_usage.timed_out is True

    async def test_tool_call_then_response(
        self, settings, mock_llm, mock_retriever, mock_tool_router,
        make_chat_response, make_tool_call,
    ):
        from claw.agent_core.agent import Agent

        # First call returns a tool call, second returns a text response
        tc = make_tool_call("tc_1", "get_weather", {"location": "Akron"})
        tool_response = make_chat_response(content=None, tool_calls=[tc])
        final_response = make_chat_response(content="The weather is sunny!")

        mock_llm.chat = AsyncMock(side_effect=[tool_response, final_response])

        agent = Agent(llm=mock_llm, retriever=mock_retriever, tool_router=mock_tool_router)
        result = await agent.process_utterance("What's the weather?", tools=[
            {"type": "function", "function": {"name": "get_weather", "description": "...", "parameters": {}}}
        ])
        assert result == "The weather is sunny!"
        mock_tool_router.call_tool.assert_called_once_with("get_weather", {"location": "Akron"})

    async def test_tool_call_error_handled(
        self, settings, mock_llm, mock_retriever, mock_tool_router,
        make_chat_response, make_tool_call,
    ):
        from claw.agent_core.agent import Agent

        tc = make_tool_call("tc_1", "broken_tool", {})
        tool_response = make_chat_response(content=None, tool_calls=[tc])
        final_response = make_chat_response(content="Something went wrong with the tool.")

        mock_tool_router.call_tool = AsyncMock(side_effect=RuntimeError("connection failed"))
        mock_llm.chat = AsyncMock(side_effect=[tool_response, final_response])

        agent = Agent(llm=mock_llm, retriever=mock_retriever, tool_router=mock_tool_router)
        result = await agent.process_utterance("test", tools=[
            {"type": "function", "function": {"name": "broken_tool", "description": "...", "parameters": {}}}
        ])
        assert result == "Something went wrong with the tool."

    async def test_no_tool_router_returns_message(
        self, settings, mock_llm, mock_retriever,
        make_chat_response, make_tool_call,
    ):
        from claw.agent_core.agent import Agent

        tc = make_tool_call("tc_1", "some_tool", {})
        tool_response = make_chat_response(content=None, tool_calls=[tc])
        final_response = make_chat_response(content="No tools available")

        mock_llm.chat = AsyncMock(side_effect=[tool_response, final_response])

        agent = Agent(llm=mock_llm, retriever=mock_retriever, tool_router=None)
        result = await agent.process_utterance("test", tools=[
            {"type": "function", "function": {"name": "some_tool", "description": "...", "parameters": {}}}
        ])
        assert result == "No tools available"


class TestSessionManagement:
    """Test session creation, expiry, and reset."""

    async def test_new_session_created_on_first_call(self, settings, mock_llm, mock_retriever):
        from claw.agent_core.agent import Agent

        agent = Agent(llm=mock_llm, retriever=mock_retriever)
        assert agent.session is None
        await agent.process_utterance("Hello")
        assert agent.session is not None

    async def test_session_persists_across_calls(self, settings, mock_llm, mock_retriever):
        from claw.agent_core.agent import Agent

        agent = Agent(llm=mock_llm, retriever=mock_retriever)
        await agent.process_utterance("Hello")
        session_id = agent.session.session_id
        await agent.process_utterance("How are you?")
        assert agent.session.session_id == session_id

    def test_new_session_clears_state(self, settings, mock_llm, mock_retriever):
        from claw.agent_core.agent import Agent

        agent = Agent(llm=mock_llm, retriever=mock_retriever)
        agent._session = MagicMock()
        agent.new_session()
        assert agent.session is None

    async def test_session_auto_rotates_on_timeout(self, settings, mock_llm, mock_retriever):
        import time
        from claw.agent_core.agent import Agent

        agent = Agent(llm=mock_llm, retriever=mock_retriever)
        await agent.process_utterance("Hello")
        old_id = agent.session.session_id

        # Simulate a very old last interaction
        agent._last_interaction = time.monotonic() - 100000
        await agent.process_utterance("Back again")
        assert agent.session.session_id != old_id


class TestKeywordRouting:
    """Test the _keyword_route fast path."""

    def _make_agent(self, settings, mock_llm, mock_retriever):
        from claw.agent_core.agent import Agent

        return Agent(llm=mock_llm, retriever=mock_retriever)

    def test_play_routes_to_play_song(self, settings, mock_llm, mock_retriever):
        agent = self._make_agent(settings, mock_llm, mock_retriever)
        tools = [{"type": "function", "function": {"name": "play_song", "description": "...", "parameters": {}}}]
        result = agent._keyword_route("play some rock music", tools)
        assert len(result) == 1
        assert result[0]["function"]["name"] == "play_song"

    def test_weather_routes_to_get_weather(self, settings, mock_llm, mock_retriever):
        agent = self._make_agent(settings, mock_llm, mock_retriever)
        tools = [{"type": "function", "function": {"name": "get_weather", "description": "...", "parameters": {}}}]
        result = agent._keyword_route("what's the weather like?", tools)
        assert result[0]["function"]["name"] == "get_weather"

    def test_conversational_returns_empty_list(self, settings, mock_llm, mock_retriever):
        agent = self._make_agent(settings, mock_llm, mock_retriever)
        tools = [{"type": "function", "function": {"name": "get_weather", "description": "...", "parameters": {}}}]
        result = agent._keyword_route("tell me a joke", tools)
        assert result == []

    def test_no_match_returns_none(self, settings, mock_llm, mock_retriever):
        agent = self._make_agent(settings, mock_llm, mock_retriever)
        tools = [{"type": "function", "function": {"name": "get_weather", "description": "...", "parameters": {}}}]
        result = agent._keyword_route("do something unusual and unique", tools)
        assert result is None

    def test_tool_not_available_returns_none(self, settings, mock_llm, mock_retriever):
        agent = self._make_agent(settings, mock_llm, mock_retriever)
        # Tools list does not include play_song
        tools = [{"type": "function", "function": {"name": "get_weather", "description": "...", "parameters": {}}}]
        result = agent._keyword_route("play me a song", tools)
        assert result is None


class TestDirectDispatch:
    """Test _try_direct_dispatch for simple commands."""

    async def test_pause_dispatch(self, settings, mock_llm, mock_retriever, mock_tool_router):
        from claw.agent_core.agent import Agent

        agent = Agent(llm=mock_llm, retriever=mock_retriever, tool_router=mock_tool_router)
        await agent._try_direct_dispatch("pause")
        mock_tool_router.call_tool.assert_called_with("pause", {})

    async def test_play_song_with_artist(self, settings, mock_llm, mock_retriever, mock_tool_router):
        from claw.agent_core.agent import Agent

        agent = Agent(llm=mock_llm, retriever=mock_retriever, tool_router=mock_tool_router)
        await agent._try_direct_dispatch("play bohemian rhapsody by queen")
        mock_tool_router.call_tool.assert_called_with(
            "play_song", {"title": "bohemian rhapsody", "artist": "queen"}
        )

    async def test_volume_set_dispatch(self, settings, mock_llm, mock_retriever, mock_tool_router):
        from claw.agent_core.agent import Agent

        agent = Agent(llm=mock_llm, retriever=mock_retriever, tool_router=mock_tool_router)
        await agent._try_direct_dispatch("volume 50")
        mock_tool_router.call_tool.assert_called_with("set_volume", {"level": 50})

    async def test_no_router_returns_none(self, settings, mock_llm, mock_retriever):
        from claw.agent_core.agent import Agent

        agent = Agent(llm=mock_llm, retriever=mock_retriever, tool_router=None)
        result = await agent._try_direct_dispatch("pause")
        assert result is None

    async def test_unmatched_text_returns_none(self, settings, mock_llm, mock_retriever, mock_tool_router):
        from claw.agent_core.agent import Agent

        agent = Agent(llm=mock_llm, retriever=mock_retriever, tool_router=mock_tool_router)
        result = await agent._try_direct_dispatch("tell me about the history of AI")
        assert result is None


class TestExtractFacts:
    """Test post-conversation fact extraction."""

    async def test_extract_facts_no_session(self, settings, mock_llm, mock_retriever):
        from claw.agent_core.agent import Agent

        agent = Agent(llm=mock_llm, retriever=mock_retriever)
        facts = await agent.extract_facts()
        assert facts == []

    async def test_extract_facts_delegates_to_retriever(self, settings, mock_llm, mock_retriever):
        from claw.agent_core.agent import Agent

        agent = Agent(llm=mock_llm, retriever=mock_retriever)
        # Create a session with some content
        await agent.process_utterance("My name is Chuck")

        mock_retriever.extract_and_store_facts = AsyncMock(return_value=["User's name is Chuck"])
        facts = await agent.extract_facts()
        assert facts == ["User's name is Chuck"]
