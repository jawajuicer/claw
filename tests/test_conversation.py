"""Tests for claw.agent_core.conversation — ConversationSession."""

from __future__ import annotations

from unittest.mock import AsyncMock

import pytest




class TestConversationSession:
    """Core session lifecycle and message management."""

    def test_session_id_generated(self, settings):
        from claw.agent_core.conversation import ConversationSession

        session = ConversationSession()
        assert len(session.session_id) == 12

    def test_initialize_sets_system_prompt(self, settings):
        from claw.agent_core.conversation import ConversationSession

        session = ConversationSession()
        session.initialize()
        msgs = session.get_messages()
        assert len(msgs) == 1
        assert msgs[0]["role"] == "system"
        assert "The Claw" in msgs[0]["content"]

    def test_initialize_appends_memory_context(self, settings):
        from claw.agent_core.conversation import ConversationSession

        session = ConversationSession()
        session.initialize(memory_context="User likes cats")
        msgs = session.get_messages()
        assert "User likes cats" in msgs[0]["content"]

    def test_add_user_and_assistant(self, settings):
        from claw.agent_core.conversation import ConversationSession

        session = ConversationSession()
        session.initialize()
        session.add_user("Hello")
        session.add_assistant("Hi there!")
        msgs = session.get_messages()
        assert msgs[1]["role"] == "user"
        assert msgs[1]["content"] == "Hello"
        assert msgs[2]["role"] == "assistant"
        assert msgs[2]["content"] == "Hi there!"

    def test_add_tool_result(self, settings):
        from claw.agent_core.conversation import ConversationSession

        session = ConversationSession()
        session.initialize()
        session.add_tool_result("call_123", "weather is sunny")
        msgs = session.get_messages()
        assert msgs[-1]["role"] == "tool"
        assert msgs[-1]["tool_call_id"] == "call_123"
        assert msgs[-1]["content"] == "weather is sunny"

    def test_get_messages_returns_copy(self, settings):
        from claw.agent_core.conversation import ConversationSession

        session = ConversationSession()
        session.initialize()
        msgs = session.get_messages()
        msgs.append({"role": "user", "content": "extra"})
        assert len(session.get_messages()) == 1  # original unchanged


class TestGetUserAssistantText:
    """Test fact extraction text generation."""

    def test_extracts_only_user_and_assistant(self, settings):
        from claw.agent_core.conversation import ConversationSession

        session = ConversationSession()
        session.initialize()
        session.add_user("My name is Chuck")
        session.add_assistant("Nice to meet you, Chuck!")
        session.add_tool_result("id1", "some tool output")
        text = session.get_user_assistant_text()
        assert "user: My name is Chuck" in text
        assert "assistant: Nice to meet you, Chuck!" in text
        assert "tool" not in text.lower().split("\n")[-1] if "tool" in text else True

    def test_empty_session_returns_empty_string(self, settings):
        from claw.agent_core.conversation import ConversationSession

        session = ConversationSession()
        session.initialize()
        assert session.get_user_assistant_text() == ""


class TestTrimToFit:
    """Test context-window trimming logic."""

    def test_no_trim_when_under_limit(self, settings):
        from claw.agent_core.conversation import ConversationSession

        session = ConversationSession()
        session.initialize()
        session.add_user("Hello")
        session.add_assistant("Hi")
        session.trim_to_fit(max_messages=40)
        assert len(session.get_messages()) == 3  # system + user + assistant

    def test_trim_preserves_system_prompt(self, settings):
        from claw.agent_core.conversation import ConversationSession

        session = ConversationSession()
        session.initialize()
        for i in range(50):
            session.add_user(f"msg {i}")
            session.add_assistant(f"reply {i}")
        session.trim_to_fit(max_messages=10)
        msgs = session.get_messages()
        assert msgs[0]["role"] == "system"
        # system + at most 10 tail messages
        assert len(msgs) <= 11

    def test_trim_drops_orphaned_tool_results(self, settings):
        from claw.agent_core.conversation import ConversationSession

        session = ConversationSession()
        session.initialize()
        # Add enough messages that trimming will occur
        for i in range(25):
            session.add_user(f"q{i}")
            session.add_tool_result(f"tc_{i}", f"result_{i}")
            session.add_assistant(f"a{i}")
        session.trim_to_fit(max_messages=6)
        msgs = session.get_messages()
        # First message after system should NOT be a tool result
        assert msgs[1].get("role") != "tool"

    def test_trim_handles_exact_boundary(self, settings):
        from claw.agent_core.conversation import ConversationSession

        session = ConversationSession()
        session.initialize()
        # max_messages + 1 (for system) = exactly at boundary
        for i in range(5):
            session.add_user(f"u{i}")
        session.trim_to_fit(max_messages=5)
        msgs = session.get_messages()
        assert msgs[0]["role"] == "system"


class TestAccountContext:
    """Test _build_account_context for multi-account Google configs."""

    def test_no_accounts_returns_empty(self, settings):
        from claw.agent_core.conversation import _build_account_context

        assert _build_account_context() == ""

    def test_single_account_with_calendar(self, tmp_config):
        import yaml
        from claw.config import Settings

        tmp_config.write_text(yaml.dump({
            "google_auth": {
                "accounts": {
                    "work": {
                        "email": "test@example.com",
                        "token_file": "t.json",
                        "calendar": {"enabled": True},
                        "gmail": {"enabled": False},
                    }
                }
            }
        }))
        Settings.load()  # populate singleton
        import claw.config as cfg_mod
        cfg_mod._settings = Settings.load()

        from claw.agent_core.conversation import _build_account_context

        ctx = _build_account_context()
        assert "work" in ctx
        assert "Google Calendar" in ctx


class TestEstimateTokens:
    """Test the rough token estimation method."""

    def test_empty_session(self, settings):
        from claw.agent_core.conversation import ConversationSession

        session = ConversationSession()
        session.initialize()
        tokens = session.estimate_tokens()
        # System prompt has some content, so tokens > 0
        assert tokens > 0

    def test_scales_with_content(self, settings):
        from claw.agent_core.conversation import ConversationSession

        session = ConversationSession()
        session.initialize()
        tokens_before = session.estimate_tokens()
        session.add_user("A" * 400)  # 400 chars = ~100 tokens
        tokens_after = session.estimate_tokens()
        assert tokens_after - tokens_before == 100

    def test_counts_all_messages(self, settings):
        from claw.agent_core.conversation import ConversationSession

        session = ConversationSession()
        session.messages = [
            {"role": "system", "content": "sys"},
            {"role": "user", "content": "AAAA"},     # 4 chars = 1 token
            {"role": "assistant", "content": "BBBBBBBB"},  # 8 chars = 2 tokens
        ]
        # sys = 3 chars // 4 = 0, user = 1, assistant = 2
        assert session.estimate_tokens() == 3

    def test_handles_missing_content(self, settings):
        from claw.agent_core.conversation import ConversationSession

        session = ConversationSession()
        session.messages = [
            {"role": "assistant", "tool_calls": [{"id": "tc1"}]},  # no content key
        ]
        assert session.estimate_tokens() == 0


class TestCompact:
    """Test LLM-driven context compaction."""

    @pytest.mark.asyncio
    async def test_compact_short_session_returns_empty(self, settings):
        """Sessions shorter than keep_recent + 2 should not be compacted."""
        from claw.agent_core.conversation import ConversationSession

        session = ConversationSession()
        session.initialize()
        session.add_user("Hello")
        session.add_assistant("Hi!")

        mock_llm = AsyncMock()
        result = await session.compact(mock_llm, keep_recent=6)
        assert result == ""
        # LLM should not have been called
        mock_llm.chat_simple.assert_not_called()

    @pytest.mark.asyncio
    async def test_compact_summarizes_old_messages(self, settings):
        """Old messages should be replaced with a summary."""
        from claw.agent_core.conversation import ConversationSession

        session = ConversationSession()
        session.initialize()
        # Add 10 user/assistant pairs (20 messages + system = 21 total)
        for i in range(10):
            session.add_user(f"Question {i}")
            session.add_assistant(f"Answer {i}")

        mock_llm = AsyncMock()
        mock_llm.chat_simple = AsyncMock(return_value="User asked 10 questions and got answers.")

        result = await session.compact(mock_llm, keep_recent=6)

        assert result == "User asked 10 questions and got answers."
        # Messages: system + summary + 6 recent
        msgs = session.get_messages()
        assert msgs[0]["role"] == "system"
        assert "[Conversation Summary]" in msgs[1]["content"]
        assert len(msgs) == 8  # system + summary + 6 recent

    @pytest.mark.asyncio
    async def test_compact_preserves_system_prompt(self, settings):
        """System prompt must be kept intact after compaction."""
        from claw.agent_core.conversation import ConversationSession

        session = ConversationSession()
        session.initialize()
        original_system = session.messages[0]["content"]
        for i in range(10):
            session.add_user(f"q{i}")
            session.add_assistant(f"a{i}")

        mock_llm = AsyncMock()
        mock_llm.chat_simple = AsyncMock(return_value="Summary text.")

        await session.compact(mock_llm, keep_recent=4)
        assert session.messages[0]["content"] == original_system

    @pytest.mark.asyncio
    async def test_compact_preserves_recent_messages(self, settings):
        """The most recent messages should be kept verbatim."""
        from claw.agent_core.conversation import ConversationSession

        session = ConversationSession()
        session.initialize()
        for i in range(10):
            session.add_user(f"q{i}")
            session.add_assistant(f"a{i}")

        mock_llm = AsyncMock()
        mock_llm.chat_simple = AsyncMock(return_value="Summary.")

        await session.compact(mock_llm, keep_recent=4)
        msgs = session.get_messages()
        # Last 4 should be the most recent conversation turns
        assert msgs[-1]["content"] == "a9"
        assert msgs[-2]["content"] == "q9"
        assert msgs[-3]["content"] == "a8"
        assert msgs[-4]["content"] == "q8"

    @pytest.mark.asyncio
    async def test_compact_does_not_split_tool_pairs(self, settings):
        """Tool call + tool result pairs must not be split across the boundary."""
        from claw.agent_core.conversation import ConversationSession

        session = ConversationSession()
        session.initialize()
        # Build: user, assistant, user, assistant(tool_calls), tool_result, user, assistant
        session.add_user("q0")
        session.add_assistant("a0")
        session.add_user("q1")
        session.add_assistant("a1")
        session.add_user("What's the weather?")
        # Simulate tool call assistant message
        session.messages.append({
            "role": "assistant",
            "content": None,
            "tool_calls": [{"id": "tc1", "type": "function", "function": {"name": "get_weather", "arguments": "{}"}}],
        })
        session.add_tool_result("tc1", "Sunny, 72F")
        session.add_user("Thanks!")
        session.add_assistant("You're welcome!")

        mock_llm = AsyncMock()
        mock_llm.chat_simple = AsyncMock(return_value="Weather was checked.")

        # keep_recent=3 would naively split at the tool result
        # The method should adjust to keep the tool_call + tool_result together
        await session.compact(mock_llm, keep_recent=3)
        msgs = session.get_messages()

        # Verify no orphaned tool results after the summary
        for i, msg in enumerate(msgs):
            if i <= 1:  # system + summary
                continue
            if msg.get("role") == "tool":
                # The message before a tool result must be the corresponding assistant with tool_calls
                prev = msgs[i - 1]
                assert prev.get("role") == "assistant" and prev.get("tool_calls"), (
                    f"Orphaned tool result at index {i}"
                )

    @pytest.mark.asyncio
    async def test_compact_empty_summary_skips(self, settings):
        """If LLM returns empty summary, messages should not change."""
        from claw.agent_core.conversation import ConversationSession

        session = ConversationSession()
        session.initialize()
        for i in range(10):
            session.add_user(f"q{i}")
            session.add_assistant(f"a{i}")

        original_count = len(session.messages)
        mock_llm = AsyncMock()
        mock_llm.chat_simple = AsyncMock(return_value="")

        result = await session.compact(mock_llm, keep_recent=4)
        assert result == ""
        assert len(session.messages) == original_count

    @pytest.mark.asyncio
    async def test_compact_formats_tool_calls_in_prompt(self, settings):
        """The summarization prompt should include formatted tool calls."""
        from claw.agent_core.conversation import ConversationSession

        session = ConversationSession()
        session.initialize()
        session.add_user("What time is it?")
        session.messages.append({
            "role": "assistant",
            "content": None,
            "tool_calls": [{"id": "tc1", "type": "function", "function": {"name": "get_time", "arguments": "{}"}}],
        })
        session.add_tool_result("tc1", "3:45 PM")
        session.add_assistant("It's 3:45 PM.")
        # Add enough recent messages
        for i in range(6):
            session.add_user(f"recent{i}")
            session.add_assistant(f"reply{i}")

        mock_llm = AsyncMock()
        mock_llm.chat_simple = AsyncMock(return_value="User asked the time, it was 3:45 PM.")

        await session.compact(mock_llm, keep_recent=6)

        # Verify the prompt sent to LLM contained tool info
        call_args = mock_llm.chat_simple.call_args[0][0]
        assert "get_time" in call_args
        assert "3:45 PM" in call_args

    @pytest.mark.asyncio
    async def test_compact_no_old_messages_returns_empty(self, settings):
        """When all messages are recent, nothing to compact."""
        from claw.agent_core.conversation import ConversationSession

        session = ConversationSession()
        session.initialize()
        # Exactly keep_recent + 1 (system) messages — no old messages
        for i in range(3):
            session.add_user(f"q{i}")
            session.add_assistant(f"a{i}")

        mock_llm = AsyncMock()
        # 7 messages total (system + 6), keep_recent=6 => recent_start=1, old=empty
        result = await session.compact(mock_llm, keep_recent=6)
        assert result == ""
        mock_llm.chat_simple.assert_not_called()

    @pytest.mark.asyncio
    async def test_compact_returns_summary_text(self, settings):
        """compact() should return the summary string for logging."""
        from claw.agent_core.conversation import ConversationSession

        session = ConversationSession()
        session.initialize()
        for i in range(10):
            session.add_user(f"q{i}")
            session.add_assistant(f"a{i}")

        mock_llm = AsyncMock()
        mock_llm.chat_simple = AsyncMock(return_value="  The key facts are X Y Z.  ")

        result = await session.compact(mock_llm, keep_recent=4)
        assert result == "The key facts are X Y Z."  # stripped
