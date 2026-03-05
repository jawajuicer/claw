"""Tests for claw.agent_core.conversation — ConversationSession."""

from __future__ import annotations




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
