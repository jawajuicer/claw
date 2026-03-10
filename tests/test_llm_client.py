"""Tests for claw.agent_core.llm_client — LLMClient wrapper."""

from __future__ import annotations

import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest


@pytest.fixture()
def llm_client(settings):
    """Create an LLMClient with a mocked AsyncOpenAI underneath."""
    with patch("claw.agent_core.llm_client.AsyncOpenAI") as MockOpenAI:
        mock_openai = AsyncMock()
        MockOpenAI.return_value = mock_openai
        from claw.agent_core.llm_client import LLMClient

        client = LLMClient()
        client._mock_openai = mock_openai
        yield client


def _fake_completion(content="Hello"):
    msg = SimpleNamespace(content=content, tool_calls=None)
    choice = SimpleNamespace(message=msg)
    usage = SimpleNamespace(prompt_tokens=10, completion_tokens=5, total_tokens=15)
    return SimpleNamespace(choices=[choice], usage=usage)


class TestChat:
    """Test the main chat() method."""

    async def test_chat_returns_response(self, llm_client):
        llm_client._mock_openai.chat.completions.create = AsyncMock(
            return_value=_fake_completion("Hi there!")
        )
        resp = await llm_client.chat(messages=[{"role": "user", "content": "Hello"}])
        assert resp.choices[0].message.content == "Hi there!"

    async def test_chat_passes_tools(self, llm_client):
        llm_client._mock_openai.chat.completions.create = AsyncMock(
            return_value=_fake_completion()
        )
        tools = [{"type": "function", "function": {"name": "test", "parameters": {}}}]
        await llm_client.chat(
            messages=[{"role": "user", "content": "test"}],
            tools=tools,
        )
        call_kwargs = llm_client._mock_openai.chat.completions.create.call_args
        assert "tools" in call_kwargs.kwargs

    async def test_chat_uses_model_override(self, llm_client):
        llm_client._mock_openai.chat.completions.create = AsyncMock(
            return_value=_fake_completion()
        )
        await llm_client.chat(
            messages=[{"role": "user", "content": "test"}],
            model="custom-model",
        )
        call_kwargs = llm_client._mock_openai.chat.completions.create.call_args
        assert call_kwargs.kwargs["model"] == "custom-model"

    async def test_chat_timeout_raises_asyncio_timeout(self, llm_client):
        from openai import APITimeoutError

        llm_client._mock_openai.chat.completions.create = AsyncMock(
            side_effect=APITimeoutError(request=MagicMock())
        )
        with pytest.raises(asyncio.TimeoutError):
            await llm_client.chat(messages=[{"role": "user", "content": "test"}])


class TestChatSimple:
    """Test the single-turn chat_simple() helper."""

    async def test_chat_simple_returns_text(self, llm_client):
        llm_client._mock_openai.chat.completions.create = AsyncMock(
            return_value=_fake_completion("Simple answer")
        )
        result = await llm_client.chat_simple("What is 2+2?")
        assert result == "Simple answer"

    async def test_chat_simple_timeout_raises(self, llm_client):
        from openai import APITimeoutError

        llm_client._mock_openai.chat.completions.create = AsyncMock(
            side_effect=APITimeoutError(request=MagicMock())
        )
        with pytest.raises(asyncio.TimeoutError):
            await llm_client.chat_simple("test")

    async def test_chat_simple_empty_content_returns_empty(self, llm_client):
        msg = SimpleNamespace(content=None, tool_calls=None)
        choice = SimpleNamespace(message=msg)
        usage = SimpleNamespace(prompt_tokens=5, completion_tokens=0, total_tokens=5)
        resp = SimpleNamespace(choices=[choice], usage=usage)
        llm_client._mock_openai.chat.completions.create = AsyncMock(return_value=resp)
        result = await llm_client.chat_simple("test")
        assert result == ""


class TestBusy:
    """Test the busy property."""

    def test_busy_false_when_idle(self, llm_client):
        assert llm_client.busy is False


class TestConfigReload:
    """Test that config reload updates LLM parameters."""

    def test_on_config_reload_updates_model(self, llm_client, settings):
        settings.llm.model = "new-model:7b"
        settings.llm.temperature = 0.2
        settings.llm.max_tokens = 2048
        settings.llm.timeout = 60
        llm_client._on_config_reload(settings)
        assert llm_client._model == "new-model:7b"
        assert llm_client._temperature == 0.2
        assert llm_client._max_tokens == 2048

    def test_on_config_reload_recreates_client_on_url_change(self, llm_client, settings):
        # The reload method checks `cfg.base_url != str(self._client.base_url).rstrip("/")`.
        # We need to set the mock client's base_url to the OLD value so the comparison
        # detects a change.
        llm_client._client.base_url = "http://localhost:8081/v1/"
        settings.llm.base_url = "http://new-host:8081/v1"
        settings.llm.api_key = "new-key"
        settings.llm.timeout = 30

        with patch("claw.agent_core.llm_client.AsyncOpenAI") as MockNewOAI:
            new_client = AsyncMock()
            MockNewOAI.return_value = new_client
            llm_client._on_config_reload(settings)
            assert llm_client._client is new_client
            MockNewOAI.assert_called_once_with(
                base_url="http://new-host:8081/v1",
                api_key="new-key",
                timeout=30,
            )
