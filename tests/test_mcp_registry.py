"""Tests for claw.mcp_handler.registry — MCPRegistry tool discovery."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest


@pytest.fixture()
def registry():
    from claw.mcp_handler.registry import MCPRegistry

    return MCPRegistry()


class TestGetOpenAITools:
    """Test tool schema retrieval."""

    def test_empty_registry_returns_empty(self, registry):
        assert registry.get_openai_tools() == []

    def test_returns_all_tools(self, registry):
        registry._openai_tools = [
            {"type": "function", "function": {"name": "tool_a", "description": "A", "parameters": {}}},
            {"type": "function", "function": {"name": "tool_b", "description": "B", "parameters": {}}},
        ]
        assert len(registry.get_openai_tools()) == 2

    def test_filter_by_server(self, registry):
        registry._openai_tools = [
            {"type": "function", "function": {"name": "tool_a", "description": "A", "parameters": {}}},
            {"type": "function", "function": {"name": "tool_b", "description": "B", "parameters": {}}},
        ]
        registry._tool_map = {"tool_a": "server_1", "tool_b": "server_2"}
        result = registry.get_openai_tools(servers=["server_1"])
        assert len(result) == 1
        assert result[0]["function"]["name"] == "tool_a"


class TestGetServerForTool:
    """Test tool-to-server mapping."""

    def test_known_tool(self, registry):
        registry._tool_map = {"get_weather": "weather"}
        assert registry.get_server_for_tool("get_weather") == "weather"

    def test_unknown_tool(self, registry):
        assert registry.get_server_for_tool("nonexistent") is None


class TestGetClient:
    """Test client retrieval."""

    def test_known_server(self, registry):
        mock_client = MagicMock()
        registry._clients = {"weather": mock_client}
        assert registry.get_client("weather") is mock_client

    def test_unknown_server(self, registry):
        assert registry.get_client("nonexistent") is None


class TestListServers:
    """Test server listing."""

    def test_empty_registry(self, registry):
        assert registry.list_servers() == {}

    def test_multiple_servers(self, registry):
        registry._tool_map = {
            "get_weather": "weather",
            "get_forecast": "weather",
            "get_time": "system_control",
        }
        servers = registry.list_servers()
        assert set(servers["weather"]) == {"get_weather", "get_forecast"}
        assert servers["system_control"] == ["get_time"]


class TestInitialize:
    """Test the initialize flow that scans for MCP servers."""

    async def test_skips_missing_server_scripts(self, settings, tmp_config, registry):
        from claw.config import get_settings

        s = get_settings()
        s.mcp.enabled_servers = ["nonexistent_server"]
        with patch("claw.mcp_handler.registry.get_settings", return_value=s):
            await registry.initialize()
        assert len(registry._clients) == 0

    async def test_handles_connection_failure(self, settings, tmp_config):
        from claw.mcp_handler.registry import MCPRegistry

        reg = MCPRegistry()
        s = settings
        s.mcp.enabled_servers = ["test_server"]

        # Create a fake server.py
        import claw.config as cfg_mod
        tools_dir = cfg_mod.PROJECT_ROOT / s.mcp.tools_dir / "test_server"
        tools_dir.mkdir(parents=True, exist_ok=True)
        (tools_dir / "server.py").write_text("# dummy")

        mock_client_cls = MagicMock()
        mock_client = AsyncMock()
        mock_client.connect = AsyncMock(side_effect=ConnectionError("refused"))
        mock_client_cls.return_value = mock_client

        with (
            patch("claw.mcp_handler.registry.get_settings", return_value=s),
            patch("claw.mcp_handler.registry.MCPClient", mock_client_cls),
        ):
            await reg.initialize()
        assert len(reg._clients) == 0


class TestShutdown:
    """Test clean shutdown of MCP servers."""

    async def test_shutdown_disconnects_all(self, registry):
        client1 = AsyncMock()
        client2 = AsyncMock()
        registry._clients = {"s1": client1, "s2": client2}
        registry._tool_map = {"tool1": "s1"}
        registry._openai_tools = [{"type": "function", "function": {"name": "tool1"}}]

        await registry.shutdown()
        client1.disconnect.assert_called_once()
        client2.disconnect.assert_called_once()
        assert len(registry._clients) == 0
        assert len(registry._tool_map) == 0
        assert len(registry._openai_tools) == 0

    async def test_shutdown_handles_disconnect_error(self, registry):
        client = AsyncMock()
        client.disconnect = AsyncMock(side_effect=RuntimeError("oops"))
        registry._clients = {"s1": client}

        # Should not raise
        await registry.shutdown()
        assert len(registry._clients) == 0
