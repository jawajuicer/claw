"""Tests for claw.mcp_handler.router — ToolRouter dispatch."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest


@pytest.fixture()
def mock_registry():
    reg = MagicMock()
    reg.get_server_for_tool = MagicMock(return_value=None)
    reg.get_client = MagicMock(return_value=None)
    return reg


@pytest.fixture()
def router(mock_registry):
    from claw.mcp_handler.router import ToolRouter

    return ToolRouter(mock_registry)


class TestCallTool:
    """Test tool call routing."""

    async def test_unknown_tool_returns_error_message(self, router, mock_registry):
        mock_registry.get_server_for_tool.return_value = None
        result = await router.call_tool("nonexistent", {})
        assert "Unknown tool" in result

    async def test_disconnected_server_returns_error(self, router, mock_registry):
        mock_registry.get_server_for_tool.return_value = "weather"
        mock_registry.get_client.return_value = None
        result = await router.call_tool("get_weather", {})
        assert "not connected" in result

    async def test_successful_routing(self, router, mock_registry):
        mock_client = AsyncMock()
        mock_client.call_tool = AsyncMock(return_value="72F, sunny")
        mock_registry.get_server_for_tool.return_value = "weather"
        mock_registry.get_client.return_value = mock_client

        result = await router.call_tool("get_weather", {"location": "Akron"})
        assert result == "72F, sunny"
        mock_client.call_tool.assert_called_once_with("get_weather", {"location": "Akron"})

    async def test_routes_to_correct_server(self, router, mock_registry):
        mock_client_weather = AsyncMock()
        mock_client_weather.call_tool = AsyncMock(return_value="sunny")
        mock_client_system = AsyncMock()
        mock_client_system.call_tool = AsyncMock(return_value="12:00")

        def get_server(name):
            return {"get_weather": "weather", "get_time": "system"}.get(name)

        def get_client(name):
            return {"weather": mock_client_weather, "system": mock_client_system}.get(name)

        mock_registry.get_server_for_tool = MagicMock(side_effect=get_server)
        mock_registry.get_client = MagicMock(side_effect=get_client)

        result = await router.call_tool("get_time", {})
        assert result == "12:00"
        mock_client_system.call_tool.assert_called_once_with("get_time", {})
        mock_client_weather.call_tool.assert_not_called()

    async def test_empty_arguments(self, router, mock_registry):
        mock_client = AsyncMock()
        mock_client.call_tool = AsyncMock(return_value="result")
        mock_registry.get_server_for_tool.return_value = "server"
        mock_registry.get_client.return_value = mock_client

        await router.call_tool("some_tool", {})
        mock_client.call_tool.assert_called_once_with("some_tool", {})
