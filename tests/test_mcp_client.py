"""Tests for claw.mcp_handler.client — MCPClient stdio session manager."""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest


class TestMCPClientInit:
    """Test MCPClient construction."""

    def test_stores_name_and_script(self, settings):
        from claw.mcp_handler.client import MCPClient

        client = MCPClient("test_server", "/path/to/server.py")
        assert client.name == "test_server"
        assert client.server_script == "/path/to/server.py"
        assert client.session is None


class TestListTools:
    """Test the list_tools method."""

    async def test_list_tools_not_connected_raises(self, settings):
        from claw.mcp_handler.client import MCPClient

        client = MCPClient("test", "/path.py")
        with pytest.raises(RuntimeError, match="not connected"):
            await client.list_tools()

    async def test_list_tools_returns_formatted(self, settings):
        from claw.mcp_handler.client import MCPClient

        client = MCPClient("test", "/path.py")
        mock_tool = SimpleNamespace(
            name="get_weather",
            description="Get current weather",
            inputSchema={"type": "object", "properties": {"location": {"type": "string"}}},
        )
        mock_response = SimpleNamespace(tools=[mock_tool])
        client.session = AsyncMock()
        client.session.list_tools = AsyncMock(return_value=mock_response)

        tools = await client.list_tools()
        assert len(tools) == 1
        assert tools[0]["name"] == "get_weather"
        assert tools[0]["description"] == "Get current weather"
        assert "properties" in tools[0]["input_schema"]

    async def test_list_tools_empty_description(self, settings):
        from claw.mcp_handler.client import MCPClient

        client = MCPClient("test", "/path.py")
        mock_tool = SimpleNamespace(name="t1", description=None, inputSchema={})
        mock_response = SimpleNamespace(tools=[mock_tool])
        client.session = AsyncMock()
        client.session.list_tools = AsyncMock(return_value=mock_response)

        tools = await client.list_tools()
        assert tools[0]["description"] == ""


class TestCallTool:
    """Test the call_tool method."""

    async def test_call_tool_not_connected_raises(self, settings):
        from claw.mcp_handler.client import MCPClient

        client = MCPClient("test", "/path.py")
        with pytest.raises(RuntimeError, match="not connected"):
            await client.call_tool("some_tool", {})

    async def test_call_tool_returns_text(self, settings):
        from claw.mcp_handler.client import MCPClient

        client = MCPClient("test", "/path.py")
        text_item = SimpleNamespace(text="Sunny and warm")
        mock_result = SimpleNamespace(content=[text_item])
        client.session = AsyncMock()
        client.session.call_tool = AsyncMock(return_value=mock_result)

        result = await client.call_tool("get_weather", {"location": "Akron"})
        assert result == "Sunny and warm"
        client.session.call_tool.assert_called_once_with("get_weather", {"location": "Akron"})

    async def test_call_tool_multiple_content_blocks(self, settings):
        from claw.mcp_handler.client import MCPClient

        client = MCPClient("test", "/path.py")
        items = [SimpleNamespace(text="Line 1"), SimpleNamespace(text="Line 2")]
        mock_result = SimpleNamespace(content=items)
        client.session = AsyncMock()
        client.session.call_tool = AsyncMock(return_value=mock_result)

        result = await client.call_tool("tool", {})
        assert result == "Line 1\nLine 2"

    async def test_call_tool_no_text_content(self, settings):
        from claw.mcp_handler.client import MCPClient

        client = MCPClient("test", "/path.py")
        # Content item without a text attribute (e.g., image)
        image_item = SimpleNamespace(type="image", data="base64data")
        mock_result = SimpleNamespace(content=[image_item])
        client.session = AsyncMock()
        client.session.call_tool = AsyncMock(return_value=mock_result)

        result = await client.call_tool("tool", {})
        assert result == ""

    async def test_call_tool_empty_content(self, settings):
        from claw.mcp_handler.client import MCPClient

        client = MCPClient("test", "/path.py")
        mock_result = SimpleNamespace(content=[])
        client.session = AsyncMock()
        client.session.call_tool = AsyncMock(return_value=mock_result)

        result = await client.call_tool("tool", {})
        assert result == ""


class TestDisconnect:
    """Test session disconnection."""

    async def test_disconnect_closes_exit_stack(self, settings):
        from claw.mcp_handler.client import MCPClient

        client = MCPClient("test", "/path.py")
        mock_stack = AsyncMock()
        client._exit_stack = mock_stack
        client.session = MagicMock()

        await client.disconnect()
        mock_stack.aclose.assert_called_once()
        assert client.session is None
        assert client._exit_stack is None

    async def test_disconnect_when_not_connected(self, settings):
        from claw.mcp_handler.client import MCPClient

        client = MCPClient("test", "/path.py")
        # Should not raise
        await client.disconnect()
        assert client.session is None
