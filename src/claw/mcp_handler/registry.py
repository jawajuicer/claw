"""MCP tool discovery, schema caching, and OpenAI function-calling format conversion."""

from __future__ import annotations

import logging
from pathlib import Path

from claw.config import PROJECT_ROOT, get_settings
from claw.mcp_handler.client import MCPClient

log = logging.getLogger(__name__)


class MCPRegistry:
    """Discovers MCP servers, connects to them, and provides tool schemas."""

    def __init__(self) -> None:
        self._clients: dict[str, MCPClient] = {}
        # tool_name → server_name mapping for routing
        self._tool_map: dict[str, str] = {}
        # Cached OpenAI-format tool definitions
        self._openai_tools: list[dict] = []

    async def initialize(self) -> None:
        """Scan for enabled MCP servers, connect, and cache tool schemas."""
        cfg = get_settings().mcp
        tools_dir = Path(cfg.tools_dir)
        if not tools_dir.is_absolute():
            tools_dir = PROJECT_ROOT / tools_dir

        for server_name in cfg.enabled_servers:
            server_path = tools_dir / server_name / "server.py"
            if not server_path.exists():
                log.warning("MCP server script not found: %s", server_path)
                continue

            client = MCPClient(server_name, str(server_path))
            try:
                await client.connect()
                self._clients[server_name] = client
                log.info("MCP server '%s' registered", server_name)
            except Exception:
                log.exception("Failed to connect to MCP server '%s'", server_name)
                continue

        await self._refresh_tools()

    async def _refresh_tools(self) -> None:
        """Fetch tool schemas from all connected servers and build the tool map."""
        self._tool_map.clear()
        self._openai_tools.clear()

        for server_name, client in self._clients.items():
            try:
                tools = await client.list_tools()
            except Exception:
                log.exception("Failed to list tools from '%s'", server_name)
                continue

            for tool in tools:
                tool_name = tool["name"]
                self._tool_map[tool_name] = server_name

                # Convert to OpenAI function-calling format
                openai_tool = {
                    "type": "function",
                    "function": {
                        "name": tool_name,
                        "description": tool["description"],
                        "parameters": tool["input_schema"],
                    },
                }
                self._openai_tools.append(openai_tool)
                log.info("Registered tool: %s (server: %s)", tool_name, server_name)

    def get_openai_tools(self) -> list[dict]:
        """Return tool definitions in OpenAI function-calling format."""
        return self._openai_tools

    def get_server_for_tool(self, tool_name: str) -> str | None:
        """Look up which server owns a given tool."""
        return self._tool_map.get(tool_name)

    def get_client(self, server_name: str) -> MCPClient | None:
        return self._clients.get(server_name)

    def list_servers(self) -> dict[str, list[str]]:
        """Return a mapping of server_name → list of tool names."""
        servers: dict[str, list[str]] = {}
        for tool_name, server_name in self._tool_map.items():
            servers.setdefault(server_name, []).append(tool_name)
        return servers

    async def shutdown(self) -> None:
        """Disconnect all MCP servers."""
        for name, client in self._clients.items():
            try:
                await client.disconnect()
            except Exception:
                log.exception("Error disconnecting MCP server '%s'", name)
        self._clients.clear()
        self._tool_map.clear()
        self._openai_tools.clear()
