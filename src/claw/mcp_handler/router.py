"""Routes tool calls to the correct MCP server."""

from __future__ import annotations

import logging

from claw.mcp_handler.registry import MCPRegistry

log = logging.getLogger(__name__)


class ToolRouter:
    """Dispatches tool calls to the appropriate MCP server by name lookup."""

    def __init__(self, registry: MCPRegistry) -> None:
        self.registry = registry

    async def call_tool(self, name: str, arguments: dict) -> str:
        """Route a tool call to the correct MCP server and return the result."""
        server_name = self.registry.get_server_for_tool(name)
        if server_name is None:
            msg = f"Unknown tool: '{name}'"
            log.warning(msg)
            return msg

        client = self.registry.get_client(server_name)
        if client is None:
            msg = f"Server '{server_name}' for tool '{name}' is not connected"
            log.warning(msg)
            return msg

        log.info("Routing tool '%s' to server '%s'", name, server_name)
        return await client.call_tool(name, arguments)
