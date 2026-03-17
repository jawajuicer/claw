"""Routes tool calls to the correct MCP server."""

from __future__ import annotations

import logging
import time

from claw.mcp_handler.registry import MCPRegistry
from claw.mcp_handler.stats import ToolStats

log = logging.getLogger(__name__)


class ToolRouter:
    """Dispatches tool calls to the appropriate MCP server by name lookup."""

    def __init__(self, registry: MCPRegistry, stats: ToolStats | None = None) -> None:
        self.registry = registry
        self.stats = stats

    async def call_tool(self, name: str, arguments: dict) -> str:
        """Route a tool call to the correct MCP server and return the result."""
        server_name = self.registry.get_server_for_tool(name)
        if server_name is None:
            msg = f"Unknown tool: '{name}'"
            log.warning(msg)
            if self.stats:
                self.stats.record(name, "", 0, success=False)
            return msg

        client = self.registry.get_client(server_name)
        if client is None:
            msg = f"Server '{server_name}' for tool '{name}' is not connected"
            log.warning(msg)
            if self.stats:
                self.stats.record(name, server_name, 0, success=False)
            return msg

        log.info("Routing tool '%s' to server '%s'", name, server_name)
        t0 = time.monotonic()
        success = True
        try:
            result = await client.call_tool(name, arguments)
        except Exception:
            success = False
            log.exception("Tool call failed: %s", name)
            raise
        finally:
            elapsed = time.monotonic() - t0
            if self.stats:
                self.stats.record(name, server_name, elapsed, success)

        return result
