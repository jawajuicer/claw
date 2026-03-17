"""Async MCP stdio client session manager."""

from __future__ import annotations

import logging
from contextlib import AsyncExitStack

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client


log = logging.getLogger(__name__)


class MCPClient:
    """Manages a persistent MCP client session to a single server subprocess."""

    def __init__(self, name: str, server_script: str) -> None:
        self.name = name
        self.server_script = server_script
        self.session: ClientSession | None = None
        self._exit_stack: AsyncExitStack | None = None

    async def connect(self) -> None:
        """Spawn the server subprocess and establish a session."""
        self._exit_stack = AsyncExitStack()

        server_params = StdioServerParameters(
            command="python",
            args=[self.server_script],
            env=None,
        )

        log.info("Connecting to MCP server '%s' at %s", self.name, self.server_script)

        try:
            transport = await self._exit_stack.enter_async_context(
                stdio_client(server_params)
            )
            read, write = transport

            self.session = await self._exit_stack.enter_async_context(
                ClientSession(read, write)
            )
            await self.session.initialize()
        except Exception:
            await self._exit_stack.aclose()
            self._exit_stack = None
            self.session = None
            raise

        log.info("MCP server '%s' connected", self.name)

    async def list_tools(self) -> list[dict]:
        """List tools exposed by this server."""
        if self.session is None:
            raise RuntimeError(f"MCP server '{self.name}' not connected")

        response = await self.session.list_tools()
        tools = []
        for tool in response.tools:
            tools.append({
                "name": tool.name,
                "description": tool.description or "",
                "input_schema": tool.inputSchema,
            })
        return tools

    async def call_tool(self, name: str, arguments: dict) -> str:
        """Call a tool on this server and return the text result."""
        if self.session is None:
            raise RuntimeError(f"MCP server '{self.name}' not connected")

        log.debug("Calling tool '%s' on server '%s'", name, self.name)
        result = await self.session.call_tool(name, arguments)

        # Extract text from content blocks
        if not result.content:
            return ""
        parts = []
        for item in result.content:
            if hasattr(item, "text"):
                parts.append(item.text)
        return "\n".join(parts) if parts else ""

    async def disconnect(self) -> None:
        """Close the session and kill the subprocess."""
        if self._exit_stack is not None:
            await self._exit_stack.aclose()
            self._exit_stack = None
            self.session = None
            log.info("MCP server '%s' disconnected", self.name)
