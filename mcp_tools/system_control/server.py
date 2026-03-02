"""System control MCP server — local system commands."""

import platform
import subprocess
from datetime import datetime, timezone

from mcp.server.fastmcp import FastMCP

mcp = FastMCP("SystemControl")


@mcp.tool()
def get_time() -> str:
    """Get the current date and time."""
    now = datetime.now(timezone.utc)
    local = datetime.now()
    return f"UTC: {now.strftime('%Y-%m-%d %H:%M:%S')}\nLocal: {local.strftime('%Y-%m-%d %H:%M:%S %Z')}"


@mcp.tool()
def get_system_info() -> str:
    """Get basic system information."""
    info = {
        "hostname": platform.node(),
        "os": f"{platform.system()} {platform.release()}",
        "architecture": platform.machine(),
        "python": platform.python_version(),
    }
    return "\n".join(f"{k}: {v}" for k, v in info.items())


@mcp.tool()
def get_uptime() -> str:
    """Get system uptime."""
    try:
        result = subprocess.run(["uptime", "-p"], capture_output=True, text=True, timeout=5)
        return result.stdout.strip() or "Unable to determine uptime"
    except Exception as e:
        return f"Error: {e}"


@mcp.tool()
def get_disk_usage() -> str:
    """Get disk usage for the root filesystem."""
    try:
        result = subprocess.run(
            ["df", "-h", "/"],
            capture_output=True, text=True, timeout=5,
        )
        return result.stdout.strip()
    except Exception as e:
        return f"Error: {e}"


@mcp.tool()
def get_memory_usage() -> str:
    """Get current memory usage."""
    try:
        result = subprocess.run(
            ["free", "-h"],
            capture_output=True, text=True, timeout=5,
        )
        return result.stdout.strip()
    except Exception as e:
        return f"Error: {e}"


if __name__ == "__main__":
    mcp.run()
