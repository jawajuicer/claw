"""MCP tool server for skill management (install/uninstall/list)."""

from __future__ import annotations

import json
import sys
from pathlib import Path

from mcp.server.fastmcp import FastMCP

mcp = FastMCP("skills")

# Add project src to path for imports
_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(_PROJECT_ROOT / "src"))


def _get_manager():
    from claw.skills.manager import SkillManager

    return SkillManager()


@mcp.tool()
def skill_install(git_url: str, name: str = "") -> str:
    """Install a new MCP tool skill from a Git repository.

    Args:
        git_url: HTTPS Git URL of the skill repository
        name: Optional override for the skill name (auto-detected from URL)
    """
    try:
        manager = _get_manager()
        result = manager.install(git_url, name=name or None)
        return json.dumps({"status": "installed", **result}, indent=2)
    except (ValueError, RuntimeError) as e:
        return f"Error: {e}"


@mcp.tool()
def skill_uninstall(name: str) -> str:
    """Uninstall an MCP tool skill.

    Args:
        name: Name of the skill to uninstall
    """
    try:
        manager = _get_manager()
        if manager.uninstall(name):
            return f"Skill '{name}' uninstalled successfully."
        else:
            return f"Skill '{name}' not found."
    except ValueError as e:
        return f"Error: {e}"


@mcp.tool()
def skill_list() -> str:
    """List all installed MCP tool skills with their versions and descriptions."""
    manager = _get_manager()
    skills = manager.list_skills()
    if not skills:
        return "No skills installed."
    return json.dumps(skills, indent=2)


if __name__ == "__main__":
    mcp.run()
