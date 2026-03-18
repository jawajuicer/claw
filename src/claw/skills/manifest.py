"""Skill manifest model for MCP tool packages."""

from __future__ import annotations

from pydantic import BaseModel, Field


class SkillManifest(BaseModel):
    """Manifest for an installable MCP tool skill (skill.json)."""

    name: str
    version: str = "0.1.0"
    description: str = ""
    author: str = ""
    dependencies: list[str] = Field(default_factory=list)  # pip packages
    config_schema: dict = Field(default_factory=dict)  # JSON Schema for tool config
    entry_point: str = "server.py"  # relative to skill directory
