"""Skill manager for installing/uninstalling MCP tool packages from Git."""

from __future__ import annotations

import json
import logging
import re
import shutil
import subprocess
import sys
from pathlib import Path

from claw.config import PROJECT_ROOT
from claw.skills.manifest import SkillManifest

log = logging.getLogger(__name__)

_TOOLS_DIR = PROJECT_ROOT / "mcp_tools"


class SkillManager:
    """Manages installation, removal, and listing of MCP tool skills.

    Skills are Git repositories cloned into mcp_tools/<name>/ that contain
    a skill.json manifest and a FastMCP server entry point.

    Security:
    - HTTPS Git URLs only (blocks file://, ssh://)
    - Install path restricted to mcp_tools/ directory
    - pip install into current venv only
    - Shallow clone (--depth 1)
    """

    def install(self, git_url: str, name: str | None = None) -> dict:
        """Install a skill from a Git repository.

        Args:
            git_url: HTTPS Git URL (e.g., https://github.com/user/claw-skill-example.git)
            name: Override the skill name (defaults to repo name)

        Returns:
            Dict with installation status and skill info.
        """
        # Validate URL
        self._validate_url(git_url)

        # Determine skill name from URL if not provided
        if not name:
            name = self._name_from_url(git_url)

        # Validate name
        if not re.match(r"^[a-z][a-z0-9_-]{0,49}$", name):
            raise ValueError(
                f"Invalid skill name '{name}'. Must be lowercase alphanumeric "
                "with hyphens/underscores, starting with a letter, max 50 chars."
            )

        target_dir = _TOOLS_DIR / name
        self._validate_path(target_dir)

        if target_dir.exists():
            raise ValueError(f"Skill '{name}' already installed at {target_dir}")

        # Clone repository
        log.info("Installing skill '%s' from %s", name, git_url)
        try:
            subprocess.run(
                ["git", "clone", "--depth", "1", git_url, str(target_dir)],
                check=True,
                capture_output=True,
                text=True,
                timeout=60,
            )
        except subprocess.CalledProcessError as e:
            raise RuntimeError(f"Git clone failed: {e.stderr}") from e
        except subprocess.TimeoutExpired:
            # Cleanup partial clone
            if target_dir.exists():
                shutil.rmtree(target_dir)
            raise RuntimeError("Git clone timed out")

        # Read manifest
        manifest_path = target_dir / "skill.json"
        if not manifest_path.exists():
            shutil.rmtree(target_dir)
            raise ValueError("No skill.json manifest found in repository")

        try:
            manifest = SkillManifest.model_validate_json(manifest_path.read_text())
        except Exception as e:
            shutil.rmtree(target_dir)
            raise ValueError(f"Invalid skill.json: {e}") from e

        # Install pip dependencies (with validation to prevent flag injection)
        if manifest.dependencies:
            for dep in manifest.dependencies:
                if dep.startswith("-") or dep.startswith("--"):
                    shutil.rmtree(target_dir)
                    raise ValueError(
                        f"Invalid dependency '{dep}': must be a package name, not a flag"
                    )
                if not re.match(r"^[a-zA-Z0-9][a-zA-Z0-9._-]*(\[.*\])?(([><=!~]=?|@).*)?\s*$", dep):
                    shutil.rmtree(target_dir)
                    raise ValueError(f"Invalid dependency specifier: '{dep}'")

            log.info("Installing dependencies: %s", manifest.dependencies)
            try:
                subprocess.run(
                    [sys.executable, "-m", "pip", "install", "--", *manifest.dependencies],
                    check=True,
                    capture_output=True,
                    text=True,
                    timeout=120,
                )
            except subprocess.CalledProcessError as e:
                shutil.rmtree(target_dir)
                raise RuntimeError(f"Dependency installation failed: {e.stderr}") from e

        # Verify entry point exists and stays within the skill directory
        entry_point = target_dir / manifest.entry_point
        try:
            resolved_ep = entry_point.resolve()
            if not str(resolved_ep).startswith(str(target_dir.resolve())):
                shutil.rmtree(target_dir)
                raise ValueError(
                    f"entry_point '{manifest.entry_point}' escapes skill directory"
                )
        except ValueError:
            raise
        except Exception as e:
            shutil.rmtree(target_dir)
            raise ValueError(f"Invalid entry_point path: {e}") from e
        if not entry_point.exists():
            shutil.rmtree(target_dir)
            raise ValueError(f"Entry point '{manifest.entry_point}' not found")

        log.info("Skill '%s' v%s installed successfully", name, manifest.version)
        return {
            "name": name,
            "version": manifest.version,
            "description": manifest.description,
            "path": str(target_dir),
        }

    def uninstall(self, name: str) -> bool:
        """Uninstall a skill by name.

        Returns True if successfully removed, False if not found.
        """
        target_dir = _TOOLS_DIR / name
        self._validate_path(target_dir)

        if not target_dir.exists():
            return False

        # Verify it's a skill (has skill.json), not a built-in tool
        if not (target_dir / "skill.json").exists():
            raise ValueError(
                f"'{name}' does not appear to be an installed skill "
                "(no skill.json). Cannot uninstall built-in tools."
            )

        shutil.rmtree(target_dir)
        log.info("Skill '%s' uninstalled", name)
        return True

    def list_skills(self) -> list[dict]:
        """List all installed skills (directories with skill.json)."""
        skills = []
        for path in sorted(_TOOLS_DIR.iterdir()):
            manifest_path = path / "skill.json"
            if path.is_dir() and manifest_path.exists():
                try:
                    manifest = SkillManifest.model_validate_json(
                        manifest_path.read_text()
                    )
                    skills.append({
                        "name": path.name,
                        "version": manifest.version,
                        "description": manifest.description,
                        "author": manifest.author,
                        "path": str(path),
                    })
                except Exception:
                    skills.append({
                        "name": path.name,
                        "version": "unknown",
                        "description": "Error reading manifest",
                        "path": str(path),
                    })
        return skills

    def _validate_url(self, url: str) -> None:
        """Validate Git URL for security."""
        if not url.startswith("https://"):
            raise ValueError(
                "Only HTTPS Git URLs are allowed. "
                f"Got: {url[:50]}"
            )
        # Block suspicious URL patterns
        if ".." in url or ";" in url or "|" in url or "&" in url:
            raise ValueError("Invalid characters in Git URL")

    def _validate_path(self, path: Path) -> None:
        """Ensure path is under mcp_tools/ to prevent directory traversal."""
        try:
            resolved = path.resolve()
            tools_resolved = _TOOLS_DIR.resolve()
            if not str(resolved).startswith(str(tools_resolved)):
                raise ValueError(f"Path traversal detected: {path}")
        except Exception as e:
            if "traversal" in str(e):
                raise
            raise ValueError(f"Invalid path: {path}") from e

    def _name_from_url(self, url: str) -> str:
        """Extract skill name from Git URL."""
        # Remove trailing .git
        name = url.rstrip("/").rsplit("/", 1)[-1]
        if name.endswith(".git"):
            name = name[:-4]
        # Remove common prefixes
        for prefix in ("claw-skill-", "claw-", "skill-"):
            if name.startswith(prefix):
                name = name[len(prefix) :]
                break
        return name.lower().replace("-", "_")
