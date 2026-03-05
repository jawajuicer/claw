"""Notes & Reminders MCP server — local JSON-backed note storage with reminders."""

from __future__ import annotations

import json
import uuid
from datetime import datetime, timezone
from pathlib import Path

import yaml
from dateutil import parser as dateutil_parser
from dateutil.relativedelta import relativedelta
from mcp.server.fastmcp import FastMCP

mcp = FastMCP("Notes")

_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
_CONFIG_YAML = _PROJECT_ROOT / "config.yaml"
_config = None


def _load_config() -> dict:
    global _config
    if _config is not None:
        return _config
    if _CONFIG_YAML.exists():
        with open(_CONFIG_YAML) as f:
            data = yaml.safe_load(f) or {}
        _config = data.get("notes", {})
    else:
        _config = {}
    return _config


def _storage_dir() -> Path:
    cfg = _load_config()
    rel = cfg.get("storage_dir", "data/notes")
    d = _PROJECT_ROOT / rel
    d.mkdir(parents=True, exist_ok=True)
    return d


def _notes_file() -> Path:
    return _storage_dir() / "notes.json"


def _reminders_file() -> Path:
    return _storage_dir() / "reminders.json"


def _read_notes() -> list[dict]:
    f = _notes_file()
    if not f.exists():
        return []
    return json.loads(f.read_text())


def _write_notes(notes: list[dict]) -> None:
    _notes_file().write_text(json.dumps(notes, indent=2))


def _read_reminders() -> list[dict]:
    f = _reminders_file()
    if not f.exists():
        return []
    return json.loads(f.read_text())


def _write_reminders(reminders: list[dict]) -> None:
    _reminders_file().write_text(json.dumps(reminders, indent=2))


def _parse_time(time_str: str) -> str:
    """Parse natural time string into ISO format."""
    now = datetime.now()
    lower = time_str.lower().strip()

    # Handle relative time ("in 30 min", "in 2 hours")
    if lower.startswith("in "):
        parts = lower[3:].strip().split()
        if len(parts) >= 2:
            try:
                amount = int(parts[0])
            except ValueError:
                amount = 1
            unit = parts[1].rstrip("s")
            kwargs = {}
            if unit in ("min", "minute"):
                kwargs["minutes"] = amount
            elif unit in ("hour", "hr"):
                kwargs["hours"] = amount
            elif unit in ("day",):
                kwargs["days"] = amount
            elif unit in ("week", "wk"):
                kwargs["weeks"] = amount
            if kwargs:
                dt = now + relativedelta(**kwargs)
                return dt.isoformat()

    # Handle "tomorrow Xam/pm"
    if lower.startswith("tomorrow"):
        rest = lower.replace("tomorrow", "").strip()
        tomorrow = now + relativedelta(days=1)
        if rest:
            try:
                t = dateutil_parser.parse(rest)
                dt = tomorrow.replace(hour=t.hour, minute=t.minute, second=0, microsecond=0)
                return dt.isoformat()
            except (ValueError, TypeError):
                pass
        return tomorrow.replace(hour=9, minute=0, second=0, microsecond=0).isoformat()

    # Fallback to dateutil parser
    try:
        dt = dateutil_parser.parse(time_str, fuzzy=True)
        if dt < now:
            dt = dt + relativedelta(days=1)
        return dt.isoformat()
    except (ValueError, TypeError):
        return time_str


def _is_enabled() -> bool:
    cfg = _load_config()
    return cfg.get("enabled", True)


def _max_notes() -> int:
    cfg = _load_config()
    return cfg.get("max_notes", 0)


_NOT_ENABLED = "Notes & Reminders is currently disabled. Enable it in Settings."


@mcp.tool()
def create_note(title: str, content: str, tags: str = "") -> str:
    """Create a new note.

    Args:
        title: Title of the note.
        content: Body text of the note.
        tags: Comma-separated tags for categorization.
    """
    if not _is_enabled():
        return _NOT_ENABLED
    notes = _read_notes()
    limit = _max_notes()
    if limit > 0 and len(notes) >= limit:
        return f"Cannot create note: maximum of {limit} notes reached."

    tag_list = [t.strip() for t in tags.split(",") if t.strip()] if tags else []
    note = {
        "id": uuid.uuid4().hex[:8],
        "title": title,
        "content": content,
        "tags": tag_list,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "updated_at": datetime.now(timezone.utc).isoformat(),
    }
    notes.append(note)
    _write_notes(notes)
    tag_str = f" [tags: {', '.join(tag_list)}]" if tag_list else ""
    return f"Note created: '{title}' (id: {note['id']}){tag_str}"


@mcp.tool()
def list_notes(tag: str = "", limit: int = 20) -> str:
    """List notes, optionally filtered by tag.

    Args:
        tag: Filter by this tag. Leave empty for all notes.
        limit: Maximum number of notes to return (1-100).
    """
    if not _is_enabled():
        return _NOT_ENABLED
    notes = _read_notes()
    limit = max(1, min(100, limit))

    if tag:
        notes = [n for n in notes if tag.lower() in [t.lower() for t in n.get("tags", [])]]

    notes = sorted(notes, key=lambda n: n.get("updated_at", ""), reverse=True)[:limit]

    if not notes:
        return "No notes found." if not tag else f"No notes found with tag '{tag}'."

    lines = [f"Notes ({len(notes)}):"]
    for n in notes:
        tags = f" [{', '.join(n['tags'])}]" if n.get("tags") else ""
        lines.append(f"- {n['title']} (id: {n['id']}){tags}")
    return "\n".join(lines)


@mcp.tool()
def search_notes(query: str, limit: int = 10) -> str:
    """Search notes by title and content.

    Args:
        query: Search text to find in titles and content.
        limit: Maximum number of results (1-50).
    """
    if not _is_enabled():
        return _NOT_ENABLED
    notes = _read_notes()
    limit = max(1, min(50, limit))
    q = query.lower()

    matches = [
        n for n in notes
        if q in n.get("title", "").lower() or q in n.get("content", "").lower()
    ]
    matches = matches[:limit]

    if not matches:
        return f"No notes matching '{query}'."

    lines = [f"Found {len(matches)} note(s) matching '{query}':"]
    for n in matches:
        preview = n.get("content", "")[:80].replace("\n", " ")
        lines.append(f"- {n['title']} (id: {n['id']}): {preview}")
    return "\n".join(lines)


@mcp.tool()
def get_note(note_id: str) -> str:
    """Get the full content of a note.

    Args:
        note_id: The note's ID.
    """
    if not _is_enabled():
        return _NOT_ENABLED
    notes = _read_notes()
    for n in notes:
        if n["id"] == note_id:
            parts = [f"Title: {n['title']}", f"ID: {n['id']}"]
            if n.get("tags"):
                parts.append(f"Tags: {', '.join(n['tags'])}")
            parts.append(f"Created: {n['created_at']}")
            parts.append(f"Updated: {n['updated_at']}")
            parts.append(f"\n{n['content']}")
            return "\n".join(parts)
    return f"Note '{note_id}' not found."


@mcp.tool()
def update_note(note_id: str, title: str = "", content: str = "") -> str:
    """Update an existing note's title and/or content.

    Args:
        note_id: The note's ID.
        title: New title (leave empty to keep current).
        content: New content (leave empty to keep current).
    """
    if not _is_enabled():
        return _NOT_ENABLED
    notes = _read_notes()
    for n in notes:
        if n["id"] == note_id:
            if title:
                n["title"] = title
            if content:
                n["content"] = content
            n["updated_at"] = datetime.now(timezone.utc).isoformat()
            _write_notes(notes)
            return f"Note '{note_id}' updated."
    return f"Note '{note_id}' not found."


@mcp.tool()
def delete_note(note_id: str) -> str:
    """Delete a note.

    Args:
        note_id: The note's ID.
    """
    if not _is_enabled():
        return _NOT_ENABLED
    notes = _read_notes()
    original_count = len(notes)
    notes = [n for n in notes if n["id"] != note_id]
    if len(notes) == original_count:
        return f"Note '{note_id}' not found."
    _write_notes(notes)
    return f"Note '{note_id}' deleted."


@mcp.tool()
def set_reminder(message: str, time_str: str) -> str:
    """Set a reminder for a specific time.

    Args:
        message: What to be reminded about.
        time_str: When to be reminded (e.g., "in 30 min", "tomorrow 9am", "2026-03-01 14:00").
    """
    if not _is_enabled():
        return _NOT_ENABLED
    parsed_time = _parse_time(time_str)
    reminder = {
        "id": uuid.uuid4().hex[:8],
        "message": message,
        "time": parsed_time,
        "created_at": datetime.now(timezone.utc).isoformat(),
    }
    reminders = _read_reminders()
    reminders.append(reminder)
    _write_reminders(reminders)
    return f"Reminder set: '{message}' at {parsed_time} (id: {reminder['id']})"


@mcp.tool()
def list_reminders(include_past: bool = False) -> str:
    """List reminders.

    Args:
        include_past: Include past reminders that have already triggered.
    """
    if not _is_enabled():
        return _NOT_ENABLED
    reminders = _read_reminders()
    now = datetime.now().isoformat()

    if not include_past:
        reminders = [r for r in reminders if r.get("time", "") >= now]

    reminders = sorted(reminders, key=lambda r: r.get("time", ""))

    if not reminders:
        return "No upcoming reminders."

    lines = [f"Reminders ({len(reminders)}):"]
    for r in reminders:
        status = "" if r.get("time", "") >= now else " (past)"
        lines.append(f"- {r['message']} — {r['time']}{status} (id: {r['id']})")
    return "\n".join(lines)


@mcp.tool()
def delete_reminder(reminder_id: str) -> str:
    """Delete a reminder.

    Args:
        reminder_id: The reminder's ID.
    """
    if not _is_enabled():
        return _NOT_ENABLED
    reminders = _read_reminders()
    original_count = len(reminders)
    reminders = [r for r in reminders if r["id"] != reminder_id]
    if len(reminders) == original_count:
        return f"Reminder '{reminder_id}' not found."
    _write_reminders(reminders)
    return f"Reminder '{reminder_id}' deleted."


if __name__ == "__main__":
    mcp.run()
