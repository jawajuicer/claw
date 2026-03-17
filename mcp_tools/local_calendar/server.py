"""Local Calendar MCP server — private on-device calendar with JSON storage."""

from __future__ import annotations

import json
import logging
import uuid
from datetime import datetime, timedelta, timezone
from pathlib import Path

import yaml
from dateutil import parser as dateutil_parser
from mcp.server.fastmcp import FastMCP

log = logging.getLogger(__name__)

mcp = FastMCP("LocalCalendar")

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
        _config = data.get("local_calendar", {})
    else:
        _config = {}
    return _config


def _is_enabled() -> bool:
    cfg = _load_config()
    return cfg.get("enabled", True)


_NOT_ENABLED = "Local Calendar is currently disabled. Enable it in Settings."


def _storage_file() -> Path:
    cfg = _load_config()
    rel = cfg.get("storage_file", "data/calendar/local.json")
    f = _PROJECT_ROOT / rel
    f.parent.mkdir(parents=True, exist_ok=True)
    return f


def _read_events() -> list[dict]:
    f = _storage_file()
    if not f.exists():
        return []
    return json.loads(f.read_text())


def _write_events(events: list[dict]) -> None:
    _storage_file().write_text(json.dumps(events, indent=2))


def _parse_datetime(s: str) -> str | None:
    """Parse a date/time string into ISO format."""
    lower = s.strip().lower()
    now = datetime.now()

    if lower == "today":
        return now.replace(hour=0, minute=0, second=0, microsecond=0).isoformat()
    if lower == "tomorrow":
        return (now + timedelta(days=1)).replace(hour=0, minute=0, second=0, microsecond=0).isoformat()
    if lower == "now":
        return now.isoformat()

    try:
        dt = dateutil_parser.parse(s, fuzzy=True)
        return dt.isoformat()
    except (ValueError, TypeError):
        log.warning("Could not parse datetime: %r", s)
        return None


@mcp.tool()
def local_add_event(title: str, start: str, end: str = "", description: str = "") -> str:
    """Add an event to the LOCAL offline calendar (private, not synced to Google). Only use this when the user explicitly asks for a local/private/offline event.

    Args:
        title: Event title.
        start: Start time (e.g., "2026-03-01 14:00", "tomorrow 3pm").
        end: End time (optional, defaults to 1 hour after start).
        description: Event description.
    """
    if not _is_enabled():
        return _NOT_ENABLED
    start_iso = _parse_datetime(start)
    if start_iso is None:
        return f"Could not parse start time: '{start}'"
    if end:
        end_iso = _parse_datetime(end)
        if end_iso is None:
            return f"Could not parse end time: '{end}'"
    else:
        try:
            start_dt = dateutil_parser.parse(start_iso)
            end_iso = (start_dt + timedelta(hours=1)).isoformat()
        except (ValueError, TypeError):
            end_iso = ""

    event = {
        "id": uuid.uuid4().hex[:8],
        "title": title,
        "start": start_iso,
        "end": end_iso,
        "description": description,
        "created_at": datetime.now(timezone.utc).isoformat(),
    }
    events = _read_events()
    events.append(event)
    _write_events(events)
    return f"Event created: '{title}' on {start_iso} (id: {event['id']})"


@mcp.tool()
def local_list_events(date: str = "today", days: int = 7) -> str:
    """List upcoming events from the LOCAL offline calendar (not Google Calendar).

    Args:
        date: Start date to list from (e.g., "today", "2026-03-01").
        days: Number of days to look ahead (1-90).
    """
    if not _is_enabled():
        return _NOT_ENABLED
    days = max(1, min(90, days))
    events = _read_events()

    try:
        if date.lower() == "today":
            start_dt = datetime.now().replace(hour=0, minute=0, second=0, microsecond=0)
        else:
            start_dt = dateutil_parser.parse(date)
    except (ValueError, TypeError):
        start_dt = datetime.now().replace(hour=0, minute=0, second=0, microsecond=0)

    end_dt = start_dt + timedelta(days=days)
    start_iso = start_dt.isoformat()
    end_iso = end_dt.isoformat()

    filtered = [
        e for e in events
        if e.get("start", "") >= start_iso and e.get("start", "") < end_iso
    ]
    filtered = sorted(filtered, key=lambda e: e.get("start", ""))

    if not filtered:
        return f"No events in the next {days} day(s)."

    lines = [f"Events ({len(filtered)}):"]
    for e in filtered:
        time_str = e["start"]
        try:
            dt = dateutil_parser.parse(time_str)
            time_str = dt.strftime("%a %b %d, %I:%M %p")
        except (ValueError, TypeError):
            pass
        desc = f" — {e['description']}" if e.get("description") else ""
        lines.append(f"- {e['title']} ({time_str}){desc} [id: {e['id']}]")
    return "\n".join(lines)


@mcp.tool()
def local_get_event(event_id: str) -> str:
    """Get full details of a LOCAL offline calendar event.

    Args:
        event_id: The event's ID.
    """
    if not _is_enabled():
        return _NOT_ENABLED
    events = _read_events()
    for e in events:
        if e["id"] == event_id:
            parts = [
                f"Title: {e['title']}",
                f"ID: {e['id']}",
                f"Start: {e['start']}",
                f"End: {e.get('end', 'N/A')}",
            ]
            if e.get("description"):
                parts.append(f"Description: {e['description']}")
            parts.append(f"Created: {e['created_at']}")
            return "\n".join(parts)
    return f"Event '{event_id}' not found."


@mcp.tool()
def local_update_event(
    event_id: str,
    title: str = "",
    start: str = "",
    end: str = "",
    description: str = "",
) -> str:
    """Update an existing LOCAL offline calendar event.

    Args:
        event_id: The event's ID.
        title: New title (leave empty to keep current).
        start: New start time (leave empty to keep current).
        end: New end time (leave empty to keep current).
        description: New description (leave empty to keep current).
    """
    if not _is_enabled():
        return _NOT_ENABLED
    events = _read_events()
    for e in events:
        if e["id"] == event_id:
            if title:
                e["title"] = title
            if start:
                parsed = _parse_datetime(start)
                if parsed is None:
                    return f"Could not parse start time: '{start}'"
                e["start"] = parsed
            if end:
                parsed = _parse_datetime(end)
                if parsed is None:
                    return f"Could not parse end time: '{end}'"
                e["end"] = parsed
            if description:
                e["description"] = description
            _write_events(events)
            return f"Event '{event_id}' updated."
    return f"Event '{event_id}' not found."


@mcp.tool()
def local_delete_event(event_id: str) -> str:
    """Delete a LOCAL offline calendar event.

    Args:
        event_id: The event's ID.
    """
    if not _is_enabled():
        return _NOT_ENABLED
    events = _read_events()
    original_count = len(events)
    events = [e for e in events if e["id"] != event_id]
    if len(events) == original_count:
        return f"Event '{event_id}' not found."
    _write_events(events)
    return f"Event '{event_id}' deleted."


@mcp.tool()
def local_search_events(query: str, limit: int = 10) -> str:
    """Search LOCAL offline calendar events by title or description.

    Args:
        query: Search text.
        limit: Maximum results (1-50).
    """
    if not _is_enabled():
        return _NOT_ENABLED
    events = _read_events()
    limit = max(1, min(50, limit))
    q = query.lower()

    matches = [
        e for e in events
        if q in e.get("title", "").lower() or q in e.get("description", "").lower()
    ]
    matches = sorted(matches, key=lambda e: e.get("start", ""))[:limit]

    if not matches:
        return f"No events matching '{query}'."

    lines = [f"Found {len(matches)} event(s) matching '{query}':"]
    for e in matches:
        lines.append(f"- {e['title']} ({e['start']}) [id: {e['id']}]")
    return "\n".join(lines)


if __name__ == "__main__":
    mcp.run()
