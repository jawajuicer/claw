"""Google Calendar MCP server — read and manage Google Calendar events."""

from __future__ import annotations

import sys
from datetime import datetime, timedelta
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import yaml
from dateutil import parser as dateutil_parser
from mcp.server.fastmcp import FastMCP

mcp = FastMCP("GoogleCalendar")

_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
_CONFIG_YAML = _PROJECT_ROOT / "config.yaml"
_services: dict[str, object] = {}


def _load_google_auth() -> dict:
    """Read google_auth from config.yaml fresh each time (no caching)."""
    if _CONFIG_YAML.exists():
        with open(_CONFIG_YAML) as f:
            data = yaml.safe_load(f) or {}
        return data.get("google_auth", {})
    return {}


def _get_accounts() -> dict:
    auth = _load_google_auth()
    return auth.get("accounts", {})


def _get_service(account_label: str):
    """Lazy-init a Google Calendar API service for a specific account."""
    if account_label in _services:
        return _services[account_label]

    from google_auth.auth import get_credentials
    from googleapiclient.discovery import build

    auth = _load_google_auth()
    accounts = auth.get("accounts", {})
    acct = accounts.get(account_label, {})

    creds = get_credentials(
        credentials_file=auth.get("credentials_file", "data/google/credentials.json"),
        token_file=acct.get("token_file", ""),
        scopes=auth.get("scopes", ["https://www.googleapis.com/auth/calendar"]),
    )
    if creds is None:
        return None

    service = build("calendar", "v3", credentials=creds)
    _services[account_label] = service
    return service


def _resolve_calendar(acct_cfg: dict, label: str) -> str:
    """Resolve a calendar label (e.g., 'work') to a calendar ID."""
    cal_cfg = acct_cfg.get("calendar", {})
    if isinstance(cal_cfg, dict):
        if label in ("default", ""):
            return cal_cfg.get("default_calendar", "primary")
        calendars = cal_cfg.get("calendars", {})
        if label in calendars:
            return calendars[label]
        # If the label isn't a known sub-calendar, fall back to primary
        # (prevents account names like "work" being used as calendar IDs)
        return cal_cfg.get("default_calendar", "primary")
    return label or "primary"


def _get_timezone(acct_cfg: dict) -> str:
    cal_cfg = acct_cfg.get("calendar", {})
    if isinstance(cal_cfg, dict):
        return cal_cfg.get("timezone", "America/New_York")
    return "America/New_York"


def _resolve(account: str, service_name: str = "calendar"):
    """Common account resolution + service init. Returns (service, acct_cfg, error)."""
    from google_auth.auth import resolve_account

    accounts = _get_accounts()
    label, err, acct_cfg = resolve_account(accounts, service_name, account)
    if err:
        return None, None, err
    svc = _get_service(label)
    if svc is None:
        return None, None, (
            "Authentication failed — please re-authorize in Settings."
        )
    return svc, acct_cfg, ""


@mcp.tool()
def list_calendars(account: str = "") -> str:
    """List all Google calendars available to the user.

    Args:
        account: Google account label (e.g., "personal", "work"). Leave empty to auto-select.
    """
    service, _, err = _resolve(account)
    if err:
        return err

    try:
        result = service.calendarList().list().execute()
        items = result.get("items", [])
        if not items:
            return "No calendars found."

        lines = [f"Google Calendars ({len(items)}):"]
        for cal in items:
            primary = " (primary)" if cal.get("primary") else ""
            role = cal.get("accessRole", "")
            lines.append(f"- {cal['summary']}{primary} [{cal['id']}] ({role})")
        return "\n".join(lines)
    except Exception as e:
        return f"Error listing calendars: {e}"


@mcp.tool()
def list_events(calendar: str = "default", date: str = "today", days: int = 7, account: str = "") -> str:
    """List upcoming events from Google Calendar.

    Args:
        calendar: Which calendar to query. Use "default" for primary calendar. Only use a specific calendar ID if the user explicitly names a sub-calendar.
        date: Start date (e.g., "today", "2026-03-01").
        days: Number of days to look ahead (1-90).
        account: Google account label (e.g., "personal", "work"). Leave empty to auto-select.
    """
    service, acct_cfg, err = _resolve(account)
    if err:
        return err

    days = max(1, min(90, days))
    cal_id = _resolve_calendar(acct_cfg, calendar)

    try:
        if date.lower() == "today":
            start_dt = datetime.now().replace(hour=0, minute=0, second=0, microsecond=0)
        else:
            start_dt = dateutil_parser.parse(date)
    except (ValueError, TypeError):
        start_dt = datetime.now().replace(hour=0, minute=0, second=0, microsecond=0)

    end_dt = start_dt + timedelta(days=days)
    time_min = start_dt.isoformat() + "Z" if start_dt.tzinfo is None else start_dt.isoformat()
    time_max = end_dt.isoformat() + "Z" if end_dt.tzinfo is None else end_dt.isoformat()

    try:
        result = service.events().list(
            calendarId=cal_id,
            timeMin=time_min,
            timeMax=time_max,
            singleEvents=True,
            orderBy="startTime",
            maxResults=50,
        ).execute()

        items = result.get("items", [])
        if not items:
            return f"No events in the next {days} day(s)."

        lines = [f"Events ({len(items)}):"]
        for event in items:
            start = event["start"].get("dateTime", event["start"].get("date", ""))
            summary = event.get("summary", "(No title)")
            try:
                dt = dateutil_parser.parse(start)
                time_str = dt.strftime("%a %b %d, %I:%M %p")
            except (ValueError, TypeError):
                time_str = start
            lines.append(f"- {summary} ({time_str}) [id: {event['id'][:12]}]")
        return "\n".join(lines)
    except Exception as e:
        return f"Error listing events: {e}"


@mcp.tool()
def create_event(
    title: str,
    start: str,
    end: str = "",
    calendar: str = "default",
    description: str = "",
    account: str = "",
) -> str:
    """Create a Google Calendar event.

    Args:
        title: Event title.
        start: Start time (e.g., "2026-03-01 14:00", "tomorrow 3pm").
        end: End time (optional, defaults to 1 hour after start).
        calendar: Which calendar to use. Use "default" for primary calendar.
        description: Event description.
        account: Google account label (e.g., "personal", "work"). Leave empty to auto-select.
    """
    service, acct_cfg, err = _resolve(account)
    if err:
        return err

    cal_id = _resolve_calendar(acct_cfg, calendar)
    tz = _get_timezone(acct_cfg)

    try:
        start_dt = dateutil_parser.parse(start, fuzzy=True)
    except (ValueError, TypeError):
        return f"Could not parse start time: '{start}'"

    if end:
        try:
            end_dt = dateutil_parser.parse(end, fuzzy=True)
        except (ValueError, TypeError):
            return f"Could not parse end time: '{end}'"
    else:
        end_dt = start_dt + timedelta(hours=1)

    event_body = {
        "summary": title,
        "start": {"dateTime": start_dt.isoformat(), "timeZone": tz},
        "end": {"dateTime": end_dt.isoformat(), "timeZone": tz},
    }
    if description:
        event_body["description"] = description

    try:
        event = service.events().insert(calendarId=cal_id, body=event_body).execute()
        return f"Event created: '{title}' on {start_dt.strftime('%b %d at %I:%M %p')} (id: {event['id'][:12]})"
    except Exception as e:
        return f"Error creating event: {e}"


@mcp.tool()
def update_event(
    event_id: str,
    calendar: str = "default",
    title: str = "",
    start: str = "",
    end: str = "",
    description: str = "",
    account: str = "",
) -> str:
    """Update an existing Google Calendar event.

    Args:
        event_id: The event ID.
        calendar: Which calendar. Use "default" for primary calendar.
        title: New title (leave empty to keep current).
        start: New start time (leave empty to keep current).
        end: New end time (leave empty to keep current).
        description: New description (leave empty to keep current).
        account: Google account label (e.g., "personal", "work"). Leave empty to auto-select.
    """
    service, acct_cfg, err = _resolve(account)
    if err:
        return err

    cal_id = _resolve_calendar(acct_cfg, calendar)
    tz = _get_timezone(acct_cfg)

    try:
        event = service.events().get(calendarId=cal_id, eventId=event_id).execute()
    except Exception as e:
        return f"Error fetching event: {e}"

    if title:
        event["summary"] = title
    if start:
        try:
            dt = dateutil_parser.parse(start, fuzzy=True)
            event["start"] = {"dateTime": dt.isoformat(), "timeZone": tz}
        except (ValueError, TypeError):
            return f"Could not parse start time: '{start}'"
    if end:
        try:
            dt = dateutil_parser.parse(end, fuzzy=True)
            event["end"] = {"dateTime": dt.isoformat(), "timeZone": tz}
        except (ValueError, TypeError):
            return f"Could not parse end time: '{end}'"
    if description:
        event["description"] = description

    try:
        updated = service.events().update(calendarId=cal_id, eventId=event_id, body=event).execute()
        return f"Event '{updated.get('summary', event_id)}' updated."
    except Exception as e:
        return f"Error updating event: {e}"


@mcp.tool()
def delete_event(event_id: str, calendar: str = "default", account: str = "") -> str:
    """Delete a Google Calendar event.

    Args:
        event_id: The event ID.
        calendar: Which calendar. Use "default" for primary calendar.
        account: Google account label (e.g., "personal", "work"). Leave empty to auto-select.
    """
    service, acct_cfg, err = _resolve(account)
    if err:
        return err

    cal_id = _resolve_calendar(acct_cfg, calendar)

    try:
        service.events().delete(calendarId=cal_id, eventId=event_id).execute()
        return f"Event '{event_id}' deleted."
    except Exception as e:
        return f"Error deleting event: {e}"


@mcp.tool()
def search_events(query: str, calendar: str = "default", limit: int = 10, account: str = "") -> str:
    """Search Google Calendar events by text.

    Args:
        query: Search text.
        calendar: Which calendar. Use "default" for primary calendar.
        limit: Maximum results (1-50).
        account: Google account label (e.g., "personal", "work"). Leave empty to auto-select.
    """
    service, acct_cfg, err = _resolve(account)
    if err:
        return err

    limit = max(1, min(50, limit))
    cal_id = _resolve_calendar(acct_cfg, calendar)

    try:
        result = service.events().list(
            calendarId=cal_id,
            q=query,
            singleEvents=True,
            orderBy="startTime",
            maxResults=limit,
        ).execute()

        items = result.get("items", [])
        if not items:
            return f"No events matching '{query}'."

        lines = [f"Found {len(items)} event(s) matching '{query}':"]
        for event in items:
            start = event["start"].get("dateTime", event["start"].get("date", ""))
            summary = event.get("summary", "(No title)")
            lines.append(f"- {summary} ({start}) [id: {event['id'][:12]}]")
        return "\n".join(lines)
    except Exception as e:
        return f"Error searching events: {e}"


if __name__ == "__main__":
    mcp.run()
