"""MCP tool server for the inter-agent inbox."""

from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4

from mcp.server.fastmcp import FastMCP

mcp = FastMCP("inbox")

_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
_INBOX_FILE = _PROJECT_ROOT / "data" / "sessions" / "inbox.json"


def _load_inbox() -> list[dict]:
    if not _INBOX_FILE.exists():
        return []
    try:
        return json.loads(_INBOX_FILE.read_text())
    except (json.JSONDecodeError, OSError):
        return []


def _save_inbox(messages: list[dict]) -> None:
    _INBOX_FILE.parent.mkdir(parents=True, exist_ok=True)
    tmp = _INBOX_FILE.with_suffix(".tmp")
    fd = os.open(str(tmp), os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
    with os.fdopen(fd, "w") as f:
        json.dump(messages, f, indent=2)
    tmp.replace(_INBOX_FILE)


@mcp.tool()
def send_to_inbox(sender: str, subject: str, body: str, priority: str = "normal") -> str:
    """Send a message to the shared inbox for later review.

    Args:
        sender: Who is sending (e.g., "cron:daily_report", "user:chuck")
        subject: Brief subject line
        body: Full message content
        priority: Message priority - "low", "normal", "high", or "urgent"
    """
    if priority not in ("low", "normal", "high", "urgent"):
        priority = "normal"

    messages = _load_inbox()

    # Cap at 100 messages
    while len(messages) >= 100:
        # Remove oldest read first
        read_indices = [i for i, m in enumerate(messages) if m.get("read")]
        if read_indices:
            messages.pop(read_indices[0])
        else:
            messages.pop(0)

    message = {
        "id": uuid4().hex[:12],
        "sender": sender,
        "subject": subject,
        "body": body,
        "priority": priority,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "read": False,
    }
    messages.append(message)
    _save_inbox(messages)

    return json.dumps({"status": "sent", "message_id": message["id"]})


@mcp.tool()
def check_inbox(unread_only: bool = True) -> str:
    """Check the shared inbox for messages.

    Args:
        unread_only: If true, only show unread messages
    """
    messages = _load_inbox()
    if unread_only:
        messages = [m for m in messages if not m.get("read")]

    if not messages:
        return "Inbox is empty." if not unread_only else "No unread messages."

    return json.dumps(messages, indent=2)


@mcp.tool()
def clear_inbox(read_only: bool = True) -> str:
    """Clear messages from the inbox.

    Args:
        read_only: If true, only clear read messages. If false, clear all messages.
    """
    messages = _load_inbox()
    before = len(messages)
    if read_only:
        messages = [m for m in messages if not m.get("read")]
    else:
        messages = []
    _save_inbox(messages)
    cleared = before - len(messages)
    return f"Cleared {cleared} message(s). {len(messages)} remaining."


if __name__ == "__main__":
    mcp.run()
