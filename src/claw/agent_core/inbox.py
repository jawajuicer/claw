"""Inter-agent inbox for cross-session message passing.

Provides a shared inbox where cron jobs, bridges, and other
subsystems can leave messages for the user to review later.
"""

from __future__ import annotations

import json
import logging
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from uuid import uuid4

from claw.config import PROJECT_ROOT

log = logging.getLogger(__name__)

MAX_MESSAGES = 100


class Inbox:
    """Shared message inbox with persistence.

    Messages are stored in data/sessions/inbox.json.
    Each message has: id, sender, subject, body, priority, timestamp, read.
    """

    def __init__(self, broadcaster=None) -> None:
        self._broadcaster = broadcaster
        self._storage_path = PROJECT_ROOT / "data" / "sessions" / "inbox.json"
        self._messages: list[dict[str, Any]] = []
        self._load()

    def _load(self) -> None:
        if not self._storage_path.exists():
            return
        try:
            data = json.loads(self._storage_path.read_text())
            if not isinstance(data, list):
                log.warning("Inbox file has invalid format (expected list), resetting")
                self._messages = []
                return
            self._messages = data
            log.info("Loaded %d inbox messages", len(self._messages))
        except (json.JSONDecodeError, OSError):
            log.warning("Failed to read inbox file")
            self._messages = []

    def _save(self) -> None:
        self._storage_path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self._storage_path.with_suffix(".tmp")
        fd = os.open(str(tmp), os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
        with os.fdopen(fd, "w") as f:
            json.dump(self._messages, f, indent=2)
        tmp.replace(self._storage_path)

    async def send(
        self,
        sender: str,
        subject: str,
        body: str,
        priority: str = "normal",
    ) -> dict:
        """Add a message to the inbox.

        Args:
            sender: Who sent the message (e.g., "cron:morning_report", "bridge:telegram")
            subject: Short subject line
            body: Full message body
            priority: "low", "normal", "high", "urgent"
        """
        if priority not in ("low", "normal", "high", "urgent"):
            priority = "normal"

        # Cap messages
        if len(self._messages) >= MAX_MESSAGES:
            # Remove oldest read messages first, then oldest unread
            read_msgs = [m for m in self._messages if m.get("read")]
            if read_msgs:
                self._messages.remove(read_msgs[0])
            else:
                self._messages.pop(0)

        message = {
            "id": uuid4().hex[:12],
            "sender": sender,
            "subject": subject,
            "body": body,
            "priority": priority,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "read": False,
        }
        self._messages.append(message)
        self._save()

        # Broadcast SSE event
        if self._broadcaster:
            await self._broadcaster.broadcast("inbox", {
                "action": "new_message",
                "message": {
                    "id": message["id"],
                    "sender": sender,
                    "subject": subject,
                    "priority": priority,
                },
                "unread_count": self.unread_count,
            })

        log.info("Inbox message from %s: %s", sender, subject)
        return message

    def check(self, unread_only: bool = False) -> list[dict]:
        """Get inbox messages, optionally filtered to unread only."""
        if unread_only:
            return [m for m in self._messages if not m.get("read")]
        return list(self._messages)

    def mark_read(self, message_id: str) -> bool:
        """Mark a message as read."""
        for msg in self._messages:
            if msg["id"] == message_id:
                msg["read"] = True
                self._save()
                return True
        return False

    def clear(self, read_only: bool = True) -> int:
        """Clear messages from inbox.

        Args:
            read_only: If True, only clear read messages. If False, clear all.

        Returns number of messages cleared.
        """
        before = len(self._messages)
        if read_only:
            self._messages = [m for m in self._messages if not m.get("read")]
        else:
            self._messages.clear()
        after = len(self._messages)
        cleared = before - after
        if cleared > 0:
            self._save()
        return cleared

    @property
    def unread_count(self) -> int:
        return sum(1 for m in self._messages if not m.get("read"))
