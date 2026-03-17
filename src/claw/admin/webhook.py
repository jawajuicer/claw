"""Webhook handler for inbound event triggers."""

from __future__ import annotations

import hashlib
import hmac
import json
import logging
from typing import TYPE_CHECKING

from pydantic import BaseModel

if TYPE_CHECKING:
    from claw.admin.sse import StatusBroadcaster
    from claw.agent_core.agent import Agent

log = logging.getLogger(__name__)


class WebhookEvent(BaseModel):
    """Inbound webhook event."""

    type: str  # "message", "reminder", "notification"
    payload: dict = {}
    source: str = ""  # optional: who sent it (e.g., "home-assistant")


def verify_signature(body: bytes, signature: str, secret: str) -> bool:
    """Verify HMAC-SHA256 signature. Returns True if valid or no secret configured."""
    if not secret:
        return True  # No verification configured
    if not signature:
        return False
    expected = hmac.new(secret.encode(), body, hashlib.sha256).hexdigest()
    # Support both "sha256=xxx" prefix and bare hex
    if signature.startswith("sha256="):
        signature = signature[7:]
    return hmac.compare_digest(expected, signature)


async def handle_message(
    event: WebhookEvent,
    agent: Agent,
    broadcaster: StatusBroadcaster,
    tools: list[dict] | None = None,
) -> dict:
    """Handle a 'message' event -- feed text to the agent."""
    if agent is None:
        return {"status": "error", "message": "Agent not initialized"}

    text = event.payload.get("text", "").strip()
    if not text:
        return {"status": "error", "message": "Missing payload.text"}

    await broadcaster.update_state("processing")
    await broadcaster.update_transcription(f"[webhook:{event.source or 'unknown'}] {text}")

    try:
        response = await agent.process_utterance(text, tools=tools)
        await broadcaster.update_response(response)
    except Exception as e:
        log.warning("Webhook message processing failed: %s", e, exc_info=True)
        response = "Sorry, something went wrong processing your request."
    finally:
        await broadcaster.update_state("idle")

    return {"status": "ok", "type": "message", "response": response}


async def handle_notification(
    event: WebhookEvent,
    broadcaster: StatusBroadcaster,
    tts=None,
) -> dict:
    """Handle a 'notification' event -- broadcast + optional TTS."""
    message = event.payload.get("message", "").strip()
    title = event.payload.get("title", "Notification")
    if not message:
        return {"status": "error", "message": "Missing payload.message"}

    # Broadcast via SSE
    await broadcaster.broadcast("notification", {
        "title": title,
        "message": message,
        "source": event.source,
    })

    # TTS announcement
    if tts and event.payload.get("speak", True):
        try:
            await tts.speak(f"{title}: {message}")
        except Exception:
            log.exception("Webhook TTS failed")

    return {"status": "ok", "type": "notification", "title": title, "message": message}


async def handle_reminder(event: WebhookEvent) -> dict:
    """Handle a 'reminder' event -- create a reminder via the notes system."""
    message = event.payload.get("message", "").strip()
    time_str = event.payload.get("time", "").strip()
    if not message:
        return {"status": "error", "message": "Missing payload.message"}
    if not time_str:
        return {"status": "error", "message": "Missing payload.time"}

    # Write directly to reminders file (same as MCP notes server)
    import os
    import uuid
    from datetime import datetime
    from pathlib import Path

    from claw.config import PROJECT_ROOT, get_settings

    storage_dir = PROJECT_ROOT / get_settings().notes.storage_dir
    storage_dir.mkdir(parents=True, exist_ok=True)
    reminders_file = storage_dir / "reminders.json"

    reminders = []
    if reminders_file.exists():
        try:
            reminders = json.loads(reminders_file.read_text())
        except (json.JSONDecodeError, OSError) as e:
            log.warning("Failed to read reminders file: %s", e)
            reminders = []

    reminder = {
        "id": uuid.uuid4().hex[:8],
        "message": message,
        "time": time_str,
        "created_at": datetime.now().isoformat(),
        "source": event.source or "webhook",
    }
    reminders.append(reminder)

    # Atomic write
    try:
        tmp = reminders_file.with_suffix(".tmp")
        fd = os.open(str(tmp), os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
        with os.fdopen(fd, "w") as f:
            json.dump(reminders, f, indent=2)
        tmp.replace(reminders_file)
    except OSError:
        log.exception("Failed to write reminders file")
        return {"status": "error", "message": "Failed to save reminder"}

    return {"status": "ok", "type": "reminder", "id": reminder["id"], "time": time_str}
