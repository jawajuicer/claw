"""Background scheduler for reminders and scheduled tasks."""

from __future__ import annotations

import asyncio
import json
import logging
import os
from datetime import datetime
from pathlib import Path

from claw.config import PROJECT_ROOT, get_settings

log = logging.getLogger(__name__)


class Scheduler:
    """Polls reminders and fires them when due via TTS and SSE."""

    def __init__(self, broadcaster, tts=None, router=None) -> None:
        self._broadcaster = broadcaster
        self._tts = tts
        self._router = router  # ToolRouter, for pausing music
        self._running = False

    @property
    def running(self) -> bool:
        return self._running

    def _reminders_file(self) -> Path:
        cfg = get_settings().notes
        return PROJECT_ROOT / cfg.storage_dir / "reminders.json"

    def _load_reminders(self) -> list[dict]:
        """Read reminders from the JSON file."""
        f = self._reminders_file()
        if not f.exists():
            return []
        try:
            return json.loads(f.read_text())
        except (json.JSONDecodeError, OSError):
            log.warning("Failed to read reminders file")
            return []

    def _remove_reminder(self, reminder_id: str) -> None:
        """Remove a fired reminder from the JSON file (atomic write)."""
        f = self._reminders_file()
        reminders = self._load_reminders()
        reminders = [r for r in reminders if r.get("id") != reminder_id]
        # Atomic write with restricted permissions
        tmp = f.with_suffix(".tmp")
        fd = os.open(str(tmp), os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
        with os.fdopen(fd, "w") as fh:
            json.dump(reminders, fh, indent=2)
        tmp.replace(f)

    async def _fire_reminder(self, reminder: dict) -> None:
        """Fire a single reminder: SSE notification + TTS announcement."""
        message = reminder.get("message", "Reminder")
        log.info("Firing reminder: %s", message)
        try:
            # Broadcast SSE event
            await self._broadcaster.broadcast("reminder", {
                "reminder_id": reminder.get("id"),
                "message": message,
                "time": reminder.get("time"),
            })

            # TTS announcement
            cfg = get_settings().scheduler
            if cfg.announce_tts and self._tts:
                try:
                    # Pause music if playing
                    was_playing = False
                    if self._router:
                        try:
                            status_json = await self._router.call_tool("get_status", {})
                            status = json.loads(status_json)
                            if status.get("playing"):
                                await self._router.call_tool("pause", {})
                                was_playing = True
                        except Exception:
                            pass  # Music may not be playing

                    await self._tts.speak(f"Reminder: {message}")

                    # Resume music only if it was playing before
                    if self._router and was_playing:
                        try:
                            await self._router.call_tool("resume", {})
                        except Exception:
                            pass
                except Exception:
                    log.exception("TTS announcement failed for reminder")
        except Exception:
            log.exception("Error firing reminder: %s", message)
        finally:
            # Always remove the fired reminder, even if broadcast/TTS failed
            self._remove_reminder(reminder.get("id", ""))

    async def _check_missed(self) -> None:
        """On startup, fire any reminders that were missed during downtime."""
        now = datetime.now()
        reminders = self._load_reminders()
        for r in reminders:
            try:
                reminder_time = datetime.fromisoformat(r["time"])
            except (ValueError, KeyError) as e:
                log.warning("Skipping reminder with invalid time: %s (error: %s)", r.get("id", "?"), e)
                continue
            if reminder_time <= now:
                log.info("Missed reminder (was due %s): %s", r["time"], r.get("message"))
                await self._fire_reminder(r)
                await asyncio.sleep(2)  # space out rapid-fire missed reminders

    async def run(self) -> None:
        """Main scheduler loop. Polls reminders every poll_interval seconds."""
        cfg = get_settings().scheduler
        if not cfg.enabled:
            log.info("Scheduler disabled")
            return

        self._running = True
        log.info("Scheduler started (poll every %ds)", cfg.poll_interval)

        # Check for missed reminders on startup
        await self._check_missed()

        try:
            while self._running:
                await asyncio.sleep(cfg.poll_interval)
                await self._poll()
        except asyncio.CancelledError:
            pass
        finally:
            self._running = False
            log.info("Scheduler stopped")

    async def _poll(self) -> None:
        """Check for due reminders and fire them."""
        now = datetime.now()
        reminders = self._load_reminders()

        for r in reminders:
            try:
                reminder_time = datetime.fromisoformat(r["time"])
            except (ValueError, KeyError) as e:
                log.warning("Skipping reminder with invalid time: %s (error: %s)", r.get("id", "?"), e)
                continue
            if reminder_time <= now:
                await self._fire_reminder(r)

    def stop(self) -> None:
        self._running = False

    def get_upcoming(self, limit: int = 10) -> list[dict]:
        """Return upcoming reminders sorted by time."""
        now = datetime.now().isoformat()
        reminders = self._load_reminders()
        upcoming = [r for r in reminders if r.get("time", "") >= now]
        upcoming.sort(key=lambda r: r.get("time", ""))
        return upcoming[:limit]
