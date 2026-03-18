"""Tests for claw.scheduler.scheduler — Scheduler loop, polling, firing, and persistence."""

from __future__ import annotations

import json
import os
from datetime import datetime, timedelta
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_reminder(
    reminder_id: str = "r1",
    message: str = "Test reminder",
    time: datetime | None = None,
) -> dict:
    """Build a reminder dict matching the JSON schema."""
    if time is None:
        time = datetime.now() - timedelta(minutes=1)  # due by default
    return {
        "id": reminder_id,
        "message": message,
        "time": time.isoformat(),
    }


def _write_reminders(path: Path, reminders: list[dict]) -> None:
    """Write a list of reminders to the JSON file, creating parent dirs."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(reminders, indent=2))


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture()
def tmp_scheduler(tmp_config):
    """Provide a Scheduler wired to a temporary directory with mocked deps.

    Patches PROJECT_ROOT in the scheduler module (which imports it by value)
    so that _reminders_file() resolves under the fixture's temporary directory.
    """
    import claw.config as cfg_mod
    import claw.scheduler.scheduler as sched_mod

    # tmp_config patches claw.config.PROJECT_ROOT, but the scheduler module
    # imported it by name (`from claw.config import PROJECT_ROOT`), so we
    # must also patch the scheduler module's own binding.
    tmp_path = cfg_mod.PROJECT_ROOT

    with patch.object(sched_mod, "PROJECT_ROOT", tmp_path):
        broadcaster = AsyncMock()
        broadcaster.broadcast = AsyncMock()
        tts = AsyncMock()
        tts.speak = AsyncMock()
        router = AsyncMock()
        router.call_tool = AsyncMock(return_value='{"playing": false}')

        scheduler = sched_mod.Scheduler(broadcaster=broadcaster, tts=tts, router=router)

        # Pre-create the notes storage dir so reminders file location exists
        notes_dir = tmp_path / "data" / "notes"
        notes_dir.mkdir(parents=True, exist_ok=True)

        yield scheduler, tmp_path, broadcaster, tts, router


# ---------------------------------------------------------------------------
# _load_reminders
# ---------------------------------------------------------------------------

class TestLoadReminders:
    """Test reading reminders from the JSON file."""

    def test_no_file_returns_empty(self, tmp_scheduler):
        scheduler, tmp_path, *_ = tmp_scheduler
        result = scheduler._load_reminders()
        assert result == []

    def test_empty_file_returns_empty(self, tmp_scheduler):
        scheduler, tmp_path, *_ = tmp_scheduler
        f = scheduler._reminders_file()
        f.parent.mkdir(parents=True, exist_ok=True)
        f.write_text("")
        result = scheduler._load_reminders()
        assert result == []

    def test_valid_json_returns_list(self, tmp_scheduler):
        scheduler, tmp_path, *_ = tmp_scheduler
        reminders = [_make_reminder("r1", "Take medication")]
        _write_reminders(scheduler._reminders_file(), reminders)
        result = scheduler._load_reminders()
        assert len(result) == 1
        assert result[0]["id"] == "r1"
        assert result[0]["message"] == "Take medication"

    def test_malformed_json_returns_empty(self, tmp_scheduler):
        scheduler, tmp_path, *_ = tmp_scheduler
        f = scheduler._reminders_file()
        f.parent.mkdir(parents=True, exist_ok=True)
        f.write_text("{invalid json!!! [[[")
        result = scheduler._load_reminders()
        assert result == []

    def test_multiple_reminders(self, tmp_scheduler):
        scheduler, tmp_path, *_ = tmp_scheduler
        reminders = [
            _make_reminder("r1", "First"),
            _make_reminder("r2", "Second"),
            _make_reminder("r3", "Third"),
        ]
        _write_reminders(scheduler._reminders_file(), reminders)
        result = scheduler._load_reminders()
        assert len(result) == 3

    def test_os_error_returns_empty(self, tmp_scheduler):
        scheduler, *_ = tmp_scheduler
        with patch.object(Path, "exists", return_value=True), \
             patch.object(Path, "read_text", side_effect=OSError("permission denied")):
            result = scheduler._load_reminders()
            assert result == []


# ---------------------------------------------------------------------------
# _remove_reminder
# ---------------------------------------------------------------------------

class TestRemoveReminder:
    """Test removing a fired reminder via atomic write."""

    def test_removes_matching_reminder(self, tmp_scheduler):
        scheduler, tmp_path, *_ = tmp_scheduler
        reminders = [
            _make_reminder("r1", "Keep me"),
            _make_reminder("r2", "Remove me"),
            _make_reminder("r3", "Keep me too"),
        ]
        _write_reminders(scheduler._reminders_file(), reminders)

        scheduler._remove_reminder("r2")

        result = scheduler._load_reminders()
        assert len(result) == 2
        ids = [r["id"] for r in result]
        assert "r2" not in ids
        assert "r1" in ids
        assert "r3" in ids

    def test_remove_nonexistent_id_is_noop(self, tmp_scheduler):
        scheduler, tmp_path, *_ = tmp_scheduler
        reminders = [_make_reminder("r1", "Stay")]
        _write_reminders(scheduler._reminders_file(), reminders)

        scheduler._remove_reminder("nonexistent")

        result = scheduler._load_reminders()
        assert len(result) == 1
        assert result[0]["id"] == "r1"

    def test_remove_from_empty_file(self, tmp_scheduler):
        scheduler, tmp_path, *_ = tmp_scheduler
        _write_reminders(scheduler._reminders_file(), [])

        scheduler._remove_reminder("r1")

        result = scheduler._load_reminders()
        assert result == []

    def test_remove_last_reminder_leaves_empty_list(self, tmp_scheduler):
        scheduler, tmp_path, *_ = tmp_scheduler
        _write_reminders(scheduler._reminders_file(), [_make_reminder("r1")])

        scheduler._remove_reminder("r1")

        result = scheduler._load_reminders()
        assert result == []

    def test_atomic_write_uses_restricted_permissions(self, tmp_scheduler):
        scheduler, tmp_path, *_ = tmp_scheduler
        _write_reminders(scheduler._reminders_file(), [_make_reminder("r1")])

        with patch("claw.scheduler.scheduler.os.open", wraps=os.open) as mock_open:
            scheduler._remove_reminder("r1")
            # Verify os.open was called with 0o600 permissions
            call_args = mock_open.call_args
            assert call_args[0][2] == 0o600


# ---------------------------------------------------------------------------
# _fire_reminder
# ---------------------------------------------------------------------------

class TestFireReminder:
    """Test firing a single reminder (SSE + TTS + cleanup)."""

    async def test_broadcasts_sse_event(self, tmp_scheduler):
        scheduler, tmp_path, broadcaster, tts, router = tmp_scheduler
        reminder = _make_reminder("r1", "Meeting in 5 minutes")
        _write_reminders(scheduler._reminders_file(), [reminder])

        await scheduler._fire_reminder(reminder)

        broadcaster.broadcast.assert_called_once_with("reminder", {
            "reminder_id": "r1",
            "message": "Meeting in 5 minutes",
            "time": reminder["time"],
        })

    async def test_speaks_via_tts_when_enabled(self, tmp_scheduler, settings):
        scheduler, tmp_path, broadcaster, tts, router = tmp_scheduler
        reminder = _make_reminder("r1", "Drink water")
        _write_reminders(scheduler._reminders_file(), [reminder])

        with patch("claw.scheduler.scheduler.get_settings", return_value=settings):
            settings.scheduler.announce_tts = True
            await scheduler._fire_reminder(reminder)

        tts.speak.assert_called_once_with("Reminder: Drink water")

    async def test_skips_tts_when_disabled(self, tmp_scheduler, settings):
        scheduler, tmp_path, broadcaster, tts, router = tmp_scheduler
        reminder = _make_reminder("r1", "Silent reminder")
        _write_reminders(scheduler._reminders_file(), [reminder])

        settings.scheduler.announce_tts = False
        with patch("claw.scheduler.scheduler.get_settings", return_value=settings):
            await scheduler._fire_reminder(reminder)

        tts.speak.assert_not_called()

    async def test_skips_tts_when_no_tts_engine(self, tmp_scheduler, settings):
        scheduler, tmp_path, broadcaster, tts, router = tmp_scheduler
        scheduler._tts = None
        reminder = _make_reminder("r1", "No TTS available")
        _write_reminders(scheduler._reminders_file(), [reminder])

        with patch("claw.scheduler.scheduler.get_settings", return_value=settings):
            await scheduler._fire_reminder(reminder)

        # No error should occur; TTS simply skipped
        broadcaster.broadcast.assert_called_once()

    async def test_removes_reminder_after_firing(self, tmp_scheduler, settings):
        scheduler, tmp_path, broadcaster, tts, router = tmp_scheduler
        reminder = _make_reminder("r1", "One-shot reminder")
        _write_reminders(scheduler._reminders_file(), [reminder])

        with patch("claw.scheduler.scheduler.get_settings", return_value=settings):
            await scheduler._fire_reminder(reminder)

        remaining = scheduler._load_reminders()
        assert len(remaining) == 0

    async def test_removes_reminder_even_on_broadcast_error(self, tmp_scheduler, settings):
        scheduler, tmp_path, broadcaster, tts, router = tmp_scheduler
        broadcaster.broadcast = AsyncMock(side_effect=RuntimeError("SSE broken"))
        reminder = _make_reminder("r1", "Should still be removed")
        _write_reminders(scheduler._reminders_file(), [reminder])

        with patch("claw.scheduler.scheduler.get_settings", return_value=settings):
            await scheduler._fire_reminder(reminder)

        remaining = scheduler._load_reminders()
        assert len(remaining) == 0

    async def test_removes_reminder_even_on_tts_error(self, tmp_scheduler, settings):
        scheduler, tmp_path, broadcaster, tts, router = tmp_scheduler
        tts.speak = AsyncMock(side_effect=RuntimeError("TTS engine crashed"))
        reminder = _make_reminder("r1", "TTS fail")
        _write_reminders(scheduler._reminders_file(), [reminder])

        with patch("claw.scheduler.scheduler.get_settings", return_value=settings):
            await scheduler._fire_reminder(reminder)

        remaining = scheduler._load_reminders()
        assert len(remaining) == 0

    async def test_pauses_and_resumes_music_for_tts(self, tmp_scheduler, settings):
        scheduler, tmp_path, broadcaster, tts, router = tmp_scheduler
        router.call_tool = AsyncMock(return_value='{"playing": true}')
        reminder = _make_reminder("r1", "Music pause test")
        _write_reminders(scheduler._reminders_file(), [reminder])

        with patch("claw.scheduler.scheduler.get_settings", return_value=settings):
            settings.scheduler.announce_tts = True
            await scheduler._fire_reminder(reminder)

        # Should have called get_status, pause, speak, resume
        calls = [c[0] for c in router.call_tool.call_args_list]
        assert ("get_status", {}) in calls
        assert ("pause", {}) in calls
        assert ("resume", {}) in calls

    async def test_no_music_pause_when_not_playing(self, tmp_scheduler, settings):
        scheduler, tmp_path, broadcaster, tts, router = tmp_scheduler
        router.call_tool = AsyncMock(return_value='{"playing": false}')
        reminder = _make_reminder("r1", "No music")
        _write_reminders(scheduler._reminders_file(), [reminder])

        with patch("claw.scheduler.scheduler.get_settings", return_value=settings):
            settings.scheduler.announce_tts = True
            await scheduler._fire_reminder(reminder)

        call_names = [c[0][0] for c in router.call_tool.call_args_list]
        assert "get_status" in call_names
        assert "pause" not in call_names
        assert "resume" not in call_names

    async def test_no_router_skips_music_control(self, tmp_scheduler, settings):
        scheduler, tmp_path, broadcaster, tts, router = tmp_scheduler
        scheduler._router = None
        reminder = _make_reminder("r1", "No router")
        _write_reminders(scheduler._reminders_file(), [reminder])

        with patch("claw.scheduler.scheduler.get_settings", return_value=settings):
            settings.scheduler.announce_tts = True
            await scheduler._fire_reminder(reminder)

        tts.speak.assert_called_once()
        router.call_tool.assert_not_called()

    async def test_missing_message_defaults_to_reminder(self, tmp_scheduler, settings):
        scheduler, tmp_path, broadcaster, tts, router = tmp_scheduler
        reminder = {"id": "r1", "time": datetime.now().isoformat()}
        _write_reminders(scheduler._reminders_file(), [reminder])

        with patch("claw.scheduler.scheduler.get_settings", return_value=settings):
            await scheduler._fire_reminder(reminder)

        broadcaster.broadcast.assert_called_once()
        call_data = broadcaster.broadcast.call_args[0][1]
        assert call_data["message"] == "Reminder"

    async def test_missing_id_still_removes(self, tmp_scheduler, settings):
        scheduler, tmp_path, broadcaster, tts, router = tmp_scheduler
        # Reminder without an id field
        reminder = {"message": "No ID", "time": datetime.now().isoformat()}
        _write_reminders(scheduler._reminders_file(), [reminder])

        with patch("claw.scheduler.scheduler.get_settings", return_value=settings):
            await scheduler._fire_reminder(reminder)

        # Should not crash; _remove_reminder gets "" as the id
        broadcaster.broadcast.assert_called_once()


# ---------------------------------------------------------------------------
# _poll
# ---------------------------------------------------------------------------

class TestPoll:
    """Test the periodic poll that checks for due reminders."""

    async def test_fires_due_reminders(self, tmp_scheduler, settings):
        scheduler, tmp_path, broadcaster, tts, router = tmp_scheduler
        past_time = datetime.now() - timedelta(minutes=5)
        reminders = [_make_reminder("r1", "Overdue", time=past_time)]
        _write_reminders(scheduler._reminders_file(), reminders)

        with patch("claw.scheduler.scheduler.get_settings", return_value=settings):
            await scheduler._poll()

        broadcaster.broadcast.assert_called_once()

    async def test_ignores_future_reminders(self, tmp_scheduler, settings):
        scheduler, tmp_path, broadcaster, tts, router = tmp_scheduler
        future_time = datetime.now() + timedelta(hours=1)
        reminders = [_make_reminder("r1", "Not yet", time=future_time)]
        _write_reminders(scheduler._reminders_file(), reminders)

        with patch("claw.scheduler.scheduler.get_settings", return_value=settings):
            await scheduler._poll()

        broadcaster.broadcast.assert_not_called()

    async def test_fires_multiple_due_reminders(self, tmp_scheduler, settings):
        scheduler, tmp_path, broadcaster, tts, router = tmp_scheduler
        past = datetime.now() - timedelta(minutes=5)
        reminders = [
            _make_reminder("r1", "First", time=past),
            _make_reminder("r2", "Second", time=past),
        ]
        _write_reminders(scheduler._reminders_file(), reminders)

        with patch("claw.scheduler.scheduler.get_settings", return_value=settings):
            await scheduler._poll()

        assert broadcaster.broadcast.call_count == 2

    async def test_skips_reminder_with_invalid_time(self, tmp_scheduler, settings):
        scheduler, tmp_path, broadcaster, tts, router = tmp_scheduler
        reminders = [{"id": "r1", "message": "Bad time", "time": "not-a-datetime"}]
        _write_reminders(scheduler._reminders_file(), reminders)

        with patch("claw.scheduler.scheduler.get_settings", return_value=settings):
            await scheduler._poll()

        broadcaster.broadcast.assert_not_called()

    async def test_skips_reminder_missing_time_key(self, tmp_scheduler, settings):
        scheduler, tmp_path, broadcaster, tts, router = tmp_scheduler
        reminders = [{"id": "r1", "message": "No time field"}]
        _write_reminders(scheduler._reminders_file(), reminders)

        with patch("claw.scheduler.scheduler.get_settings", return_value=settings):
            await scheduler._poll()

        broadcaster.broadcast.assert_not_called()

    async def test_no_reminders_file_is_noop(self, tmp_scheduler, settings):
        scheduler, *_ = tmp_scheduler
        # No file written — _load_reminders returns []

        with patch("claw.scheduler.scheduler.get_settings", return_value=settings):
            await scheduler._poll()

        # Should complete without error
        scheduler._broadcaster.broadcast.assert_not_called()

    async def test_mixed_due_and_future(self, tmp_scheduler, settings):
        scheduler, tmp_path, broadcaster, tts, router = tmp_scheduler
        past = datetime.now() - timedelta(minutes=5)
        future = datetime.now() + timedelta(hours=1)
        reminders = [
            _make_reminder("r1", "Due", time=past),
            _make_reminder("r2", "Future", time=future),
        ]
        _write_reminders(scheduler._reminders_file(), reminders)

        with patch("claw.scheduler.scheduler.get_settings", return_value=settings):
            await scheduler._poll()

        # Only the due reminder should fire
        assert broadcaster.broadcast.call_count == 1
        call_data = broadcaster.broadcast.call_args[0][1]
        assert call_data["reminder_id"] == "r1"


# ---------------------------------------------------------------------------
# _check_missed
# ---------------------------------------------------------------------------

class TestCheckMissed:
    """Test startup catch-up for missed reminders."""

    async def test_fires_missed_reminders(self, tmp_scheduler, settings):
        scheduler, tmp_path, broadcaster, tts, router = tmp_scheduler
        past = datetime.now() - timedelta(hours=2)
        reminders = [_make_reminder("r1", "Missed one", time=past)]
        _write_reminders(scheduler._reminders_file(), reminders)

        with patch("claw.scheduler.scheduler.get_settings", return_value=settings), \
             patch("claw.scheduler.scheduler.asyncio.sleep", new_callable=AsyncMock):
            await scheduler._check_missed()

        broadcaster.broadcast.assert_called_once()

    async def test_ignores_future_reminders(self, tmp_scheduler, settings):
        scheduler, tmp_path, broadcaster, tts, router = tmp_scheduler
        future = datetime.now() + timedelta(hours=1)
        reminders = [_make_reminder("r1", "Not missed", time=future)]
        _write_reminders(scheduler._reminders_file(), reminders)

        with patch("claw.scheduler.scheduler.get_settings", return_value=settings), \
             patch("claw.scheduler.scheduler.asyncio.sleep", new_callable=AsyncMock):
            await scheduler._check_missed()

        broadcaster.broadcast.assert_not_called()

    async def test_spaces_out_multiple_missed(self, tmp_scheduler, settings):
        scheduler, tmp_path, broadcaster, tts, router = tmp_scheduler
        past = datetime.now() - timedelta(hours=1)
        reminders = [
            _make_reminder("r1", "Missed 1", time=past),
            _make_reminder("r2", "Missed 2", time=past),
        ]
        _write_reminders(scheduler._reminders_file(), reminders)

        mock_sleep = AsyncMock()
        with patch("claw.scheduler.scheduler.get_settings", return_value=settings), \
             patch("claw.scheduler.scheduler.asyncio.sleep", mock_sleep):
            await scheduler._check_missed()

        # sleep(2) should be called between each missed reminder
        assert mock_sleep.call_count == 2
        mock_sleep.assert_any_call(2)

    async def test_skips_invalid_time_gracefully(self, tmp_scheduler, settings):
        scheduler, tmp_path, broadcaster, tts, router = tmp_scheduler
        reminders = [
            {"id": "r1", "message": "Bad time", "time": "garbage"},
            _make_reminder("r2", "Good one", time=datetime.now() - timedelta(minutes=5)),
        ]
        _write_reminders(scheduler._reminders_file(), reminders)

        with patch("claw.scheduler.scheduler.get_settings", return_value=settings), \
             patch("claw.scheduler.scheduler.asyncio.sleep", new_callable=AsyncMock):
            await scheduler._check_missed()

        # Only the valid overdue reminder should fire
        assert broadcaster.broadcast.call_count == 1
        call_data = broadcaster.broadcast.call_args[0][1]
        assert call_data["reminder_id"] == "r2"

    async def test_skips_missing_time_key(self, tmp_scheduler, settings):
        scheduler, tmp_path, broadcaster, tts, router = tmp_scheduler
        reminders = [{"id": "r1", "message": "No time key"}]
        _write_reminders(scheduler._reminders_file(), reminders)

        with patch("claw.scheduler.scheduler.get_settings", return_value=settings), \
             patch("claw.scheduler.scheduler.asyncio.sleep", new_callable=AsyncMock):
            await scheduler._check_missed()

        broadcaster.broadcast.assert_not_called()

    async def test_no_reminders_is_noop(self, tmp_scheduler, settings):
        scheduler, *_ = tmp_scheduler

        with patch("claw.scheduler.scheduler.get_settings", return_value=settings), \
             patch("claw.scheduler.scheduler.asyncio.sleep", new_callable=AsyncMock):
            await scheduler._check_missed()

        scheduler._broadcaster.broadcast.assert_not_called()


# ---------------------------------------------------------------------------
# get_upcoming
# ---------------------------------------------------------------------------

class TestGetUpcoming:
    """Test retrieving upcoming (future) reminders sorted by time."""

    def test_returns_future_reminders_sorted(self, tmp_scheduler):
        scheduler, *_ = tmp_scheduler
        now = datetime.now()
        reminders = [
            _make_reminder("r3", "Third", time=now + timedelta(hours=3)),
            _make_reminder("r1", "First", time=now + timedelta(hours=1)),
            _make_reminder("r2", "Second", time=now + timedelta(hours=2)),
        ]
        _write_reminders(scheduler._reminders_file(), reminders)

        upcoming = scheduler.get_upcoming()
        assert len(upcoming) == 3
        assert upcoming[0]["id"] == "r1"
        assert upcoming[1]["id"] == "r2"
        assert upcoming[2]["id"] == "r3"

    def test_excludes_past_reminders(self, tmp_scheduler):
        scheduler, *_ = tmp_scheduler
        now = datetime.now()
        reminders = [
            _make_reminder("r1", "Past", time=now - timedelta(hours=1)),
            _make_reminder("r2", "Future", time=now + timedelta(hours=1)),
        ]
        _write_reminders(scheduler._reminders_file(), reminders)

        upcoming = scheduler.get_upcoming()
        assert len(upcoming) == 1
        assert upcoming[0]["id"] == "r2"

    def test_respects_limit(self, tmp_scheduler):
        scheduler, *_ = tmp_scheduler
        now = datetime.now()
        reminders = [
            _make_reminder(f"r{i}", f"Reminder {i}", time=now + timedelta(hours=i + 1))
            for i in range(15)
        ]
        _write_reminders(scheduler._reminders_file(), reminders)

        upcoming = scheduler.get_upcoming(limit=5)
        assert len(upcoming) == 5

    def test_default_limit_is_10(self, tmp_scheduler):
        scheduler, *_ = tmp_scheduler
        now = datetime.now()
        reminders = [
            _make_reminder(f"r{i}", f"Reminder {i}", time=now + timedelta(hours=i + 1))
            for i in range(15)
        ]
        _write_reminders(scheduler._reminders_file(), reminders)

        upcoming = scheduler.get_upcoming()
        assert len(upcoming) == 10

    def test_empty_file_returns_empty(self, tmp_scheduler):
        scheduler, *_ = tmp_scheduler
        upcoming = scheduler.get_upcoming()
        assert upcoming == []

    def test_all_past_returns_empty(self, tmp_scheduler):
        scheduler, *_ = tmp_scheduler
        past = datetime.now() - timedelta(hours=1)
        reminders = [
            _make_reminder("r1", "Old 1", time=past),
            _make_reminder("r2", "Old 2", time=past),
        ]
        _write_reminders(scheduler._reminders_file(), reminders)

        upcoming = scheduler.get_upcoming()
        assert upcoming == []

    def test_reminder_missing_time_excluded(self, tmp_scheduler):
        scheduler, *_ = tmp_scheduler
        now = datetime.now()
        reminders = [
            {"id": "r1", "message": "No time"},
            _make_reminder("r2", "Has time", time=now + timedelta(hours=1)),
        ]
        _write_reminders(scheduler._reminders_file(), reminders)

        upcoming = scheduler.get_upcoming()
        assert len(upcoming) == 1
        assert upcoming[0]["id"] == "r2"


# ---------------------------------------------------------------------------
# run
# ---------------------------------------------------------------------------

class TestRun:
    """Test the main scheduler loop lifecycle."""

    async def test_disabled_scheduler_returns_immediately(self, tmp_scheduler, settings):
        scheduler, *_ = tmp_scheduler
        settings.scheduler.enabled = False

        with patch("claw.scheduler.scheduler.get_settings", return_value=settings):
            await scheduler.run()

        assert scheduler.running is False

    async def test_sets_running_flag(self, tmp_scheduler, settings):
        scheduler, *_ = tmp_scheduler
        settings.scheduler.enabled = True

        call_count = 0

        async def fake_sleep(interval):
            nonlocal call_count
            call_count += 1
            if call_count >= 1:
                scheduler.stop()

        with patch("claw.scheduler.scheduler.get_settings", return_value=settings), \
             patch("claw.scheduler.scheduler.asyncio.sleep", side_effect=fake_sleep):
            await scheduler.run()

        # After run completes, running should be False
        assert scheduler.running is False

    async def test_polls_on_interval(self, tmp_scheduler, settings):
        scheduler, *_ = tmp_scheduler
        settings.scheduler.enabled = True
        settings.scheduler.poll_interval = 10

        call_count = 0

        async def fake_sleep(interval):
            nonlocal call_count
            assert interval == 10  # poll_interval
            call_count += 1
            if call_count >= 2:
                scheduler.stop()

        with patch("claw.scheduler.scheduler.get_settings", return_value=settings), \
             patch("claw.scheduler.scheduler.asyncio.sleep", side_effect=fake_sleep), \
             patch.object(scheduler, "_poll", new_callable=AsyncMock) as mock_poll, \
             patch.object(scheduler, "_check_missed", new_callable=AsyncMock):
            await scheduler.run()

        assert mock_poll.call_count == 2

    async def test_checks_missed_on_startup(self, tmp_scheduler, settings):
        scheduler, *_ = tmp_scheduler
        settings.scheduler.enabled = True

        async def stop_after_start(interval):
            scheduler.stop()

        with patch("claw.scheduler.scheduler.get_settings", return_value=settings), \
             patch("claw.scheduler.scheduler.asyncio.sleep", side_effect=stop_after_start), \
             patch.object(scheduler, "_check_missed", new_callable=AsyncMock) as mock_check, \
             patch.object(scheduler, "_poll", new_callable=AsyncMock):
            await scheduler.run()

        mock_check.assert_called_once()

    async def test_handles_cancellation(self, tmp_scheduler, settings):
        import asyncio

        scheduler, *_ = tmp_scheduler
        settings.scheduler.enabled = True

        with patch("claw.scheduler.scheduler.get_settings", return_value=settings), \
             patch("claw.scheduler.scheduler.asyncio.sleep",
                   side_effect=asyncio.CancelledError()), \
             patch.object(scheduler, "_check_missed", new_callable=AsyncMock):
            await scheduler.run()

        assert scheduler.running is False


# ---------------------------------------------------------------------------
# stop
# ---------------------------------------------------------------------------

class TestStop:
    """Test the stop method."""

    def test_stop_clears_running_flag(self, tmp_scheduler):
        scheduler, *_ = tmp_scheduler
        scheduler._running = True
        scheduler.stop()
        assert scheduler.running is False

    def test_stop_is_idempotent(self, tmp_scheduler):
        scheduler, *_ = tmp_scheduler
        scheduler._running = False
        scheduler.stop()
        assert scheduler.running is False


# ---------------------------------------------------------------------------
# Constructor and properties
# ---------------------------------------------------------------------------

class TestSchedulerInit:
    """Test constructor and property access."""

    def test_initial_state(self, settings):
        from claw.scheduler.scheduler import Scheduler

        broadcaster = AsyncMock()
        scheduler = Scheduler(broadcaster=broadcaster)
        assert scheduler.running is False
        assert scheduler._tts is None
        assert scheduler._router is None

    def test_with_all_dependencies(self, settings):
        from claw.scheduler.scheduler import Scheduler

        broadcaster = AsyncMock()
        tts = AsyncMock()
        router = AsyncMock()
        scheduler = Scheduler(broadcaster=broadcaster, tts=tts, router=router)
        assert scheduler._tts is tts
        assert scheduler._router is router
        assert scheduler._broadcaster is broadcaster

    def test_running_property_reflects_internal_state(self, settings):
        from claw.scheduler.scheduler import Scheduler

        scheduler = Scheduler(broadcaster=AsyncMock())
        assert scheduler.running is False
        scheduler._running = True
        assert scheduler.running is True


# ---------------------------------------------------------------------------
# _reminders_file
# ---------------------------------------------------------------------------

class TestRemindersFile:
    """Test that _reminders_file resolves to the correct path."""

    def test_resolves_from_config(self, tmp_scheduler, settings):
        scheduler, tmp_path, *_ = tmp_scheduler
        import claw.config as cfg_mod

        with patch.object(cfg_mod, "PROJECT_ROOT", tmp_path):
            f = scheduler._reminders_file()

        assert f.name == "reminders.json"
        assert "notes" in str(f) or "data" in str(f)


# ---------------------------------------------------------------------------
# Edge cases and integration-like scenarios
# ---------------------------------------------------------------------------

class TestEdgeCases:
    """Test unusual and edge-case scenarios."""

    async def test_reminder_exactly_at_current_time(self, tmp_scheduler, settings):
        """A reminder whose time == now should fire (<=)."""
        scheduler, tmp_path, broadcaster, tts, router = tmp_scheduler
        now = datetime.now()
        reminders = [_make_reminder("r1", "Exact time", time=now)]
        _write_reminders(scheduler._reminders_file(), reminders)

        # Use a frozen "now" that matches the reminder time
        with patch("claw.scheduler.scheduler.get_settings", return_value=settings), \
             patch("claw.scheduler.scheduler.datetime") as mock_dt:
            mock_dt.now.return_value = now
            mock_dt.fromisoformat = datetime.fromisoformat
            await scheduler._poll()

        broadcaster.broadcast.assert_called_once()

    async def test_poll_fires_multiple_due_reminders(self, tmp_scheduler, settings):
        """Multiple due reminders are all fired in a single poll."""
        scheduler, tmp_path, broadcaster, tts, router = tmp_scheduler
        past = datetime.now() - timedelta(minutes=5)
        reminders = [
            _make_reminder("r1", "First", time=past),
            _make_reminder("r2", "Second", time=past),
        ]
        _write_reminders(scheduler._reminders_file(), reminders)

        with patch("claw.scheduler.scheduler.get_settings", return_value=settings):
            # Fire both polls without error
            await scheduler._poll()

        assert broadcaster.broadcast.call_count == 2

    async def test_reminder_with_extra_fields_is_tolerated(self, tmp_scheduler, settings):
        """Reminders with extra/unknown fields should not cause errors."""
        scheduler, tmp_path, broadcaster, tts, router = tmp_scheduler
        past = datetime.now() - timedelta(minutes=1)
        reminder = {
            "id": "r1",
            "message": "Extra fields",
            "time": past.isoformat(),
            "category": "health",
            "repeat": "daily",
            "unknown_field": 42,
        }
        _write_reminders(scheduler._reminders_file(), [reminder])

        with patch("claw.scheduler.scheduler.get_settings", return_value=settings):
            await scheduler._poll()

        broadcaster.broadcast.assert_called_once()

    async def test_empty_reminders_list_in_file(self, tmp_scheduler, settings):
        """An empty JSON array should be handled gracefully."""
        scheduler, *_ = tmp_scheduler
        _write_reminders(scheduler._reminders_file(), [])

        with patch("claw.scheduler.scheduler.get_settings", return_value=settings):
            await scheduler._poll()

        scheduler._broadcaster.broadcast.assert_not_called()

    async def test_music_status_error_does_not_block_tts(self, tmp_scheduler, settings):
        """If get_status throws, TTS should still fire."""
        scheduler, tmp_path, broadcaster, tts, router = tmp_scheduler
        router.call_tool = AsyncMock(side_effect=RuntimeError("Music server down"))
        reminder = _make_reminder("r1", "Music error test")
        _write_reminders(scheduler._reminders_file(), [reminder])

        with patch("claw.scheduler.scheduler.get_settings", return_value=settings):
            settings.scheduler.announce_tts = True
            await scheduler._fire_reminder(reminder)

        # TTS should still be called despite music error
        tts.speak.assert_called_once()

    async def test_resume_error_does_not_crash(self, tmp_scheduler, settings):
        """If resume throws after TTS, the reminder should still be removed."""
        scheduler, tmp_path, broadcaster, tts, router = tmp_scheduler

        call_count = 0

        async def mock_call_tool(name, args):
            nonlocal call_count
            call_count += 1
            if name == "get_status":
                return '{"playing": true}'
            if name == "pause":
                return "{}"
            if name == "resume":
                raise RuntimeError("Resume failed")
            return "{}"

        router.call_tool = AsyncMock(side_effect=mock_call_tool)
        reminder = _make_reminder("r1", "Resume error test")
        _write_reminders(scheduler._reminders_file(), [reminder])

        with patch("claw.scheduler.scheduler.get_settings", return_value=settings):
            settings.scheduler.announce_tts = True
            await scheduler._fire_reminder(reminder)

        remaining = scheduler._load_reminders()
        assert len(remaining) == 0
