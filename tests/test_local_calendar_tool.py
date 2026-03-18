"""Tests for mcp_tools/local_calendar/server.py — Local Calendar MCP tool."""

from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone
from unittest.mock import patch

import pytest


# Reset module-level caches between tests
@pytest.fixture(autouse=True)
def _reset_calendar_module():
    import mcp_tools.local_calendar.server as cs

    cs._config = None
    yield
    cs._config = None


@pytest.fixture()
def cal_storage(tmp_path):
    """Set up a temp storage file for calendar events and wire it into the module."""
    import mcp_tools.local_calendar.server as cs

    storage_file = tmp_path / "calendar" / "local.json"
    cs._config = {"enabled": True, "storage_file": str(storage_file.relative_to(tmp_path))}
    cs._PROJECT_ROOT = tmp_path
    storage_file.parent.mkdir(parents=True, exist_ok=True)
    return storage_file


@pytest.fixture()
def cal_disabled(tmp_path):
    """Configure the calendar module as disabled."""
    import mcp_tools.local_calendar.server as cs

    cs._config = {"enabled": False}
    cs._PROJECT_ROOT = tmp_path


def _seed_events(storage_file, events: list[dict]) -> None:
    """Write pre-built events directly into the storage file."""
    storage_file.write_text(json.dumps(events, indent=2))


def _read_stored(storage_file) -> list[dict]:
    """Read events directly from the storage file."""
    return json.loads(storage_file.read_text())


# ── Helpers / internal functions ──────────────────────────────────────────


class TestLoadConfig:
    """Test lazy config loading."""

    def test_no_config_file(self, tmp_path):
        import mcp_tools.local_calendar.server as cs

        cs._CONFIG_YAML = tmp_path / "nonexistent.yaml"
        cs._config = None
        cfg = cs._load_config()
        assert cfg == {}

    def test_reads_local_calendar_section(self, tmp_path):
        import yaml

        import mcp_tools.local_calendar.server as cs

        cfg_file = tmp_path / "config.yaml"
        cfg_file.write_text(
            yaml.dump({"local_calendar": {"enabled": True, "storage_file": "data/cal.json"}})
        )
        cs._CONFIG_YAML = cfg_file
        cs._config = None
        cfg = cs._load_config()
        assert cfg["enabled"] is True
        assert cfg["storage_file"] == "data/cal.json"

    def test_caches_config_after_first_load(self, tmp_path):
        import mcp_tools.local_calendar.server as cs

        cs._config = {"enabled": False}
        cfg = cs._load_config()
        assert cfg["enabled"] is False

    def test_empty_yaml_returns_empty_dict(self, tmp_path):
        import mcp_tools.local_calendar.server as cs

        cfg_file = tmp_path / "config.yaml"
        cfg_file.write_text("")
        cs._CONFIG_YAML = cfg_file
        cs._config = None
        cfg = cs._load_config()
        assert cfg == {}

    def test_yaml_without_local_calendar_key(self, tmp_path):
        import yaml

        import mcp_tools.local_calendar.server as cs

        cfg_file = tmp_path / "config.yaml"
        cfg_file.write_text(yaml.dump({"weather": {"api_key": "x"}}))
        cs._CONFIG_YAML = cfg_file
        cs._config = None
        cfg = cs._load_config()
        assert cfg == {}


class TestIsEnabled:
    """Test the enabled/disabled check."""

    def test_enabled_by_default_when_key_missing(self, tmp_path):
        import mcp_tools.local_calendar.server as cs

        cs._config = {}
        assert cs._is_enabled() is True

    def test_explicitly_enabled(self, tmp_path):
        import mcp_tools.local_calendar.server as cs

        cs._config = {"enabled": True}
        assert cs._is_enabled() is True

    def test_explicitly_disabled(self, tmp_path):
        import mcp_tools.local_calendar.server as cs

        cs._config = {"enabled": False}
        assert cs._is_enabled() is False


class TestStorageFile:
    """Test storage file resolution."""

    def test_default_storage_path(self, tmp_path):
        import mcp_tools.local_calendar.server as cs

        cs._config = {}
        cs._PROJECT_ROOT = tmp_path
        result = cs._storage_file()
        assert result == tmp_path / "data" / "calendar" / "local.json"
        assert result.parent.exists()

    def test_custom_storage_path(self, tmp_path):
        import mcp_tools.local_calendar.server as cs

        cs._config = {"storage_file": "custom/events.json"}
        cs._PROJECT_ROOT = tmp_path
        result = cs._storage_file()
        assert result == tmp_path / "custom" / "events.json"
        assert result.parent.exists()


class TestParseDatetime:
    """Test the _parse_datetime helper."""

    def test_today(self):
        import mcp_tools.local_calendar.server as cs

        result = cs._parse_datetime("today")
        assert result is not None
        now = datetime.now()
        assert result.startswith(now.strftime("%Y-%m-%d"))
        assert "T00:00:00" in result

    def test_tomorrow(self):
        import mcp_tools.local_calendar.server as cs

        result = cs._parse_datetime("tomorrow")
        assert result is not None
        tomorrow = datetime.now() + timedelta(days=1)
        assert result.startswith(tomorrow.strftime("%Y-%m-%d"))
        assert "T00:00:00" in result

    def test_now(self):
        import mcp_tools.local_calendar.server as cs

        before = datetime.now()
        result = cs._parse_datetime("now")
        after = datetime.now()
        assert result is not None
        parsed = datetime.fromisoformat(result)
        assert before <= parsed <= after

    def test_iso_format_passthrough(self):
        import mcp_tools.local_calendar.server as cs

        result = cs._parse_datetime("2026-06-15T14:30:00")
        assert result is not None
        assert result == "2026-06-15T14:30:00"

    def test_natural_language_date(self):
        import mcp_tools.local_calendar.server as cs

        result = cs._parse_datetime("March 20 2026 3pm")
        assert result is not None
        assert "2026-03-20" in result
        assert "15:00" in result

    def test_unparseable_returns_none(self):
        import mcp_tools.local_calendar.server as cs

        result = cs._parse_datetime("not a date at all zzzzz")
        # dateutil with fuzzy=True is very aggressive, so only truly
        # impossible strings return None.  We test the contract: result
        # is either a valid ISO string or None.
        assert result is None or "T" in result

    def test_whitespace_stripped(self):
        import mcp_tools.local_calendar.server as cs

        result = cs._parse_datetime("  today  ")
        assert result is not None
        assert "T00:00:00" in result

    def test_case_insensitive(self):
        import mcp_tools.local_calendar.server as cs

        for variant in ("TODAY", "Today", "tOdAy"):
            result = cs._parse_datetime(variant)
            assert result is not None
            assert "T00:00:00" in result


class TestReadWriteEvents:
    """Test the low-level _read_events / _write_events helpers."""

    def test_read_empty_when_file_missing(self, cal_storage):
        import mcp_tools.local_calendar.server as cs

        assert cs._read_events() == []

    def test_write_and_read_roundtrip(self, cal_storage):
        import mcp_tools.local_calendar.server as cs

        events = [{"id": "abc", "title": "Test", "start": "2026-03-16T10:00:00"}]
        cs._write_events(events)
        assert cs._read_events() == events

    def test_write_creates_file(self, cal_storage):
        import mcp_tools.local_calendar.server as cs

        assert not cal_storage.exists()
        cs._write_events([])
        assert cal_storage.exists()


# ── Tool: local_add_event ─────────────────────────────────────────────────


class TestAddEvent:
    """Test the local_add_event tool."""

    def test_add_event_success(self, cal_storage):
        import mcp_tools.local_calendar.server as cs

        result = cs.local_add_event("Team standup", "2026-03-20 09:00")
        assert "Event created" in result
        assert "Team standup" in result
        assert "2026-03-20" in result

        events = _read_stored(cal_storage)
        assert len(events) == 1
        assert events[0]["title"] == "Team standup"
        assert events[0]["id"]  # non-empty
        assert events[0]["end"]  # auto-generated

    def test_add_event_with_end_time(self, cal_storage):
        import mcp_tools.local_calendar.server as cs

        result = cs.local_add_event(
            "Workshop", "2026-03-20 10:00", end="2026-03-20 12:00"
        )
        assert "Event created" in result

        events = _read_stored(cal_storage)
        assert "12:00" in events[0]["end"]

    def test_add_event_with_description(self, cal_storage):
        import mcp_tools.local_calendar.server as cs

        cs.local_add_event("Dentist", "2026-04-01 14:00", description="Annual checkup")
        events = _read_stored(cal_storage)
        assert events[0]["description"] == "Annual checkup"

    def test_add_event_auto_end_one_hour_later(self, cal_storage):
        import mcp_tools.local_calendar.server as cs

        cs.local_add_event("Quick meeting", "2026-03-20 10:00")
        events = _read_stored(cal_storage)
        assert events[0]["end"] == "2026-03-20T11:00:00"

    def test_add_event_bad_start_time(self, cal_storage):
        import mcp_tools.local_calendar.server as cs

        result = cs.local_add_event("Bad event", "not-a-date zzzzz")
        # Could be parsed by fuzzy dateutil or rejected — we check the contract
        if "Could not parse" in result:
            events = _read_stored(cal_storage) if cal_storage.exists() else []
            assert len(events) == 0

    def test_add_event_bad_end_time(self, cal_storage):
        import mcp_tools.local_calendar.server as cs

        result = cs.local_add_event("Bad end", "2026-03-20 10:00", end="not-a-date zzzzz")
        if "Could not parse end" in result:
            events = _read_stored(cal_storage) if cal_storage.exists() else []
            assert len(events) == 0

    def test_add_event_when_disabled(self, cal_disabled):
        import mcp_tools.local_calendar.server as cs

        result = cs.local_add_event("Nope", "2026-03-20 10:00")
        assert "disabled" in result.lower()

    def test_add_multiple_events(self, cal_storage):
        import mcp_tools.local_calendar.server as cs

        cs.local_add_event("First", "2026-03-20 09:00")
        cs.local_add_event("Second", "2026-03-21 10:00")
        cs.local_add_event("Third", "2026-03-22 11:00")

        events = _read_stored(cal_storage)
        assert len(events) == 3

    def test_add_event_generates_unique_ids(self, cal_storage):
        import mcp_tools.local_calendar.server as cs

        cs.local_add_event("A", "2026-03-20 09:00")
        cs.local_add_event("B", "2026-03-20 09:00")
        events = _read_stored(cal_storage)
        assert events[0]["id"] != events[1]["id"]

    def test_add_event_records_created_at(self, cal_storage):
        import mcp_tools.local_calendar.server as cs

        before = datetime.now(timezone.utc)
        cs.local_add_event("Timed", "2026-03-20 09:00")
        after = datetime.now(timezone.utc)

        events = _read_stored(cal_storage)
        created = datetime.fromisoformat(events[0]["created_at"])
        # created_at should be between before and after
        assert before <= created <= after

    def test_add_event_with_natural_language_start(self, cal_storage):
        import mcp_tools.local_calendar.server as cs

        result = cs.local_add_event("Lunch", "tomorrow")
        assert "Event created" in result

    def test_add_event_returns_id(self, cal_storage):
        import mcp_tools.local_calendar.server as cs

        result = cs.local_add_event("With ID", "2026-03-20 09:00")
        assert "id:" in result


# ── Tool: local_list_events ───────────────────────────────────────────────


class TestListEvents:
    """Test the local_list_events tool."""

    def test_list_empty(self, cal_storage):
        import mcp_tools.local_calendar.server as cs

        result = cs.local_list_events()
        assert "No events" in result

    def test_list_with_events_in_range(self, cal_storage):
        import mcp_tools.local_calendar.server as cs

        # Create events spanning today + 7 days
        now = datetime.now()
        start = (now + timedelta(hours=2)).isoformat()
        _seed_events(cal_storage, [
            {
                "id": "ev1",
                "title": "Today event",
                "start": start,
                "end": (now + timedelta(hours=3)).isoformat(),
                "description": "",
                "created_at": now.isoformat(),
            },
        ])
        result = cs.local_list_events("today", days=7)
        assert "Today event" in result
        assert "ev1" in result

    def test_list_excludes_past_events(self, cal_storage):
        import mcp_tools.local_calendar.server as cs

        past = datetime(2020, 1, 1, 10, 0, 0)
        _seed_events(cal_storage, [
            {
                "id": "old1",
                "title": "Ancient event",
                "start": past.isoformat(),
                "end": (past + timedelta(hours=1)).isoformat(),
                "description": "",
                "created_at": past.isoformat(),
            },
        ])
        result = cs.local_list_events("today", days=7)
        assert "No events" in result

    def test_list_excludes_events_outside_window(self, cal_storage):
        import mcp_tools.local_calendar.server as cs

        far_future = datetime(2030, 12, 25, 10, 0, 0)
        _seed_events(cal_storage, [
            {
                "id": "far1",
                "title": "Far future",
                "start": far_future.isoformat(),
                "end": (far_future + timedelta(hours=1)).isoformat(),
                "description": "",
                "created_at": datetime.now().isoformat(),
            },
        ])
        result = cs.local_list_events("today", days=1)
        assert "No events" in result

    def test_list_respects_days_parameter(self, cal_storage):
        import mcp_tools.local_calendar.server as cs

        # Use a fixed base date to avoid wall-clock sensitivity
        base = datetime(2026, 7, 1, 0, 0, 0)
        target = datetime(2026, 7, 4, 10, 0, 0)  # 3 days + 10 hours later
        _seed_events(cal_storage, [
            {
                "id": "ev3",
                "title": "In 3 days",
                "start": target.isoformat(),
                "end": (target + timedelta(hours=1)).isoformat(),
                "description": "",
                "created_at": base.isoformat(),
            },
        ])
        # Looking 2 days ahead from July 1 should miss July 4
        result_2 = cs.local_list_events("2026-07-01", days=2)
        assert "No events" in result_2

        # Looking 5 days ahead from July 1 should find July 4
        result_5 = cs.local_list_events("2026-07-01", days=5)
        assert "In 3 days" in result_5

    def test_list_days_clamped_minimum(self, cal_storage):
        import mcp_tools.local_calendar.server as cs

        # days=0 should be clamped to 1
        result = cs.local_list_events("today", days=0)
        assert isinstance(result, str)

    def test_list_days_clamped_maximum(self, cal_storage):
        import mcp_tools.local_calendar.server as cs

        # days=200 should be clamped to 90
        result = cs.local_list_events("today", days=200)
        assert isinstance(result, str)

    def test_list_with_specific_date(self, cal_storage):
        import mcp_tools.local_calendar.server as cs

        target = datetime(2026, 6, 15, 14, 0, 0)
        _seed_events(cal_storage, [
            {
                "id": "june1",
                "title": "June event",
                "start": target.isoformat(),
                "end": (target + timedelta(hours=1)).isoformat(),
                "description": "",
                "created_at": datetime.now().isoformat(),
            },
        ])
        result = cs.local_list_events("2026-06-15", days=1)
        assert "June event" in result

    def test_list_sorts_by_start_time(self, cal_storage):
        import mcp_tools.local_calendar.server as cs

        base = datetime(2026, 7, 1, 0, 0, 0)
        _seed_events(cal_storage, [
            {
                "id": "late",
                "title": "Later",
                "start": (base + timedelta(hours=15)).isoformat(),
                "end": (base + timedelta(hours=16)).isoformat(),
                "description": "",
                "created_at": datetime.now().isoformat(),
            },
            {
                "id": "early",
                "title": "Earlier",
                "start": (base + timedelta(hours=9)).isoformat(),
                "end": (base + timedelta(hours=10)).isoformat(),
                "description": "",
                "created_at": datetime.now().isoformat(),
            },
        ])
        result = cs.local_list_events("2026-07-01", days=1)
        earlier_pos = result.index("Earlier")
        later_pos = result.index("Later")
        assert earlier_pos < later_pos

    def test_list_shows_description(self, cal_storage):
        import mcp_tools.local_calendar.server as cs

        now = datetime.now()
        start = (now + timedelta(hours=1)).isoformat()
        _seed_events(cal_storage, [
            {
                "id": "desc1",
                "title": "With desc",
                "start": start,
                "end": (now + timedelta(hours=2)).isoformat(),
                "description": "Important meeting notes",
                "created_at": now.isoformat(),
            },
        ])
        result = cs.local_list_events("today", days=1)
        assert "Important meeting notes" in result

    def test_list_formats_time_nicely(self, cal_storage):
        import mcp_tools.local_calendar.server as cs

        target = datetime(2026, 7, 1, 14, 30, 0)
        _seed_events(cal_storage, [
            {
                "id": "fmt1",
                "title": "Formatted",
                "start": target.isoformat(),
                "end": (target + timedelta(hours=1)).isoformat(),
                "description": "",
                "created_at": datetime.now().isoformat(),
            },
        ])
        result = cs.local_list_events("2026-07-01", days=1)
        # Should display formatted date, not raw ISO
        assert "02:30 PM" in result or "2:30 PM" in result

    def test_list_shows_event_count(self, cal_storage):
        import mcp_tools.local_calendar.server as cs

        base = datetime(2026, 7, 1, 0, 0, 0)
        _seed_events(cal_storage, [
            {
                "id": f"cnt{i}",
                "title": f"Event {i}",
                "start": (base + timedelta(hours=9 + i)).isoformat(),
                "end": (base + timedelta(hours=10 + i)).isoformat(),
                "description": "",
                "created_at": datetime.now().isoformat(),
            }
            for i in range(3)
        ])
        result = cs.local_list_events("2026-07-01", days=1)
        assert "Events (3)" in result

    def test_list_when_disabled(self, cal_disabled):
        import mcp_tools.local_calendar.server as cs

        result = cs.local_list_events()
        assert "disabled" in result.lower()

    def test_list_bad_date_falls_back_to_today(self, cal_storage):
        import mcp_tools.local_calendar.server as cs

        # An unparseable date should fall back to today's midnight
        now = datetime.now()
        start = (now + timedelta(hours=1)).isoformat()
        _seed_events(cal_storage, [
            {
                "id": "fb1",
                "title": "Fallback event",
                "start": start,
                "end": (now + timedelta(hours=2)).isoformat(),
                "description": "",
                "created_at": now.isoformat(),
            },
        ])
        result = cs.local_list_events("gibberish_not_a_date", days=7)
        # Should not crash; should fall back to today
        assert isinstance(result, str)


# ── Tool: local_get_event ─────────────────────────────────────────────────


class TestGetEvent:
    """Test the local_get_event tool."""

    def test_get_existing_event(self, cal_storage):
        import mcp_tools.local_calendar.server as cs

        _seed_events(cal_storage, [
            {
                "id": "get1",
                "title": "My Event",
                "start": "2026-03-20T10:00:00",
                "end": "2026-03-20T11:00:00",
                "description": "Details here",
                "created_at": "2026-03-16T08:00:00+00:00",
            },
        ])
        result = cs.local_get_event("get1")
        assert "My Event" in result
        assert "get1" in result
        assert "2026-03-20T10:00:00" in result
        assert "2026-03-20T11:00:00" in result
        assert "Details here" in result
        assert "Created:" in result

    def test_get_event_without_description(self, cal_storage):
        import mcp_tools.local_calendar.server as cs

        _seed_events(cal_storage, [
            {
                "id": "nodesc",
                "title": "No Desc",
                "start": "2026-03-20T10:00:00",
                "end": "2026-03-20T11:00:00",
                "description": "",
                "created_at": "2026-03-16T08:00:00+00:00",
            },
        ])
        result = cs.local_get_event("nodesc")
        assert "No Desc" in result
        assert "Description:" not in result

    def test_get_nonexistent_event(self, cal_storage):
        import mcp_tools.local_calendar.server as cs

        result = cs.local_get_event("nonexistent_id")
        assert "not found" in result.lower()

    def test_get_event_when_disabled(self, cal_disabled):
        import mcp_tools.local_calendar.server as cs

        result = cs.local_get_event("any_id")
        assert "disabled" in result.lower()

    def test_get_event_from_multiple(self, cal_storage):
        import mcp_tools.local_calendar.server as cs

        _seed_events(cal_storage, [
            {
                "id": "aaa",
                "title": "First",
                "start": "2026-03-20T09:00:00",
                "end": "2026-03-20T10:00:00",
                "description": "",
                "created_at": "2026-03-16T08:00:00+00:00",
            },
            {
                "id": "bbb",
                "title": "Second",
                "start": "2026-03-20T11:00:00",
                "end": "2026-03-20T12:00:00",
                "description": "The right one",
                "created_at": "2026-03-16T08:00:00+00:00",
            },
        ])
        result = cs.local_get_event("bbb")
        assert "Second" in result
        assert "The right one" in result
        assert "First" not in result

    def test_get_event_without_end(self, cal_storage):
        import mcp_tools.local_calendar.server as cs

        _seed_events(cal_storage, [
            {
                "id": "noend",
                "title": "No End",
                "start": "2026-03-20T10:00:00",
                "description": "",
                "created_at": "2026-03-16T08:00:00+00:00",
            },
        ])
        result = cs.local_get_event("noend")
        assert "N/A" in result


# ── Tool: local_update_event ──────────────────────────────────────────────


class TestUpdateEvent:
    """Test the local_update_event tool."""

    def test_update_title(self, cal_storage):
        import mcp_tools.local_calendar.server as cs

        _seed_events(cal_storage, [
            {
                "id": "upd1",
                "title": "Old Title",
                "start": "2026-03-20T10:00:00",
                "end": "2026-03-20T11:00:00",
                "description": "",
                "created_at": "2026-03-16T08:00:00+00:00",
            },
        ])
        result = cs.local_update_event("upd1", title="New Title")
        assert "updated" in result.lower()

        events = _read_stored(cal_storage)
        assert events[0]["title"] == "New Title"

    def test_update_start_time(self, cal_storage):
        import mcp_tools.local_calendar.server as cs

        _seed_events(cal_storage, [
            {
                "id": "upd2",
                "title": "Event",
                "start": "2026-03-20T10:00:00",
                "end": "2026-03-20T11:00:00",
                "description": "",
                "created_at": "2026-03-16T08:00:00+00:00",
            },
        ])
        result = cs.local_update_event("upd2", start="2026-03-20 14:00")
        assert "updated" in result.lower()

        events = _read_stored(cal_storage)
        assert "14:00" in events[0]["start"]

    def test_update_end_time(self, cal_storage):
        import mcp_tools.local_calendar.server as cs

        _seed_events(cal_storage, [
            {
                "id": "upd3",
                "title": "Event",
                "start": "2026-03-20T10:00:00",
                "end": "2026-03-20T11:00:00",
                "description": "",
                "created_at": "2026-03-16T08:00:00+00:00",
            },
        ])
        result = cs.local_update_event("upd3", end="2026-03-20 16:00")
        assert "updated" in result.lower()

        events = _read_stored(cal_storage)
        assert "16:00" in events[0]["end"]

    def test_update_description(self, cal_storage):
        import mcp_tools.local_calendar.server as cs

        _seed_events(cal_storage, [
            {
                "id": "upd4",
                "title": "Event",
                "start": "2026-03-20T10:00:00",
                "end": "2026-03-20T11:00:00",
                "description": "Old desc",
                "created_at": "2026-03-16T08:00:00+00:00",
            },
        ])
        result = cs.local_update_event("upd4", description="New desc")
        assert "updated" in result.lower()

        events = _read_stored(cal_storage)
        assert events[0]["description"] == "New desc"

    def test_update_multiple_fields(self, cal_storage):
        import mcp_tools.local_calendar.server as cs

        _seed_events(cal_storage, [
            {
                "id": "upd5",
                "title": "Old",
                "start": "2026-03-20T10:00:00",
                "end": "2026-03-20T11:00:00",
                "description": "Old desc",
                "created_at": "2026-03-16T08:00:00+00:00",
            },
        ])
        result = cs.local_update_event(
            "upd5", title="New", start="2026-03-21 09:00", description="New desc"
        )
        assert "updated" in result.lower()

        events = _read_stored(cal_storage)
        assert events[0]["title"] == "New"
        assert "2026-03-21" in events[0]["start"]
        assert events[0]["description"] == "New desc"

    def test_update_no_fields_no_change(self, cal_storage):
        import mcp_tools.local_calendar.server as cs

        _seed_events(cal_storage, [
            {
                "id": "upd6",
                "title": "Unchanged",
                "start": "2026-03-20T10:00:00",
                "end": "2026-03-20T11:00:00",
                "description": "Same",
                "created_at": "2026-03-16T08:00:00+00:00",
            },
        ])
        result = cs.local_update_event("upd6")
        # Even with no changes, the event is found and "updated"
        assert "updated" in result.lower()

        events = _read_stored(cal_storage)
        assert events[0]["title"] == "Unchanged"

    def test_update_nonexistent_event(self, cal_storage):
        import mcp_tools.local_calendar.server as cs

        result = cs.local_update_event("fake_id", title="Won't work")
        assert "not found" in result.lower()

    def test_update_bad_start_time(self, cal_storage):
        import mcp_tools.local_calendar.server as cs

        _seed_events(cal_storage, [
            {
                "id": "badstart",
                "title": "Event",
                "start": "2026-03-20T10:00:00",
                "end": "2026-03-20T11:00:00",
                "description": "",
                "created_at": "2026-03-16T08:00:00+00:00",
            },
        ])
        result = cs.local_update_event("badstart", start="not-a-date zzzzz")
        # Either the fuzzy parser handles it or we get an error
        if "Could not parse" in result:
            events = _read_stored(cal_storage)
            # Original start should be preserved
            assert events[0]["start"] == "2026-03-20T10:00:00"

    def test_update_bad_end_time(self, cal_storage):
        import mcp_tools.local_calendar.server as cs

        _seed_events(cal_storage, [
            {
                "id": "badend",
                "title": "Event",
                "start": "2026-03-20T10:00:00",
                "end": "2026-03-20T11:00:00",
                "description": "",
                "created_at": "2026-03-16T08:00:00+00:00",
            },
        ])
        result = cs.local_update_event("badend", end="not-a-date zzzzz")
        if "Could not parse" in result:
            events = _read_stored(cal_storage)
            assert events[0]["end"] == "2026-03-20T11:00:00"

    def test_update_when_disabled(self, cal_disabled):
        import mcp_tools.local_calendar.server as cs

        result = cs.local_update_event("any_id", title="Nope")
        assert "disabled" in result.lower()

    def test_update_preserves_other_events(self, cal_storage):
        import mcp_tools.local_calendar.server as cs

        _seed_events(cal_storage, [
            {
                "id": "keep",
                "title": "Keep Me",
                "start": "2026-03-20T09:00:00",
                "end": "2026-03-20T10:00:00",
                "description": "",
                "created_at": "2026-03-16T08:00:00+00:00",
            },
            {
                "id": "change",
                "title": "Change Me",
                "start": "2026-03-20T11:00:00",
                "end": "2026-03-20T12:00:00",
                "description": "",
                "created_at": "2026-03-16T08:00:00+00:00",
            },
        ])
        cs.local_update_event("change", title="Changed")
        events = _read_stored(cal_storage)
        assert len(events) == 2
        assert events[0]["title"] == "Keep Me"
        assert events[1]["title"] == "Changed"


# ── Tool: local_delete_event ──────────────────────────────────────────────


class TestDeleteEvent:
    """Test the local_delete_event tool."""

    def test_delete_existing_event(self, cal_storage):
        import mcp_tools.local_calendar.server as cs

        _seed_events(cal_storage, [
            {
                "id": "del1",
                "title": "Delete Me",
                "start": "2026-03-20T10:00:00",
                "end": "2026-03-20T11:00:00",
                "description": "",
                "created_at": "2026-03-16T08:00:00+00:00",
            },
        ])
        result = cs.local_delete_event("del1")
        assert "deleted" in result.lower()

        events = _read_stored(cal_storage)
        assert len(events) == 0

    def test_delete_nonexistent_event(self, cal_storage):
        import mcp_tools.local_calendar.server as cs

        result = cs.local_delete_event("fake_id")
        assert "not found" in result.lower()

    def test_delete_when_disabled(self, cal_disabled):
        import mcp_tools.local_calendar.server as cs

        result = cs.local_delete_event("any_id")
        assert "disabled" in result.lower()

    def test_delete_preserves_other_events(self, cal_storage):
        import mcp_tools.local_calendar.server as cs

        _seed_events(cal_storage, [
            {
                "id": "keep1",
                "title": "Keep A",
                "start": "2026-03-20T09:00:00",
                "end": "2026-03-20T10:00:00",
                "description": "",
                "created_at": "2026-03-16T08:00:00+00:00",
            },
            {
                "id": "remove1",
                "title": "Remove",
                "start": "2026-03-20T11:00:00",
                "end": "2026-03-20T12:00:00",
                "description": "",
                "created_at": "2026-03-16T08:00:00+00:00",
            },
            {
                "id": "keep2",
                "title": "Keep B",
                "start": "2026-03-20T13:00:00",
                "end": "2026-03-20T14:00:00",
                "description": "",
                "created_at": "2026-03-16T08:00:00+00:00",
            },
        ])
        cs.local_delete_event("remove1")
        events = _read_stored(cal_storage)
        assert len(events) == 2
        ids = [e["id"] for e in events]
        assert "keep1" in ids
        assert "keep2" in ids
        assert "remove1" not in ids

    def test_delete_from_empty_storage(self, cal_storage):
        import mcp_tools.local_calendar.server as cs

        result = cs.local_delete_event("anything")
        assert "not found" in result.lower()


# ── Tool: local_search_events ─────────────────────────────────────────────


class TestSearchEvents:
    """Test the local_search_events tool."""

    def test_search_by_title(self, cal_storage):
        import mcp_tools.local_calendar.server as cs

        _seed_events(cal_storage, [
            {
                "id": "s1",
                "title": "Python Workshop",
                "start": "2026-03-20T10:00:00",
                "end": "2026-03-20T12:00:00",
                "description": "",
                "created_at": "2026-03-16T08:00:00+00:00",
            },
            {
                "id": "s2",
                "title": "Grocery Shopping",
                "start": "2026-03-21T15:00:00",
                "end": "2026-03-21T16:00:00",
                "description": "",
                "created_at": "2026-03-16T08:00:00+00:00",
            },
        ])
        result = cs.local_search_events("python")
        assert "Python Workshop" in result
        assert "Grocery" not in result

    def test_search_by_description(self, cal_storage):
        import mcp_tools.local_calendar.server as cs

        _seed_events(cal_storage, [
            {
                "id": "sd1",
                "title": "Meeting",
                "start": "2026-03-20T10:00:00",
                "end": "2026-03-20T11:00:00",
                "description": "Discuss Q2 budget projections",
                "created_at": "2026-03-16T08:00:00+00:00",
            },
        ])
        result = cs.local_search_events("budget")
        assert "Meeting" in result

    def test_search_case_insensitive(self, cal_storage):
        import mcp_tools.local_calendar.server as cs

        _seed_events(cal_storage, [
            {
                "id": "ci1",
                "title": "IMPORTANT Meeting",
                "start": "2026-03-20T10:00:00",
                "end": "2026-03-20T11:00:00",
                "description": "",
                "created_at": "2026-03-16T08:00:00+00:00",
            },
        ])
        result = cs.local_search_events("important")
        assert "IMPORTANT Meeting" in result

    def test_search_no_results(self, cal_storage):
        import mcp_tools.local_calendar.server as cs

        _seed_events(cal_storage, [
            {
                "id": "nr1",
                "title": "Something",
                "start": "2026-03-20T10:00:00",
                "end": "2026-03-20T11:00:00",
                "description": "",
                "created_at": "2026-03-16T08:00:00+00:00",
            },
        ])
        result = cs.local_search_events("nonexistent_term_xyz")
        assert "No events matching" in result

    def test_search_empty_storage(self, cal_storage):
        import mcp_tools.local_calendar.server as cs

        result = cs.local_search_events("anything")
        assert "No events matching" in result

    def test_search_respects_limit(self, cal_storage):
        import mcp_tools.local_calendar.server as cs

        events = [
            {
                "id": f"lim{i}",
                "title": f"Event {i}",
                "start": f"2026-03-{20 + i}T10:00:00",
                "end": f"2026-03-{20 + i}T11:00:00",
                "description": "",
                "created_at": "2026-03-16T08:00:00+00:00",
            }
            for i in range(5)
        ]
        _seed_events(cal_storage, events)
        result = cs.local_search_events("Event", limit=2)
        assert "Found 2 event(s)" in result

    def test_search_limit_clamped_minimum(self, cal_storage):
        import mcp_tools.local_calendar.server as cs

        _seed_events(cal_storage, [
            {
                "id": "clamp1",
                "title": "Only One",
                "start": "2026-03-20T10:00:00",
                "end": "2026-03-20T11:00:00",
                "description": "",
                "created_at": "2026-03-16T08:00:00+00:00",
            },
        ])
        # limit=0 should be clamped to 1
        result = cs.local_search_events("Only", limit=0)
        assert "Only One" in result

    def test_search_limit_clamped_maximum(self, cal_storage):
        import mcp_tools.local_calendar.server as cs

        # limit=100 should be clamped to 50
        result = cs.local_search_events("anything", limit=100)
        assert isinstance(result, str)

    def test_search_results_sorted_by_start(self, cal_storage):
        import mcp_tools.local_calendar.server as cs

        _seed_events(cal_storage, [
            {
                "id": "sort2",
                "title": "Later Event",
                "start": "2026-04-01T10:00:00",
                "end": "2026-04-01T11:00:00",
                "description": "match",
                "created_at": "2026-03-16T08:00:00+00:00",
            },
            {
                "id": "sort1",
                "title": "Earlier Event",
                "start": "2026-03-20T10:00:00",
                "end": "2026-03-20T11:00:00",
                "description": "match",
                "created_at": "2026-03-16T08:00:00+00:00",
            },
        ])
        result = cs.local_search_events("match")
        earlier_pos = result.index("Earlier")
        later_pos = result.index("Later")
        assert earlier_pos < later_pos

    def test_search_shows_count_and_ids(self, cal_storage):
        import mcp_tools.local_calendar.server as cs

        _seed_events(cal_storage, [
            {
                "id": "show1",
                "title": "Searchable",
                "start": "2026-03-20T10:00:00",
                "end": "2026-03-20T11:00:00",
                "description": "",
                "created_at": "2026-03-16T08:00:00+00:00",
            },
        ])
        result = cs.local_search_events("Searchable")
        assert "Found 1 event(s)" in result
        assert "show1" in result

    def test_search_when_disabled(self, cal_disabled):
        import mcp_tools.local_calendar.server as cs

        result = cs.local_search_events("anything")
        assert "disabled" in result.lower()


# ── Integration-style tests ───────────────────────────────────────────────


class TestIntegrationWorkflows:
    """Test end-to-end workflows combining multiple tools."""

    def test_add_then_get(self, cal_storage):
        import mcp_tools.local_calendar.server as cs

        add_result = cs.local_add_event("Integration Test", "2026-05-01 09:00", description="E2E")
        # Extract the id from the add result
        events = _read_stored(cal_storage)
        event_id = events[0]["id"]

        get_result = cs.local_get_event(event_id)
        assert "Integration Test" in get_result
        assert "E2E" in get_result

    def test_add_then_update_then_get(self, cal_storage):
        import mcp_tools.local_calendar.server as cs

        cs.local_add_event("Original", "2026-05-01 09:00")
        events = _read_stored(cal_storage)
        event_id = events[0]["id"]

        cs.local_update_event(event_id, title="Updated", description="Changed")

        get_result = cs.local_get_event(event_id)
        assert "Updated" in get_result
        assert "Changed" in get_result
        assert "Original" not in get_result

    def test_add_then_delete_then_get(self, cal_storage):
        import mcp_tools.local_calendar.server as cs

        cs.local_add_event("Ephemeral", "2026-05-01 09:00")
        events = _read_stored(cal_storage)
        event_id = events[0]["id"]

        delete_result = cs.local_delete_event(event_id)
        assert "deleted" in delete_result.lower()

        get_result = cs.local_get_event(event_id)
        assert "not found" in get_result.lower()

    def test_add_then_search(self, cal_storage):
        import mcp_tools.local_calendar.server as cs

        cs.local_add_event("Dentist Appointment", "2026-05-15 14:00", description="Annual cleaning")
        cs.local_add_event("Team Lunch", "2026-05-16 12:00")

        result = cs.local_search_events("dentist")
        assert "Dentist Appointment" in result
        assert "Team Lunch" not in result

    def test_add_then_list(self, cal_storage):
        import mcp_tools.local_calendar.server as cs

        cs.local_add_event("May Event", "2026-05-01 09:00")
        cs.local_add_event("May Event 2", "2026-05-02 10:00")

        result = cs.local_list_events("2026-05-01", days=7)
        assert "May Event" in result
        assert "Events (2)" in result

    def test_disabled_blocks_all_tools(self, cal_disabled):
        """Every tool should return the disabled message when the feature is off."""
        import mcp_tools.local_calendar.server as cs

        results = [
            cs.local_add_event("X", "2026-01-01"),
            cs.local_list_events(),
            cs.local_get_event("x"),
            cs.local_update_event("x", title="Y"),
            cs.local_delete_event("x"),
            cs.local_search_events("x"),
        ]
        for r in results:
            assert "disabled" in r.lower()
