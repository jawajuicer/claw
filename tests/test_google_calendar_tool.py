"""Tests for mcp_tools/google_calendar/server.py — Google Calendar MCP tool."""

from __future__ import annotations

from datetime import datetime, timedelta
from unittest.mock import MagicMock, patch

import pytest
import yaml


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(autouse=True)
def _isolate_calendar(tmp_path):
    """Reset module-level caches and redirect config paths to tmp_path."""
    import mcp_tools.google_calendar.server as cs

    cs._services.clear()

    with (
        patch.object(cs, "_PROJECT_ROOT", tmp_path),
        patch.object(cs, "_CONFIG_YAML", tmp_path / "config.yaml"),
    ):
        yield

    cs._services.clear()


def _write_config(tmp_path, google_auth: dict | None = None):
    """Write a config.yaml with the given google_auth block."""
    cfg = {}
    if google_auth is not None:
        cfg["google_auth"] = google_auth
    (tmp_path / "config.yaml").write_text(yaml.dump(cfg))


def _sample_account_config() -> dict:
    """Return a typical account config dict with calendar settings."""
    return {
        "token_file": "data/google/token_personal.json",
        "calendar": {
            "enabled": True,
            "default_calendar": "user@gmail.com",
            "timezone": "America/Chicago",
            "calendars": {
                "work": "work_calendar_id@group.calendar.google.com",
                "family": "family_calendar_id@group.calendar.google.com",
            },
        },
    }


def _sample_google_auth(accounts: dict | None = None) -> dict:
    """Return a typical google_auth config block."""
    if accounts is None:
        accounts = {"personal": _sample_account_config()}
    return {
        "credentials_file": "data/google/credentials.json",
        "scopes": ["https://www.googleapis.com/auth/calendar"],
        "accounts": accounts,
    }


@pytest.fixture()
def mock_service():
    """Build a deeply-mocked Google Calendar API service object."""
    svc = MagicMock()

    # calendarList().list().execute()
    svc.calendarList.return_value.list.return_value.execute.return_value = {
        "items": []
    }

    # events().list().execute()
    svc.events.return_value.list.return_value.execute.return_value = {
        "items": []
    }

    # events().insert().execute()
    svc.events.return_value.insert.return_value.execute.return_value = {
        "id": "abc123def456",
        "summary": "Test Event",
    }

    # events().get().execute()
    svc.events.return_value.get.return_value.execute.return_value = {
        "id": "abc123def456",
        "summary": "Existing Event",
        "start": {"dateTime": "2026-03-20T14:00:00", "timeZone": "America/Chicago"},
        "end": {"dateTime": "2026-03-20T15:00:00", "timeZone": "America/Chicago"},
    }

    # events().update().execute()
    svc.events.return_value.update.return_value.execute.return_value = {
        "id": "abc123def456",
        "summary": "Updated Event",
    }

    # events().delete().execute()
    svc.events.return_value.delete.return_value.execute.return_value = None

    return svc


def _patch_resolve(mock_service, acct_cfg=None, error=""):
    """Patch _resolve to return the mock service, account config, and error."""
    if acct_cfg is None:
        acct_cfg = _sample_account_config()
    return patch(
        "mcp_tools.google_calendar.server._resolve",
        return_value=(mock_service, acct_cfg, error),
    )


# ---------------------------------------------------------------------------
# Config loading
# ---------------------------------------------------------------------------


class TestLoadGoogleAuth:
    """Test _load_google_auth config reader."""

    def test_no_config_file(self, tmp_path):
        import mcp_tools.google_calendar.server as cs

        # CONFIG_YAML already points to tmp_path/config.yaml which doesn't exist
        result = cs._load_google_auth()
        assert result == {}

    def test_empty_config_file(self, tmp_path):
        import mcp_tools.google_calendar.server as cs

        (tmp_path / "config.yaml").write_text("")
        result = cs._load_google_auth()
        assert result == {}

    def test_config_without_google_auth(self, tmp_path):
        import mcp_tools.google_calendar.server as cs

        _write_config(tmp_path)
        result = cs._load_google_auth()
        assert result == {}

    def test_config_with_google_auth(self, tmp_path):
        import mcp_tools.google_calendar.server as cs

        _write_config(tmp_path, _sample_google_auth())
        result = cs._load_google_auth()
        assert "accounts" in result
        assert "personal" in result["accounts"]


class TestGetAccounts:
    """Test _get_accounts helper."""

    def test_returns_empty_when_no_config(self, tmp_path):
        import mcp_tools.google_calendar.server as cs

        result = cs._get_accounts()
        assert result == {}

    def test_returns_accounts_dict(self, tmp_path):
        import mcp_tools.google_calendar.server as cs

        _write_config(tmp_path, _sample_google_auth())
        result = cs._get_accounts()
        assert "personal" in result


# ---------------------------------------------------------------------------
# Calendar / timezone resolution
# ---------------------------------------------------------------------------


class TestResolveCalendar:
    """Test _resolve_calendar label-to-ID mapping."""

    def test_default_label(self):
        import mcp_tools.google_calendar.server as cs

        acct = _sample_account_config()
        assert cs._resolve_calendar(acct, "default") == "user@gmail.com"

    def test_empty_label(self):
        import mcp_tools.google_calendar.server as cs

        acct = _sample_account_config()
        assert cs._resolve_calendar(acct, "") == "user@gmail.com"

    def test_named_sub_calendar(self):
        import mcp_tools.google_calendar.server as cs

        acct = _sample_account_config()
        assert cs._resolve_calendar(acct, "work") == "work_calendar_id@group.calendar.google.com"

    def test_named_sub_calendar_family(self):
        import mcp_tools.google_calendar.server as cs

        acct = _sample_account_config()
        assert cs._resolve_calendar(acct, "family") == "family_calendar_id@group.calendar.google.com"

    def test_unknown_label_falls_back_to_default(self):
        import mcp_tools.google_calendar.server as cs

        acct = _sample_account_config()
        # An unknown label should NOT be used as a calendar ID
        assert cs._resolve_calendar(acct, "nonexistent") == "user@gmail.com"

    def test_no_calendar_config_uses_primary(self):
        import mcp_tools.google_calendar.server as cs

        acct = {}  # no calendar key at all
        assert cs._resolve_calendar(acct, "default") == "primary"

    def test_non_dict_calendar_config(self):
        import mcp_tools.google_calendar.server as cs

        acct = {"calendar": "some_calendar_id"}
        assert cs._resolve_calendar(acct, "") == "primary"

    def test_non_dict_calendar_config_with_label(self):
        import mcp_tools.google_calendar.server as cs

        acct = {"calendar": "some_calendar_id"}
        assert cs._resolve_calendar(acct, "work") == "work"

    def test_no_default_calendar_key_falls_back_to_primary(self):
        import mcp_tools.google_calendar.server as cs

        acct = {"calendar": {"calendars": {"work": "work_id"}}}
        # no default_calendar key
        assert cs._resolve_calendar(acct, "default") == "primary"


class TestGetTimezone:
    """Test _get_timezone helper."""

    def test_returns_configured_timezone(self):
        import mcp_tools.google_calendar.server as cs

        acct = _sample_account_config()
        assert cs._get_timezone(acct) == "America/Chicago"

    def test_defaults_to_eastern(self):
        import mcp_tools.google_calendar.server as cs

        assert cs._get_timezone({}) == "America/New_York"

    def test_non_dict_calendar_defaults_to_eastern(self):
        import mcp_tools.google_calendar.server as cs

        assert cs._get_timezone({"calendar": "string_value"}) == "America/New_York"

    def test_no_timezone_key_defaults_to_eastern(self):
        import mcp_tools.google_calendar.server as cs

        acct = {"calendar": {"default_calendar": "primary"}}
        assert cs._get_timezone(acct) == "America/New_York"


# ---------------------------------------------------------------------------
# _resolve (account resolution + service init)
# ---------------------------------------------------------------------------


class TestResolve:
    """Test the _resolve helper that wires account resolution and service init."""

    def test_account_not_configured(self, tmp_path):
        import mcp_tools.google_calendar.server as cs

        # No config file => no accounts
        svc, acct, err = cs._resolve("")
        assert svc is None
        assert err != ""
        assert "No Google accounts" in err

    def test_auth_failure_returns_error(self, tmp_path):
        import mcp_tools.google_calendar.server as cs

        _write_config(tmp_path, _sample_google_auth())
        with patch.object(cs, "_get_service", return_value=None):
            svc, acct, err = cs._resolve("")
            assert svc is None
            assert "re-authorize" in err.lower()

    def test_successful_resolution(self, tmp_path, mock_service):
        import mcp_tools.google_calendar.server as cs

        _write_config(tmp_path, _sample_google_auth())
        with patch.object(cs, "_get_service", return_value=mock_service):
            svc, acct, err = cs._resolve("")
            assert svc is mock_service
            assert err == ""
            assert acct is not None


# ---------------------------------------------------------------------------
# _get_service
# ---------------------------------------------------------------------------


class TestGetService:
    """Test the lazy service initializer."""

    def test_cached_service_returned(self):
        import mcp_tools.google_calendar.server as cs

        sentinel = object()
        cs._services["personal"] = sentinel
        assert cs._get_service("personal") is sentinel

    def test_credentials_none_returns_none(self, tmp_path):
        import mcp_tools.google_calendar.server as cs

        _write_config(tmp_path, _sample_google_auth())
        with (
            patch("mcp_tools.google_calendar.server.get_credentials", return_value=None, create=True),
            patch.dict("sys.modules", {
                "google_auth": MagicMock(),
                "google_auth.auth": MagicMock(get_credentials=MagicMock(return_value=None)),
                "googleapiclient": MagicMock(),
                "googleapiclient.discovery": MagicMock(),
            }),
        ):
            result = cs._get_service("personal")
            assert result is None

    def test_successful_build(self, tmp_path):
        import mcp_tools.google_calendar.server as cs

        _write_config(tmp_path, _sample_google_auth())
        fake_creds = MagicMock()
        fake_service = MagicMock()

        mock_auth_mod = MagicMock()
        mock_auth_mod.get_credentials.return_value = fake_creds
        mock_discovery = MagicMock()
        mock_discovery.build.return_value = fake_service

        with patch.dict("sys.modules", {
            "google_auth": MagicMock(),
            "google_auth.auth": mock_auth_mod,
            "googleapiclient": MagicMock(),
            "googleapiclient.discovery": mock_discovery,
        }):
            result = cs._get_service("personal")
            assert result is fake_service
            assert cs._services["personal"] is fake_service


# ---------------------------------------------------------------------------
# list_calendars
# ---------------------------------------------------------------------------


class TestListCalendars:
    """Test the list_calendars MCP tool."""

    def test_resolve_error_propagated(self):
        import mcp_tools.google_calendar.server as cs

        with _patch_resolve(None, error="No accounts configured"):
            result = cs.list_calendars()
        assert result == "No accounts configured"

    def test_no_calendars(self, mock_service):
        import mcp_tools.google_calendar.server as cs

        mock_service.calendarList.return_value.list.return_value.execute.return_value = {
            "items": []
        }
        with _patch_resolve(mock_service):
            result = cs.list_calendars()
        assert result == "No calendars found."

    def test_single_primary_calendar(self, mock_service):
        import mcp_tools.google_calendar.server as cs

        mock_service.calendarList.return_value.list.return_value.execute.return_value = {
            "items": [
                {
                    "summary": "My Calendar",
                    "id": "user@gmail.com",
                    "primary": True,
                    "accessRole": "owner",
                }
            ]
        }
        with _patch_resolve(mock_service):
            result = cs.list_calendars()
        assert "Google Calendars (1):" in result
        assert "My Calendar" in result
        assert "(primary)" in result
        assert "user@gmail.com" in result
        assert "owner" in result

    def test_multiple_calendars(self, mock_service):
        import mcp_tools.google_calendar.server as cs

        mock_service.calendarList.return_value.list.return_value.execute.return_value = {
            "items": [
                {"summary": "Personal", "id": "personal@gmail.com", "primary": True, "accessRole": "owner"},
                {"summary": "Work", "id": "work@group.calendar.google.com", "accessRole": "writer"},
                {"summary": "Holidays", "id": "holidays@calendar.google.com", "accessRole": "reader"},
            ]
        }
        with _patch_resolve(mock_service):
            result = cs.list_calendars()
        assert "Google Calendars (3):" in result
        assert "Personal" in result
        assert "Work" in result
        assert "Holidays" in result

    def test_calendar_without_primary_flag(self, mock_service):
        import mcp_tools.google_calendar.server as cs

        mock_service.calendarList.return_value.list.return_value.execute.return_value = {
            "items": [
                {"summary": "Shared", "id": "shared@group.calendar.google.com", "accessRole": "reader"},
            ]
        }
        with _patch_resolve(mock_service):
            result = cs.list_calendars()
        assert "(primary)" not in result
        assert "Shared" in result

    def test_api_error(self, mock_service):
        import mcp_tools.google_calendar.server as cs

        mock_service.calendarList.return_value.list.return_value.execute.side_effect = (
            Exception("API quota exceeded")
        )
        with _patch_resolve(mock_service):
            result = cs.list_calendars()
        assert "Error listing calendars" in result
        assert "API quota exceeded" in result

    def test_passes_account_parameter(self, mock_service):
        import mcp_tools.google_calendar.server as cs

        with patch.object(cs, "_resolve", return_value=(mock_service, _sample_account_config(), "")) as mock_resolve:
            cs.list_calendars(account="work")
            mock_resolve.assert_called_once_with("work")


# ---------------------------------------------------------------------------
# list_events
# ---------------------------------------------------------------------------


class TestListEvents:
    """Test the list_events MCP tool."""

    def test_resolve_error(self):
        import mcp_tools.google_calendar.server as cs

        with _patch_resolve(None, error="Auth failed"):
            result = cs.list_events()
        assert result == "Auth failed"

    def test_no_events(self, mock_service):
        import mcp_tools.google_calendar.server as cs

        with _patch_resolve(mock_service):
            result = cs.list_events(days=3)
        assert "No events in the next 3 day(s)." == result

    def test_events_returned(self, mock_service):
        import mcp_tools.google_calendar.server as cs

        mock_service.events.return_value.list.return_value.execute.return_value = {
            "items": [
                {
                    "id": "event123456789",
                    "summary": "Team Standup",
                    "start": {"dateTime": "2026-03-20T09:00:00-05:00"},
                    "end": {"dateTime": "2026-03-20T09:30:00-05:00"},
                },
                {
                    "id": "event987654321",
                    "summary": "Lunch",
                    "start": {"dateTime": "2026-03-20T12:00:00-05:00"},
                    "end": {"dateTime": "2026-03-20T13:00:00-05:00"},
                },
            ]
        }
        with _patch_resolve(mock_service):
            result = cs.list_events()
        assert "Events (2):" in result
        assert "Team Standup" in result
        assert "Lunch" in result
        assert "event123456" in result  # first 12 chars

    def test_all_day_event(self, mock_service):
        import mcp_tools.google_calendar.server as cs

        mock_service.events.return_value.list.return_value.execute.return_value = {
            "items": [
                {
                    "id": "allday123456",
                    "summary": "Company Holiday",
                    "start": {"date": "2026-03-20"},
                    "end": {"date": "2026-03-21"},
                },
            ]
        }
        with _patch_resolve(mock_service):
            result = cs.list_events()
        assert "Company Holiday" in result

    def test_event_without_summary(self, mock_service):
        import mcp_tools.google_calendar.server as cs

        mock_service.events.return_value.list.return_value.execute.return_value = {
            "items": [
                {
                    "id": "nosummary1234",
                    "start": {"dateTime": "2026-03-20T14:00:00-05:00"},
                    "end": {"dateTime": "2026-03-20T15:00:00-05:00"},
                },
            ]
        }
        with _patch_resolve(mock_service):
            result = cs.list_events()
        assert "(No title)" in result

    def test_date_today(self, mock_service):
        import mcp_tools.google_calendar.server as cs

        with _patch_resolve(mock_service):
            result = cs.list_events(date="today")
        # Should not error, should call the API
        mock_service.events.return_value.list.assert_called_once()
        call_kwargs = mock_service.events.return_value.list.call_args[1]
        # time_min should end with Z (naive datetime)
        assert call_kwargs["timeMin"].endswith("Z")

    def test_date_specific(self, mock_service):
        import mcp_tools.google_calendar.server as cs

        with _patch_resolve(mock_service):
            cs.list_events(date="2026-04-01")
        call_kwargs = mock_service.events.return_value.list.call_args[1]
        assert "2026-04-01" in call_kwargs["timeMin"]

    def test_date_invalid_falls_back_to_today(self, mock_service):
        import mcp_tools.google_calendar.server as cs

        with _patch_resolve(mock_service):
            cs.list_events(date="not-a-date-at-all!!!")
        # Should not raise, falls back to today
        mock_service.events.return_value.list.assert_called_once()

    def test_days_clamped_minimum(self, mock_service):
        import mcp_tools.google_calendar.server as cs

        with _patch_resolve(mock_service):
            result = cs.list_events(days=0)
        # days clamped to 1
        assert "1 day(s)" in result

    def test_days_clamped_maximum(self, mock_service):
        import mcp_tools.google_calendar.server as cs

        with _patch_resolve(mock_service):
            result = cs.list_events(days=200)
        # days clamped to 90
        assert "90 day(s)" in result

    def test_calendar_resolution(self, mock_service):
        import mcp_tools.google_calendar.server as cs

        acct = _sample_account_config()
        with patch.object(cs, "_resolve", return_value=(mock_service, acct, "")):
            cs.list_events(calendar="work")
        call_kwargs = mock_service.events.return_value.list.call_args[1]
        assert call_kwargs["calendarId"] == "work_calendar_id@group.calendar.google.com"

    def test_api_error(self, mock_service):
        import mcp_tools.google_calendar.server as cs

        mock_service.events.return_value.list.return_value.execute.side_effect = (
            Exception("Calendar not found")
        )
        with _patch_resolve(mock_service):
            result = cs.list_events()
        assert "Error listing events" in result
        assert "Calendar not found" in result

    def test_max_results_50(self, mock_service):
        import mcp_tools.google_calendar.server as cs

        with _patch_resolve(mock_service):
            cs.list_events()
        call_kwargs = mock_service.events.return_value.list.call_args[1]
        assert call_kwargs["maxResults"] == 50
        assert call_kwargs["singleEvents"] is True
        assert call_kwargs["orderBy"] == "startTime"


# ---------------------------------------------------------------------------
# create_event
# ---------------------------------------------------------------------------


class TestCreateEvent:
    """Test the create_event MCP tool."""

    def test_resolve_error(self):
        import mcp_tools.google_calendar.server as cs

        with _patch_resolve(None, error="No accounts"):
            result = cs.create_event("Test", "2026-03-20 14:00")
        assert result == "No accounts"

    def test_create_basic_event(self, mock_service):
        import mcp_tools.google_calendar.server as cs

        with _patch_resolve(mock_service):
            result = cs.create_event("Meeting", "2026-03-20 14:00")
        assert "Event created" in result
        assert "Meeting" in result
        assert "abc123def456"[:12] in result

        # Verify the event body
        call_kwargs = mock_service.events.return_value.insert.call_args[1]
        body = call_kwargs["body"]
        assert body["summary"] == "Meeting"
        assert "2026-03-20T14:00:00" in body["start"]["dateTime"]
        assert body["start"]["timeZone"] == "America/Chicago"
        # End should be 1 hour after start
        assert "2026-03-20T15:00:00" in body["end"]["dateTime"]

    def test_create_event_with_explicit_end(self, mock_service):
        import mcp_tools.google_calendar.server as cs

        with _patch_resolve(mock_service):
            result = cs.create_event("Workshop", "2026-03-20 09:00", end="2026-03-20 17:00")
        assert "Event created" in result

        call_kwargs = mock_service.events.return_value.insert.call_args[1]
        body = call_kwargs["body"]
        assert "2026-03-20T09:00:00" in body["start"]["dateTime"]
        assert "2026-03-20T17:00:00" in body["end"]["dateTime"]

    def test_create_event_with_description(self, mock_service):
        import mcp_tools.google_calendar.server as cs

        with _patch_resolve(mock_service):
            cs.create_event("Review", "2026-03-20 10:00", description="Q1 review meeting")
        call_kwargs = mock_service.events.return_value.insert.call_args[1]
        assert call_kwargs["body"]["description"] == "Q1 review meeting"

    def test_create_event_no_description_omits_key(self, mock_service):
        import mcp_tools.google_calendar.server as cs

        with _patch_resolve(mock_service):
            cs.create_event("Quick Sync", "2026-03-20 11:00")
        call_kwargs = mock_service.events.return_value.insert.call_args[1]
        assert "description" not in call_kwargs["body"]

    def test_create_event_invalid_start_time(self, mock_service):
        import mcp_tools.google_calendar.server as cs

        with _patch_resolve(mock_service):
            result = cs.create_event("Bad Event", "not a time at all !!!")
        assert "Could not parse start time" in result

    def test_create_event_invalid_end_time(self, mock_service):
        import mcp_tools.google_calendar.server as cs

        with _patch_resolve(mock_service):
            result = cs.create_event("Bad Event", "2026-03-20 14:00", end="not a time !!!")
        assert "Could not parse end time" in result

    def test_create_event_uses_correct_calendar(self, mock_service):
        import mcp_tools.google_calendar.server as cs

        acct = _sample_account_config()
        with patch.object(cs, "_resolve", return_value=(mock_service, acct, "")):
            cs.create_event("Work Meeting", "2026-03-20 14:00", calendar="work")
        call_kwargs = mock_service.events.return_value.insert.call_args[1]
        assert call_kwargs["calendarId"] == "work_calendar_id@group.calendar.google.com"

    def test_create_event_api_error(self, mock_service):
        import mcp_tools.google_calendar.server as cs

        mock_service.events.return_value.insert.return_value.execute.side_effect = (
            Exception("Insufficient permissions")
        )
        with _patch_resolve(mock_service):
            result = cs.create_event("Test", "2026-03-20 14:00")
        assert "Error creating event" in result
        assert "Insufficient permissions" in result

    def test_create_event_default_end_is_one_hour(self, mock_service):
        import mcp_tools.google_calendar.server as cs

        with _patch_resolve(mock_service):
            cs.create_event("Short Meeting", "2026-03-20 14:30")
        call_kwargs = mock_service.events.return_value.insert.call_args[1]
        body = call_kwargs["body"]
        assert "2026-03-20T15:30:00" in body["end"]["dateTime"]

    def test_create_event_timezone_from_config(self, mock_service):
        import mcp_tools.google_calendar.server as cs

        acct = _sample_account_config()
        acct["calendar"]["timezone"] = "US/Pacific"
        with patch.object(cs, "_resolve", return_value=(mock_service, acct, "")):
            cs.create_event("Pacific Meeting", "2026-03-20 14:00")
        call_kwargs = mock_service.events.return_value.insert.call_args[1]
        assert call_kwargs["body"]["start"]["timeZone"] == "US/Pacific"
        assert call_kwargs["body"]["end"]["timeZone"] == "US/Pacific"


# ---------------------------------------------------------------------------
# update_event
# ---------------------------------------------------------------------------


class TestUpdateEvent:
    """Test the update_event MCP tool."""

    def test_resolve_error(self):
        import mcp_tools.google_calendar.server as cs

        with _patch_resolve(None, error="Auth failed"):
            result = cs.update_event("evt123")
        assert result == "Auth failed"

    def test_update_title(self, mock_service):
        import mcp_tools.google_calendar.server as cs

        with _patch_resolve(mock_service):
            result = cs.update_event("evt123", title="New Title")
        assert "updated" in result.lower()

        call_kwargs = mock_service.events.return_value.update.call_args[1]
        assert call_kwargs["body"]["summary"] == "New Title"

    def test_update_start_time(self, mock_service):
        import mcp_tools.google_calendar.server as cs

        with _patch_resolve(mock_service):
            result = cs.update_event("evt123", start="2026-03-21 10:00")
        assert "updated" in result.lower()

        call_kwargs = mock_service.events.return_value.update.call_args[1]
        assert "2026-03-21T10:00:00" in call_kwargs["body"]["start"]["dateTime"]
        assert call_kwargs["body"]["start"]["timeZone"] == "America/Chicago"

    def test_update_end_time(self, mock_service):
        import mcp_tools.google_calendar.server as cs

        with _patch_resolve(mock_service):
            result = cs.update_event("evt123", end="2026-03-21 17:00")
        assert "updated" in result.lower()

        call_kwargs = mock_service.events.return_value.update.call_args[1]
        assert "2026-03-21T17:00:00" in call_kwargs["body"]["end"]["dateTime"]

    def test_update_description(self, mock_service):
        import mcp_tools.google_calendar.server as cs

        with _patch_resolve(mock_service):
            result = cs.update_event("evt123", description="Updated description")
        assert "updated" in result.lower()

        call_kwargs = mock_service.events.return_value.update.call_args[1]
        assert call_kwargs["body"]["description"] == "Updated description"

    def test_update_no_changes(self, mock_service):
        import mcp_tools.google_calendar.server as cs

        with _patch_resolve(mock_service):
            result = cs.update_event("evt123")
        # Should still succeed, just re-saves the existing event
        assert "updated" in result.lower()

    def test_update_multiple_fields(self, mock_service):
        import mcp_tools.google_calendar.server as cs

        with _patch_resolve(mock_service):
            result = cs.update_event(
                "evt123",
                title="New Title",
                start="2026-03-21 10:00",
                end="2026-03-21 11:00",
                description="New desc",
            )
        assert "updated" in result.lower()

    def test_update_invalid_start_time(self, mock_service):
        import mcp_tools.google_calendar.server as cs

        with _patch_resolve(mock_service):
            result = cs.update_event("evt123", start="not a time !!!")
        assert "Could not parse start time" in result

    def test_update_invalid_end_time(self, mock_service):
        import mcp_tools.google_calendar.server as cs

        with _patch_resolve(mock_service):
            result = cs.update_event("evt123", end="not a time !!!")
        assert "Could not parse end time" in result

    def test_update_fetch_error(self, mock_service):
        import mcp_tools.google_calendar.server as cs

        mock_service.events.return_value.get.return_value.execute.side_effect = (
            Exception("Event not found")
        )
        with _patch_resolve(mock_service):
            result = cs.update_event("evt123", title="New")
        assert "Error fetching event" in result
        assert "Event not found" in result

    def test_update_api_error(self, mock_service):
        import mcp_tools.google_calendar.server as cs

        mock_service.events.return_value.update.return_value.execute.side_effect = (
            Exception("Permission denied")
        )
        with _patch_resolve(mock_service):
            result = cs.update_event("evt123", title="New")
        assert "Error updating event" in result
        assert "Permission denied" in result

    def test_update_uses_correct_calendar_and_event_id(self, mock_service):
        import mcp_tools.google_calendar.server as cs

        acct = _sample_account_config()
        with patch.object(cs, "_resolve", return_value=(mock_service, acct, "")):
            cs.update_event("my_event_id", calendar="family", title="Family Dinner")

        # Verify get used correct IDs
        get_kwargs = mock_service.events.return_value.get.call_args[1]
        assert get_kwargs["calendarId"] == "family_calendar_id@group.calendar.google.com"
        assert get_kwargs["eventId"] == "my_event_id"

        # Verify update used correct IDs
        update_kwargs = mock_service.events.return_value.update.call_args[1]
        assert update_kwargs["calendarId"] == "family_calendar_id@group.calendar.google.com"
        assert update_kwargs["eventId"] == "my_event_id"

    def test_update_response_uses_summary(self, mock_service):
        import mcp_tools.google_calendar.server as cs

        mock_service.events.return_value.update.return_value.execute.return_value = {
            "id": "evt123",
            "summary": "Renamed Event",
        }
        with _patch_resolve(mock_service):
            result = cs.update_event("evt123", title="Renamed Event")
        assert "Renamed Event" in result


# ---------------------------------------------------------------------------
# delete_event
# ---------------------------------------------------------------------------


class TestDeleteEvent:
    """Test the delete_event MCP tool."""

    def test_resolve_error(self):
        import mcp_tools.google_calendar.server as cs

        with _patch_resolve(None, error="Auth error"):
            result = cs.delete_event("evt123")
        assert result == "Auth error"

    def test_delete_success(self, mock_service):
        import mcp_tools.google_calendar.server as cs

        with _patch_resolve(mock_service):
            result = cs.delete_event("evt123")
        assert "deleted" in result.lower()
        assert "evt123" in result

    def test_delete_uses_correct_calendar(self, mock_service):
        import mcp_tools.google_calendar.server as cs

        acct = _sample_account_config()
        with patch.object(cs, "_resolve", return_value=(mock_service, acct, "")):
            cs.delete_event("evt123", calendar="work")
        call_kwargs = mock_service.events.return_value.delete.call_args[1]
        assert call_kwargs["calendarId"] == "work_calendar_id@group.calendar.google.com"
        assert call_kwargs["eventId"] == "evt123"

    def test_delete_api_error(self, mock_service):
        import mcp_tools.google_calendar.server as cs

        mock_service.events.return_value.delete.return_value.execute.side_effect = (
            Exception("Not found")
        )
        with _patch_resolve(mock_service):
            result = cs.delete_event("evt123")
        assert "Error deleting event" in result
        assert "Not found" in result

    def test_delete_default_calendar(self, mock_service):
        import mcp_tools.google_calendar.server as cs

        with _patch_resolve(mock_service):
            cs.delete_event("evt123")
        call_kwargs = mock_service.events.return_value.delete.call_args[1]
        assert call_kwargs["calendarId"] == "user@gmail.com"


# ---------------------------------------------------------------------------
# search_events
# ---------------------------------------------------------------------------


class TestSearchEvents:
    """Test the search_events MCP tool."""

    def test_resolve_error(self):
        import mcp_tools.google_calendar.server as cs

        with _patch_resolve(None, error="No account"):
            result = cs.search_events("meeting")
        assert result == "No account"

    def test_no_results(self, mock_service):
        import mcp_tools.google_calendar.server as cs

        with _patch_resolve(mock_service):
            result = cs.search_events("nonexistent_event")
        assert "No events matching 'nonexistent_event'." == result

    def test_results_found(self, mock_service):
        import mcp_tools.google_calendar.server as cs

        mock_service.events.return_value.list.return_value.execute.return_value = {
            "items": [
                {
                    "id": "search_result_1",
                    "summary": "Team Meeting",
                    "start": {"dateTime": "2026-03-20T10:00:00-05:00"},
                },
                {
                    "id": "search_result_2",
                    "summary": "Board Meeting",
                    "start": {"dateTime": "2026-03-21T14:00:00-05:00"},
                },
            ]
        }
        with _patch_resolve(mock_service):
            result = cs.search_events("meeting")
        assert "Found 2 event(s) matching 'meeting':" in result
        assert "Team Meeting" in result
        assert "Board Meeting" in result
        assert "search_resul" in result  # first 12 chars

    def test_event_with_date_only(self, mock_service):
        import mcp_tools.google_calendar.server as cs

        mock_service.events.return_value.list.return_value.execute.return_value = {
            "items": [
                {
                    "id": "allday_search1",
                    "summary": "Holiday",
                    "start": {"date": "2026-12-25"},
                },
            ]
        }
        with _patch_resolve(mock_service):
            result = cs.search_events("holiday")
        assert "Holiday" in result
        assert "2026-12-25" in result

    def test_event_without_summary(self, mock_service):
        import mcp_tools.google_calendar.server as cs

        mock_service.events.return_value.list.return_value.execute.return_value = {
            "items": [
                {
                    "id": "nosummary_evt1",
                    "start": {"dateTime": "2026-03-20T09:00:00-05:00"},
                },
            ]
        }
        with _patch_resolve(mock_service):
            result = cs.search_events("test")
        assert "(No title)" in result

    def test_limit_clamped_minimum(self, mock_service):
        import mcp_tools.google_calendar.server as cs

        with _patch_resolve(mock_service):
            cs.search_events("test", limit=0)
        call_kwargs = mock_service.events.return_value.list.call_args[1]
        assert call_kwargs["maxResults"] == 1

    def test_limit_clamped_maximum(self, mock_service):
        import mcp_tools.google_calendar.server as cs

        with _patch_resolve(mock_service):
            cs.search_events("test", limit=100)
        call_kwargs = mock_service.events.return_value.list.call_args[1]
        assert call_kwargs["maxResults"] == 50

    def test_query_passed_to_api(self, mock_service):
        import mcp_tools.google_calendar.server as cs

        with _patch_resolve(mock_service):
            cs.search_events("dentist appointment")
        call_kwargs = mock_service.events.return_value.list.call_args[1]
        assert call_kwargs["q"] == "dentist appointment"

    def test_search_uses_correct_calendar(self, mock_service):
        import mcp_tools.google_calendar.server as cs

        acct = _sample_account_config()
        with patch.object(cs, "_resolve", return_value=(mock_service, acct, "")):
            cs.search_events("test", calendar="family")
        call_kwargs = mock_service.events.return_value.list.call_args[1]
        assert call_kwargs["calendarId"] == "family_calendar_id@group.calendar.google.com"

    def test_api_error(self, mock_service):
        import mcp_tools.google_calendar.server as cs

        mock_service.events.return_value.list.return_value.execute.side_effect = (
            Exception("Rate limit exceeded")
        )
        with _patch_resolve(mock_service):
            result = cs.search_events("test")
        assert "Error searching events" in result
        assert "Rate limit exceeded" in result

    def test_search_api_params(self, mock_service):
        import mcp_tools.google_calendar.server as cs

        with _patch_resolve(mock_service):
            cs.search_events("query", limit=25)
        call_kwargs = mock_service.events.return_value.list.call_args[1]
        assert call_kwargs["singleEvents"] is True
        assert call_kwargs["orderBy"] == "startTime"
        assert call_kwargs["maxResults"] == 25


# ---------------------------------------------------------------------------
# Integration / edge-case scenarios
# ---------------------------------------------------------------------------


class TestEdgeCases:
    """Miscellaneous edge cases and integration-style tests."""

    def test_event_id_truncation_short_id(self, mock_service):
        """Event IDs shorter than 12 chars should not crash."""
        import mcp_tools.google_calendar.server as cs

        mock_service.events.return_value.list.return_value.execute.return_value = {
            "items": [
                {
                    "id": "short",
                    "summary": "Short ID Event",
                    "start": {"dateTime": "2026-03-20T10:00:00"},
                },
            ]
        }
        with _patch_resolve(mock_service):
            result = cs.list_events()
        assert "short" in result

    def test_event_id_truncation_in_search(self, mock_service):
        """Event IDs shorter than 12 chars should not crash in search."""
        import mcp_tools.google_calendar.server as cs

        mock_service.events.return_value.list.return_value.execute.return_value = {
            "items": [
                {
                    "id": "tiny",
                    "summary": "Tiny",
                    "start": {"dateTime": "2026-03-20T10:00:00"},
                },
            ]
        }
        with _patch_resolve(mock_service):
            result = cs.search_events("tiny")
        assert "tiny" in result

    def test_services_cache_isolation(self):
        """Verify _services cache is cleared between tests by the fixture."""
        import mcp_tools.google_calendar.server as cs

        assert len(cs._services) == 0

    def test_list_events_with_timezone_aware_date(self, mock_service):
        """When a date string produces a tz-aware datetime, no trailing Z is appended."""
        import mcp_tools.google_calendar.server as cs

        with _patch_resolve(mock_service):
            cs.list_events(date="2026-03-20T00:00:00-05:00")
        call_kwargs = mock_service.events.return_value.list.call_args[1]
        # Should NOT have trailing Z since it's tz-aware
        assert not call_kwargs["timeMin"].endswith("Z")
        assert "-05:00" in call_kwargs["timeMin"]

    def test_resolve_calendar_empty_calendars_dict(self):
        """Calendar config with empty calendars sub-dict."""
        import mcp_tools.google_calendar.server as cs

        acct = {"calendar": {"default_calendar": "my@calendar.com", "calendars": {}}}
        assert cs._resolve_calendar(acct, "work") == "my@calendar.com"
        assert cs._resolve_calendar(acct, "default") == "my@calendar.com"

    def test_create_event_formatted_output(self, mock_service):
        """Verify the output format includes the expected date formatting."""
        import mcp_tools.google_calendar.server as cs

        with _patch_resolve(mock_service):
            result = cs.create_event("Dentist", "2026-07-04 14:30")
        assert "Jul 04 at 02:30 PM" in result

    def test_list_events_event_time_formatting(self, mock_service):
        """Verify event times are formatted as expected."""
        import mcp_tools.google_calendar.server as cs

        mock_service.events.return_value.list.return_value.execute.return_value = {
            "items": [
                {
                    "id": "format_test_12",
                    "summary": "Format Test",
                    "start": {"dateTime": "2026-03-20T14:30:00-05:00"},
                },
            ]
        }
        with _patch_resolve(mock_service):
            result = cs.list_events()
        # dateutil parses the TZ offset, strftime formats it
        assert "02:30 PM" in result

    def test_multiple_accounts_error(self, tmp_path):
        """Multiple accounts with calendar enabled should ask user to specify."""
        import mcp_tools.google_calendar.server as cs

        acct1 = _sample_account_config()
        acct2 = _sample_account_config()
        acct2["token_file"] = "data/google/token_work.json"
        auth = _sample_google_auth({"personal": acct1, "work": acct2})
        _write_config(tmp_path, auth)

        with patch.object(cs, "_get_service", return_value=MagicMock()):
            svc, acct, err = cs._resolve("")
        assert "Multiple accounts" in err or "Please specify" in err

    def test_requested_account_not_found(self, tmp_path):
        """Requesting a nonexistent account label should return an error."""
        import mcp_tools.google_calendar.server as cs

        _write_config(tmp_path, _sample_google_auth())
        svc, acct, err = cs._resolve("nonexistent")
        assert "not found" in err.lower()
        assert "personal" in err  # should list available accounts
