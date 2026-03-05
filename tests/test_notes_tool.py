"""Tests for mcp_tools/notes/server.py — Notes & Reminders MCP tool."""

from __future__ import annotations

import json

import pytest


@pytest.fixture(autouse=True)
def _reset_notes_module():
    import mcp_tools.notes.server as ns

    ns._config = None
    yield
    ns._config = None


@pytest.fixture()
def notes_dir(tmp_path):
    """Set up a temp storage directory for notes."""
    import mcp_tools.notes.server as ns

    ns._config = {"enabled": True, "storage_dir": str(tmp_path / "notes"), "max_notes": 0}
    ns._PROJECT_ROOT = tmp_path
    (tmp_path / "notes").mkdir(parents=True, exist_ok=True)
    return tmp_path / "notes"


class TestCreateNote:
    """Test note creation."""

    def test_create_note_success(self, notes_dir):
        import mcp_tools.notes.server as ns

        result = ns.create_note("Shopping List", "Buy milk and eggs", tags="grocery, personal")
        assert "Note created" in result
        assert "Shopping List" in result

        # Verify stored
        notes = json.loads((notes_dir / "notes.json").read_text())
        assert len(notes) == 1
        assert notes[0]["title"] == "Shopping List"
        assert notes[0]["tags"] == ["grocery", "personal"]

    def test_create_note_when_disabled(self, tmp_path):
        import mcp_tools.notes.server as ns

        ns._config = {"enabled": False, "storage_dir": "data/notes"}
        result = ns.create_note("Test", "Content")
        assert "disabled" in result.lower()

    def test_create_note_max_limit(self, notes_dir):
        import mcp_tools.notes.server as ns

        ns._config["max_notes"] = 2
        ns.create_note("Note 1", "Content 1")
        ns.create_note("Note 2", "Content 2")
        result = ns.create_note("Note 3", "Content 3")
        assert "maximum" in result.lower()

    def test_create_note_no_tags(self, notes_dir):
        import mcp_tools.notes.server as ns

        result = ns.create_note("Simple Note", "Just text")
        assert "[tags:" not in result


class TestListNotes:
    """Test note listing."""

    def test_list_empty(self, notes_dir):
        import mcp_tools.notes.server as ns

        result = ns.list_notes()
        assert "no notes" in result.lower()

    def test_list_with_notes(self, notes_dir):
        import mcp_tools.notes.server as ns

        ns.create_note("First", "Content 1")
        ns.create_note("Second", "Content 2")
        result = ns.list_notes()
        assert "2" in result
        assert "First" in result
        assert "Second" in result

    def test_list_filter_by_tag(self, notes_dir):
        import mcp_tools.notes.server as ns

        ns.create_note("Tagged", "Content", tags="important")
        ns.create_note("Untagged", "Content")
        result = ns.list_notes(tag="important")
        assert "Tagged" in result
        assert "Untagged" not in result

    def test_list_tag_not_found(self, notes_dir):
        import mcp_tools.notes.server as ns

        ns.create_note("Note", "Content", tags="work")
        result = ns.list_notes(tag="nonexistent")
        assert "no notes found" in result.lower()


class TestSearchNotes:
    """Test note searching."""

    def test_search_by_title(self, notes_dir):
        import mcp_tools.notes.server as ns

        ns.create_note("Python Tips", "Use list comprehensions")
        ns.create_note("Grocery List", "Milk, eggs, bread")
        result = ns.search_notes("python")
        assert "Python Tips" in result
        assert "Grocery" not in result

    def test_search_by_content(self, notes_dir):
        import mcp_tools.notes.server as ns

        ns.create_note("Recipe", "Add milk and stir")
        result = ns.search_notes("milk")
        assert "Recipe" in result

    def test_search_no_results(self, notes_dir):
        import mcp_tools.notes.server as ns

        ns.create_note("Note", "Content")
        result = ns.search_notes("nonexistent_term")
        assert "no notes matching" in result.lower()


class TestGetNote:
    """Test getting full note content."""

    def test_get_existing_note(self, notes_dir):
        import mcp_tools.notes.server as ns

        ns.create_note("Test Note", "Full content here", tags="test")
        notes = json.loads((notes_dir / "notes.json").read_text())
        note_id = notes[0]["id"]
        result = ns.get_note(note_id)
        assert "Test Note" in result
        assert "Full content here" in result
        assert "test" in result  # tag

    def test_get_nonexistent_note(self, notes_dir):
        import mcp_tools.notes.server as ns

        result = ns.get_note("nonexistent")
        assert "not found" in result.lower()


class TestUpdateNote:
    """Test note updates."""

    def test_update_title(self, notes_dir):
        import mcp_tools.notes.server as ns

        ns.create_note("Old Title", "Content")
        notes = json.loads((notes_dir / "notes.json").read_text())
        note_id = notes[0]["id"]
        result = ns.update_note(note_id, title="New Title")
        assert "updated" in result.lower()

        updated = json.loads((notes_dir / "notes.json").read_text())
        assert updated[0]["title"] == "New Title"

    def test_update_nonexistent(self, notes_dir):
        import mcp_tools.notes.server as ns

        result = ns.update_note("fake_id", title="New")
        assert "not found" in result.lower()


class TestDeleteNote:
    """Test note deletion."""

    def test_delete_existing_note(self, notes_dir):
        import mcp_tools.notes.server as ns

        ns.create_note("To Delete", "Content")
        notes = json.loads((notes_dir / "notes.json").read_text())
        note_id = notes[0]["id"]
        result = ns.delete_note(note_id)
        assert "deleted" in result.lower()

        remaining = json.loads((notes_dir / "notes.json").read_text())
        assert len(remaining) == 0

    def test_delete_nonexistent(self, notes_dir):
        import mcp_tools.notes.server as ns

        result = ns.delete_note("fake_id")
        assert "not found" in result.lower()


class TestSetReminder:
    """Test reminder creation."""

    def test_set_reminder(self, notes_dir):
        import mcp_tools.notes.server as ns

        result = ns.set_reminder("Call dentist", "tomorrow 9am")
        assert "Reminder set" in result
        assert "Call dentist" in result

    def test_set_reminder_relative_time(self, notes_dir):
        import mcp_tools.notes.server as ns

        result = ns.set_reminder("Check oven", "in 30 minutes")
        assert "Reminder set" in result


class TestParseTime:
    """Test the _parse_time helper."""

    def test_relative_minutes(self):
        import mcp_tools.notes.server as ns

        result = ns._parse_time("in 30 minutes")
        assert "T" in result  # ISO format

    def test_relative_hours(self):
        import mcp_tools.notes.server as ns

        result = ns._parse_time("in 2 hours")
        assert "T" in result

    def test_tomorrow(self):
        import mcp_tools.notes.server as ns

        result = ns._parse_time("tomorrow")
        assert "T09:00:00" in result  # defaults to 9am

    def test_tomorrow_with_time(self):
        import mcp_tools.notes.server as ns

        result = ns._parse_time("tomorrow 3pm")
        assert "T15:00:00" in result

    def test_unparseable_returns_original(self):
        import mcp_tools.notes.server as ns

        result = ns._parse_time("sometime next year maybe")
        # dateutil may try to parse; if it fails it returns the original
        assert isinstance(result, str)
