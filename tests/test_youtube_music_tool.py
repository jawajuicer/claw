"""Tests for mcp_tools/youtube_music/server.py — YouTube Music MCP tool."""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
import yaml


# ---------------------------------------------------------------------------
# Module-level reset fixture — clear cached config and player between tests
# ---------------------------------------------------------------------------

@pytest.fixture(autouse=True)
def _reset_server_module():
    import mcp_tools.youtube_music.server as yt_srv

    orig_yaml = yt_srv._CONFIG_YAML
    yt_srv._config = None
    yt_srv._player = None
    yield
    yt_srv._config = None
    yt_srv._player = None
    yt_srv._CONFIG_YAML = orig_yaml


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _enable(srv_mod, *, extra: dict | None = None):
    """Set the module config to enabled with optional overrides."""
    cfg = {"enabled": True}
    if extra:
        cfg.update(extra)
    srv_mod._config = cfg


def _disable(srv_mod):
    """Set the module config to disabled."""
    srv_mod._config = {"enabled": False}


def _mock_player() -> MagicMock:
    """Create a MagicMock standing in for MusicPlayer with sensible defaults."""
    player = MagicMock()
    player.play_search.return_value = "Now playing: Test Song by Test Artist"
    player.play_playlist.return_value = "Now playing playlist 'My Playlist' (3 tracks)"
    player.search.return_value = []
    player.pause.return_value = "Paused: Test Song by Test Artist"
    player.resume.return_value = "Resumed: Test Song by Test Artist"
    player.skip.return_value = "Skipped 'Test Song'. Now playing: Next Song by Next Artist"
    player.stop.return_value = "Stopped playing Test Song"
    player.set_volume.return_value = "Volume set to 75%"
    player.seek.return_value = "Seeked to 30s"
    player.get_status.return_value = {"playing": True, "paused": False}
    player.get_now_playing.return_value = None
    player.get_queue.return_value = []
    player.get_history.return_value = []
    player.pin_song.return_value = 'Pinned \'Song\' by Artist as "my phrase"'
    player.pin_playlist.return_value = 'Pinned playlist \'Chill Mix\' as "chill"'
    player.unpin_song.return_value = 'Unpinned "my phrase"'
    player.get_pins.return_value = {}
    return player


@pytest.fixture()
def mock_player():
    """Inject a mock MusicPlayer into the server module and return it."""
    import mcp_tools.youtube_music.server as srv

    player = _mock_player()
    srv._player = player
    _enable(srv)
    return player


# ===================================================================
# Config loading
# ===================================================================

class TestLoadConfig:
    """Test _load_config reads and caches the youtube_music section."""

    def test_no_config_file(self, tmp_path):
        import mcp_tools.youtube_music.server as srv

        srv._CONFIG_YAML = tmp_path / "nonexistent.yaml"
        cfg = srv._load_config()
        assert cfg == {}

    def test_reads_youtube_music_section(self, tmp_path):
        import mcp_tools.youtube_music.server as srv

        cfg_file = tmp_path / "config.yaml"
        cfg_file.write_text(yaml.dump({
            "youtube_music": {"enabled": True, "default_volume": 60},
        }))
        srv._CONFIG_YAML = cfg_file
        cfg = srv._load_config()
        assert cfg["enabled"] is True
        assert cfg["default_volume"] == 60

    def test_empty_yaml_returns_empty_dict(self, tmp_path):
        import mcp_tools.youtube_music.server as srv

        cfg_file = tmp_path / "config.yaml"
        cfg_file.write_text("")
        srv._CONFIG_YAML = cfg_file
        cfg = srv._load_config()
        assert cfg == {}

    def test_missing_youtube_music_key(self, tmp_path):
        import mcp_tools.youtube_music.server as srv

        cfg_file = tmp_path / "config.yaml"
        cfg_file.write_text(yaml.dump({"weather": {"api_key": "x"}}))
        srv._CONFIG_YAML = cfg_file
        cfg = srv._load_config()
        assert cfg == {}

    def test_config_is_cached(self):
        import mcp_tools.youtube_music.server as srv

        srv._config = {"enabled": True, "cached": True}
        cfg = srv._load_config()
        assert cfg["cached"] is True


# ===================================================================
# Enabled / disabled guard
# ===================================================================

class TestIsEnabled:
    """Test _is_enabled correctly reads config."""

    def test_enabled_true(self):
        import mcp_tools.youtube_music.server as srv

        srv._config = {"enabled": True}
        assert srv._is_enabled() is True

    def test_enabled_false(self):
        import mcp_tools.youtube_music.server as srv

        srv._config = {"enabled": False}
        assert srv._is_enabled() is False

    def test_missing_enabled_key_defaults_false(self):
        import mcp_tools.youtube_music.server as srv

        srv._config = {}
        assert srv._is_enabled() is False


class TestNotConfiguredGuard:
    """Every tool must return _NOT_CONFIGURED when disabled."""

    _TOOLS_RETURNING_STRING = [
        ("play_song", {"title": "test"}),
        ("play_playlist", {"playlist_id": "PLtest"}),
        ("search_music", {"query": "test"}),
        ("pause", {}),
        ("resume", {}),
        ("skip", {}),
        ("stop", {}),
        ("set_volume", {"level": 50}),
        ("seek", {"position": 30.0}),
        ("now_playing", {}),
        ("get_queue", {}),
        ("listen_history", {}),
        ("pin_song", {"phrase": "test"}),
        ("pin_playlist", {"phrase": "test", "playlist_id": "PLtest"}),
        ("unpin_song", {"phrase": "test"}),
        ("list_pins", {}),
    ]

    @pytest.mark.parametrize("tool_name,kwargs", _TOOLS_RETURNING_STRING)
    def test_disabled_tools_return_not_configured(self, tool_name, kwargs):
        import mcp_tools.youtube_music.server as srv

        _disable(srv)
        func = getattr(srv, tool_name)
        result = func(**kwargs)
        assert "not configured" in result.lower()

    def test_get_status_disabled_returns_json(self):
        """get_status returns JSON even when disabled."""
        import mcp_tools.youtube_music.server as srv

        _disable(srv)
        result = srv.get_status()
        data = json.loads(result)
        assert data["playing"] is False
        assert data["error"] == "not_configured"


# ===================================================================
# _get_player lazy init
# ===================================================================

class TestGetPlayer:
    """Test lazy player initialization."""

    def test_returns_cached_player(self):
        import mcp_tools.youtube_music.server as srv

        fake_player = MagicMock()
        srv._player = fake_player
        assert srv._get_player() is fake_player

    def test_creates_player_with_config_defaults(self, tmp_path):
        import mcp_tools.youtube_music.server as srv

        srv._config = {"enabled": True}
        srv._PROJECT_ROOT = tmp_path

        MockMP = MagicMock()
        mock_player_mod = MagicMock()
        mock_player_mod.MusicPlayer = MockMP
        with patch.dict("sys.modules", {"player": mock_player_mod}):
            srv._get_player()

        MockMP.assert_called_once()
        call_kwargs = MockMP.call_args[1]
        assert call_kwargs["default_volume"] == 80
        assert call_kwargs["auto_radio"] is True
        assert call_kwargs["max_history"] == 500
        assert call_kwargs["max_search_results"] == 5

    def test_creates_player_with_custom_config(self, tmp_path):
        import mcp_tools.youtube_music.server as srv

        srv._config = {
            "enabled": True,
            "default_volume": 50,
            "auto_radio": False,
            "max_history": 100,
            "max_search_results": 10,
            "auth_file": "custom/auth.json",
            "history_file": "custom/history.json",
            "pins_file": "custom/pins.json",
            "client_id": "my_id",
            "client_secret": "my_secret",
        }
        srv._PROJECT_ROOT = tmp_path

        MockMP = MagicMock()
        mock_player_mod = MagicMock()
        mock_player_mod.MusicPlayer = MockMP
        with patch.dict("sys.modules", {"player": mock_player_mod}):
            srv._get_player()

        call_kwargs = MockMP.call_args[1]
        assert call_kwargs["default_volume"] == 50
        assert call_kwargs["auto_radio"] is False
        assert call_kwargs["max_history"] == 100
        assert call_kwargs["max_search_results"] == 10
        assert call_kwargs["client_id"] == "my_id"
        assert call_kwargs["client_secret"] == "my_secret"
        # Relative paths resolved against PROJECT_ROOT
        assert call_kwargs["auth_file"] == str(tmp_path / "custom/auth.json")
        assert call_kwargs["history_file"] == str(tmp_path / "custom/history.json")
        assert call_kwargs["pins_file"] == str(tmp_path / "custom/pins.json")

    def test_absolute_paths_not_resolved(self, tmp_path):
        import mcp_tools.youtube_music.server as srv

        srv._config = {
            "enabled": True,
            "auth_file": "/absolute/auth.json",
            "history_file": "/absolute/history.json",
            "pins_file": "/absolute/pins.json",
        }
        srv._PROJECT_ROOT = tmp_path

        MockMP = MagicMock()
        mock_player_mod = MagicMock()
        mock_player_mod.MusicPlayer = MockMP
        with patch.dict("sys.modules", {"player": mock_player_mod}):
            srv._get_player()

        call_kwargs = MockMP.call_args[1]
        assert call_kwargs["auth_file"] == "/absolute/auth.json"
        assert call_kwargs["history_file"] == "/absolute/history.json"
        assert call_kwargs["pins_file"] == "/absolute/pins.json"


# ===================================================================
# play_song
# ===================================================================

class TestPlaySong:
    """Test the play_song MCP tool."""

    def test_play_with_title_only(self, mock_player):
        import mcp_tools.youtube_music.server as srv

        result = srv.play_song("Bohemian Rhapsody")
        mock_player.play_search.assert_called_once_with("Bohemian Rhapsody")
        assert "Now playing" in result

    def test_play_with_title_and_artist(self, mock_player):
        import mcp_tools.youtube_music.server as srv

        result = srv.play_song("Bohemian Rhapsody", artist="Queen")
        mock_player.play_search.assert_called_once_with("Bohemian Rhapsody Queen")
        assert "Now playing" in result

    def test_play_with_empty_artist(self, mock_player):
        import mcp_tools.youtube_music.server as srv

        result = srv.play_song("Yesterday", artist="")
        mock_player.play_search.assert_called_once_with("Yesterday")

    def test_play_when_disabled(self):
        import mcp_tools.youtube_music.server as srv

        _disable(srv)
        result = srv.play_song("Test")
        assert "not configured" in result.lower()


# ===================================================================
# play_playlist
# ===================================================================

class TestPlayPlaylist:
    """Test the play_playlist MCP tool."""

    def test_play_playlist_success(self, mock_player):
        import mcp_tools.youtube_music.server as srv

        result = srv.play_playlist("PLabc123")
        mock_player.play_playlist.assert_called_once_with("PLabc123")
        assert "Now playing playlist" in result

    def test_play_playlist_when_disabled(self):
        import mcp_tools.youtube_music.server as srv

        _disable(srv)
        result = srv.play_playlist("PLabc123")
        assert "not configured" in result.lower()


# ===================================================================
# search_music
# ===================================================================

class TestSearchMusic:
    """Test the search_music MCP tool."""

    def test_search_returns_formatted_results(self, mock_player):
        import mcp_tools.youtube_music.server as srv

        mock_player.search.return_value = [
            {"title": "Song A", "artist": "Artist A", "album": "Album A", "duration": "3:45"},
            {"title": "Song B", "artist": "Artist B", "album": "", "duration": ""},
        ]
        result = srv.search_music("test query", limit=5)
        assert "Search results for 'test query'" in result
        assert "1. Song A by Artist A" in result
        assert "Album A" in result
        assert "(3:45)" in result
        assert "2. Song B by Artist B" in result
        mock_player.search.assert_called_once_with("test query", limit=5)

    def test_search_no_results(self, mock_player):
        import mcp_tools.youtube_music.server as srv

        mock_player.search.return_value = []
        result = srv.search_music("obscure query")
        assert "No results found for 'obscure query'" in result

    def test_search_no_album(self, mock_player):
        import mcp_tools.youtube_music.server as srv

        mock_player.search.return_value = [
            {"title": "Single", "artist": "Solo", "album": "", "duration": "2:30"},
        ]
        result = srv.search_music("single")
        # Album should be omitted, not " -- "
        assert " -- " not in result.replace(" --- ", "")
        assert "Single by Solo" in result

    def test_search_no_duration(self, mock_player):
        import mcp_tools.youtube_music.server as srv

        mock_player.search.return_value = [
            {"title": "Live", "artist": "Band", "album": "Concert", "duration": ""},
        ]
        result = srv.search_music("live")
        # Duration parens should not appear
        assert "()" not in result

    def test_search_limit_clamped_high(self, mock_player):
        import mcp_tools.youtube_music.server as srv

        mock_player.search.return_value = []
        srv.search_music("test", limit=50)
        mock_player.search.assert_called_once_with("test", limit=20)

    def test_search_limit_clamped_low(self, mock_player):
        import mcp_tools.youtube_music.server as srv

        mock_player.search.return_value = []
        srv.search_music("test", limit=0)
        mock_player.search.assert_called_once_with("test", limit=1)

    def test_search_limit_negative(self, mock_player):
        import mcp_tools.youtube_music.server as srv

        mock_player.search.return_value = []
        srv.search_music("test", limit=-5)
        mock_player.search.assert_called_once_with("test", limit=1)


# ===================================================================
# pause / resume / skip / stop
# ===================================================================

class TestPlaybackControls:
    """Test pause, resume, skip, stop tools."""

    def test_pause(self, mock_player):
        import mcp_tools.youtube_music.server as srv

        result = srv.pause()
        mock_player.pause.assert_called_once()
        assert "Paused" in result

    def test_resume(self, mock_player):
        import mcp_tools.youtube_music.server as srv

        result = srv.resume()
        mock_player.resume.assert_called_once()
        assert "Resumed" in result

    def test_skip(self, mock_player):
        import mcp_tools.youtube_music.server as srv

        result = srv.skip()
        mock_player.skip.assert_called_once()
        assert "Skipped" in result

    def test_stop(self, mock_player):
        import mcp_tools.youtube_music.server as srv

        result = srv.stop()
        mock_player.stop.assert_called_once()
        assert "Stopped" in result


# ===================================================================
# set_volume
# ===================================================================

class TestSetVolume:
    """Test the set_volume MCP tool."""

    def test_set_volume(self, mock_player):
        import mcp_tools.youtube_music.server as srv

        result = srv.set_volume(75)
        mock_player.set_volume.assert_called_once_with(75)
        assert "75%" in result


# ===================================================================
# seek
# ===================================================================

class TestSeek:
    """Test the seek MCP tool."""

    def test_seek(self, mock_player):
        import mcp_tools.youtube_music.server as srv

        result = srv.seek(30.5)
        mock_player.seek.assert_called_once_with(30.5)
        assert "30s" in result


# ===================================================================
# get_status
# ===================================================================

class TestGetStatus:
    """Test the get_status MCP tool — always returns JSON."""

    def test_status_enabled(self, mock_player):
        import mcp_tools.youtube_music.server as srv

        mock_player.get_status.return_value = {
            "playing": True,
            "paused": False,
            "title": "Song",
            "artist": "Artist",
            "time_pos": 45.2,
            "duration": 210.0,
            "volume": 80,
        }
        result = srv.get_status()
        data = json.loads(result)
        assert data["playing"] is True
        assert data["title"] == "Song"

    def test_status_not_playing(self, mock_player):
        import mcp_tools.youtube_music.server as srv

        mock_player.get_status.return_value = {"playing": False}
        result = srv.get_status()
        data = json.loads(result)
        assert data["playing"] is False

    def test_status_disabled_returns_json(self):
        import mcp_tools.youtube_music.server as srv

        _disable(srv)
        result = srv.get_status()
        data = json.loads(result)
        assert data["playing"] is False
        assert data["error"] == "not_configured"


# ===================================================================
# now_playing
# ===================================================================

class TestNowPlaying:
    """Test the now_playing MCP tool."""

    def test_nothing_playing(self, mock_player):
        import mcp_tools.youtube_music.server as srv

        mock_player.get_now_playing.return_value = None
        result = srv.now_playing()
        assert "Nothing is currently playing" in result

    def test_track_with_all_fields(self, mock_player):
        import mcp_tools.youtube_music.server as srv

        mock_player.get_now_playing.return_value = {
            "title": "Stairway to Heaven",
            "artist": "Led Zeppelin",
            "album": "Led Zeppelin IV",
            "duration": "8:02",
            "started_at": "2026-03-16T12:00:00+00:00",
        }
        result = srv.now_playing()
        assert "Now playing: Stairway to Heaven by Led Zeppelin" in result
        assert "Album: Led Zeppelin IV" in result
        assert "Duration: 8:02" in result
        assert "Started: 2026-03-16T12:00:00+00:00" in result

    def test_track_without_optional_fields(self, mock_player):
        import mcp_tools.youtube_music.server as srv

        mock_player.get_now_playing.return_value = {
            "title": "Instrumental",
            "artist": "Unknown",
        }
        result = srv.now_playing()
        assert "Now playing: Instrumental by Unknown" in result
        assert "Album:" not in result
        assert "Duration:" not in result
        assert "Started:" not in result

    def test_track_with_empty_album(self, mock_player):
        import mcp_tools.youtube_music.server as srv

        mock_player.get_now_playing.return_value = {
            "title": "Loose Single",
            "artist": "Indie Band",
            "album": "",
            "duration": "3:15",
        }
        result = srv.now_playing()
        assert "Album:" not in result
        assert "Duration: 3:15" in result


# ===================================================================
# get_queue
# ===================================================================

class TestGetQueue:
    """Test the get_queue MCP tool."""

    def test_empty_queue(self, mock_player):
        import mcp_tools.youtube_music.server as srv

        mock_player.get_queue.return_value = []
        result = srv.get_queue()
        assert "Queue is empty" in result

    def test_queue_with_tracks(self, mock_player):
        import mcp_tools.youtube_music.server as srv

        mock_player.get_queue.return_value = [
            {"title": "Next Song", "artist": "Artist A", "duration": "4:00"},
            {"title": "After That", "artist": "Artist B"},
        ]
        result = srv.get_queue()
        assert "Upcoming songs:" in result
        assert "1. Next Song by Artist A (4:00)" in result
        assert "2. After That by Artist B" in result

    def test_queue_limit_clamped_high(self, mock_player):
        import mcp_tools.youtube_music.server as srv

        mock_player.get_queue.return_value = []
        srv.get_queue(limit=100)
        mock_player.get_queue.assert_called_once_with(50)

    def test_queue_limit_clamped_low(self, mock_player):
        import mcp_tools.youtube_music.server as srv

        mock_player.get_queue.return_value = []
        srv.get_queue(limit=0)
        mock_player.get_queue.assert_called_once_with(1)

    def test_queue_track_without_duration(self, mock_player):
        import mcp_tools.youtube_music.server as srv

        mock_player.get_queue.return_value = [
            {"title": "Live Jam", "artist": "Improv"},
        ]
        result = srv.get_queue()
        assert "()" not in result
        assert "1. Live Jam by Improv" in result


# ===================================================================
# listen_history
# ===================================================================

class TestListenHistory:
    """Test the listen_history MCP tool."""

    def test_no_history(self, mock_player):
        import mcp_tools.youtube_music.server as srv

        mock_player.get_history.return_value = []
        result = srv.listen_history()
        assert "No listen history yet" in result

    def test_history_with_entries(self, mock_player):
        import mcp_tools.youtube_music.server as srv

        mock_player.get_history.return_value = [
            {
                "title": "Song 1",
                "artist": "Artist 1",
                "played_at": "2026-03-16T10:00:00Z",
                "source": "user_request",
            },
            {
                "title": "Song 2",
                "artist": "Artist 2",
                "played_at": "2026-03-16T10:05:00Z",
                "source": "auto_radio",
            },
        ]
        result = srv.listen_history()
        assert "Recent listen history:" in result
        assert "1. Song 1 by Artist 1" in result
        assert "[radio]" not in result.split("\n")[1]  # first entry is user_request
        assert "[radio]" in result.split("\n")[2]  # second entry is auto_radio

    def test_history_limit_clamped_high(self, mock_player):
        import mcp_tools.youtube_music.server as srv

        mock_player.get_history.return_value = []
        srv.listen_history(limit=200)
        mock_player.get_history.assert_called_once_with(100)

    def test_history_limit_clamped_low(self, mock_player):
        import mcp_tools.youtube_music.server as srv

        mock_player.get_history.return_value = []
        srv.listen_history(limit=0)
        mock_player.get_history.assert_called_once_with(1)


# ===================================================================
# pin_song
# ===================================================================

class TestPinSong:
    """Test the pin_song MCP tool."""

    def test_pin_song_delegates(self, mock_player):
        import mcp_tools.youtube_music.server as srv

        result = srv.pin_song("mario 64 ost")
        mock_player.pin_song.assert_called_once_with("mario 64 ost")

    def test_pin_song_when_disabled(self):
        import mcp_tools.youtube_music.server as srv

        _disable(srv)
        result = srv.pin_song("test")
        assert "not configured" in result.lower()


# ===================================================================
# pin_playlist
# ===================================================================

class TestPinPlaylist:
    """Test the pin_playlist MCP tool."""

    def test_pin_playlist_delegates(self, mock_player):
        import mcp_tools.youtube_music.server as srv

        result = srv.pin_playlist("chill", "PLchill123", title="Chill Mix")
        mock_player.pin_playlist.assert_called_once_with("chill", "PLchill123", "Chill Mix")

    def test_pin_playlist_no_title(self, mock_player):
        import mcp_tools.youtube_music.server as srv

        srv.pin_playlist("vibes", "PLvibes456")
        mock_player.pin_playlist.assert_called_once_with("vibes", "PLvibes456", "")


# ===================================================================
# unpin_song
# ===================================================================

class TestUnpinSong:
    """Test the unpin_song MCP tool."""

    def test_unpin_delegates(self, mock_player):
        import mcp_tools.youtube_music.server as srv

        result = srv.unpin_song("mario 64 ost")
        mock_player.unpin_song.assert_called_once_with("mario 64 ost")

    def test_unpin_when_disabled(self):
        import mcp_tools.youtube_music.server as srv

        _disable(srv)
        result = srv.unpin_song("test")
        assert "not configured" in result.lower()


# ===================================================================
# list_pins
# ===================================================================

class TestListPins:
    """Test the list_pins MCP tool."""

    def test_no_pins(self, mock_player):
        import mcp_tools.youtube_music.server as srv

        mock_player.get_pins.return_value = {}
        result = srv.list_pins()
        assert "No pinned songs" in result

    def test_song_pins(self, mock_player):
        import mcp_tools.youtube_music.server as srv

        mock_player.get_pins.return_value = {
            "mario 64 ost": {
                "video_id": "abc",
                "title": "SM64 Slide Theme",
                "artist": "Koji Kondo",
            },
        }
        result = srv.list_pins()
        assert "Pinned songs:" in result
        assert '"mario 64 ost"' in result
        assert "SM64 Slide Theme by Koji Kondo" in result

    def test_playlist_pins(self, mock_player):
        import mcp_tools.youtube_music.server as srv

        mock_player.get_pins.return_value = {
            "chill vibes": {
                "playlist_id": "PLchill",
                "title": "Chill Mix",
            },
        }
        result = srv.list_pins()
        assert "Pinned songs:" in result
        assert '"chill vibes"' in result
        assert "playlist: Chill Mix" in result

    def test_mixed_pins(self, mock_player):
        import mcp_tools.youtube_music.server as srv

        mock_player.get_pins.return_value = {
            "song phrase": {
                "video_id": "v1",
                "title": "My Song",
                "artist": "My Artist",
            },
            "playlist phrase": {
                "playlist_id": "PLtest",
                "title": "My Playlist",
            },
        }
        result = srv.list_pins()
        assert "My Song by My Artist" in result
        assert "playlist: My Playlist" in result


# ===================================================================
# Edge cases: search result formatting
# ===================================================================

class TestSearchFormatEdgeCases:
    """Test edge cases in search result formatting."""

    def test_all_fields_present(self, mock_player):
        import mcp_tools.youtube_music.server as srv

        mock_player.search.return_value = [
            {
                "title": "Complete Song",
                "artist": "Full Artist",
                "album": "Great Album",
                "duration": "5:30",
            },
        ]
        result = srv.search_music("complete")
        expected = "1. Complete Song by Full Artist \u2014 Great Album (5:30)"
        assert expected in result

    def test_single_result(self, mock_player):
        import mcp_tools.youtube_music.server as srv

        mock_player.search.return_value = [
            {"title": "Only One", "artist": "Solo", "album": "", "duration": "2:00"},
        ]
        result = srv.search_music("only")
        lines = result.strip().split("\n")
        assert len(lines) == 2  # header + 1 result
        assert "1. Only One by Solo" in lines[1]

    def test_multiple_results_numbered(self, mock_player):
        import mcp_tools.youtube_music.server as srv

        mock_player.search.return_value = [
            {"title": f"Song {i}", "artist": f"Artist {i}", "album": "", "duration": ""}
            for i in range(1, 6)
        ]
        result = srv.search_music("songs")
        for i in range(1, 6):
            assert f"{i}. Song {i} by Artist {i}" in result


# ===================================================================
# Integration-style: config file -> player init path
# ===================================================================

class TestConfigToPlayerInit:
    """Test config.yaml values flow correctly to MusicPlayer constructor."""

    def test_full_config_path(self, tmp_path):
        """Write a real config.yaml and verify _get_player reads it correctly."""
        import mcp_tools.youtube_music.server as srv

        cfg_data = {
            "youtube_music": {
                "enabled": True,
                "auth_file": "data/yt/auth.json",
                "history_file": "data/yt/history.json",
                "pins_file": "data/yt/pins.json",
                "default_volume": 42,
                "auto_radio": False,
                "max_history": 99,
                "max_search_results": 3,
                "client_id": "cid",
                "client_secret": "csec",
            }
        }
        cfg_file = tmp_path / "config.yaml"
        cfg_file.write_text(yaml.dump(cfg_data))

        srv._CONFIG_YAML = cfg_file
        srv._PROJECT_ROOT = tmp_path
        srv._config = None  # force reload

        MockMP = MagicMock()
        mock_player_mod = MagicMock()
        mock_player_mod.MusicPlayer = MockMP
        with patch.dict("sys.modules", {"player": mock_player_mod}):
            srv._get_player()

        kw = MockMP.call_args[1]
        assert kw["auth_file"] == str(tmp_path / "data/yt/auth.json")
        assert kw["history_file"] == str(tmp_path / "data/yt/history.json")
        assert kw["pins_file"] == str(tmp_path / "data/yt/pins.json")
        assert kw["default_volume"] == 42
        assert kw["auto_radio"] is False
        assert kw["max_history"] == 99
        assert kw["max_search_results"] == 3
        assert kw["client_id"] == "cid"
        assert kw["client_secret"] == "csec"
