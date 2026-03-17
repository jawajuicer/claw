"""Tests for mcp_tools.youtube_music.player — search ranking and pinned songs."""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import patch

from mcp_tools.youtube_music.player import MusicPlayer


class TestRankResults:
    """Test the _rank_results static method for search relevance scoring."""

    def test_exact_title_match_preferred(self):
        results = [
            {"title": "Mario 64 Remix", "artist": "DJ Someone"},
            {"title": "Super Mario 64 OST", "artist": "Koji Kondo"},
            {"title": "Mario Party Music", "artist": "Nintendo"},
        ]
        ranked = MusicPlayer._rank_results(results, "mario 64 ost")
        assert ranked[0]["title"] == "Super Mario 64 OST"

    def test_artist_contributes_to_score(self):
        results = [
            {"title": "Song A", "artist": "Unknown"},
            {"title": "Song B", "artist": "Beatles"},
        ]
        ranked = MusicPlayer._rank_results(results, "beatles song b")
        # "Song B" by "Beatles" should rank highest (title match + artist match)
        assert ranked[0]["title"] == "Song B"

    def test_single_result_returned_as_is(self):
        results = [{"title": "Only Song", "artist": "Only Artist"}]
        ranked = MusicPlayer._rank_results(results, "anything")
        assert len(ranked) == 1
        assert ranked[0]["title"] == "Only Song"

    def test_empty_results(self):
        ranked = MusicPlayer._rank_results([], "test query")
        assert ranked == []

    def test_title_terms_score_higher_than_artist(self):
        results = [
            {"title": "Something Else", "artist": "Bohemian"},
            {"title": "Bohemian Rhapsody", "artist": "Someone"},
        ]
        ranked = MusicPlayer._rank_results(results, "bohemian rhapsody")
        # Title match for both terms (4pts) beats artist match for one (1pt)
        assert ranked[0]["title"] == "Bohemian Rhapsody"


class TestNormalizePhrase:
    """Test the _normalize_phrase static method."""

    def test_lowercase_and_strip(self):
        assert MusicPlayer._normalize_phrase("  Mario 64 OST  ") == "mario 64 ost"

    def test_removes_leading_the(self):
        assert MusicPlayer._normalize_phrase("the mario 64 ost") == "mario 64 ost"

    def test_preserves_inner_the(self):
        assert MusicPlayer._normalize_phrase("under the bridge") == "under the bridge"

    def test_empty_string(self):
        assert MusicPlayer._normalize_phrase("") == ""

    def test_just_the(self):
        # "the" alone doesn't strip (no trailing space), which is fine —
        # nobody pins a song as just "the"
        assert MusicPlayer._normalize_phrase("the") == "the"


def _make_player(tmp_path: Path, pins: dict | None = None) -> MusicPlayer:
    """Create a MusicPlayer with fake auth/history/pins files for testing."""
    auth = tmp_path / "auth.json"
    auth.write_text("{}")
    history = tmp_path / "history.json"
    history.write_text("[]")
    pins_file = tmp_path / "pins.json"
    if pins:
        pins_file.write_text(json.dumps(pins))

    return MusicPlayer(
        auth_file=str(auth),
        history_file=str(history),
        pins_file=str(pins_file),
    )


class TestPinSong:
    """Test pinning the currently playing track."""

    def test_pin_song_saves_current_track(self, tmp_path):
        player = _make_player(tmp_path)
        player._current = {
            "video_id": "abc123",
            "title": "Mario 64 Slide Theme",
            "artist": "Koji Kondo",
        }
        result = player.pin_song("mario 64 ost")
        assert "Pinned" in result
        assert "mario 64 ost" in result
        assert player._pinned["mario 64 ost"]["video_id"] == "abc123"
        # Verify persisted to disk
        saved = json.loads(player._pins_file.read_text())
        assert saved["mario 64 ost"]["video_id"] == "abc123"

    def test_pin_song_nothing_playing(self, tmp_path):
        player = _make_player(tmp_path)
        result = player.pin_song("mario 64 ost")
        assert "Nothing is playing" in result
        assert len(player._pinned) == 0

    def test_pin_normalizes_phrase(self, tmp_path):
        player = _make_player(tmp_path)
        player._current = {
            "video_id": "xyz",
            "title": "Test Song",
            "artist": "Test Artist",
        }
        player.pin_song("The Mario 64 OST")
        assert "mario 64 ost" in player._pinned


class TestUnpinSong:
    """Test removing a pinned phrase."""

    def test_unpin_existing(self, tmp_path):
        pins = {"mario 64 ost": {"video_id": "abc", "title": "T", "artist": "A"}}
        player = _make_player(tmp_path, pins=pins)
        result = player.unpin_song("mario 64 ost")
        assert "Unpinned" in result
        assert "mario 64 ost" not in player._pinned

    def test_unpin_nonexistent(self, tmp_path):
        player = _make_player(tmp_path)
        result = player.unpin_song("nothing here")
        assert "No pin found" in result


class TestPlaySearchWithPins:
    """Test that play_search checks pins before searching."""

    def test_play_search_uses_pin(self, tmp_path):
        pins = {
            "mario 64 ost": {
                "video_id": "pinned123",
                "title": "SM64 OST",
                "artist": "Koji Kondo",
            }
        }
        player = _make_player(tmp_path, pins=pins)
        with patch.object(player, "play", return_value="Now playing: SM64 OST by Koji Kondo") as mock_play:
            result = player.play_search("mario 64 ost")
            mock_play.assert_called_once_with("pinned123", "SM64 OST", "Koji Kondo")
            assert "SM64 OST" in result

    def test_play_search_uses_pin_with_the_prefix(self, tmp_path):
        pins = {
            "mario 64 ost": {
                "video_id": "pinned123",
                "title": "SM64 OST",
                "artist": "Koji Kondo",
            }
        }
        player = _make_player(tmp_path, pins=pins)
        with patch.object(player, "play", return_value="Now playing: SM64 OST") as mock_play:
            result = player.play_search("the mario 64 ost")
            mock_play.assert_called_once_with("pinned123", "SM64 OST", "Koji Kondo")

    def test_play_search_falls_through(self, tmp_path):
        player = _make_player(tmp_path)
        with patch.object(player, "search", return_value=[
            {"title": "Result", "artist": "Artist", "video_id": "v1", "album": "", "duration": ""}
        ]):
            with patch.object(player, "play", return_value="Now playing: Result by Artist") as mock_play:
                result = player.play_search("unknown query")
                mock_play.assert_called_once_with("v1", "Result", "Artist")
