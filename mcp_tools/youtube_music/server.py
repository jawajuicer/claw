"""YouTube Music MCP server — search, play, queue, and history via ytmusicapi + mpv."""

from __future__ import annotations

import sys
from pathlib import Path

# Ensure sibling modules (player.py) are importable when launched as subprocess
sys.path.insert(0, str(Path(__file__).resolve().parent))

import json
import yaml
from mcp.server.fastmcp import FastMCP

mcp = FastMCP("YouTubeMusic")

# Resolved lazily on first tool call
_player = None
_config = None

# Path to project root (server.py is at mcp_tools/youtube_music/server.py)
_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
_CONFIG_YAML = _PROJECT_ROOT / "config.yaml"


def _load_config() -> dict:
    """Read youtube_music config from config.yaml."""
    global _config
    if _config is not None:
        return _config
    if _CONFIG_YAML.exists():
        with open(_CONFIG_YAML) as f:
            data = yaml.safe_load(f) or {}
        _config = data.get("youtube_music", {})
    else:
        _config = {}
    return _config


def _is_enabled() -> bool:
    cfg = _load_config()
    return cfg.get("enabled", False)


def _get_player():
    """Lazy-init the MusicPlayer singleton."""
    global _player
    if _player is not None:
        return _player

    from player import MusicPlayer

    cfg = _load_config()
    # Resolve relative paths against project root
    auth_file = cfg.get("auth_file", "data/youtube_music/auth.json")
    if not Path(auth_file).is_absolute():
        auth_file = str(_PROJECT_ROOT / auth_file)
    history_file = cfg.get("history_file", "data/youtube_music/history.json")
    if not Path(history_file).is_absolute():
        history_file = str(_PROJECT_ROOT / history_file)
    pins_file = cfg.get("pins_file", "data/youtube_music/pins.json")
    if not Path(pins_file).is_absolute():
        pins_file = str(_PROJECT_ROOT / pins_file)

    _player = MusicPlayer(
        auth_file=auth_file,
        history_file=history_file,
        default_volume=cfg.get("default_volume", 80),
        auto_radio=cfg.get("auto_radio", True),
        max_history=cfg.get("max_history", 500),
        max_search_results=cfg.get("max_search_results", 5),
        client_id=cfg.get("client_id", ""),
        client_secret=cfg.get("client_secret", ""),
        pins_file=pins_file,
    )
    return _player


_NOT_CONFIGURED = (
    "YouTube Music is not configured. "
    "Enable it in Settings and link a Google account with YouTube Music enabled."
)


@mcp.tool()
def play_song(title: str, artist: str = "") -> str:
    """Play a song on YouTube Music. Searches by title and optional artist, then starts playback.

    Args:
        title: The title of the song to play.
        artist: Optional artist name to narrow the search.
    """
    if not _is_enabled():
        return _NOT_CONFIGURED
    query = f"{title} {artist}".strip() if artist else title
    return _get_player().play_search(query)


@mcp.tool()
def play_playlist(playlist_id: str) -> str:
    """Play all tracks from a YouTube Music playlist.

    Args:
        playlist_id: The YouTube Music playlist ID (e.g. from a playlist URL).
    """
    if not _is_enabled():
        return _NOT_CONFIGURED
    return _get_player().play_playlist(playlist_id)


@mcp.tool()
def search_music(query: str, limit: int = 5) -> str:
    """Search for songs on YouTube Music without playing them.

    Args:
        query: Search query (song name, artist, or lyrics).
        limit: Maximum number of results to return (1-20).
    """
    if not _is_enabled():
        return _NOT_CONFIGURED
    limit = max(1, min(20, limit))
    results = _get_player().search(query, limit=limit)
    if not results:
        return f"No results found for '{query}'"
    lines = [f"Search results for '{query}':"]
    for i, r in enumerate(results, 1):
        dur = f" ({r['duration']})" if r["duration"] else ""
        album = f" — {r['album']}" if r["album"] else ""
        lines.append(f"{i}. {r['title']} by {r['artist']}{album}{dur}")
    return "\n".join(lines)


@mcp.tool()
def pause() -> str:
    """Pause the currently playing song."""
    if not _is_enabled():
        return _NOT_CONFIGURED
    return _get_player().pause()


@mcp.tool()
def resume() -> str:
    """Resume playback of the paused song."""
    if not _is_enabled():
        return _NOT_CONFIGURED
    return _get_player().resume()


@mcp.tool()
def skip() -> str:
    """Skip to the next song in the queue."""
    if not _is_enabled():
        return _NOT_CONFIGURED
    return _get_player().skip()


@mcp.tool()
def stop() -> str:
    """Stop playback entirely and clear the queue."""
    if not _is_enabled():
        return _NOT_CONFIGURED
    return _get_player().stop()


@mcp.tool()
def set_volume(level: int) -> str:
    """Set the playback volume.

    Args:
        level: Volume level from 0 (mute) to 100 (max).
    """
    if not _is_enabled():
        return _NOT_CONFIGURED
    return _get_player().set_volume(level)


@mcp.tool()
def seek(position: float) -> str:
    """Seek to a position in seconds within the currently playing song.

    Args:
        position: The position in seconds to seek to.
    """
    if not _is_enabled():
        return _NOT_CONFIGURED
    return _get_player().seek(position)


@mcp.tool()
def get_status() -> str:
    """Get current playback status including position, duration, and track info. Returns JSON."""
    if not _is_enabled():
        return json.dumps({"playing": False, "error": "not_configured"})
    return json.dumps(_get_player().get_status())


@mcp.tool()
def now_playing() -> str:
    """Get information about the currently playing song."""
    if not _is_enabled():
        return _NOT_CONFIGURED
    track = _get_player().get_now_playing()
    if track is None:
        return "Nothing is currently playing"
    parts = [f"Now playing: {track['title']} by {track['artist']}"]
    if track.get("album"):
        parts.append(f"Album: {track['album']}")
    if track.get("duration"):
        parts.append(f"Duration: {track['duration']}")
    if track.get("started_at"):
        parts.append(f"Started: {track['started_at']}")
    return "\n".join(parts)


@mcp.tool()
def get_queue(limit: int = 10) -> str:
    """Show upcoming songs in the queue.

    Args:
        limit: Maximum number of upcoming songs to show (1-50).
    """
    if not _is_enabled():
        return _NOT_CONFIGURED
    limit = max(1, min(50, limit))
    queue = _get_player().get_queue(limit)
    if not queue:
        return "Queue is empty"
    lines = ["Upcoming songs:"]
    for i, t in enumerate(queue, 1):
        dur = f" ({t['duration']})" if t.get("duration") else ""
        lines.append(f"{i}. {t['title']} by {t['artist']}{dur}")
    return "\n".join(lines)


@mcp.tool()
def listen_history(limit: int = 20) -> str:
    """Get recent listen history — what songs have been played.

    Args:
        limit: Number of recent songs to show (1-100).
    """
    if not _is_enabled():
        return _NOT_CONFIGURED
    limit = max(1, min(100, limit))
    history = _get_player().get_history(limit)
    if not history:
        return "No listen history yet"
    lines = ["Recent listen history:"]
    for i, h in enumerate(history, 1):
        source_tag = " [radio]" if h.get("source") == "auto_radio" else ""
        lines.append(f"{i}. {h['title']} by {h['artist']}{source_tag} — {h['played_at']}")
    return "\n".join(lines)


@mcp.tool()
def pin_song(phrase: str) -> str:
    """Pin the currently playing song to a phrase for instant playback later.

    When you say "play <phrase>" in the future, it will always play this exact track
    instead of searching. Use this after playing a song to save it.

    Args:
        phrase: The phrase to associate with the current song (e.g. "mario 64 ost").
    """
    if not _is_enabled():
        return _NOT_CONFIGURED
    return _get_player().pin_song(phrase)


@mcp.tool()
def pin_playlist(phrase: str, playlist_id: str, title: str = "") -> str:
    """Pin a YouTube Music playlist to a phrase for instant playback later.

    When you say "play <phrase>" in the future, it will play this entire playlist.

    Args:
        phrase: The phrase to associate with the playlist (e.g. "mario 64 ost").
        playlist_id: The YouTube Music playlist ID.
        title: Optional human-readable name for the playlist.
    """
    if not _is_enabled():
        return _NOT_CONFIGURED
    return _get_player().pin_playlist(phrase, playlist_id, title)


@mcp.tool()
def unpin_song(phrase: str) -> str:
    """Remove a pinned phrase so it goes back to searching normally.

    Args:
        phrase: The pinned phrase to remove.
    """
    if not _is_enabled():
        return _NOT_CONFIGURED
    return _get_player().unpin_song(phrase)


@mcp.tool()
def list_pins() -> str:
    """List all pinned phrases and their associated songs."""
    if not _is_enabled():
        return _NOT_CONFIGURED
    pins = _get_player().get_pins()
    if not pins:
        return "No pinned songs"
    lines = ["Pinned songs:"]
    for phrase, info in pins.items():
        if "playlist_id" in info:
            lines.append(f"  \"{phrase}\" → playlist: {info['title']}")
        else:
            lines.append(f"  \"{phrase}\" → {info['title']} by {info['artist']}")
    return "\n".join(lines)


if __name__ == "__main__":
    mcp.run()
