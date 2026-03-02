"""YouTube Music player engine — mpv wrapper, queue management, and listen history."""

from __future__ import annotations

import json
import logging
import subprocess
import threading
from datetime import datetime, timezone
from pathlib import Path

log = logging.getLogger(__name__)


class MusicPlayer:
    """Manages mpv playback, song queue, radio generation, and listen history."""

    def __init__(
        self,
        auth_file: str,
        history_file: str,
        default_volume: int = 80,
        auto_radio: bool = True,
        max_history: int = 500,
        max_search_results: int = 5,
    ) -> None:
        self._auth_file = auth_file
        self._history_file = Path(history_file)
        self._default_volume = max(0, min(100, default_volume))
        self._auto_radio = auto_radio
        self._max_history = max_history
        self._max_search_results = max_search_results

        self._mpv = None
        self._volume = self._default_volume
        self._current: dict | None = None
        self._queue: list[dict] = []
        self._history: list[dict] = []
        self._yt = None
        self._lock = threading.Lock()
        self._generation = 0  # tracks which play() call is current
        self._has_played = False  # guard against mpv's initial idle state

        self._load_history()

    def _ensure_yt(self):
        """Lazy-init ytmusicapi client. Thread-safe via double-checked locking."""
        if self._yt is not None:
            return self._yt
        with self._lock:
            if self._yt is not None:
                return self._yt
            from ytmusicapi import YTMusic

            auth_path = Path(self._auth_file)
            if auth_path.exists():
                self._yt = YTMusic(str(auth_path))
            else:
                self._yt = YTMusic()
        return self._yt

    def _ensure_mpv(self):
        """Lazy-init mpv player with idle observer for auto-advance. Thread-safe."""
        if self._mpv is not None:
            return self._mpv
        with self._lock:
            if self._mpv is not None:
                return self._mpv
            import mpv

            player = mpv.MPV(
                video=False,
                ytdl=False,
                input_default_bindings=False,
                input_vo_keyboard=False,
            )
            player.volume = self._volume

            self._has_played = False  # guard against initial idle state

            @player.property_observer("idle-active")
            def _on_idle(_name, value):
                """Fired when mpv transitions to idle after finishing a track."""
                if value and self._has_played:
                    try:
                        self._on_track_end()
                    except Exception:
                        log.exception("Error in track-end handler")

            self._mpv = player
        return self._mpv

    def _on_track_end(self):
        """Called when mpv becomes idle. Advances the queue."""
        with self._lock:
            if self._queue:
                next_track = self._queue.pop(0)
                log.info("Auto-advancing to: %s - %s", next_track["title"], next_track["artist"])
                threading.Thread(
                    target=self._play_track, args=(next_track, "auto_radio"), daemon=True
                ).start()
            else:
                self._current = None
                log.info("Queue empty, playback finished")

    def _load_history(self):
        """Load listen history from JSON file."""
        if self._history_file.exists():
            try:
                self._history = json.loads(self._history_file.read_text())
            except (json.JSONDecodeError, OSError):
                log.warning("Failed to load history from %s, starting fresh", self._history_file)
                self._history = []

    def _save_history(self):
        """Persist listen history to JSON file atomically."""
        self._history_file.parent.mkdir(parents=True, exist_ok=True)
        tmp = self._history_file.with_suffix(".tmp")
        try:
            tmp.write_text(json.dumps(self._history, indent=2))
            tmp.replace(self._history_file)  # atomic on POSIX
        except OSError:
            log.exception("Failed to save history to %s", self._history_file)

    def _log_to_history(self, track: dict, source: str = "user_request"):
        """Append track to history and trim to max. Must be called with _lock held."""
        entry = {
            "video_id": track.get("video_id", ""),
            "title": track.get("title", "Unknown"),
            "artist": track.get("artist", "Unknown"),
            "album": track.get("album", ""),
            "duration": track.get("duration", ""),
            "played_at": datetime.now(timezone.utc).isoformat(),
            "source": source,
        }
        self._history.append(entry)
        if len(self._history) > self._max_history:
            self._history = self._history[-self._max_history :]
        self._save_history()

    def search(self, query: str, limit: int | None = None) -> list[dict]:
        """Search YouTube Music, return list of track info dicts."""
        if limit is None:
            limit = self._max_search_results
        yt = self._ensure_yt()
        try:
            results = yt.search(query, filter="songs", limit=limit)
        except Exception:
            log.exception("YouTube Music search failed for: %s", query)
            return []

        tracks = []
        for r in results[:limit]:
            artists = r.get("artists") or []
            first_artist = artists[0] if artists else None
            artist_name = (first_artist.get("name", "Unknown") if isinstance(first_artist, dict)
                           else "Unknown")
            duration = r.get("duration") or ""
            album_data = r.get("album")
            album_name = (album_data.get("name", "") if isinstance(album_data, dict)
                          else "")
            tracks.append({
                "video_id": r.get("videoId", ""),
                "title": r.get("title", "Unknown"),
                "artist": artist_name,
                "album": album_name,
                "duration": duration,
            })
        return tracks

    def _get_stream_url(self, video_id: str) -> str:
        """Use yt-dlp to extract the best audio stream URL."""
        url = f"https://music.youtube.com/watch?v={video_id}"
        try:
            result = subprocess.run(
                ["yt-dlp", "-g", "-f", "bestaudio", url],
                capture_output=True,
                text=True,
                timeout=30,
            )
            if result.returncode != 0:
                raise RuntimeError(f"yt-dlp error: {result.stderr.strip()}")
            return result.stdout.strip()
        except subprocess.TimeoutExpired:
            raise RuntimeError("yt-dlp timed out extracting stream URL")

    def _play_track(self, track: dict, source: str = "user_request"):
        """Internal: stream a track via mpv. Called from play() or auto-advance."""
        video_id = track["video_id"]
        try:
            stream_url = self._get_stream_url(video_id)
        except RuntimeError as e:
            log.error("Failed to get stream URL for %s: %s", video_id, e)
            return

        player = self._ensure_mpv()
        with self._lock:
            self._current = {
                **track,
                "started_at": datetime.now(timezone.utc).isoformat(),
                "source": source,
            }
            self._log_to_history(track, source)

        try:
            self._has_played = True
            player.play(stream_url)
        except Exception:
            log.exception("mpv failed to play stream URL")
            with self._lock:
                self._current = None

    def _generate_radio_queue(self, video_id: str, generation: int):
        """Use ytmusicapi watch playlist to fill the queue with similar tracks."""
        if not self._auto_radio:
            return
        yt = self._ensure_yt()
        try:
            watch = yt.get_watch_playlist(videoId=video_id, radio=True)
            tracks_data = watch.get("tracks", [])
            new_tracks = []
            for t in tracks_data[1:]:  # skip first (current song)
                vid = t.get("videoId")
                if not vid:
                    continue
                artists = t.get("artists") or []
                first_artist = artists[0] if artists else None
                artist = (first_artist.get("name", "Unknown") if isinstance(first_artist, dict)
                          else "Unknown")
                duration = t.get("length") or t.get("duration") or ""
                album_data = t.get("album")
                album_name = (album_data.get("name", "") if isinstance(album_data, dict)
                              else "")
                new_tracks.append({
                    "video_id": vid,
                    "title": t.get("title", "Unknown"),
                    "artist": artist,
                    "album": album_name,
                    "duration": duration,
                })
            with self._lock:
                if self._generation != generation:
                    return  # stale — a newer play() call superseded this one
                self._queue.extend(new_tracks)
            log.info("Radio queue generated: %d tracks", len(new_tracks))
        except Exception:
            log.exception("Failed to generate radio queue for %s", video_id)

    def play(self, video_id: str, title: str, artist: str) -> str:
        """Play a specific track by video ID. Generates radio queue."""
        with self._lock:
            self._queue.clear()
            self._generation += 1
            gen = self._generation
        track = {"video_id": video_id, "title": title, "artist": artist, "album": "", "duration": ""}
        self._play_track(track, source="user_request")
        threading.Thread(
            target=self._generate_radio_queue, args=(video_id, gen), daemon=True
        ).start()
        return f"Now playing: {title} by {artist}"

    def play_search(self, query: str) -> str:
        """Search and play the top result."""
        results = self.search(query, limit=1)
        if not results:
            return f"No results found for '{query}'"
        top = results[0]
        return self.play(top["video_id"], top["title"], top["artist"])

    def pause(self) -> str:
        """Pause playback."""
        with self._lock:
            if self._mpv is None or self._current is None:
                return "Nothing is playing"
            title = self._current["title"]
            artist = self._current["artist"]
        self._mpv.pause = True
        return f"Paused: {title} by {artist}"

    def resume(self) -> str:
        """Resume playback."""
        with self._lock:
            if self._mpv is None or self._current is None:
                return "Nothing to resume"
            title = self._current["title"]
            artist = self._current["artist"]
        self._mpv.pause = False
        return f"Resumed: {title} by {artist}"

    def skip(self) -> str:
        """Skip to the next song in the queue."""
        if self._mpv is None:
            return "Nothing is playing"
        with self._lock:
            if not self._queue:
                # Snapshot title before releasing lock, stop mpv outside lock
                self._has_played = False
                title = self._current["title"] if self._current else "music"
                self._current = None
                empty = True
                next_track = None
            else:
                next_track = self._queue.pop(0)
                title = self._current["title"] if self._current else "Unknown"
                empty = False
        if empty:
            self._mpv.stop()  # outside lock to avoid deadlock with observer
            return "Queue is empty, playback stopped"
        self._play_track(next_track, source="auto_radio")
        return f"Skipped '{title}'. Now playing: {next_track['title']} by {next_track['artist']}"

    def stop(self) -> str:
        """Stop playback and clear queue."""
        if self._mpv is None:
            return "Nothing is playing"
        with self._lock:
            self._queue.clear()
            self._generation += 1  # invalidate any pending radio queue generation
            self._has_played = False  # prevent idle observer from auto-advancing
            title = self._current["title"] if self._current else "music"
            self._current = None
        self._mpv.stop()  # outside lock to avoid deadlock with observer
        return f"Stopped playing {title}"

    def set_volume(self, level: int) -> str:
        """Set volume 0-100."""
        level = max(0, min(100, level))
        self._volume = level
        if self._mpv is not None:
            self._mpv.volume = level
        return f"Volume set to {level}%"

    def get_now_playing(self) -> dict | None:
        """Return current track info or None."""
        return self._current

    def get_queue(self, limit: int = 10) -> list[dict]:
        """Return upcoming tracks."""
        with self._lock:
            return self._queue[:limit]

    def get_history(self, limit: int = 20) -> list[dict]:
        """Return recent listen history, newest first."""
        with self._lock:
            return list(reversed(self._history[-limit:]))

    def shutdown(self):
        """Clean up mpv process."""
        if self._mpv is not None:
            try:
                self._mpv.terminate()
            except Exception:
                pass
            self._mpv = None
        self._current = None
