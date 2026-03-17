"""YouTube Music player engine — mpv wrapper, queue management, and listen history."""

from __future__ import annotations

import json
import logging
import os
import socket
import subprocess
import threading
import time
from datetime import datetime, timezone
from pathlib import Path

log = logging.getLogger(__name__)


class MusicPlayer:
    """Manages mpv playback, song queue, radio generation, and listen history.

    Uses mpv as a separate process with JSON IPC over a Unix socket rather than
    libmpv in-process.  This avoids conflicts with the MCP stdio transport
    (piped stdin/stdout) that cause libmpv to immediately go idle.
    """

    def __init__(
        self,
        auth_file: str,
        history_file: str,
        default_volume: int = 80,
        auto_radio: bool = True,
        max_history: int = 500,
        max_search_results: int = 5,
        client_id: str = "",
        client_secret: str = "",
        pins_file: str = "",
    ) -> None:
        self._auth_file = auth_file
        self._history_file = Path(history_file)
        self._pins_file = Path(pins_file) if pins_file else None
        self._default_volume = max(0, min(100, default_volume))
        self._auto_radio = auto_radio
        self._max_history = max_history
        self._max_search_results = max_search_results
        self._client_id = client_id
        self._client_secret = client_secret

        self._mpv_proc: subprocess.Popen | None = None
        self._ipc_path: str | None = None
        self._volume = self._default_volume
        self._current: dict | None = None
        self._queue: list[dict] = []
        self._history: list[dict] = []
        self._pinned: dict[str, dict] = {}
        self._yt = None
        self._lock = threading.Lock()
        self._generation = 0  # tracks which play() call is current
        self._has_played = False  # guard against mpv's initial idle state

        self._load_history()
        self._load_pins()

    # ── ytmusicapi ──────────────────────────────────────────────

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
                if self._client_id and self._client_secret:
                    from ytmusicapi.auth.oauth.credentials import OAuthCredentials
                    oauth_creds = OAuthCredentials(self._client_id, self._client_secret)
                    self._yt = YTMusic(str(auth_path), oauth_credentials=oauth_creds)
                else:
                    self._yt = YTMusic(str(auth_path))
            else:
                self._yt = YTMusic()
        return self._yt

    # ── mpv process + IPC ───────────────────────────────────────

    def _ensure_mpv(self):
        """Launch mpv as a separate process with JSON IPC. Thread-safe."""
        if self._mpv_proc is not None and self._mpv_proc.poll() is None:
            return
        with self._lock:
            if self._mpv_proc is not None and self._mpv_proc.poll() is None:
                return

            self._ipc_path = f"/tmp/claw-mpv-{os.getpid()}.sock"
            # Clean up stale socket
            if os.path.exists(self._ipc_path):
                os.unlink(self._ipc_path)

            # Explicitly pass env vars needed for PipeWire/PulseAudio audio output.
            # The MCP subprocess may not always inherit the full environment.
            env = os.environ.copy()
            env.setdefault("XDG_RUNTIME_DIR", f"/run/user/{os.getuid()}")
            env.setdefault("HOME", str(Path.home()))

            self._mpv_proc = subprocess.Popen(
                [
                    "mpv",
                    "--idle=yes",
                    "--no-video",
                    "--no-terminal",
                    "--ao=pipewire,pulse,alsa",
                    f"--volume={self._volume}",
                    f"--input-ipc-server={self._ipc_path}",
                ],
                stdin=subprocess.DEVNULL,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                env=env,
                start_new_session=True,
            )

            # Wait for IPC socket to appear
            for _ in range(50):
                if os.path.exists(self._ipc_path):
                    break
                time.sleep(0.1)
            else:
                log.error("mpv IPC socket did not appear at %s", self._ipc_path)
                return

            self._has_played = False
            self._start_idle_watcher()
            log.info("mpv process started (pid=%d, ipc=%s)", self._mpv_proc.pid, self._ipc_path)

    def _mpv_command(self, *args):
        """Send a command to mpv via IPC and return the parsed response.

        Skips interleaved event messages (mpv sends events like start-file,
        end-file to all connected clients).
        """
        if self._ipc_path is None:
            return None
        sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        try:
            sock.settimeout(5)
            sock.connect(self._ipc_path)
            cmd = json.dumps({"command": list(args)}) + "\n"
            sock.sendall(cmd.encode())
            buf = b""
            while True:
                while b"\n" not in buf:
                    chunk = sock.recv(4096)
                    if not chunk:
                        return None
                    buf += chunk
                line, buf = buf.split(b"\n", 1)
                if not line.strip():
                    continue
                msg = json.loads(line)
                # Skip event messages — only return command responses
                if "event" not in msg:
                    return msg
        except (OSError, json.JSONDecodeError) as e:
            log.debug("mpv IPC command %s failed: %s", args, e)
            return None
        finally:
            sock.close()

    def _mpv_set_property(self, name: str, value):
        return self._mpv_command("set_property", name, value)

    def _mpv_get_property(self, name: str):
        resp = self._mpv_command("get_property", name)
        if resp and resp.get("error") == "success":
            return resp.get("data")
        return None

    def _start_idle_watcher(self):
        """Background thread that watches mpv idle-active for auto-advance."""
        def watcher():
            with self._lock:
                ipc_path = self._ipc_path
            if not ipc_path:
                return
            sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
            try:
                sock.settimeout(None)
                sock.connect(ipc_path)
                # Register property observer
                sock.sendall(
                    json.dumps({"command": ["observe_property", 1, "idle-active"]}).encode() + b"\n"
                )
                buf = b""
                prev_idle = None  # Track previous state to detect transitions
                while self._mpv_proc and self._mpv_proc.poll() is None:
                    chunk = sock.recv(4096)
                    if not chunk:
                        break
                    buf += chunk
                    while b"\n" in buf:
                        line, buf = buf.split(b"\n", 1)
                        if not line.strip():
                            continue
                        try:
                            msg = json.loads(line)
                        except json.JSONDecodeError:
                            continue
                        if (
                            msg.get("event") == "property-change"
                            and msg.get("name") == "idle-active"
                        ):
                            is_idle = msg.get("data")
                            # Only trigger on transition from playing (False) to idle (True).
                            # This avoids firing on mpv's initial idle state.
                            if is_idle and prev_idle is False and self._has_played:
                                try:
                                    self._on_track_end()
                                except Exception:
                                    log.exception("Error in track-end handler")
                            prev_idle = is_idle
            except Exception:
                log.debug("Idle watcher thread ended")
            finally:
                sock.close()

        t = threading.Thread(target=watcher, daemon=True)
        t.start()

    # ── Track-end / auto-advance ────────────────────────────────

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

    # ── History ─────────────────────────────────────────────────

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

    # ── Pinned songs ─────────────────────────────────────────────

    @staticmethod
    def _normalize_phrase(phrase: str) -> str:
        """Normalize a pin phrase: lowercase, strip, remove leading 'the '."""
        phrase = phrase.lower().strip()
        if phrase.startswith("the "):
            phrase = phrase[4:]
        return phrase

    def _load_pins(self):
        """Load pinned songs from JSON file."""
        if self._pins_file and self._pins_file.exists():
            try:
                self._pinned = json.loads(self._pins_file.read_text())
            except (json.JSONDecodeError, OSError):
                log.warning("Failed to load pins from %s, starting fresh", self._pins_file)
                self._pinned = {}

    def _save_pins(self):
        """Persist pinned songs to JSON file atomically."""
        if self._pins_file is None:
            return
        self._pins_file.parent.mkdir(parents=True, exist_ok=True)
        tmp = self._pins_file.with_suffix(".tmp")
        try:
            tmp.write_text(json.dumps(self._pinned, indent=2))
            tmp.replace(self._pins_file)
        except OSError:
            log.exception("Failed to save pins to %s", self._pins_file)

    def pin_song(self, phrase: str) -> str:
        """Pin the currently playing track to a phrase."""
        with self._lock:
            if self._current is None:
                return "Nothing is playing — play a song first, then pin it."
            track = self._current
        normalized = self._normalize_phrase(phrase)
        if not normalized:
            return "Please provide a phrase to pin this song to."
        self._pinned[normalized] = {
            "video_id": track.get("video_id", ""),
            "title": track.get("title", "Unknown"),
            "artist": track.get("artist", "Unknown"),
        }
        self._save_pins()
        return f"Pinned '{track['title']}' by {track['artist']} as \"{normalized}\""

    def pin_playlist(self, phrase: str, playlist_id: str, title: str = "") -> str:
        """Pin a playlist to a phrase."""
        normalized = self._normalize_phrase(phrase)
        if not normalized:
            return "Please provide a phrase to pin this playlist to."
        self._pinned[normalized] = {
            "playlist_id": playlist_id,
            "title": title or playlist_id,
        }
        self._save_pins()
        return f"Pinned playlist '{title or playlist_id}' as \"{normalized}\""

    def pin_song_by_id(self, phrase: str, video_id: str, title: str, artist: str) -> str:
        """Pin a specific track to a phrase without needing playback."""
        normalized = self._normalize_phrase(phrase)
        if not normalized:
            return "Please provide a phrase to pin this song to."
        self._pinned[normalized] = {
            "video_id": video_id,
            "title": title,
            "artist": artist,
        }
        self._save_pins()
        return f"Pinned '{title}' by {artist} as \"{normalized}\""

    def unpin_song(self, phrase: str) -> str:
        """Remove a pinned phrase."""
        normalized = self._normalize_phrase(phrase)
        if normalized not in self._pinned:
            return f"No pin found for \"{normalized}\""
        removed = self._pinned.pop(normalized)
        self._save_pins()
        return f"Unpinned \"{normalized}\" (was: {removed.get('title', '?')} by {removed.get('artist', 'unknown')})"

    def get_pins(self) -> dict:
        """Return all pinned phrases."""
        return dict(self._pinned)

    # ── Search / stream ─────────────────────────────────────────

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

    # ── Playback controls ───────────────────────────────────────

    def _play_track(self, track: dict, source: str = "user_request"):
        """Internal: stream a track via mpv. Called from play() or auto-advance."""
        video_id = track["video_id"]
        try:
            stream_url = self._get_stream_url(video_id)
        except RuntimeError as e:
            log.error("Failed to get stream URL for %s: %s", video_id, e)
            return

        self._ensure_mpv()
        with self._lock:
            self._current = {
                **track,
                "started_at": datetime.now(timezone.utc).isoformat(),
                "source": source,
            }
            self._log_to_history(track, source)

        try:
            self._has_played = True
            self._mpv_command("loadfile", stream_url)
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
        return f"Now playing: {title} by {artist} [video:{video_id}]"

    def play_playlist(self, playlist_id: str) -> str:
        """Play all tracks from a YouTube Music playlist."""
        yt = self._ensure_yt()
        try:
            playlist = yt.get_playlist(playlist_id, limit=200)
        except Exception:
            log.exception("Failed to fetch playlist %s", playlist_id)
            return f"Failed to load playlist '{playlist_id}'"

        playlist_title = playlist.get("title", playlist_id)
        raw_tracks = playlist.get("tracks", [])
        if not raw_tracks:
            return f"Playlist '{playlist_title}' is empty or unavailable"

        tracks = []
        for t in raw_tracks:
            vid = t.get("videoId")
            if not vid:
                continue
            artists = t.get("artists") or []
            first_artist = artists[0] if artists else None
            artist = (first_artist.get("name", "Unknown") if isinstance(first_artist, dict)
                      else "Unknown")
            duration = t.get("duration") or ""
            album_data = t.get("album")
            album_name = (album_data.get("name", "") if isinstance(album_data, dict)
                          else "")
            tracks.append({
                "video_id": vid,
                "title": t.get("title", "Unknown"),
                "artist": artist,
                "album": album_name,
                "duration": duration,
            })

        if not tracks:
            return f"No playable tracks in playlist '{playlist_title}'"

        first = tracks[0]
        with self._lock:
            self._queue.clear()
            self._generation += 1
            self._queue.extend(tracks[1:])

        self._play_track(first, source="user_request")
        return (
            f"Now playing playlist '{playlist_title}' ({len(tracks)} tracks). "
            f"Starting with: {first['title']} by {first['artist']}"
        )

    @staticmethod
    def _rank_results(results: list[dict], query: str) -> list[dict]:
        """Rank search results by relevance to the query.

        Scoring: title term match = 2pts, artist term match = 1pt,
        exact title substring = 3pt bonus.
        """
        query_lower = query.lower()
        terms = query_lower.split()

        scored = []
        for r in results:
            score = 0.0
            title = r.get("title", "").lower()
            artist = r.get("artist", "").lower()
            for term in terms:
                if term in title:
                    score += 2.0
                if term in artist:
                    score += 1.0
            if query_lower in title:
                score += 3.0
            scored.append((score, r))

        scored.sort(key=lambda x: x[0], reverse=True)
        return [r for _, r in scored]

    def play_search(self, query: str) -> str:
        """Search and play the best-matching result. Checks pinned songs first."""
        normalized = self._normalize_phrase(query)
        pin = self._pinned.get(normalized)
        if pin:
            if "playlist_id" in pin:
                log.info("Playing pinned playlist for '%s': %s", normalized, pin.get("title", pin["playlist_id"]))
                return self.play_playlist(pin["playlist_id"])
            log.info("Playing pinned track for '%s': %s", normalized, pin["title"])
            return self.play(pin["video_id"], pin["title"], pin["artist"])
        results = self.search(query)
        if not results:
            return f"No results found for '{query}'"
        ranked = self._rank_results(results, query)
        top = ranked[0]
        return self.play(top["video_id"], top["title"], top["artist"])

    def pause(self) -> str:
        """Pause playback."""
        with self._lock:
            if self._mpv_proc is None or self._current is None:
                return "Nothing is playing"
            title = self._current["title"]
            artist = self._current["artist"]
        self._mpv_set_property("pause", True)
        return f"Paused: {title} by {artist}"

    def resume(self) -> str:
        """Resume playback."""
        with self._lock:
            if self._mpv_proc is None or self._current is None:
                return "Nothing to resume"
            title = self._current["title"]
            artist = self._current["artist"]
        self._mpv_set_property("pause", False)
        return f"Resumed: {title} by {artist}"

    def skip(self) -> str:
        """Skip to the next song in the queue."""
        if self._mpv_proc is None:
            return "Nothing is playing"
        with self._lock:
            if not self._queue:
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
            self._mpv_command("stop")
            return "Queue is empty, playback stopped"
        self._play_track(next_track, source="auto_radio")
        return f"Skipped '{title}'. Now playing: {next_track['title']} by {next_track['artist']} [video:{next_track['video_id']}]"

    def stop(self) -> str:
        """Stop playback and clear queue."""
        if self._mpv_proc is None:
            return "Nothing is playing"
        with self._lock:
            self._queue.clear()
            self._generation += 1
            self._has_played = False
            title = self._current["title"] if self._current else "music"
            self._current = None
        self._mpv_command("stop")
        return f"Stopped playing {title}"

    def set_volume(self, level: int) -> str:
        """Set volume 0-100."""
        level = max(0, min(100, level))
        self._volume = level
        if self._mpv_proc is not None and self._mpv_proc.poll() is None:
            self._mpv_set_property("volume", float(level))
        return f"Volume set to {level}%"

    def seek(self, position: float) -> str:
        """Seek to an absolute position in seconds."""
        if self._mpv_proc is None or self._mpv_proc.poll() is not None or self._current is None:
            return "Nothing is playing"
        position = max(0.0, position)
        self._mpv_command("seek", position, "absolute")
        return f"Seeked to {int(position)}s"

    def get_status(self) -> dict:
        """Return current playback status for the now-playing widget."""
        if self._mpv_proc is None or self._mpv_proc.poll() is not None:
            return {"playing": False}

        idle = self._mpv_get_property("idle-active")
        if idle is True or idle is None:
            return {"playing": False}

        paused = self._mpv_get_property("pause")
        time_pos = self._mpv_get_property("time-pos")
        duration = self._mpv_get_property("duration")
        if paused is None or time_pos is None:
            return {"playing": False}
        duration = float(duration) if duration is not None else 0.0

        with self._lock:
            current = self._current
        if current is None:
            return {"playing": False}

        return {
            "playing": True,
            "paused": paused,
            "title": current.get("title", "Unknown"),
            "artist": current.get("artist", "Unknown"),
            "album": current.get("album", ""),
            "time_pos": round(time_pos, 1),
            "duration": round(duration, 1),
            "volume": self._volume,
        }

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
        """Clean up mpv process and IPC socket."""
        if self._mpv_proc is not None:
            try:
                self._mpv_proc.terminate()
                self._mpv_proc.wait(timeout=5)
            except Exception:
                try:
                    self._mpv_proc.kill()
                except Exception:
                    pass
            self._mpv_proc = None
        if self._ipc_path and os.path.exists(self._ipc_path):
            try:
                os.unlink(self._ipc_path)
            except OSError:
                pass
        self._current = None
