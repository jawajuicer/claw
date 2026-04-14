"""Remote API for accessing The Claw from outside the local network.

Provides REST endpoints and a WebSocket for voice interaction, all
authenticated via per-device API keys (X-API-Key header). Transport
encryption is handled by WireGuard — this layer handles identity.

Endpoints:
    GET  /api/remote/ping           — Health check
    POST /api/remote/chat           — Text chat
    POST /api/remote/tts            — Text-to-speech (returns WAV)
    GET  /api/remote/status         — System status
    GET  /api/remote/events         — SSE real-time updates
    GET  /api/remote/stream/{id}    — YouTube audio proxy
    WS   /api/remote/audio          — Voice interaction

WebSocket protocol:
    Connect with ?key=<API_KEY>.
    First message (JSON): {"mode": "push_to_talk"|"client_wake"|"server_wake"}
    Binary frames: 16kHz 16-bit signed LE mono PCM (80ms = 2560 bytes).
    Text frames: JSON control messages.
"""

from __future__ import annotations

import asyncio
import json
import logging
import re

import numpy as np
from fastapi import APIRouter, Request, WebSocket, WebSocketDisconnect
from fastapi.responses import JSONResponse, StreamingResponse
from starlette.responses import Response

log = logging.getLogger(__name__)

remote_router = APIRouter(prefix="/api/remote", tags=["remote"])

_VIDEO_ID_RE = re.compile(r"^[a-zA-Z0-9_-]{11}$")
_WAKE_PREFIX_RE = re.compile(r"^(?:hey\s+claw|claw)[,;:.\s!]*", re.IGNORECASE)
_VIDEO_TAG_RE = re.compile(r"\[video:([a-zA-Z0-9_-]{11})\]")


# ── Health check ────────────────────────────────────────────────────────

@remote_router.get("/ping")
async def ping():
    return {"status": "ok", "service": "claw"}


@remote_router.get("/health")
async def remote_health(request: Request):
    """Authenticated health check with subsystem detail for Android app."""
    import httpx
    from claw.config import get_settings

    settings = get_settings()
    checks: dict[str, dict] = {}

    # LLM
    try:
        async with httpx.AsyncClient(timeout=5.0) as client:
            resp = await client.get(f"{settings.llm.base_url.rstrip('/')}/models")
            checks["llm"] = {"status": "ok", "model": settings.llm.model}
    except Exception as exc:
        checks["llm"] = {"status": "error", "detail": str(exc)}

    # MCP
    registry = request.app.state.registry
    if registry:
        servers = registry.list_servers()
        checks["mcp"] = {"status": "ok", "servers": list(servers.keys()) if isinstance(servers, dict) else []}
    else:
        checks["mcp"] = {"status": "warning"}

    # TTS
    tts = getattr(request.app.state, "tts", None)
    checks["tts"] = {"status": "ok" if tts else "unavailable"}

    statuses = [c["status"] for c in checks.values()]
    overall = "healthy" if all(s == "ok" for s in statuses) else "degraded" if "error" not in statuses else "unhealthy"
    code = 200 if overall == "healthy" else 503
    return JSONResponse({"status": overall, "checks": checks}, status_code=code)


# ── Text chat ───────────────────────────────────────────────────────────

@remote_router.post("/chat")
async def remote_chat(request: Request):
    """Process a text message through the agent."""
    body = await request.json()
    message = body.get("message", "").strip()
    image_b64 = body.get("image")

    # Prepare image for LLM if provided
    images = None
    if image_b64:
        import base64 as b64mod
        from claw.agent_core.image_utils import prepare_images_for_llm
        try:
            raw = b64mod.b64decode(image_b64)
        except Exception:
            return JSONResponse({"error": "Invalid base64 image data"}, status_code=400)
        images = prepare_images_for_llm([raw]) or None

    if not message and not images:
        return JSONResponse({"error": "Empty message"}, status_code=400)

    agent = request.app.state.agent
    registry = request.app.state.registry
    broadcaster = request.app.state.broadcaster
    device = getattr(request.state, "device_name", "unknown")

    if agent is None:
        return JSONResponse({"error": "Agent not available"}, status_code=503)

    chat_lock: asyncio.Lock = request.app.state.chat_lock

    async with chat_lock:
        msg_count_before = len(agent.session.messages) if agent.session else 0
        await broadcaster.update_state("processing")
        await broadcaster.update_transcription(message)

        tools = registry.get_openai_tools() if registry else None
        try:
            response = await agent.process_utterance(
                message, tools=tools or None, interactive=True, images=images,
            )
            tools_used = _extract_tools(agent, msg_count_before)
            await broadcaster.update_response(response)
        except Exception:
            log.exception("Remote chat failed (device: %s)", device)
            response = "Sorry, something went wrong processing your message."
            tools_used = []
        finally:
            await broadcaster.update_state("idle")

    # Background fact extraction
    task = asyncio.create_task(_extract_facts(agent))
    request.app.state.bg_tasks.add(task)
    task.add_done_callback(request.app.state.bg_tasks.discard)

    from claw.config import get_settings
    settings = get_settings()
    usage = agent.last_usage.to_dict() if agent.last_usage else {}
    usage["context_window"] = settings.llm.context_window

    audio_output = settings.remote.audio_output

    result = {
        "role": "assistant",
        "content": response,
        "tools_used": sorted(tools_used),
        "usage": usage,
        "device": device,
        "claude_mode": agent.claude_code_active,
    }

    # Extract music info — route based on audio_output setting
    music = _extract_music_from_session(agent, msg_count_before)
    if music:
        if audio_output == "phone":
            result["music"] = music
            # Stop local mpv — phone will stream
            tool_router = getattr(request.app.state, "tool_router", None)
            if tool_router:
                try:
                    await tool_router.call_tool("stop", {})
                except Exception:
                    log.debug("Could not stop local mpv (may not be running)")
        # else "computer": mpv plays locally, no stream URL sent to phone

    return result


# ── TTS ──────────────────────────────────────────────────────────────────

@remote_router.post("/tts")
async def remote_tts(request: Request):
    """Synthesize text to speech. Returns WAV audio bytes or plays on computer."""
    body = await request.json()
    text = body.get("text", "").strip()
    if not text:
        return JSONResponse({"error": "Empty text"}, status_code=400)

    tts = getattr(request.app.state, "tts", None)
    if tts is None:
        return JSONResponse({"error": "TTS not available"}, status_code=503)

    from claw.config import get_settings

    if get_settings().remote.audio_output == "computer":
        # Play on computer speakers instead of returning audio
        try:
            await tts.speak(text)
            return JSONResponse({"status": "played_locally"})
        except Exception:
            log.exception("Local TTS playback failed")
            return JSONResponse({"error": "TTS playback failed"}, status_code=500)

    try:
        wav_bytes = await tts.synthesize_wav(text)
        if not wav_bytes:
            return JSONResponse({"error": "TTS produced no audio"}, status_code=500)
        return Response(content=wav_bytes, media_type="audio/wav")
    except Exception:
        log.exception("Remote TTS failed")
        return JSONResponse({"error": "TTS synthesis failed"}, status_code=500)


# ── Status & Events ─────────────────────────────────────────────────────

@remote_router.get("/status")
async def remote_status(request: Request):
    broadcaster = request.app.state.broadcaster
    return broadcaster.get_status()


@remote_router.get("/events")
async def remote_events(request: Request):
    """SSE endpoint for real-time status updates."""
    from sse_starlette.sse import EventSourceResponse

    broadcaster = request.app.state.broadcaster
    queue = broadcaster.subscribe()

    async def generate():
        try:
            async for data in broadcaster.event_generator(queue):
                parsed = json.loads(data)
                event_name = parsed.get("event", "message")
                yield {"event": event_name, "data": data}
        finally:
            broadcaster.unsubscribe(queue)

    return EventSourceResponse(generate())


# ── YouTube Audio Proxy ──────────────────────────────────────────────────

@remote_router.get("/stream/{video_id}")
async def stream_youtube(video_id: str, request: Request):
    """Proxy YouTube audio so the client never contacts Google directly.

    Server resolves the stream URL via yt-dlp and pipes the audio bytes
    through to the client. Supports Range requests for seeking.
    """
    if not _VIDEO_ID_RE.match(video_id):
        return JSONResponse({"error": "Invalid video ID"}, status_code=400)

    import subprocess

    try:
        result = await asyncio.to_thread(
            subprocess.run,
            [
                "yt-dlp",
                "-f", "bestaudio[acodec=opus]/bestaudio",
                "--get-url",
                f"https://www.youtube.com/watch?v={video_id}",
            ],
            capture_output=True,
            text=True,
            timeout=15,
        )
    except subprocess.TimeoutExpired:
        return JSONResponse({"error": "Stream resolution timed out"}, status_code=504)

    if result.returncode != 0:
        log.warning("yt-dlp failed for %s: %s", video_id, result.stderr[:200])
        return JSONResponse({"error": "Failed to resolve stream"}, status_code=502)

    stream_url = result.stdout.strip().split("\n")[0]
    if not stream_url:
        return JSONResponse({"error": "No stream URL returned"}, status_code=502)

    # Proxy the upstream audio stream
    import httpx

    proxy_headers: dict[str, str] = {}
    range_header = request.headers.get("range")
    if range_header:
        proxy_headers["Range"] = range_header

    client = httpx.AsyncClient(timeout=httpx.Timeout(30.0, connect=10.0))
    try:
        upstream = await client.send(
            client.build_request("GET", stream_url, headers=proxy_headers),
            stream=True,
        )
    except httpx.HTTPError as exc:
        await client.aclose()
        log.warning("Upstream stream fetch failed: %s", exc)
        return JSONResponse({"error": "Stream fetch failed"}, status_code=502)

    response_headers: dict[str, str] = {"Accept-Ranges": "bytes"}
    for hdr in ("content-type", "content-length", "content-range"):
        val = upstream.headers.get(hdr)
        if val:
            response_headers[hdr.title()] = val

    async def proxy_generator():
        try:
            async for chunk in upstream.aiter_bytes(chunk_size=65536):
                yield chunk
        finally:
            await upstream.aclose()
            await client.aclose()

    return StreamingResponse(
        proxy_generator(),
        status_code=upstream.status_code,
        headers=response_headers,
    )


# ── YouTube Search (no playback) ────────────────────────────────────────

@remote_router.get("/music/search")
async def music_search(request: Request, q: str = "", limit: int = 5):
    """Search YouTube Music and return results with stream URLs.

    Does NOT start playback — the client can stream via /stream/{video_id}.
    """
    if not q.strip():
        return JSONResponse({"error": "Empty query"}, status_code=400)

    router = getattr(request.app.state, "tool_router", None)
    if router is None:
        return JSONResponse({"error": "MCP not available"}, status_code=503)

    try:
        result_str = await router.call_tool("search_music", {"query": q, "limit": limit})
    except Exception:
        log.exception("Music search failed")
        return JSONResponse({"error": "Search failed"}, status_code=502)

    return {"query": q, "raw_results": result_str}


# ── App Updates ──────────────────────────────────────────────────────────

@remote_router.get("/app/version")
async def app_version():
    """Return the latest available APK version."""
    from claw.config import PROJECT_ROOT

    version_file = PROJECT_ROOT / "data" / "remote" / "app" / "version.json"
    if not version_file.exists():
        return JSONResponse({"error": "No app version published"}, status_code=404)

    import json as _json
    try:
        info = _json.loads(version_file.read_text())
        return {
            "version_code": info.get("version_code", 0),
            "version_name": info.get("version_name", "unknown"),
            "size": info.get("size", 0),
            "changelog": info.get("changelog", ""),
        }
    except Exception:
        return JSONResponse({"error": "Invalid version info"}, status_code=500)


@remote_router.get("/app/download")
async def app_download():
    """Download the latest APK."""
    from claw.config import PROJECT_ROOT

    apk_dir = PROJECT_ROOT / "data" / "remote" / "app"
    apk_files = sorted(apk_dir.glob("*.apk"), key=lambda p: p.stat().st_mtime, reverse=True)
    if not apk_files:
        return JSONResponse({"error": "No APK available"}, status_code=404)

    apk_path = apk_files[0]
    from starlette.responses import FileResponse
    return FileResponse(
        path=str(apk_path),
        media_type="application/vnd.android.package-archive",
        filename="claw.apk",
    )


# ── WebSocket Audio ──────────────────────────────────────────────────────

@remote_router.websocket("/audio")
async def audio_websocket(ws: WebSocket):
    """Bidirectional audio streaming for voice interaction.

    Auth: Connect with ?key=<API_KEY> query parameter.

    Protocol after accept:
    1. Client sends JSON config: {"mode": "push_to_talk"|"client_wake"|"server_wake"}
    2. Based on mode:
       push_to_talk / client_wake:
         Client sends {"type":"start"}, binary PCM frames, {"type":"stop"}
       server_wake:
         Client streams continuous binary PCM, server detects wake word
    3. Server responds with JSON events and binary TTS audio.

    Audio format: 16kHz, 16-bit signed LE, mono PCM.
    Recommended chunk: 2560 bytes (1280 samples = 80ms).
    """
    # Auth via query param (WebSocket JS API doesn't support custom headers)
    api_key = ws.query_params.get("key", "")
    if not api_key:
        await ws.close(code=4001, reason="API key required")
        return

    from claw.config import get_settings

    if not get_settings().remote.enabled:
        await ws.close(code=4003, reason="Remote access disabled")
        return

    from claw.admin.api_key import verify_key

    device = verify_key(api_key)
    if device is None:
        await ws.close(code=4001, reason="Invalid API key")
        return

    await ws.accept()
    log.info("Remote audio session opened (device: %s)", device)

    try:
        # Wait for config message
        config_data = await asyncio.wait_for(ws.receive_json(), timeout=10)
        mode = config_data.get("mode", "push_to_talk")

        if mode not in ("push_to_talk", "client_wake", "server_wake"):
            await ws.send_json({"type": "error", "message": f"Unknown mode: {mode}"})
            return

        await ws.send_json({"type": "connected", "device": device, "mode": mode})

        if mode == "server_wake":
            await _handle_server_wake_mode(ws, device)
        else:
            await _handle_client_initiated_mode(ws, device)

    except WebSocketDisconnect:
        log.info("Remote audio session disconnected (device: %s)", device)
    except asyncio.TimeoutError:
        log.warning("Config timeout (device: %s)", device)
        try:
            await ws.send_json({"type": "error", "message": "Config timeout"})
        except Exception:
            pass
    except Exception:
        log.exception("Remote audio session error (device: %s)", device)
        try:
            await ws.send_json({"type": "error", "message": "Internal error"})
        except Exception:
            pass
    finally:
        log.info("Remote audio session closed (device: %s)", device)


# ── WebSocket mode handlers ─────────────────────────────────────────────

async def _handle_server_wake_mode(ws: WebSocket, device: str) -> None:
    """Server-side wake word detection.

    Client streams continuous audio. Server runs wake word detection,
    records until silence after detection, then processes and responds.
    """
    from claw.audio_pipeline.wake_word import WakeWordDetector
    from claw.config import get_settings

    settings = get_settings()

    # Each connection needs its own detector (internal state per stream)
    wake = WakeWordDetector()
    wake.load()

    sample_rate = settings.audio.sample_rate
    chunk_samples = settings.audio.block_size
    chunks_per_sec = sample_rate / chunk_samples
    silence_limit = int(settings.audio.silence_duration * chunks_per_sec)
    max_chunks = int(settings.audio.max_record_seconds * chunks_per_sec)
    silence_threshold = settings.audio.agc_silence_threshold

    recording = False
    audio_buffer: list[np.ndarray] = []
    silence_count = 0

    while True:
        message = await ws.receive()

        if message["type"] == "websocket.disconnect":
            break

        if "bytes" in message and message["bytes"]:
            pcm_bytes = message["bytes"]
            audio_int16 = np.frombuffer(pcm_bytes, dtype=np.int16)
            audio_float = audio_int16.astype(np.float32) / 32768.0

            if not recording:
                result = wake.process_chunk(audio_int16)
                if result:
                    recording = True
                    audio_buffer = []
                    silence_count = 0
                    log.info("Wake word detected remotely: %s (device: %s)", result, device)
                    await ws.send_json({"type": "wake_word", "word": result})
                    await ws.send_json({"type": "listening"})
            else:
                audio_buffer.append(audio_float)
                rms = float(np.sqrt(np.mean(audio_float ** 2)))
                if rms < silence_threshold:
                    silence_count += 1
                else:
                    silence_count = 0

                if silence_count >= silence_limit or len(audio_buffer) >= max_chunks:
                    recording = False
                    if len(audio_buffer) < int(0.5 * chunks_per_sec):
                        await ws.send_json({"type": "error", "message": "Recording too short"})
                        continue
                    full_audio = np.concatenate(audio_buffer)
                    audio_buffer = []
                    await _process_audio_and_respond(ws, full_audio, device)

        elif "text" in message and message["text"]:
            try:
                data = json.loads(message["text"])
                if data.get("type") == "ping":
                    await ws.send_json({"type": "pong"})
            except json.JSONDecodeError:
                pass


async def _handle_client_initiated_mode(ws: WebSocket, device: str) -> None:
    """Client-initiated recording (push_to_talk or client_wake).

    Client sends {"type":"start"}, streams binary PCM, then {"type":"stop"}.
    Server accumulates audio, processes, and responds.
    """
    recording = False
    audio_buffer: list[np.ndarray] = []

    while True:
        message = await ws.receive()

        if message["type"] == "websocket.disconnect":
            break

        if "bytes" in message and message["bytes"] and recording:
            audio_int16 = np.frombuffer(message["bytes"], dtype=np.int16)
            audio_float = audio_int16.astype(np.float32) / 32768.0
            audio_buffer.append(audio_float)

        elif "text" in message and message["text"]:
            try:
                data = json.loads(message["text"])
            except json.JSONDecodeError:
                continue

            msg_type = data.get("type")

            if msg_type in ("start", "wake_word"):
                recording = True
                audio_buffer = []
                log.info("Remote recording started (device: %s, trigger: %s)", device, msg_type)
                await ws.send_json({"type": "listening"})

            elif msg_type == "stop" and recording:
                recording = False
                log.info("Remote recording stopped (device: %s, chunks: %d)", device, len(audio_buffer))

                if not audio_buffer:
                    await ws.send_json({"type": "error", "message": "No audio received"})
                    continue

                full_audio = np.concatenate(audio_buffer)
                audio_buffer = []

                from claw.config import get_settings
                min_samples = int(get_settings().audio.sample_rate * 0.5)
                if len(full_audio) < min_samples:
                    await ws.send_json({"type": "error", "message": "Recording too short"})
                    continue

                await _process_audio_and_respond(ws, full_audio, device)

            elif msg_type == "chat":
                text = data.get("text", "").strip()
                if text:
                    await _process_text_and_respond(ws, text, device)

            elif msg_type == "ping":
                await ws.send_json({"type": "pong"})

            elif msg_type == "new_session":
                agent = ws.app.state.agent
                if agent:
                    agent.new_session()
                    await ws.send_json({"type": "session_reset"})


# ── Audio + text processing ─────────────────────────────────────────────

async def _process_audio_and_respond(
    ws: WebSocket, audio: np.ndarray, device: str,
) -> None:
    """Transcribe audio, run agent, send response + TTS."""
    transcriber = ws.app.state.remote_transcriber
    if transcriber is None:
        await ws.send_json({"type": "error", "message": "Transcriber not available"})
        return

    await ws.send_json({"type": "processing", "stage": "transcribing"})

    try:
        text = await transcriber.transcribe(audio)
    except Exception:
        log.exception("Remote transcription failed (device: %s)", device)
        await ws.send_json({"type": "error", "message": "Transcription failed"})
        return

    text = text.strip()
    if not text:
        await ws.send_json({"type": "error", "message": "Empty transcription"})
        return

    # Strip wake word prefix
    text = _WAKE_PREFIX_RE.sub("", text).strip() or text

    await ws.send_json({"type": "transcription", "text": text})
    log.info("Remote transcription (device: %s): %s", device, text)

    await _process_text_and_respond(ws, text, device)


async def _process_text_and_respond(
    ws: WebSocket, text: str, device: str,
) -> None:
    """Process text through agent, send response + optional TTS audio."""
    agent = ws.app.state.agent
    registry = ws.app.state.registry
    broadcaster = ws.app.state.broadcaster
    tts = getattr(ws.app.state, "tts", None)

    if agent is None:
        await ws.send_json({"type": "error", "message": "Agent not available"})
        return

    await ws.send_json({"type": "processing", "stage": "thinking"})
    await broadcaster.update_state("processing")
    await broadcaster.update_transcription(text)

    msg_count_before = len(agent.session.messages) if agent.session else 0
    tools = registry.get_openai_tools() if registry else None

    try:
        response = await agent.process_utterance(text, tools=tools or None, interactive=True, voice_mode=True)
        tools_used = _extract_tools(agent, msg_count_before)
    except Exception:
        log.exception("Remote agent processing failed (device: %s)", device)
        await ws.send_json({"type": "error", "message": "Processing failed"})
        await broadcaster.update_state("idle")
        return

    await broadcaster.update_response(response)
    await broadcaster.update_state("idle")

    # In Claude Code mode, agent sets tts_override with a voice summary.
    # Use that for TTS instead of the full response.
    tts_text = agent.tts_override or response
    agent.tts_override = None

    from claw.config import get_settings
    settings = get_settings()
    usage = agent.last_usage.to_dict() if agent.last_usage else {}

    audio_output = settings.remote.audio_output

    result: dict = {
        "type": "response",
        "text": response,
        "tools_used": sorted(tools_used),
        "usage": usage,
        "claude_mode": agent.claude_code_active,
    }

    music = _extract_music_from_session(agent, msg_count_before)
    if music:
        if audio_output == "phone":
            # Send stream URL to phone and stop local mpv playback
            result["music"] = music
            tool_router = getattr(ws.app.state, "tool_router", None)
            if tool_router:
                try:
                    await tool_router.call_tool("stop", {})
                    log.info("Stopped local mpv — music streaming to phone")
                except Exception:
                    log.debug("Could not stop local mpv (may not be running)")
        # else "computer": let mpv play locally, don't send stream URL to phone

    await ws.send_json(result)
    log.info("Remote response (device: %s, audio: %s): %s", device, audio_output, response[:100])

    # TTS — route based on audio_output setting
    # Use tts_text (voice summary in Claude Code mode, or full response otherwise)
    if tts and settings.tts.enabled:
        if audio_output == "phone":
            # Send WAV bytes over WebSocket to phone
            try:
                wav_bytes = await tts.synthesize_wav(tts_text)
                if wav_bytes:
                    await ws.send_json({"type": "tts_start", "size": len(wav_bytes)})
                    chunk_size = 32768
                    for i in range(0, len(wav_bytes), chunk_size):
                        await ws.send_bytes(wav_bytes[i : i + chunk_size])
                    await ws.send_json({"type": "tts_end"})
            except Exception:
                log.exception("Remote TTS failed (device: %s)", device)
        else:
            # Play TTS on computer speakers
            try:
                await tts.speak(tts_text)
                log.info("TTS played on computer speakers (device: %s)", device)
            except Exception:
                log.exception("Local TTS playback failed (device: %s)", device)


# ── Helpers ──────────────────────────────────────────────────────────────

def _extract_tools(agent, start_idx: int) -> list[str]:
    """Extract tool names from messages added after start_idx."""
    if agent.session is None:
        return []
    names: list[str] = []
    for msg in agent.session.messages[start_idx:]:
        if msg.get("role") == "assistant" and msg.get("tool_calls"):
            for tc in msg["tool_calls"]:
                name = tc.get("function", {}).get("name")
                if name and name not in names:
                    names.append(name)
    return names


def _extract_music_from_session(agent, start_idx: int) -> dict | None:
    """Extract music info (video_id) from tool call results in the session.

    Looks for [video:<id>] tags that the player embeds in play responses.
    """
    if agent.session is None:
        return None
    for msg in agent.session.messages[start_idx:]:
        if msg.get("role") == "tool":
            content = msg.get("content", "")
            match = _VIDEO_TAG_RE.search(content)
            if match:
                video_id = match.group(1)
                # Also try to extract title/artist from "Now playing: X by Y"
                title_match = re.search(r"Now playing:\s*(.+?)(?:\s+by\s+(.+?))?(?:\s*\[video:)", content)
                title = title_match.group(1).strip() if title_match and title_match.group(1) else ""
                artist = title_match.group(2).strip() if title_match and title_match.group(2) else ""
                return {
                    "video_id": video_id,
                    "title": title,
                    "artist": artist,
                    "stream_url": f"/api/remote/stream/{video_id}",
                }
    return None


async def _extract_facts(agent) -> None:
    """Background fact extraction after chat."""
    try:
        facts = await agent.extract_facts()
        if facts:
            log.info("Extracted %d facts from remote chat", len(facts))
    except Exception:
        log.exception("Remote chat fact extraction failed")
