"""Admin panel API and page routes."""

from __future__ import annotations

import asyncio
import base64
import io
import json
import logging
import logging.handlers
import os
import subprocess
import time
from collections import deque
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import TYPE_CHECKING
from uuid import uuid4

from fastapi import APIRouter, Request
from fastapi.responses import HTMLResponse, JSONResponse, RedirectResponse, Response
from sse_starlette.sse import EventSourceResponse

if TYPE_CHECKING:
    from claw.admin.sse import StatusBroadcaster

log = logging.getLogger(__name__)

router = APIRouter()

# ── Pending OAuth state store ────────────────────────────────────────────────
# NOTE: Process-local store — single-worker only.

_pending_auth: dict[str, dict] = {}
_PENDING_AUTH_TTL = 600  # 10 minutes
_PENDING_AUTH_MAX = 50


# ── Log buffer ──────────────────────────────────────────────────────────────

class LogBuffer(logging.Handler):
    """Ring-buffer logging handler for the admin log viewer."""

    def __init__(self, maxlen: int = 1000, fmt: str = "text") -> None:
        super().__init__()
        self.records: deque[str] = deque(maxlen=maxlen)
        self._json_mode = fmt == "json"
        if not self._json_mode:
            self.setFormatter(logging.Formatter(
                "%(asctime)s %(levelname)-7s %(name)s: %(message)s",
                datefmt="%H:%M:%S",
            ))

    def emit(self, record: logging.LogRecord) -> None:
        if self._json_mode:
            entry = {
                "timestamp": datetime.fromtimestamp(record.created, tz=timezone.utc).isoformat(),
                "level": record.levelname,
                "logger": record.name,
                "message": record.getMessage(),
            }
            if record.exc_info and record.exc_info[1]:
                entry["exc_info"] = logging.Formatter().formatException(record.exc_info)
            self.records.append(json.dumps(entry))
        else:
            self.records.append(self.format(record))


class JsonLineFormatter(logging.Formatter):
    """Structured JSON formatter for file logging."""

    def format(self, record: logging.LogRecord) -> str:
        entry = {
            "timestamp": datetime.fromtimestamp(record.created, tz=timezone.utc).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
        }
        if record.exc_info and record.exc_info[1]:
            entry["exc_info"] = self.formatException(record.exc_info)
        return json.dumps(entry)


# ── Page routes ─────────────────────────────────────────────────────────────

@router.get("/app", response_class=HTMLResponse)
async def pwa_app(request: Request):
    """The Claw PWA — remote client interface. No admin auth required."""
    return request.app.state.templates.TemplateResponse("app.html", {"request": request})


@router.get("/", response_class=HTMLResponse)
async def chat_page(request: Request):
    from claw.config import get_settings
    settings = get_settings()
    tts_enabled = (
        settings.tts.enabled
        and settings.tts.admin_chat_enabled
        and getattr(request.app.state, "tts", None) is not None
    )
    return request.app.state.templates.TemplateResponse(
        "chat.html", {
            "request": request,
            "save_conversations": settings.chat.save_conversations,
            "context_window": settings.llm.context_window,
            "tts_enabled": tts_enabled,
            "tts_autoplay": tts_enabled and settings.tts.admin_chat_autoplay,
        },
    )


@router.get("/dashboard", response_class=HTMLResponse)
async def dashboard(request: Request):
    broadcaster: StatusBroadcaster = request.app.state.broadcaster
    return request.app.state.templates.TemplateResponse(
        "dashboard.html",
        {"request": request, "status": broadcaster.get_status()},
    )


@router.get("/settings", response_class=HTMLResponse)
async def settings_page(request: Request):
    from claw.config import PROJECT_ROOT, get_settings
    settings = get_settings()

    # Check if Google OAuth credentials file exists
    creds_path = PROJECT_ROOT / settings.google_auth.credentials_file
    has_google_credentials = creds_path.exists()

    # Build per-account auth status (does token file exist + have refresh token?)
    acct_status: dict[str, bool] = {}
    for label, acct_data in settings.google_auth.accounts.items():
        tf = acct_data.token_file
        if tf:
            token_path = PROJECT_ROOT / tf if not Path(tf).is_absolute() else Path(tf)
            if token_path.exists():
                try:
                    data = json.loads(token_path.read_text())
                    acct_status[label] = bool(data.get("refresh_token"))
                except (json.JSONDecodeError, KeyError):
                    acct_status[label] = False
            else:
                acct_status[label] = False
        else:
            acct_status[label] = False

    # Gemini API key status
    from claw.secret_store import exists as secret_exists, mask as secret_mask
    gemini_key_set = secret_exists("gemini_api_key")
    gemini_key_masked = secret_mask("gemini_api_key") if gemini_key_set else "Not set"

    return request.app.state.templates.TemplateResponse(
        "settings.html",
        {
            "request": request,
            "settings": _redact_settings(settings.model_dump()),
            "has_google_credentials": has_google_credentials,
            "acct_status": acct_status,
            "gemini_key_set": gemini_key_set,
            "gemini_key_masked": gemini_key_masked,
        },
    )


@router.get("/logs", response_class=HTMLResponse)
async def logs_page(request: Request):
    log_buffer: LogBuffer = request.app.state.log_buffer
    return request.app.state.templates.TemplateResponse(
        "logs.html",
        {"request": request, "logs": list(log_buffer.records)},
    )


@router.get("/audio", response_class=HTMLResponse)
async def audio_page(request: Request):
    return request.app.state.templates.TemplateResponse(
        "audio.html", {"request": request},
    )


@router.get("/conversations", response_class=HTMLResponse)
async def conversations_page(request: Request):
    return request.app.state.templates.TemplateResponse(
        "conversations.html", {"request": request},
    )


# ── API routes ──────────────────────────────────────────────────────────────

@router.get("/api/status")
async def api_status(request: Request):
    broadcaster: StatusBroadcaster = request.app.state.broadcaster
    return broadcaster.get_status()


@router.get("/api/events")
async def api_events(request: Request):
    """SSE endpoint for real-time status updates."""
    broadcaster: StatusBroadcaster = request.app.state.broadcaster
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


@router.get("/api/logs")
async def api_logs(request: Request):
    log_buffer: LogBuffer = request.app.state.log_buffer
    return {"logs": list(log_buffer.records)}


@router.post("/api/chat")
async def api_chat(request: Request):
    """Process a chat message through the agent."""
    body = await request.json()
    message = body.get("message", "").strip()
    model = body.get("model")  # optional per-chat model override
    if not message:
        return JSONResponse({"error": "Empty message"}, status_code=400)

    agent = request.app.state.agent
    registry = request.app.state.registry
    broadcaster: StatusBroadcaster = request.app.state.broadcaster

    if agent is None:
        return {"role": "assistant", "content": "Agent not available.", "tools_used": []}

    # Handle slash commands before agent processing
    if message.startswith("/"):
        from claw.agent_core.commands import dispatch_command
        result = await dispatch_command(message, agent)
        if result is not None:
            return result

    # Serialize chat requests to protect shared session state
    chat_lock: asyncio.Lock = request.app.state.chat_lock

    async with chat_lock:
        msg_count_before = len(agent.session.messages) if agent.session else 0

        await broadcaster.update_state("processing")
        await broadcaster.update_transcription(message)

        tools = registry.get_openai_tools() if registry else None
        try:
            response = await agent.process_utterance(message, tools=tools or None, model=model or None)
            tools_used = _extract_tools_from_new_messages(agent, msg_count_before)
            await broadcaster.update_response(response)
        except Exception:
            log.exception("Chat processing failed")
            response = "Sorry, something went wrong processing your message."
            tools_used = []
        finally:
            await broadcaster.update_state("idle")

    # Background fact extraction (tracked to avoid GC)
    task = asyncio.create_task(_extract_facts(agent))
    request.app.state.bg_tasks.add(task)
    task.add_done_callback(request.app.state.bg_tasks.discard)

    from claw.config import get_settings
    usage = agent.last_usage.to_dict() if agent.last_usage else {}
    usage["context_window"] = get_settings().llm.context_window
    return {"role": "assistant", "content": response, "tools_used": sorted(tools_used), "usage": usage}


@router.post("/api/chat/stream")
async def api_chat_stream(request: Request):
    """Stream a chat response as Server-Sent Events."""
    body = await request.json()
    message = body.get("message", "").strip()
    model = body.get("model")
    if not message:
        return JSONResponse({"error": "Empty message"}, status_code=400)

    agent = request.app.state.agent
    registry = request.app.state.registry
    broadcaster: StatusBroadcaster = request.app.state.broadcaster

    if agent is None:
        return JSONResponse({"error": "Agent not available"}, status_code=503)

    # Handle slash commands before streaming — return plain JSON
    if message.startswith("/"):
        from claw.agent_core.commands import dispatch_command
        result = await dispatch_command(message, agent)
        if result is not None:
            # Return as a single SSE event so the client gets the response
            async def command_stream():
                yield json.dumps({"type": "token", "content": result["content"]})
                yield json.dumps({
                    "type": "done",
                    "content": result["content"],
                    "tools_used": [],
                    "usage": {},
                    "is_command": True,
                })
            return EventSourceResponse(command_stream())

    chat_lock: asyncio.Lock = request.app.state.chat_lock
    bg_tasks: set = request.app.state.bg_tasks

    async def generate():
        async with chat_lock:
            await broadcaster.update_state("processing")
            await broadcaster.update_transcription(message)

            tools = registry.get_openai_tools() if registry else None
            try:
                async for event in agent.process_utterance_stream(
                    message, tools=tools or None, model=model or None,
                ):
                    yield json.dumps(event)

                    # Broadcast tokens to other SSE clients (Android, dashboard)
                    if event["type"] == "token":
                        await broadcaster.broadcast_token(event["content"])
                    elif event["type"] == "done":
                        await broadcaster.update_response(event["content"])
                        await broadcaster.broadcast_token("", done=True)

            except Exception:
                log.exception("Stream processing failed")
                yield json.dumps({
                    "type": "error",
                    "message": "Sorry, something went wrong processing your message.",
                })
            finally:
                await broadcaster.update_state("idle")

        # After lock released: background fact extraction
        task = asyncio.create_task(_extract_facts(agent))
        bg_tasks.add(task)
        task.add_done_callback(bg_tasks.discard)

    return EventSourceResponse(generate())


@router.post("/api/chat/new")
async def api_chat_new(request: Request):
    """Start a new conversation session."""
    agent = request.app.state.agent
    if agent:
        agent.new_session()
    return {"status": "ok"}


@router.post("/api/chat/load")
async def api_chat_load(request: Request):
    """Load a past conversation into the agent session for resumption."""
    from claw.agent_core.conversation import ConversationSession

    body = await request.json()
    messages = body.get("messages", [])

    agent = request.app.state.agent
    if agent is None:
        return JSONResponse({"error": "Agent not available"}, status_code=503)

    # Create a fresh session and replay user/assistant turns
    session = ConversationSession()
    session.initialize()

    for msg in messages:
        role = msg.get("role")
        content = msg.get("content", "")
        if role == "user":
            session.add_user(content)
        elif role == "assistant" and content:
            session.add_assistant(content)

    session.trim_to_fit()
    agent._session = session

    # Report how much context was loaded
    prompt_tokens = sum(len(m.get("content", "")) // 4 for m in session.messages)
    return {"status": "ok", "loaded_messages": len(session.messages), "estimated_tokens": prompt_tokens}


@router.post("/api/tts")
async def api_tts(request: Request):
    """Synthesize text to speech and return WAV audio."""
    body = await request.json()
    text = body.get("text", "").strip()
    if not text:
        return JSONResponse({"error": "Empty text"}, status_code=400)

    tts = getattr(request.app.state, "tts", None)
    if tts is None:
        return JSONResponse({"error": "TTS not available"}, status_code=503)

    try:
        wav_bytes = await tts.synthesize_wav(text)
        if not wav_bytes:
            return JSONResponse({"error": "TTS synthesis produced no audio"}, status_code=500)
        return Response(content=wav_bytes, media_type="audio/wav")
    except Exception:
        log.exception("TTS synthesis failed")
        return JSONResponse({"error": "TTS synthesis failed"}, status_code=500)


def _extract_tools_from_new_messages(agent, start_idx: int) -> list[str]:
    """Extract tool names from messages added after start_idx."""
    if agent.session is None:
        return []
    names = []
    for msg in agent.session.messages[start_idx:]:
        if msg.get("role") == "assistant" and msg.get("tool_calls"):
            for tc in msg["tool_calls"]:
                fn = tc.get("function", {})
                name = fn.get("name")
                if name and name not in names:
                    names.append(name)
    return names


async def _extract_facts(agent) -> None:
    """Background fact extraction after chat."""
    try:
        facts = await agent.extract_facts()
        if facts:
            log.info("Extracted %d facts from chat", len(facts))
    except Exception:
        log.exception("Chat fact extraction failed")


# ── Usage tracking ──────────────────────────────────────────────────────────

@router.get("/api/usage")
async def api_usage(request: Request):
    """Get token usage summary."""
    tracker = getattr(request.app.state, "usage_tracker", None)
    if tracker is None:
        return {"session": {}, "today": {}, "total": {}}
    return {
        "session": tracker.get_session_summary(),
        "today": tracker.get_daily_summary(),
        "total": tracker.get_total_summary(),
    }


@router.get("/api/usage/history")
async def api_usage_history(request: Request):
    """Get daily usage history."""
    tracker = getattr(request.app.state, "usage_tracker", None)
    days = int(request.query_params.get("days", "7"))
    if tracker is None:
        return {"history": []}
    return {"history": tracker.get_history(days)}


# ── Conversation persistence ───────────────────────────────────────────────

def _conversations_dir() -> Path:
    from claw.config import PROJECT_ROOT, get_settings
    rel = get_settings().chat.conversations_dir.lstrip("/")
    d = PROJECT_ROOT / rel
    d.mkdir(parents=True, exist_ok=True)
    return d


def _valid_convo_id(convo_id: str) -> bool:
    return bool(re.fullmatch(r"[a-zA-Z0-9_-]{1,64}", convo_id))


@router.get("/api/conversations/search")
async def api_search_conversations(q: str = ""):
    """Substring search across saved conversations."""
    if not q.strip():
        return []
    query = q.strip().lower()
    d = _conversations_dir()
    results = []
    for f in sorted(d.glob("*.json"), key=lambda p: p.stat().st_mtime, reverse=True):
        try:
            data = json.loads(f.read_text())
            # Search in title and message content
            title = data.get("title", "")
            messages = data.get("messages", [])
            text_blob = title.lower() + " " + " ".join(
                (m.get("content") or "").lower() for m in messages
            )
            if query in text_blob:
                results.append({
                    "id": data["id"],
                    "title": title,
                    "created_at": data.get("created_at"),
                    "updated_at": data.get("updated_at"),
                    "message_count": len(messages),
                })
        except (json.JSONDecodeError, KeyError):
            continue
    return results


@router.post("/api/conversations/bulk-delete")
async def api_bulk_delete_conversations(request: Request):
    """Delete multiple conversations at once."""
    body = await request.json()
    ids = body.get("ids", [])
    if not ids:
        return JSONResponse({"error": "No IDs provided"}, status_code=400)
    d = _conversations_dir()
    deleted = 0
    for convo_id in ids:
        if not _valid_convo_id(convo_id):
            continue
        f = d / f"{convo_id}.json"
        if f.exists():
            f.unlink()
            deleted += 1
    return {"status": "ok", "deleted": deleted}


@router.get("/api/conversations")
async def api_list_conversations():
    """List saved conversations (summary only)."""
    convos = []
    d = _conversations_dir()
    for f in sorted(d.glob("*.json"), key=lambda p: p.stat().st_mtime, reverse=True):
        try:
            data = json.loads(f.read_text())
            convos.append({
                "id": data["id"],
                "title": data.get("title", "Untitled"),
                "created_at": data.get("created_at"),
                "updated_at": data.get("updated_at"),
                "message_count": len(data.get("messages", [])),
            })
        except (json.JSONDecodeError, KeyError):
            continue
    return convos


@router.post("/api/conversations/save")
async def api_save_conversation(request: Request):
    """Save or update a conversation."""
    body = await request.json()
    messages = body.get("messages", [])
    convo_id = body.get("id") or uuid4().hex[:12]
    if not _valid_convo_id(convo_id):
        return JSONResponse({"error": "Invalid conversation ID"}, status_code=400)
    now = datetime.now(timezone.utc).isoformat()

    # Auto-generate title from first user message
    title = "Untitled"
    for msg in messages:
        if msg.get("role") == "user" and msg.get("content"):
            title = msg["content"][:50]
            break

    d = _conversations_dir()
    existing_path = d / f"{convo_id}.json"

    created_at = now
    if existing_path.exists():
        try:
            old = json.loads(existing_path.read_text())
            created_at = old.get("created_at", now)
        except (json.JSONDecodeError, KeyError):
            pass

    data = {
        "id": convo_id,
        "title": title,
        "created_at": created_at,
        "updated_at": now,
        "messages": messages,
    }
    existing_path.write_text(json.dumps(data, indent=2))
    return {"status": "ok", "id": convo_id, "title": title}


@router.get("/api/conversations/{convo_id}")
async def api_get_conversation(convo_id: str):
    """Get a full conversation with messages."""
    if not _valid_convo_id(convo_id):
        return JSONResponse({"error": "Invalid conversation ID"}, status_code=400)
    f = _conversations_dir() / f"{convo_id}.json"
    if not f.exists():
        return JSONResponse({"error": "Not found"}, status_code=404)
    return json.loads(f.read_text())


@router.delete("/api/conversations/{convo_id}")
async def api_delete_conversation(convo_id: str):
    """Delete a saved conversation."""
    if not _valid_convo_id(convo_id):
        return JSONResponse({"error": "Invalid conversation ID"}, status_code=400)
    f = _conversations_dir() / f"{convo_id}.json"
    if not f.exists():
        return JSONResponse({"error": "Not found"}, status_code=404)
    f.unlink()
    return {"status": "ok"}


@router.get("/api/conversations/{convo_id}/export")
async def api_export_conversation(convo_id: str, format: str = "json"):
    """Export a conversation as JSON download or Markdown."""
    if not _valid_convo_id(convo_id):
        return JSONResponse({"error": "Invalid conversation ID"}, status_code=400)
    f = _conversations_dir() / f"{convo_id}.json"
    if not f.exists():
        return JSONResponse({"error": "Not found"}, status_code=404)

    data = json.loads(f.read_text())

    if format == "md":
        lines = [f"# {data.get('title', 'Conversation')}\n"]
        if data.get("created_at"):
            lines.append(f"*{data['created_at']}*\n")
        lines.append("")
        for msg in data.get("messages", []):
            content_text = msg.get("content") or ""
            if not content_text or msg.get("role") == "tool":
                continue
            role = "User" if msg.get("role") == "user" else "Claw"
            lines.append(f"**{role}:** {content_text}\n")
        content = "\n".join(lines)
        return Response(
            content=content,
            media_type="text/markdown",
            headers={"Content-Disposition": f'attachment; filename="{convo_id}.md"'},
        )

    # Default: JSON download
    return Response(
        content=json.dumps(data, indent=2),
        media_type="application/json",
        headers={"Content-Disposition": f'attachment; filename="{convo_id}.json"'},
    )


def _friendly_device_name(raw: str, direction: str = "input") -> str:
    """Turn raw ALSA/PipeWire device names into human-readable labels.

    direction: "input" or "output" — used for generic labels.
    """
    low = raw.lower()

    # PipeWire virtual devices
    if raw == "default":
        return "System Default"
    if raw == "pipewire":
        return "System Audio (PipeWire)"

    # USB devices — pull brand if present
    if "usb" in low:
        # Strip hw:X,Y suffix for cleaner display
        clean = re.sub(r'\s*\(hw:\d+,\d+\)', '', raw).strip()
        # Common USB mic brands
        for brand in ("fifine", "blue", "hyperx", "rode", "samson", "yeti", "snowball"):
            if brand in low:
                label = "USB Microphone" if direction == "input" else "USB Audio"
                return f"{label} ({clean})"
        label = "USB Microphone" if direction == "input" else "USB Audio"
        return f"{label} ({clean})" if clean.lower() != "usb audio" else label

    # HDMI outputs
    if "hdmi" in low:
        clean = re.sub(r'\s*\(hw:\d+,\d+\)', '', raw).strip()
        # Shorten "HDA NVidia: HDMI 0" → "HDMI 1 (NVIDIA)"
        if "nvidia" in low:
            hdmi_num = re.search(r'HDMI\s*(\d+)', raw)
            num = int(hdmi_num.group(1)) + 1 if hdmi_num else ""
            return f"HDMI {num} (NVIDIA GPU)"
        if "generic" in low or "hd-audio" in low:
            hdmi_num = re.search(r'HDMI\s*(\d+)', raw)
            num = int(hdmi_num.group(1)) + 1 if hdmi_num else ""
            return f"HDMI {num} (Integrated)"
        return f"HDMI ({clean})"

    # Internal analog (Realtek ALC, etc.)
    if "analog" in low or "alc" in low:
        codec = re.search(r'(ALC\w+)', raw)
        codec_str = f" ({codec.group(1)})" if codec else ""
        if direction == "input":
            return f"Internal Microphone{codec_str}"
        return f"Internal Speaker{codec_str}"

    # Bluetooth
    if "bluetooth" in low or "bluez" in low:
        clean = re.sub(r'\s*\(hw:\d+,\d+\)', '', raw).strip()
        return f"Bluetooth ({clean})"

    # Fallback: strip hw:X,Y and return as-is
    return re.sub(r'\s*\(hw:\d+,\d+\)', '', raw).strip()


@router.get("/api/audio/devices")
async def api_audio_devices():
    """List available audio input devices via sounddevice."""
    import sounddevice as sd

    devices = []
    try:
        all_devs = sd.query_devices()
        default_input = sd.default.device[0]  # default input device index
        for i, d in enumerate(all_devs):
            if d["max_input_channels"] > 0:
                devices.append({
                    "index": i,
                    "name": _friendly_device_name(d["name"], "input"),
                    "channels": d["max_input_channels"],
                    "sample_rate": int(d["default_samplerate"]),
                })
    except Exception as e:
        log.warning("Failed to query audio devices: %s", e)
        return {"devices": [], "default": None, "error": str(e)}

    return {"devices": devices, "default": default_input}


@router.get("/api/audio/devices/output")
async def api_audio_output_devices():
    """List available audio output devices via sounddevice."""
    import sounddevice as sd

    devices = []
    try:
        all_devs = sd.query_devices()
        default_output = sd.default.device[1]  # default output device index
        for i, d in enumerate(all_devs):
            if d["max_output_channels"] > 0:
                devices.append({
                    "index": i,
                    "name": _friendly_device_name(d["name"], "output"),
                    "channels": d["max_output_channels"],
                    "sample_rate": int(d["default_samplerate"]),
                })
    except Exception as e:
        log.warning("Failed to query audio output devices: %s", e)
        return {"devices": [], "default": None, "error": str(e)}

    return {"devices": devices, "default": default_output}


@router.get("/api/models")
async def api_models():
    """List models available in the connected LLM server (OpenAI-compatible)."""
    import httpx
    from claw.config import get_settings

    base_url = get_settings().llm.base_url.rstrip("/")

    try:
        async with httpx.AsyncClient(timeout=5.0) as client:
            resp = await client.get(f"{base_url}/models")
            resp.raise_for_status()
            data = resp.json()
            models = []
            for m in data.get("data", []):
                meta = m.get("meta", {})
                # llama-swap nests metadata under meta.llamaswap
                ls_meta = meta.get("llamaswap", meta)
                models.append({
                    "name": m.get("id", ""),
                    "size": 0,
                    "parameter_size": ls_meta.get("parameter_size", ""),
                    "family": ls_meta.get("family", ""),
                })
            return {"models": models}
    except Exception as e:
        log.warning("Failed to query LLM models: %s", e)
        return {"models": [], "error": str(e)}


@router.get("/api/ollama/models")
async def api_ollama_models_redirect():
    """Deprecated: redirects to /api/models."""
    return RedirectResponse("/api/models", status_code=301)


@router.get("/api/wake/models")
async def api_wake_models():
    """List available wake word models (built-in + custom)."""
    from claw.config import PROJECT_ROOT, get_settings

    BUILTIN_MODELS = [
        "alexa", "hey_mycroft", "hey_jarvis_v0.1",
        "hey_rhasspy", "timer", "weather",
    ]

    cfg = get_settings().wake
    custom_dir = PROJECT_ROOT / cfg.custom_models_dir
    custom = []
    if custom_dir.is_dir():
        custom = [p.stem for p in custom_dir.glob("*.onnx")]

    return {
        "builtin": BUILTIN_MODELS,
        "custom": sorted(custom),
        "active": list(cfg.model_paths),
    }


_REDACT_KEYS = {"api_key"}


def _redact_settings(data: dict) -> dict:
    """Deep-copy settings dict with sensitive values masked."""
    out = {}
    for k, v in data.items():
        if isinstance(v, dict):
            out[k] = _redact_settings(v)
        elif k in _REDACT_KEYS and isinstance(v, str) and v and v != "no-key":
            if len(v) <= 8:
                out[k] = v[:2] + "***"
            else:
                out[k] = v[:4] + "***" + v[-4:]
        else:
            out[k] = v
    return out


@router.get("/api/settings")
async def api_get_settings():
    """Return current settings as JSON (used by frontend polling)."""
    from claw.config import get_settings
    return _redact_settings(get_settings().model_dump())


@router.post("/api/settings")
async def api_update_settings(request: Request):
    """Update settings from the admin panel."""
    from pydantic import ValidationError
    from claw.config import get_settings, reload_settings

    body = await request.json()
    settings = get_settings()
    old_model = settings.llm.model

    # Merge updates into current settings, skipping redacted masked values
    current = settings.model_dump()
    for section, values in body.items():
        if section in current and isinstance(values, dict):
            # Deep merge for google_auth.accounts
            if section == "google_auth" and "accounts" in values:
                current[section]["accounts"] = values["accounts"]
                for k, v in values.items():
                    if k != "accounts":
                        current[section][k] = v
            else:
                for k, v in values.items():
                    # Skip masked values — don't overwrite real secrets with "xxxx***xxxx"
                    if isinstance(v, str) and "***" in v:
                        continue
                    current[section][k] = v

    # Save and reload
    try:
        new_settings = type(settings)(**current)
    except ValidationError as exc:
        errors = "; ".join(f"{e['loc']}: {e['msg']}" for e in exc.errors())
        return JSONResponse({"error": f"Invalid settings: {errors}"}, status_code=422)

    new_settings.save_yaml()
    reload_settings()

    new_model = new_settings.llm.model
    if new_model != old_model:
        log.info("Model changed from %s to %s (server handles unloading via TTL)", old_model, new_model)

    return {"status": "ok", "message": "Settings updated and reloaded"}


# ── Health check ────────────────────────────────────────────────────────

@router.get("/api/health")
async def api_health(request: Request):
    """System health check. Returns 200 if healthy, 503 if degraded/unhealthy."""
    import httpx
    from claw.config import get_settings

    settings = get_settings()
    checks: dict[str, dict] = {}

    # LLM reachable
    try:
        async with httpx.AsyncClient(timeout=5.0) as client:
            resp = await client.get(f"{settings.llm.base_url.rstrip('/')}/models")
            checks["llm"] = {"status": "ok" if resp.status_code == 200 else "error"}
    except Exception as exc:
        checks["llm"] = {"status": "error", "detail": str(exc)}

    # ChromaDB — check via the agent's retriever → store
    try:
        agent = request.app.state.agent
        store = getattr(getattr(agent, "retriever", None), "_store", None)
        if store and getattr(store, "_initialized", False):
            checks["chromadb"] = {"status": "ok"}
        else:
            checks["chromadb"] = {"status": "warning", "detail": "Store not initialized"}
    except Exception as exc:
        checks["chromadb"] = {"status": "error", "detail": str(exc)}

    # Audio device (run off event loop — PortAudio can block)
    try:
        import sounddevice as sd
        devs = await asyncio.to_thread(sd.query_devices)
        input_devs = [d for d in devs if d["max_input_channels"] > 0]
        checks["audio"] = {"status": "ok" if input_devs else "warning", "devices": len(input_devs)}
    except Exception as exc:
        checks["audio"] = {"status": "error", "detail": str(exc)}

    # MCP servers
    registry = request.app.state.registry
    if registry:
        servers = registry.list_servers()
        checks["mcp"] = {"status": "ok", "servers": len(servers)}
    else:
        checks["mcp"] = {"status": "warning", "detail": "No registry"}

    # Overall status
    statuses = [c["status"] for c in checks.values()]
    if all(s == "ok" for s in statuses):
        overall = "healthy"
    elif "error" in statuses:
        overall = "unhealthy"
    else:
        overall = "degraded"

    code = 200 if overall == "healthy" else 503
    return JSONResponse({"status": overall, "checks": checks}, status_code=code)


# ── Webhook ────────────────────────────────────────────────────────────────

@router.post("/api/webhook")
async def api_webhook(request: Request):
    """Inbound webhook endpoint for external integrations.

    Expects JSON body with: {"type": "message|notification|reminder", "payload": {...}, "source": "..."}
    Optionally verified via X-Webhook-Signature header (HMAC-SHA256).
    Auth: uses HMAC signature verification, NOT HTTP Basic Auth.
    """
    from claw.admin.webhook import (
        WebhookEvent,
        handle_message,
        handle_notification,
        handle_reminder,
        verify_signature,
    )
    from claw.config import get_settings

    cfg = get_settings().webhook
    if not cfg.enabled:
        return JSONResponse({"error": "Webhooks are disabled"}, status_code=403)

    # Verify HMAC signature
    body = await request.body()
    signature = request.headers.get("X-Webhook-Signature", "")
    if cfg.secret and not verify_signature(body, signature, cfg.secret):
        return JSONResponse({"error": "Invalid signature"}, status_code=401)

    # Parse event
    try:
        data = json.loads(body)
        event = WebhookEvent(**data)
    except Exception as e:
        return JSONResponse({"error": f"Invalid event: {e}"}, status_code=400)

    # Check allowed event types
    if event.type not in cfg.allowed_events:
        return JSONResponse({"error": f"Event type '{event.type}' not allowed"}, status_code=400)

    # Dispatch
    agent = request.app.state.agent
    broadcaster = request.app.state.broadcaster
    tts = getattr(request.app.state, "tts", None)

    if event.type == "message":
        registry = request.app.state.registry
        tools = registry.get_openai_tools() if registry else None
        result = await handle_message(event, agent, broadcaster, tools=tools)
    elif event.type == "notification":
        result = await handle_notification(event, broadcaster, tts)
    elif event.type == "reminder":
        result = await handle_reminder(event)
    else:
        return JSONResponse({"error": f"Unknown event type: {event.type}"}, status_code=400)

    return result


# ── Tool usage stats ──────────────────────────────────────────────────────

@router.get("/api/tools/stats")
async def api_tool_stats(request: Request):
    """Per-tool usage statistics."""
    stats = getattr(request.app.state, "tool_stats", None)
    if stats is None:
        return {"tools": {}, "recent": []}
    return {"tools": stats.summary(), "recent": stats.recent()}


# ── Scheduled reminders ───────────────────────────────────────────────────

@router.get("/api/reminders")
async def api_reminders(request: Request):
    """Get upcoming scheduled reminders."""
    scheduler = getattr(request.app.state, "scheduler", None)
    if scheduler is None:
        return {"reminders": []}
    return {"reminders": scheduler.get_upcoming()}


# ── Audio diagnostics ─────────────────────────────────────────────────────

@router.get("/api/audio/stats")
async def api_audio_stats(request: Request):
    """Return current audio capture metrics for the diagnostics page."""
    capture = getattr(request.app.state, "audio_capture", None)
    vad = getattr(request.app.state, "vad", None)

    if capture is None:
        return {"active": False}

    metrics = capture.get_metrics()
    metrics["active"] = True
    if vad:
        metrics["vad_speech_prob"] = round(vad.last_speech_prob, 4)
        metrics["vad_threshold"] = vad.threshold
    return metrics


# ── Backup ────────────────────────────────────────────────────────────────

@router.post("/api/admin/backup")
async def api_trigger_backup(request: Request):
    """Trigger a data backup."""
    from claw.config import PROJECT_ROOT

    script = PROJECT_ROOT / "scripts" / "backup.sh"
    if not script.exists():
        return JSONResponse({"error": "Backup script not found"}, status_code=404)

    try:
        result = await asyncio.to_thread(
            subprocess.run,
            ["bash", str(script)],
            capture_output=True,
            text=True,
            timeout=120,
            cwd=str(PROJECT_ROOT),
        )
        if result.returncode == 0:
            return {"status": "ok", "output": result.stdout[-500:]}
        return JSONResponse(
            {"error": "Backup failed", "output": result.stderr[-500:]},
            status_code=500,
        )
    except subprocess.TimeoutExpired:
        return JSONResponse({"error": "Backup timed out"}, status_code=504)


# ── Device management (remote access) ────────────────────────────────────

@router.get("/api/devices")
async def api_list_devices():
    """List all registered remote devices."""
    from claw.admin.api_key import list_devices
    return {"devices": list_devices()}


@router.post("/api/devices")
async def api_create_device(request: Request):
    """Register a new remote device. Returns API key + provisioning code exactly once."""
    body = await request.json()
    name = body.get("name", "").strip()
    if not name:
        return JSONResponse({"error": "Device name is required"}, status_code=400)

    from claw.admin.api_key import create_device
    try:
        result = create_device(name)
    except ValueError as exc:
        return JSONResponse({"error": str(exc)}, status_code=400)

    response = {
        "status": "ok",
        "name": name,
        "api_key": result["api_key"],
        "wg_available": result["wg_available"],
        "warning": "Save this information now — it cannot be retrieved later.",
    }
    if result["provisioning_code"]:
        response["provisioning_code"] = result["provisioning_code"]
        # Generate QR code as SVG data URI for the provisioning code
        try:
            import segno

            qr = segno.make(result["provisioning_code"], error="L")
            buf = io.BytesIO()
            qr.save(buf, kind="svg", dark="#ffffff", light="#0d1117", scale=12, border=2)
            buf.seek(0)
            svg_b64 = base64.b64encode(buf.read()).decode()
            response["qr_data_uri"] = f"data:image/svg+xml;base64,{svg_b64}"
        except Exception:
            log.warning("Failed to generate QR code for device '%s'", name)

    return response


@router.delete("/api/devices/{name}")
async def api_revoke_device(name: str):
    """Revoke a device's remote access."""
    from claw.admin.api_key import revoke_device
    if not revoke_device(name):
        return JSONResponse({"error": "Device not found"}, status_code=404)
    return {"status": "ok", "message": f"Device '{name}' revoked"}


# ── Google OAuth flow ────────────────────────────────────────────────────────

def _token_is_valid(token_file: str) -> bool:
    """Check if a token file exists, has a refresh_token, and covers all configured scopes."""
    from claw.config import PROJECT_ROOT, get_settings
    p = Path(token_file)
    token_path = p if p.is_absolute() else PROJECT_ROOT / p
    if not token_path.exists():
        return False
    try:
        data = json.loads(token_path.read_text())
        if not data.get("refresh_token"):
            return False
        # Check that token has all scopes from config
        token_scopes = set(data.get("scopes", []))
        config_scopes = set(get_settings().google_auth.scopes)
        if not config_scopes.issubset(token_scopes):
            return False  # Missing scopes — re-auth needed
        return True
    except (json.JSONDecodeError, KeyError):
        return False


def _clean_pending_auth() -> None:
    """Remove expired pending auth entries and enforce cap."""
    now = time.monotonic()
    expired = [k for k, v in _pending_auth.items() if now - v["timestamp"] > _PENDING_AUTH_TTL]
    for k in expired:
        del _pending_auth[k]
    while len(_pending_auth) > _PENDING_AUTH_MAX:
        oldest = min(_pending_auth, key=lambda k: _pending_auth[k]["timestamp"])
        del _pending_auth[oldest]


@router.get("/auth/google/start")
async def auth_google_start(request: Request, label: str = ""):
    from claw.config import PROJECT_ROOT, get_settings

    # Validate label
    label = label.strip()
    if not label or not re.fullmatch(r"[a-zA-Z0-9_-]{1,32}", label):
        return RedirectResponse(
            "/settings?error=" + "Invalid+account+label.+Use+letters,+numbers,+hyphens,+or+underscores.",
            status_code=302,
        )

    settings = get_settings()
    # Allow re-auth if account exists but token is missing or invalid
    if label in settings.google_auth.accounts:
        acct = settings.google_auth.accounts[label]
        token_file = acct.token_file if hasattr(acct, "token_file") else acct.get("token_file", "")
        if _token_is_valid(token_file):
            return RedirectResponse(f"/settings?error=Account+'{label}'+already+linked.", status_code=302)

    # Check credentials.json exists
    creds_path = PROJECT_ROOT / settings.google_auth.credentials_file
    if not creds_path.exists():
        return RedirectResponse("/settings?error=no_credentials", status_code=302)

    try:
        from google_auth_oauthlib.flow import Flow
    except ImportError:
        return RedirectResponse(
            "/settings?error=Google+auth+libraries+not+installed.+Run:+pip+install+'claw[google]'",
            status_code=302,
        )

    # Desktop app credentials require http://localhost redirect URIs.
    # Use the configured admin port so the callback handler can process it.
    admin_port = settings.admin.port
    redirect_uri = f"http://localhost:{admin_port}/auth/google/callback"

    scopes = list(settings.google_auth.scopes)

    flow = Flow.from_client_secrets_file(
        str(creds_path),
        scopes=scopes,
        redirect_uri=redirect_uri,
    )
    auth_url, state = flow.authorization_url(
        access_type="offline",
        prompt="consent",
    )

    # Capture PKCE code_verifier generated by the library (required for token exchange)
    code_verifier = getattr(flow, "code_verifier", None)

    # Store pending auth
    _clean_pending_auth()
    _pending_auth[state] = {
        "label": label,
        "redirect_uri": redirect_uri,
        "scopes": scopes,
        "code_verifier": code_verifier,
        "timestamp": time.monotonic(),
    }

    return RedirectResponse(auth_url, status_code=302)


@router.get("/auth/google/callback")
async def auth_google_callback(request: Request, code: str = "", state: str = ""):
    from claw.config import PROJECT_ROOT, get_settings, reload_settings

    _clean_pending_auth()

    if not state or state not in _pending_auth:
        return RedirectResponse("/settings?error=OAuth+session+expired.+Please+try+again.", status_code=302)

    pending = _pending_auth[state]  # peek, don't pop yet
    label = pending["label"]
    redirect_uri = pending["redirect_uri"]
    scopes = pending["scopes"]
    code_verifier = pending.get("code_verifier")

    if not code:
        _pending_auth.pop(state, None)
        return RedirectResponse("/settings?error=Authorization+was+cancelled.", status_code=302)

    settings = get_settings()

    # For re-auth: allow if token is missing/invalid; block only if fully linked concurrently
    is_reauth = label in settings.google_auth.accounts
    if is_reauth:
        acct = settings.google_auth.accounts[label]
        token_file = acct.token_file if hasattr(acct, "token_file") else acct.get("token_file", "")
        if _token_is_valid(token_file):
            _pending_auth.pop(state, None)
            return RedirectResponse("/settings?error=Account+already+linked.+Another+flow+completed+first.", status_code=302)

    creds_path = PROJECT_ROOT / settings.google_auth.credentials_file

    try:
        from google_auth_oauthlib.flow import Flow

        flow = Flow.from_client_secrets_file(
            str(creds_path),
            scopes=scopes,
            redirect_uri=redirect_uri,
        )
        flow.fetch_token(code=code, code_verifier=code_verifier)
        creds = flow.credentials
    except Exception as exc:
        log.exception("OAuth token exchange failed")
        _pending_auth.pop(state, None)  # only pop after failure
        return RedirectResponse(f"/settings?error=Token+exchange+failed:+{type(exc).__name__}", status_code=302)

    # Token exchange succeeded — now pop the state
    _pending_auth.pop(state, None)

    # Save token file with restricted permissions (contains refresh token)
    token_dir = PROJECT_ROOT / "data" / "google" / "tokens"
    token_dir.mkdir(parents=True, exist_ok=True)
    token_path = token_dir / f"{label}.json"
    fd = os.open(str(token_path), os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
    with os.fdopen(fd, "w") as f:
        f.write(creds.to_json())
    log.info("Saved Google OAuth token for account '%s' at %s", label, token_path)

    # Best-effort: fetch email via Gmail API
    email = ""
    try:
        from googleapiclient.discovery import build
        service = build("gmail", "v1", credentials=creds)
        profile = service.users().getProfile(userId="me").execute()
        email = profile.get("emailAddress", "")
    except Exception:
        log.warning("Could not fetch email for account '%s'", label)

    # Build account config and merge into settings
    token_rel = f"data/google/tokens/{label}.json"
    current = settings.model_dump()

    if is_reauth:
        # Re-auth: preserve existing calendar/gmail settings, just update token + email
        existing = current["google_auth"]["accounts"][label]
        existing["token_file"] = token_rel
        if email:
            existing["email"] = email
    else:
        # New account: create fresh config with defaults
        from claw.config import GoogleAccountCalendarConfig, GoogleAccountGmailConfig
        cal_defaults = GoogleAccountCalendarConfig()
        gmail_defaults = GoogleAccountGmailConfig()
        current["google_auth"]["accounts"][label] = {
            "email": email,
            "token_file": token_rel,
            "calendar": {
                "enabled": True,
                "default_calendar": cal_defaults.default_calendar,
                "calendars": {},
                "timezone": cal_defaults.timezone,
            },
            "gmail": {
                "enabled": True,
                "max_results": gmail_defaults.max_results,
                "default_label": gmail_defaults.default_label,
            },
            "youtube_music": False,
        }

    new_settings = type(settings)(**current)
    new_settings.save_yaml()
    reload_settings()

    action = "re-linked" if is_reauth else "linked"
    log.info("Google account '%s' (%s) %s successfully", label, email or "unknown email", action)
    return RedirectResponse(f"/settings?linked={label}", status_code=302)


@router.post("/api/auth/google/complete")
async def auth_google_complete(request: Request):
    """Fallback for remote access: user pastes the redirect URL containing code+state."""
    from urllib.parse import parse_qs, urlparse

    body = await request.json()
    url = body.get("url", "")
    if not url:
        return JSONResponse({"error": "No URL provided"}, status_code=400)

    parsed = urlparse(url)
    params = parse_qs(parsed.query)
    code = params.get("code", [None])[0]
    state = params.get("state", [None])[0]

    if not code or not state:
        return JSONResponse({"error": "URL does not contain code and state parameters"}, status_code=400)

    # Reuse the callback logic
    from claw.config import PROJECT_ROOT, get_settings, reload_settings

    _clean_pending_auth()

    if state not in _pending_auth:
        return JSONResponse({"error": "OAuth session expired. Please try again."}, status_code=400)

    pending = _pending_auth[state]  # peek, don't pop yet
    label = pending["label"]
    redirect_uri = pending["redirect_uri"]
    scopes = pending["scopes"]
    code_verifier = pending.get("code_verifier")

    settings = get_settings()
    if label in settings.google_auth.accounts:
        _pending_auth.pop(state, None)
        return JSONResponse({"error": f"Account '{label}' already exists."}, status_code=400)

    creds_path = PROJECT_ROOT / settings.google_auth.credentials_file

    try:
        from google_auth_oauthlib.flow import Flow

        flow = Flow.from_client_secrets_file(str(creds_path), scopes=scopes, redirect_uri=redirect_uri)
        flow.fetch_token(code=code, code_verifier=code_verifier)
        creds = flow.credentials
    except Exception as exc:
        log.exception("OAuth token exchange failed (manual)")
        _pending_auth.pop(state, None)  # only pop after failure
        return JSONResponse({"error": f"Token exchange failed: {type(exc).__name__}: {exc}"}, status_code=500)

    # Token exchange succeeded — now pop the state
    _pending_auth.pop(state, None)

    # Save token
    token_dir = PROJECT_ROOT / "data" / "google" / "tokens"
    token_dir.mkdir(parents=True, exist_ok=True)
    token_path = token_dir / f"{label}.json"
    fd = os.open(str(token_path), os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
    with os.fdopen(fd, "w") as f:
        f.write(creds.to_json())

    email = ""
    try:
        from googleapiclient.discovery import build
        service = build("gmail", "v1", credentials=creds)
        profile = service.users().getProfile(userId="me").execute()
        email = profile.get("emailAddress", "")
    except Exception:
        log.warning("Could not fetch email for account '%s'", label)

    from claw.config import GoogleAccountCalendarConfig, GoogleAccountGmailConfig
    cal_defaults = GoogleAccountCalendarConfig()
    gmail_defaults = GoogleAccountGmailConfig()
    token_rel = f"data/google/tokens/{label}.json"
    acct_config = {
        "email": email,
        "token_file": token_rel,
        "calendar": {
            "enabled": True,
            "default_calendar": cal_defaults.default_calendar,
            "calendars": {},
            "timezone": cal_defaults.timezone,
        },
        "gmail": {
            "enabled": True,
            "max_results": gmail_defaults.max_results,
            "default_label": gmail_defaults.default_label,
        },
        "youtube_music": False,
    }

    current = settings.model_dump()
    current["google_auth"]["accounts"][label] = acct_config
    new_settings = type(settings)(**current)
    new_settings.save_yaml()
    reload_settings()

    log.info("Google account '%s' (%s) linked successfully (manual)", label, email or "unknown email")
    return {"status": "ok", "label": label, "email": email}


# ── Admin auth endpoints ──────────────────────────────────────────────────────

@router.post("/api/admin/setup")
async def api_admin_setup(request: Request):
    """Set the initial admin password. Only works when no password is configured."""
    from claw.secret_store import exists as secret_exists, store as secret_store

    if secret_exists("admin_password"):
        return JSONResponse({"error": "Password already configured"}, status_code=409)

    body = await request.json()
    password = body.get("password", "")
    if len(password) < 8:
        return JSONResponse({"error": "Password must be at least 8 characters"}, status_code=400)

    secret_store("admin_password", password)
    log.info("Admin password configured via setup endpoint")
    return {"status": "ok", "message": "Admin password set"}


@router.post("/api/admin/password")
async def api_admin_password(request: Request):
    """Change the admin password. Requires current authentication."""
    from claw.secret_store import store as secret_store

    body = await request.json()
    new_password = body.get("password", "")
    if len(new_password) < 8:
        return JSONResponse({"error": "Password must be at least 8 characters"}, status_code=400)

    secret_store("admin_password", new_password)
    log.info("Admin password changed")
    return {"status": "ok", "message": "Password updated"}


# ── Weather / OWM API key endpoints ──────────────────────────────────────────

@router.post("/api/weather/key")
async def api_weather_key_save(request: Request):
    """Store OWM API key in secret store and clear plaintext from config."""
    from claw.secret_store import store as secret_store

    body = await request.json()
    key = body.get("key", "").strip()
    if not key:
        return JSONResponse({"error": "No API key provided"}, status_code=400)

    secret_store("owm_api_key", key)

    # Clear plaintext key from config.yaml
    from claw.config import get_settings, reload_settings
    settings = get_settings()
    if settings.weather.api_key:
        current = settings.model_dump()
        current["weather"]["api_key"] = ""
        new_settings = type(settings)(**current)
        new_settings.save_yaml()
        reload_settings()

    masked = key[:4] + "***" + key[-4:] if len(key) > 8 else key[:2] + "***"
    log.info("OWM API key stored in secret store (masked: %s)", masked)
    return {"status": "ok", "masked": masked}


@router.delete("/api/weather/key")
async def api_weather_key_delete():
    """Remove the stored OWM API key."""
    from claw.secret_store import delete as secret_delete
    secret_delete("owm_api_key")
    log.info("OWM API key removed")
    return {"status": "ok"}


# ── Gemini API key + log endpoints ────────────────────────────────────────────

@router.post("/api/gemini/key")
async def api_gemini_key_save(request: Request):
    """Validate, test, and store a Gemini API key."""
    from claw.secret_store import store as secret_store

    body = await request.json()
    key = body.get("key", "").strip()

    if not key:
        return JSONResponse({"error": "No API key provided"}, status_code=400)

    if not key.startswith("AIza"):
        return JSONResponse(
            {"error": "Invalid key format. Gemini API keys start with 'AIza'."},
            status_code=400,
        )

    # Test the key by listing models
    try:
        import google.generativeai as genai
        genai.configure(api_key=key)
        models = list(genai.list_models())
        if not models:
            return JSONResponse({"error": "Key is valid but no models are accessible."}, status_code=400)
    except ImportError:
        return JSONResponse(
            {"error": "google-generativeai not installed. Run: pip install 'claw[gemini]'"},
            status_code=500,
        )
    except Exception as exc:
        return JSONResponse(
            {"error": f"API key test failed: {exc}"},
            status_code=400,
        )

    secret_store("gemini_api_key", key)
    masked = key[:4] + "***" + key[-4:]
    log.info("Gemini API key stored (masked: %s)", masked)
    return {"status": "ok", "masked": masked}


@router.delete("/api/gemini/key")
async def api_gemini_key_delete():
    """Remove the stored Gemini API key."""
    from claw.secret_store import delete as secret_delete
    secret_delete("gemini_api_key")
    log.info("Gemini API key removed")
    return {"status": "ok"}


@router.get("/api/gemini/logs")
async def api_gemini_logs():
    """Return recent Gemini API call logs and today's usage."""
    try:
        from claw.config import PROJECT_ROOT, get_settings
        cfg = get_settings().gemini

        log_dir = PROJECT_ROOT / cfg.log_dir
        entries = []
        if log_dir.exists():
            for log_file in sorted(log_dir.glob("*.jsonl"), reverse=True):
                for line in reversed(log_file.read_text().splitlines()):
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        entries.append(json.loads(line))
                    except json.JSONDecodeError:
                        continue
                    if len(entries) >= 100:
                        break
                if len(entries) >= 100:
                    break

        # Count today's usage from log entries
        from datetime import date as date_type
        today = date_type.today().isoformat()
        today_requests = sum(1 for e in entries if e.get("timestamp", "").startswith(today))
        today_grounding = sum(
            1 for e in entries
            if e.get("timestamp", "").startswith(today) and e.get("tool") == "gemini_web_search"
        )

        return {
            "entries": entries,
            "usage": {
                "requests_today": today_requests,
                "grounding_today": today_grounding,
                "daily_limit": cfg.daily_request_limit,
                "grounding_limit": cfg.grounding_daily_limit,
            },
        }
    except Exception:
        log.exception("Failed to read Gemini logs")
        return {"entries": [], "usage": {}}


@router.delete("/api/gemini/logs")
async def api_gemini_logs_delete():
    """Wipe all Gemini API call logs."""
    from claw.config import PROJECT_ROOT, get_settings
    cfg = get_settings().gemini
    log_dir = PROJECT_ROOT / cfg.log_dir
    count = 0
    if log_dir.exists():
        for log_file in log_dir.glob("*.jsonl"):
            log_file.unlink()
            count += 1
    log.info("Wiped %d Gemini log file(s)", count)
    return {"status": "ok", "files_deleted": count}


# ── Cloud LLM endpoints ──────────────────────────────────────────────────────


@router.post("/api/cloud-llm/key")
async def api_cloud_llm_key(request: Request):
    """Store a cloud LLM API key in the secret store."""
    from claw import secret_store
    from claw.config import get_settings

    body = await request.json()
    provider = body.get("provider", "")  # "claude" or "gemini"
    api_key = body.get("api_key", "").strip()

    cloud_cfg = get_settings().cloud_llm
    if provider not in cloud_cfg.providers:
        return JSONResponse({"error": f"Unknown provider: {provider}"}, status_code=400)

    secret_name = cloud_cfg.providers[provider].api_key_secret
    if not api_key:
        return JSONResponse({"error": "API key is required"}, status_code=400)

    secret_store.store(secret_name, api_key)

    # Re-initialize cloud clients
    from claw.agent_core.llm_client import LLMClient
    llm: LLMClient = request.app.state.agent.llm
    llm._init_cloud_clients()

    return {"status": "ok", "provider": provider, "masked": secret_store.mask(secret_name)}


@router.delete("/api/cloud-llm/key")
async def api_cloud_llm_key_delete(request: Request):
    """Remove a cloud LLM API key."""
    from claw import secret_store
    from claw.config import get_settings

    body = await request.json()
    provider = body.get("provider", "")

    cloud_cfg = get_settings().cloud_llm
    if provider not in cloud_cfg.providers:
        return JSONResponse({"error": f"Unknown provider: {provider}"}, status_code=400)

    secret_name = cloud_cfg.providers[provider].api_key_secret
    secret_store.delete(secret_name)

    # Re-initialize cloud clients (will remove this provider)
    from claw.agent_core.llm_client import LLMClient
    llm: LLMClient = request.app.state.agent.llm
    llm._init_cloud_clients()

    # If this was the active provider, switch back to local
    if llm.active_provider == provider:
        llm.active_provider = "local"

    return {"status": "ok"}


@router.post("/api/cloud-llm/switch")
async def api_cloud_llm_switch(request: Request):
    """Switch the active LLM backend."""
    from claw.config import get_settings

    body = await request.json()
    provider = body.get("provider", "local")

    from claw.agent_core.llm_client import LLMClient
    llm: LLMClient = request.app.state.agent.llm
    old_provider = llm.active_provider
    llm.active_provider = provider

    # Update config and trigger reload callbacks
    from claw.config import reload_settings
    settings = get_settings()
    settings.cloud_llm.active_provider = provider
    settings.save_yaml()
    reload_settings()

    # Broadcast change
    broadcaster = request.app.state.broadcaster
    await broadcaster.broadcast("llm_provider", {"provider": provider})

    return {"status": "ok", "provider": provider, "previous": old_provider}


@router.get("/api/cloud-llm/status")
async def api_cloud_llm_status(request: Request):
    """Get cloud LLM configuration status."""
    from claw import secret_store
    from claw.config import get_settings

    cloud_cfg = get_settings().cloud_llm

    from claw.agent_core.llm_client import LLMClient
    llm: LLMClient = request.app.state.agent.llm

    providers = {}
    for name, provider in cloud_cfg.providers.items():
        has_key = secret_store.exists(provider.api_key_secret) if provider.api_key_secret else False
        providers[name] = {
            "model": provider.model,
            "has_key": has_key,
            "key_masked": secret_store.mask(provider.api_key_secret) if has_key else "Not set",
            "available": name in llm._cloud_clients,
        }

    return {
        "active_provider": llm.active_provider,
        "failover_to_local": cloud_cfg.failover_to_local,
        "providers": providers,
    }


# ── Compute backend endpoints ────────────────────────────────────────────────

@router.get("/api/compute/hardware")
async def api_compute_hardware():
    """Detect available compute hardware (GPU, iGPU, CPU)."""
    from claw.compute import detect_all

    try:
        result = await asyncio.to_thread(detect_all)
        return result
    except Exception as exc:
        log.exception("Hardware detection failed")
        return JSONResponse({"error": str(exc)}, status_code=500)


@router.post("/api/compute/switch")
async def api_compute_switch(request: Request):
    """Switch compute backend: install deps, rebuild llama.cpp, update config."""
    from claw.compute import VALID_BACKENDS, is_build_running, switch_backend

    body = await request.json()
    target = body.get("backend", "").strip()

    if target not in VALID_BACKENDS:
        return JSONResponse({"error": f"Invalid backend: {target}"}, status_code=400)

    if is_build_running():
        return JSONResponse({"error": "A build is already in progress"}, status_code=409)

    async def _run_switch():
        result = await switch_backend(target)
        if result["status"] == "ok":
            # Save new backend to config
            from claw.config import get_settings, reload_settings
            settings = get_settings()
            current = settings.model_dump()
            current["compute"]["backend"] = target
            new_settings = type(settings)(**current)
            new_settings.save_yaml()
            reload_settings()
            log.info("Compute backend switched to %s", target)

    task = asyncio.create_task(_run_switch())
    request.app.state.compute_build_task = task
    request.app.state.bg_tasks.add(task)
    task.add_done_callback(request.app.state.bg_tasks.discard)

    return {"status": "ok", "message": f"Switching to {target}..."}


@router.get("/api/compute/progress")
async def api_compute_progress():
    """SSE endpoint for build progress updates."""
    from claw.compute import get_build_progress, is_build_running

    async def generate():
        last_idx = 0
        while True:
            progress = get_build_progress()
            while last_idx < len(progress):
                entry = progress[last_idx]
                yield {"event": "build_progress", "data": json.dumps(entry)}
                last_idx += 1
                # Check if done
                if entry.get("percent", 0) >= 100 or entry.get("phase") == "error":
                    yield {"event": "build_complete", "data": json.dumps(entry)}
                    return

            if not is_build_running() and last_idx >= len(progress):
                # Build finished but we may have missed the 100% event
                yield {"event": "build_complete", "data": json.dumps(
                    {"phase": "done", "percent": 100, "message": "Complete"}
                )}
                return

            await asyncio.sleep(0.5)

    return EventSourceResponse(generate())


# ── Music control endpoints ──────────────────────────────────────────────────

async def _music_call(request: Request, tool_name: str, arguments: dict | None = None) -> str | None:
    """Call a YouTube Music MCP tool via the shared tool router. Returns result string or None."""
    router_ = getattr(request.app.state, "tool_router", None)
    if router_ is None:
        # Fallback: create a bare router without stats
        from claw.mcp_handler.router import ToolRouter
        registry = request.app.state.registry
        if registry is None:
            return None
        router_ = ToolRouter(registry)
    try:
        return await router_.call_tool(tool_name, arguments or {})
    except Exception:
        log.exception("Music tool call '%s' failed", tool_name)
        return None


@router.get("/api/music/status")
async def api_music_status(request: Request):
    """Return current music playback status for the now-playing widget."""
    result = await _music_call(request, "get_status")
    if result is None:
        return {"playing": False}
    try:
        return json.loads(result)
    except (json.JSONDecodeError, TypeError):
        return {"playing": False}


@router.post("/api/music/control")
async def api_music_control(request: Request):
    """Control music playback: pause, resume, skip, prev, seek."""
    body = await request.json()
    action = body.get("action", "")
    value = body.get("value")

    action_map = {
        "pause": ("pause", {}),
        "resume": ("resume", {}),
        "skip": ("skip", {}),
        "prev": ("seek", {"position": 0.0}),
    }

    if action == "seek" and value is not None:
        result = await _music_call(request, "seek", {"position": float(value)})
    elif action in action_map:
        tool_name, args = action_map[action]
        result = await _music_call(request, tool_name, args)
    else:
        return JSONResponse({"error": f"Unknown action: {action}"}, status_code=400)

    return {"status": "ok", "result": result or ""}


@router.get("/api/music/pins")
async def api_music_pins(request: Request):
    """Return all pinned songs."""
    result = await _music_call(request, "list_pins")
    if result is None:
        return JSONResponse({"pins": {}})
    # Parse the player's get_pins() dict via the MCP list_pins tool
    # We need the raw dict, so call the player directly if possible
    from claw.config import PROJECT_ROOT, get_settings
    settings = get_settings()
    pins_path = PROJECT_ROOT / settings.youtube_music.pins_file
    if pins_path.exists():
        try:
            pins = json.loads(pins_path.read_text())
            return JSONResponse({"pins": pins})
        except (json.JSONDecodeError, OSError):
            pass
    return JSONResponse({"pins": {}})


@router.delete("/api/music/pins/{phrase:path}")
async def api_music_unpin(phrase: str, request: Request):
    """Remove a pinned song phrase."""
    result = await _music_call(request, "unpin_song", {"phrase": phrase})
    if result and "No pin found" in result:
        return JSONResponse({"error": result}, status_code=404)
    return {"status": "ok", "result": result or ""}


@router.delete("/api/google-accounts/{label}")
async def api_delete_google_account(label: str):
    """Remove a Google account from config."""
    from claw.config import get_settings, reload_settings

    settings = get_settings()
    if label not in settings.google_auth.accounts:
        return JSONResponse({"error": f"Account '{label}' not found"}, status_code=404)

    current = settings.model_dump()
    del current["google_auth"]["accounts"][label]

    new_settings = type(settings)(**current)
    new_settings.save_yaml()
    reload_settings()

    return {"status": "ok", "message": f"Account '{label}' removed"}


# ── Cron Jobs API ──────────────────────────────────────────────────────────

@router.get("/api/cron")
async def api_list_cron_jobs(request: Request):
    """List all cron jobs."""
    cron_manager = getattr(request.app.state, "cron_manager", None)
    if not cron_manager:
        return JSONResponse({"error": "Cron manager not initialized"}, status_code=503)
    return cron_manager.list_jobs()


@router.post("/api/cron")
async def api_create_cron_job(request: Request):
    """Create a new cron job."""
    cron_manager = getattr(request.app.state, "cron_manager", None)
    if not cron_manager:
        return JSONResponse({"error": "Cron manager not initialized"}, status_code=503)

    body = await request.json()
    try:
        job = cron_manager.create(
            name=body.get("name", ""),
            schedule=body.get("schedule", ""),
            job_type=body.get("type", "notification"),
            payload=body.get("payload"),
        )
        return {"status": "created", "job": job.to_dict()}
    except ValueError as e:
        return JSONResponse({"error": str(e)}, status_code=400)


@router.delete("/api/cron/{job_id}")
async def api_delete_cron_job(job_id: str, request: Request):
    """Delete a cron job."""
    cron_manager = getattr(request.app.state, "cron_manager", None)
    if not cron_manager:
        return JSONResponse({"error": "Cron manager not initialized"}, status_code=503)

    if cron_manager.delete(job_id):
        return {"status": "deleted"}
    return JSONResponse({"error": "Job not found"}, status_code=404)


# ── Inbox API ──────────────────────────────────────────────────────────────

@router.get("/api/inbox")
async def api_check_inbox(request: Request):
    """Check inbox messages."""
    inbox = getattr(request.app.state, "inbox", None)
    if not inbox:
        return JSONResponse({"error": "Inbox not initialized"}, status_code=503)

    unread_only = request.query_params.get("unread_only", "false").lower() == "true"
    messages = inbox.check(unread_only=unread_only)
    return {"messages": messages, "unread_count": inbox.unread_count}


@router.post("/api/inbox")
async def api_send_to_inbox(request: Request):
    """Send a message to the inbox."""
    inbox = getattr(request.app.state, "inbox", None)
    if not inbox:
        return JSONResponse({"error": "Inbox not initialized"}, status_code=503)

    body = await request.json()
    message = await inbox.send(
        sender=body.get("sender", "admin"),
        subject=body.get("subject", ""),
        body=body.get("body", ""),
        priority=body.get("priority", "normal"),
    )
    return {"status": "sent", "message": message}


@router.post("/api/inbox/{message_id}/read")
async def api_mark_inbox_read(message_id: str, request: Request):
    """Mark an inbox message as read."""
    inbox = getattr(request.app.state, "inbox", None)
    if not inbox:
        return JSONResponse({"error": "Inbox not initialized"}, status_code=503)

    if inbox.mark_read(message_id):
        return {"status": "ok"}
    return JSONResponse({"error": "Message not found"}, status_code=404)


@router.delete("/api/inbox")
async def api_clear_inbox(request: Request):
    """Clear inbox messages."""
    inbox = getattr(request.app.state, "inbox", None)
    if not inbox:
        return JSONResponse({"error": "Inbox not initialized"}, status_code=503)

    read_only = request.query_params.get("read_only", "true").lower() == "true"
    cleared = inbox.clear(read_only=read_only)
    return {"status": "ok", "cleared": cleared}


# ── Skills API ─────────────────────────────────────────────────────────────

@router.get("/skills", response_class=HTMLResponse)
async def skills_page(request: Request):
    """Skills management page."""
    templates = request.app.state.templates
    return templates.TemplateResponse("skills.html", {"request": request})


@router.get("/api/skills")
async def api_list_skills(request: Request):
    """List installed skills."""
    skill_manager = getattr(request.app.state, "skill_manager", None)
    if not skill_manager:
        return JSONResponse({"error": "Skill manager not initialized"}, status_code=503)
    return skill_manager.list_skills()


@router.post("/api/skills/install")
async def api_install_skill(request: Request):
    """Install a skill from a Git URL."""
    skill_manager = getattr(request.app.state, "skill_manager", None)
    if not skill_manager:
        return JSONResponse({"error": "Skill manager not initialized"}, status_code=503)

    body = await request.json()
    try:
        result = skill_manager.install(
            git_url=body.get("git_url", ""),
            name=body.get("name") or None,
        )
        return {"status": "installed", **result}
    except (ValueError, RuntimeError) as e:
        return JSONResponse({"error": str(e)}, status_code=400)


@router.delete("/api/skills/{name}")
async def api_uninstall_skill(name: str, request: Request):
    """Uninstall a skill."""
    skill_manager = getattr(request.app.state, "skill_manager", None)
    if not skill_manager:
        return JSONResponse({"error": "Skill manager not initialized"}, status_code=503)

    try:
        if skill_manager.uninstall(name):
            return {"status": "uninstalled"}
        return JSONResponse({"error": "Skill not found"}, status_code=404)
    except ValueError as e:
        return JSONResponse({"error": str(e)}, status_code=400)


# ── Pairing API ────────────────────────────────────────────────────────────

@router.post("/api/remote/pair")
async def api_generate_pairing_code(request: Request):
    """Generate a pairing code for device registration (authenticated)."""
    pairing_manager = getattr(request.app.state, "pairing_manager", None)
    if not pairing_manager:
        return JSONResponse({"error": "Pairing not initialized"}, status_code=503)

    body = await request.json()
    try:
        code = pairing_manager.generate(device_name=body.get("device_name", ""))
        return {"code": code, "ttl_seconds": 300}
    except ValueError as e:
        return JSONResponse({"error": str(e)}, status_code=429)


@router.post("/api/remote/pair/claim")
async def api_claim_pairing_code(request: Request):
    """Claim a pairing code to register a device (unauthenticated).

    The code itself acts as the authentication credential.
    """
    pairing_manager = getattr(request.app.state, "pairing_manager", None)
    if not pairing_manager:
        return JSONResponse({"error": "Pairing not initialized"}, status_code=503)

    body = await request.json()
    code = body.get("code", "")
    if not code:
        return JSONResponse({"error": "code is required"}, status_code=400)

    client_ip = request.client.host if request.client else ""

    try:
        pairing = pairing_manager.claim(code, client_ip=client_ip)
    except ValueError as e:
        return JSONResponse({"error": str(e)}, status_code=429)

    if pairing is None:
        return JSONResponse({"error": "Invalid or expired code"}, status_code=401)

    # Create a device using the existing device registration logic
    from claw.admin.api_key import create_device
    device_name = pairing.device_name or f"paired-{code[:4]}"
    try:
        device_info = create_device(device_name)
        return {"status": "paired", "device": device_info}
    except Exception as e:
        return JSONResponse({"error": f"Device creation failed: {e}"}, status_code=500)


# ── Bridge Status API ──────────────────────────────────────────────────────

@router.get("/api/bridge/status")
async def api_bridge_status(request: Request):
    """Get status of all messaging bridge adapters."""
    bridge_manager = getattr(request.app.state, "bridge_manager", None)
    if not bridge_manager:
        return {"adapters": {}, "running": False}
    return {"adapters": bridge_manager.get_status(), "running": bridge_manager.running}
