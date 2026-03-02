"""Admin panel API and page routes."""

from __future__ import annotations

import asyncio
import json
import logging
from collections import deque
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import TYPE_CHECKING
from uuid import uuid4

from fastapi import APIRouter, Request
from fastapi.responses import HTMLResponse, JSONResponse, Response
from sse_starlette.sse import EventSourceResponse

if TYPE_CHECKING:
    from claw.admin.sse import StatusBroadcaster
    from claw.config import Settings

log = logging.getLogger(__name__)

router = APIRouter()


# ── Log buffer ──────────────────────────────────────────────────────────────

class LogBuffer(logging.Handler):
    """Ring-buffer logging handler for the admin log viewer."""

    def __init__(self, maxlen: int = 1000) -> None:
        super().__init__()
        self.records: deque[str] = deque(maxlen=maxlen)
        self.setFormatter(logging.Formatter(
            "%(asctime)s %(levelname)-7s %(name)s: %(message)s",
            datefmt="%H:%M:%S",
        ))

    def emit(self, record: logging.LogRecord) -> None:
        self.records.append(self.format(record))


# ── Page routes ─────────────────────────────────────────────────────────────

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
    from claw.config import get_settings
    settings = get_settings()
    return request.app.state.templates.TemplateResponse(
        "settings.html",
        {"request": request, "settings": settings.model_dump()},
    )


@router.get("/logs", response_class=HTMLResponse)
async def logs_page(request: Request):
    log_buffer: LogBuffer = request.app.state.log_buffer
    return request.app.state.templates.TemplateResponse(
        "logs.html",
        {"request": request, "logs": list(log_buffer.records)},
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
    if not message:
        return JSONResponse({"error": "Empty message"}, status_code=400)

    agent = request.app.state.agent
    registry = request.app.state.registry
    broadcaster: StatusBroadcaster = request.app.state.broadcaster

    if agent is None:
        return {"role": "assistant", "content": "Agent not available.", "tools_used": []}

    # Serialize chat requests to protect shared session state
    chat_lock: asyncio.Lock = request.app.state.chat_lock

    async with chat_lock:
        msg_count_before = len(agent.session.messages) if agent.session else 0

        await broadcaster.update_state("processing")
        await broadcaster.update_transcription(message)

        tools = registry.get_openai_tools() if registry else None
        try:
            response = await agent.process_utterance(message, tools=tools or None)
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


# ── Conversation persistence ───────────────────────────────────────────────

def _conversations_dir() -> Path:
    from claw.config import PROJECT_ROOT, get_settings
    rel = get_settings().chat.conversations_dir.lstrip("/")
    d = PROJECT_ROOT / rel
    d.mkdir(parents=True, exist_ok=True)
    return d


def _valid_convo_id(convo_id: str) -> bool:
    return bool(re.fullmatch(r"[a-zA-Z0-9_-]{1,64}", convo_id))


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


@router.post("/api/settings")
async def api_update_settings(request: Request):
    """Update settings from the admin panel."""
    from claw.config import get_settings, reload_settings
    body = await request.json()
    settings = get_settings()

    # Merge updates into current settings
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
                current[section].update(values)

    # Save and reload
    new_settings = type(settings)(**current)
    new_settings.save_yaml()
    reload_settings()

    return {"status": "ok", "message": "Settings updated and reloaded"}


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
