"""Admin panel API and page routes."""

from __future__ import annotations

import asyncio
import json
import logging
import os
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
            "settings": settings.model_dump(),
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


@router.get("/api/settings")
async def api_get_settings():
    """Return current settings as JSON (used by frontend polling)."""
    from claw.config import get_settings
    return get_settings().model_dump()


@router.post("/api/settings")
async def api_update_settings(request: Request):
    """Update settings from the admin panel."""
    from pydantic import ValidationError
    from claw.config import get_settings, reload_settings

    body = await request.json()
    settings = get_settings()
    old_model = settings.llm.model

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


# ── Google OAuth flow ────────────────────────────────────────────────────────

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
    if label in settings.google_auth.accounts:
        return RedirectResponse(f"/settings?error=Account+'{label}'+already+exists.", status_code=302)

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

    # Re-check label hasn't been claimed by a concurrent flow
    if label in settings.google_auth.accounts:
        _pending_auth.pop(state, None)
        return RedirectResponse("/settings?error=Account+already+exists.+Another+linking+completed+first.", status_code=302)

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

    log.info("Google account '%s' (%s) linked successfully", label, email or "unknown email")
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


# ── Compute backend endpoints ────────────────────────────────────────────────

_compute_build_task: asyncio.Task | None = None


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
    global _compute_build_task
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

    _compute_build_task = asyncio.create_task(_run_switch())
    request.app.state.bg_tasks.add(_compute_build_task)
    _compute_build_task.add_done_callback(request.app.state.bg_tasks.discard)

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
    """Call a YouTube Music MCP tool via the registry. Returns result string or None."""
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
