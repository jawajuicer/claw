"""Configuration system with YAML + env var loading and hot-reload support."""

from __future__ import annotations

import asyncio
import logging
from pathlib import Path
from typing import Any, Callable

import yaml
from pydantic import BaseModel, Field, field_validator
from pydantic_settings import BaseSettings, PydanticBaseSettingsSource, SettingsConfigDict

log = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
CONFIG_YAML = PROJECT_ROOT / "config.yaml"


# ── Section models ──────────────────────────────────────────────────────────

class AudioConfig(BaseModel):
    device_index: int | None = None
    sample_rate: int = 16000

    @field_validator("device_index", mode="before")
    @classmethod
    def empty_str_to_none(cls, v):
        if v == "" or v == "null":
            return None
        return v
    channels: int = 1
    block_size: int = 1280
    silence_threshold: float = 0.01
    silence_duration: float = 2.0
    max_record_seconds: int = 30
    chime_enabled: bool = True
    chime_volume: float = 0.5       # 0.0–1.0
    chime_frequency: int = 880      # Hz (base tone; chime plays 0.75x then 1x)
    chime_duration_ms: int = 200    # total duration in milliseconds


class WakeConfig(BaseModel):
    model_paths: list[str] = Field(default_factory=lambda: ["hey_jarvis_v0.1"])
    thresholds: dict[str, float] = Field(default_factory=dict)
    default_threshold: float = 0.5
    inference_framework: str = "onnx"
    custom_models_dir: str = "data/models/wake"


class WhisperConfig(BaseModel):
    model_size: str = "base"
    compute_type: str = "int8"
    language: str = "en"
    beam_size: int = 5


class TTSConfig(BaseModel):
    enabled: bool = True
    engine: str = "piper"  # "piper" or "fish_speech"
    piper_model: str = "data/models/tts/piper/en_US-lessac-medium.onnx"
    piper_speaker_id: int | None = None
    fish_speech_url: str = "http://localhost:8090"
    fish_speech_reference_audio: str = ""
    fish_speech_reference_text: str = ""
    speed: float = 1.0  # 0.5-2.0
    voice_loop_enabled: bool = True
    admin_chat_enabled: bool = True
    admin_chat_autoplay: bool = False  # auto-play voice responses in admin chat

    @field_validator("piper_speaker_id", mode="before")
    @classmethod
    def empty_str_to_none(cls, v):
        if v == "" or v == "null":
            return None
        return v


class LLMConfig(BaseModel):
    base_url: str = "http://localhost:11434/v1"
    api_key: str = "ollama"
    model: str = "qwen2.5:14b"
    temperature: float = 0.7
    max_tokens: int = 1024
    max_iterations: int = 5
    context_window: int = 32768
    system_prompt: str = (
        "You are The Claw, a helpful voice assistant. You are running locally "
        "on the user's machine. You have access to tools via MCP servers. "
        "Be concise in your responses since they will be spoken aloud."
    )


class MemoryConfig(BaseModel):
    chroma_path: str = "data/chromadb"
    embedding_model: str = "all-MiniLM-L6-v2"
    max_results: int = 5
    conversation_collection: str = "conversations"
    facts_collection: str = "facts"
    categories_collection: str = "categories"


class YouTubeMusicConfig(BaseModel):
    enabled: bool = False
    auth_file: str = "data/youtube_music/auth.json"
    history_file: str = "data/youtube_music/history.json"
    auto_radio: bool = True
    default_volume: int = 80
    max_history: int = 500
    max_search_results: int = 5


class NotesConfig(BaseModel):
    enabled: bool = True
    storage_dir: str = "data/notes"
    max_notes: int = 1000


class LocalCalendarConfig(BaseModel):
    enabled: bool = True
    storage_file: str = "data/calendar/local.json"


class GoogleAccountCalendarConfig(BaseModel):
    enabled: bool = False
    default_calendar: str = "primary"
    calendars: dict[str, str] = Field(default_factory=dict)
    timezone: str = "America/New_York"


class GoogleAccountGmailConfig(BaseModel):
    enabled: bool = False
    max_results: int = 10
    default_label: str = "INBOX"


class GoogleAccountConfig(BaseModel):
    email: str = ""
    token_file: str = ""
    calendar: GoogleAccountCalendarConfig = Field(default_factory=GoogleAccountCalendarConfig)
    gmail: GoogleAccountGmailConfig = Field(default_factory=GoogleAccountGmailConfig)
    youtube_music: bool = False


class GoogleAuthConfig(BaseModel):
    credentials_file: str = "data/google/credentials.json"
    scopes: list[str] = Field(default_factory=lambda: [
        "https://www.googleapis.com/auth/gmail.compose",
        "https://www.googleapis.com/auth/gmail.readonly",
        "https://www.googleapis.com/auth/calendar",
    ])
    accounts: dict[str, GoogleAccountConfig] = Field(default_factory=dict)


class MCPConfig(BaseModel):
    tools_dir: str = "mcp_tools"
    startup_timeout: int = 10
    enabled_servers: list[str] = Field(
        default_factory=lambda: [
            "youtube_music", "system_control", "notes",
            "local_calendar", "google_calendar", "gmail",
        ]
    )

    @field_validator("enabled_servers", mode="before")
    @classmethod
    def split_csv(cls, v):
        if isinstance(v, str):
            return [s.strip() for s in v.split(",") if s.strip()]
        return v


class ChatConfig(BaseModel):
    save_conversations: bool = False
    conversations_dir: str = "data/conversations"
    session_timeout: int = 86400  # seconds of inactivity before auto-rotating session (default 24h)


class AdminConfig(BaseModel):
    host: str = "0.0.0.0"
    port: int = 8080
    log_buffer_size: int = 1000


# ── Root settings ───────────────────────────────────────────────────────────

def _migrate_google_config(data: dict[str, Any]) -> dict[str, Any]:
    """Migrate old flat google_calendar/gmail config into google_auth.accounts."""
    ga = data.get("google_auth", {})
    old_cal = data.pop("google_calendar", None)
    old_gmail = data.pop("gmail", None)
    old_token = ga.pop("token_file", None)

    # Nothing to migrate
    if old_cal is None and old_gmail is None and old_token is None:
        return data

    # Already has accounts — don't overwrite
    if ga.get("accounts"):
        return data

    acct: dict[str, Any] = {}
    if old_token:
        acct["token_file"] = old_token
    elif old_cal or old_gmail:
        acct["token_file"] = "data/google/token.json"

    if old_cal and isinstance(old_cal, dict):
        acct["calendar"] = {
            "enabled": old_cal.get("enabled", False),
            "default_calendar": old_cal.get("default_calendar", "primary"),
            "calendars": old_cal.get("calendars", {}),
            "timezone": old_cal.get("timezone", "America/New_York"),
        }

    if old_gmail and isinstance(old_gmail, dict):
        acct["gmail"] = {
            "enabled": old_gmail.get("enabled", False),
            "max_results": old_gmail.get("max_results", 10),
            "default_label": old_gmail.get("default_label", "INBOX"),
        }

    if acct:
        ga.setdefault("accounts", {})["default"] = acct
        data["google_auth"] = ga

    return data


class YamlSettingsSource(PydanticBaseSettingsSource):
    """Custom settings source that reads from config.yaml."""

    def get_field_value(self, field, field_name):
        # Not used — we override __call__ directly
        return None, None, False

    def __call__(self) -> dict[str, Any]:
        if CONFIG_YAML.exists():
            with open(CONFIG_YAML) as f:
                data = yaml.safe_load(f) or {}
            data = _migrate_google_config(data)
            return data
        return {}


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_prefix="CLAW_",
        env_nested_delimiter="__",
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    audio: AudioConfig = Field(default_factory=AudioConfig)
    wake: WakeConfig = Field(default_factory=WakeConfig)
    whisper: WhisperConfig = Field(default_factory=WhisperConfig)
    tts: TTSConfig = Field(default_factory=TTSConfig)
    llm: LLMConfig = Field(default_factory=LLMConfig)
    memory: MemoryConfig = Field(default_factory=MemoryConfig)
    mcp: MCPConfig = Field(default_factory=MCPConfig)
    youtube_music: YouTubeMusicConfig = Field(default_factory=YouTubeMusicConfig)
    notes: NotesConfig = Field(default_factory=NotesConfig)
    local_calendar: LocalCalendarConfig = Field(default_factory=LocalCalendarConfig)
    google_auth: GoogleAuthConfig = Field(default_factory=GoogleAuthConfig)
    chat: ChatConfig = Field(default_factory=ChatConfig)
    admin: AdminConfig = Field(default_factory=AdminConfig)

    @classmethod
    def settings_customise_sources(
        cls,
        settings_cls: type[BaseSettings],
        init_settings: PydanticBaseSettingsSource,
        env_settings: PydanticBaseSettingsSource,
        dotenv_settings: PydanticBaseSettingsSource,
        file_secret_settings: PydanticBaseSettingsSource,
    ) -> tuple[PydanticBaseSettingsSource, ...]:
        """Priority: init kwargs > env vars > .env > config.yaml > defaults."""
        return (
            init_settings,
            env_settings,
            dotenv_settings,
            YamlSettingsSource(settings_cls),
            file_secret_settings,
        )

    @classmethod
    def load(cls) -> Settings:
        """Load settings with priority: env vars > .env > config.yaml > defaults."""
        return cls()

    def save_yaml(self) -> None:
        """Write current settings back to config.yaml."""
        data = self.model_dump()
        with open(CONFIG_YAML, "w") as f:
            yaml.dump(data, f, default_flow_style=False, sort_keys=False)
        log.info("Settings saved to %s", CONFIG_YAML)


# ── Singleton + hot-reload ──────────────────────────────────────────────────

_settings: Settings | None = None
_reload_callbacks: list[Callable[[Settings], Any]] = []


def get_settings() -> Settings:
    global _settings
    if _settings is None:
        _settings = Settings.load()
    return _settings


def reload_settings() -> Settings:
    """Force-reload settings from disk."""
    global _settings
    _settings = Settings.load()
    log.info("Settings reloaded")
    for cb in _reload_callbacks:
        try:
            cb(_settings)
        except Exception:
            log.exception("Error in reload callback %s", cb)
    return _settings


def on_reload(callback: Callable[[Settings], Any]) -> None:
    """Register a callback invoked after settings reload."""
    _reload_callbacks.append(callback)


async def watch_config() -> None:
    """Watch config.yaml for changes and auto-reload. Run as background task."""
    from watchfiles import awatch

    log.info("Watching %s for changes", CONFIG_YAML)
    async for _changes in awatch(CONFIG_YAML):
        await asyncio.sleep(0.3)  # debounce: wait for writes to settle
        log.info("Config file changed, reloading...")
        reload_settings()
