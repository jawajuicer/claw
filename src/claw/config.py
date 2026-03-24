"""Configuration system with YAML + env var loading and hot-reload support."""

from __future__ import annotations

import asyncio
import logging
import os
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
    output_device_index: int | None = None
    sample_rate: int = 16000

    @field_validator("device_index", "output_device_index", mode="before")
    @classmethod
    def empty_str_to_none(cls, v):
        if v == "" or v == "null":
            return None
        return v
    channels: int = 1
    block_size: int = 1280
    silence_duration: float = 1.0
    agc_silence_threshold: float = 0.05  # post-AGC RMS below this = silence
    agc_gain_ceiling: float = 3.5        # AGC gain above this = boosted noise, not speech
    vad_threshold: float = 0.5           # Silero VAD speech probability threshold (0.0–1.0)
    max_record_seconds: int = 30
    chime_enabled: bool = True
    chime_volume: float = 0.5       # 0.0–1.0
    chime_frequency: int = 880      # Hz (base tone; chime plays 0.75x then 1x)
    chime_duration_ms: int = 200    # total duration in milliseconds

    @field_validator("chime_volume")
    @classmethod
    def clamp_chime_volume(cls, v):
        return max(0.0, min(1.0, float(v)))


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

    @field_validator("beam_size")
    @classmethod
    def beam_size_positive(cls, v):
        if v < 1:
            raise ValueError("beam_size must be >= 1")
        return v


class TTSConfig(BaseModel):
    enabled: bool = True
    engine: str = "piper"  # "piper" or "fish_speech"
    piper_model: str = "data/models/tts/piper/en_US-lessac-medium.onnx"
    piper_speaker_id: int | None = None
    fish_speech_url: str = "http://localhost:8090"
    fish_speech_reference_audio: str = ""
    fish_speech_reference_text: str = ""
    speed: float = 1.0  # 0.5-2.0

    @field_validator("speed")
    @classmethod
    def clamp_speed(cls, v):
        return max(0.5, min(2.0, float(v)))
    voice_loop_enabled: bool = True
    admin_chat_enabled: bool = True
    admin_chat_autoplay: bool = False  # auto-play voice responses in admin chat
    music_announcement: str = "before"  # "before" = pause music, speak, resume; "none" = skip TTS for music

    @field_validator("piper_speaker_id", mode="before")
    @classmethod
    def empty_str_to_none(cls, v):
        if v == "" or v == "null":
            return None
        return v

    @field_validator("music_announcement", mode="before")
    @classmethod
    def validate_music_announcement(cls, v):
        allowed = {"before", "none"}
        if v not in allowed:
            raise ValueError(f"music_announcement must be one of {allowed}, got {v!r}")
        return v


class LLMConfig(BaseModel):
    base_url: str = "http://localhost:8081/v1"
    api_key: str = "no-key"
    model: str = "qwen3.5:4b"
    fast_model: str | None = None  # lighter model for conversational (no-tool) queries
    temperature: float = 0.7
    max_tokens: int = 1024
    max_iterations: int = 5
    context_window: int = 32768
    timeout: int = 45  # seconds per LLM call
    thinking: str = "off"  # "off", "low", "medium", "high"
    compact_threshold: float = 0.8  # auto-compact at this % of context_window (0 = disabled)
    compact_keep_recent: int = 6    # messages to keep verbatim during compaction

    @field_validator("thinking", mode="before")
    @classmethod
    def normalize_thinking(cls, v):
        if v is True:
            return "medium"
        if v is False:
            return "off"
        if isinstance(v, str) and v.lower() in ("off", "low", "medium", "high"):
            return v.lower()
        return "off"

    system_prompt: str = (
        "You are The Claw, a helpful AI assistant running locally on the user's machine. "
        "You have access to tools via MCP servers. You interact through voice and messaging (Signal). "
        "You have memory of past conversations, group chats, and user facts. "
        "Be CONCISE. Short answers. No fluff. Never ask 'anything else?' or 'can I help with something else?' "
        "ACT FIRST, ask questions only when truly ambiguous. When told 'nevermind' or 'forget it', "
        "just acknowledge briefly. DO NOT OVERTHINK. DO NOT over-explain."
    )


class CloudProvider(BaseModel):
    """Configuration for a single cloud LLM provider."""
    enabled: bool = False
    base_url: str = ""
    model: str = ""
    api_key_secret: str = ""  # name in SecretStore (e.g., "claude_api_key")
    max_tokens: int = 4096
    temperature: float = 0.7
    timeout: int = 60


class CloudLLMConfig(BaseModel):
    """Optional cloud LLM backends (Claude, Gemini) as alternatives to local inference."""
    active_provider: str = "local"  # "local", "claude", "claude-cli", "gemini"
    escalation_mode: str = "auto"  # "auto" (silent), "ask" (confirm), "off"
    escalation_provider: str = "auto"  # "auto" (first available), "claude", "gemini", etc.
    providers: dict[str, CloudProvider] = Field(default_factory=lambda: {
        "claude": CloudProvider(
            base_url="https://api.anthropic.com/v1/",
            model="claude-sonnet-4-6",
            api_key_secret="claude_api_key",
            max_tokens=4096,
        ),
        "claude-cli": CloudProvider(
            model="sonnet",
            timeout=120,
        ),
        "gemini": CloudProvider(
            base_url="https://generativelanguage.googleapis.com/v1beta/openai/",
            model="gemini-2.5-flash",
            api_key_secret="gemini_api_key",
            max_tokens=4096,
        ),
    })
    failover_to_local: bool = True  # if cloud fails, fall back to local
    failover_chain: list[str] = Field(default_factory=lambda: [])  # e.g. ["claude", "local"]; empty = use failover_to_local

    @field_validator("failover_chain", mode="before")
    @classmethod
    def parse_failover_chain(cls, v):
        if isinstance(v, str):
            return [s.strip() for s in v.split(",") if s.strip()]
        return v


class ComputeConfig(BaseModel):
    backend: str = "cpu"       # "cpu" | "cuda" | "rocm" | "vulkan"
    gpu_layers: int = 99       # --n-gpu-layers value (0 for CPU)
    speculative: bool = False  # enable speculative decoding with draft model
    speculative_model: str = ""  # path to draft model GGUF (e.g. "models/Qwen3.5-0.8B.gguf")
    speculative_main_model: str = ""  # path to main model GGUF (selected when speculative enabled)
    speculative_draft_max: int = 16  # max tokens to draft per step


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
    pins_file: str = "data/youtube_music/pins.json"
    auto_radio: bool = True
    default_volume: int = 80
    max_history: int = 500
    max_search_results: int = 5


class NotesConfig(BaseModel):
    enabled: bool = True
    storage_dir: str = "data/notes"
    max_notes: int = 0  # 0 = unlimited


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
        "https://www.googleapis.com/auth/contacts.readonly",
        "https://www.googleapis.com/auth/directory.readonly",
    ])
    accounts: dict[str, GoogleAccountConfig] = Field(default_factory=dict)


class WeatherConfig(BaseModel):
    api_key: str = ""  # OpenWeatherMap API key (free tier)
    default_location: str = ""  # User's location, e.g. "Akron, OH" — set in settings


class GeminiConfig(BaseModel):
    enabled: bool = False
    model: str = "gemini-2.5-flash"
    pro_model: str = "gemini-2.5-pro"
    temperature: float = 0.7
    max_output_tokens: int = 2048
    web_search: bool = True
    document_analysis: bool = True
    image_understanding: bool = True
    reasoning_fallback: bool = False
    escalation_mode: str = "ask"  # "ask", "auto", or "off"
    daily_request_limit: int = 200
    grounding_daily_limit: int = 400
    log_requests: bool = True
    log_dir: str = "data/gemini/logs"


class SchedulerConfig(BaseModel):
    enabled: bool = True
    poll_interval: int = 30  # seconds between checks
    announce_tts: bool = True  # speak reminders via TTS
    cron_storage: str = "data/scheduler/cron_jobs.json"


# ── Channel profiles ───────────────────────────────────────────────────────

class ChannelProfile(BaseModel):
    memory_scope: str = "shared"
    system_prompt_addon: str = ""
    tools_enabled: bool = True


class ChannelProfilesConfig(BaseModel):
    profiles: dict[str, ChannelProfile] = Field(
        default_factory=lambda: {"default": ChannelProfile()}
    )
    default_profile: str = "default"


# ── Bridge configs ─────────────────────────────────────────────────────────

class TelegramBridgeConfig(BaseModel):
    enabled: bool = False
    token_secret: str = "telegram_bot_token"
    mode: str = "polling"  # "polling" or "webhook"
    webhook_secret: str = ""
    allowed_users: list[str] = Field(default_factory=list)

    @field_validator("mode", mode="before")
    @classmethod
    def validate_mode(cls, v):
        if v not in ("polling", "webhook"):
            raise ValueError(f"mode must be 'polling' or 'webhook', got {v!r}")
        return v


class DiscordBridgeConfig(BaseModel):
    enabled: bool = False
    token_secret: str = "discord_bot_token"
    allowed_channels: list[str] = Field(default_factory=list)
    allowed_users: list[str] = Field(default_factory=list)
    profile: str = ""
    channel_profiles: dict[str, str] = Field(default_factory=dict)
    group_sessions: bool = True


class SlackBridgeConfig(BaseModel):
    enabled: bool = False
    bot_token_secret: str = "slack_bot_token"
    app_token_secret: str = "slack_app_token"
    allowed_channels: list[str] = Field(default_factory=list)
    allowed_users: list[str] = Field(default_factory=list)


class TwilioBridgeConfig(BaseModel):
    enabled: bool = False
    account_sid_secret: str = "twilio_account_sid"
    auth_token_secret: str = "twilio_auth_token"
    from_number: str = ""
    allowed_numbers: list[str] = Field(default_factory=list)


class MatrixBridgeConfig(BaseModel):
    enabled: bool = False
    homeserver: str = "https://matrix.org"
    user_id: str = ""
    token_secret: str = "matrix_access_token"
    allowed_rooms: list[str] = Field(default_factory=list)
    allowed_users: list[str] = Field(default_factory=list)


class IRCBridgeConfig(BaseModel):
    enabled: bool = False
    server: str = "irc.libera.chat"
    port: int = 6697
    nickname: str = "claw-bot"
    channels: list[str] = Field(default_factory=list)
    password_secret: str = ""
    use_tls: bool = True
    allowed_users: list[str] = Field(default_factory=list)


class SignalBridgeConfig(BaseModel):
    enabled: bool = False
    api_url: str = "http://localhost:8080"  # signal-cli-rest-api endpoint
    phone_number: str = ""  # registered Signal phone number (E.164 format)
    bot_name: str = "Claw"  # name to match for mentions in groups
    allowed_groups: list[str] = Field(default_factory=list)  # group IDs (empty = all)
    allowed_users: list[str] = Field(default_factory=list)  # phone numbers for DMs (empty = all)
    admin_users: list[str] = Field(default_factory=list)  # UUIDs or phone numbers that can execute tools (empty = all)
    observe_groups: bool = True  # passively ingest non-mention group messages into memory
    poll_interval: float = 1.0  # seconds between receive polls
    profile: str = ""
    channel_profiles: dict[str, str] = Field(default_factory=dict)
    group_sessions: bool = True


class BridgesConfig(BaseModel):
    telegram: TelegramBridgeConfig = Field(default_factory=TelegramBridgeConfig)
    discord: DiscordBridgeConfig = Field(default_factory=DiscordBridgeConfig)
    slack: SlackBridgeConfig = Field(default_factory=SlackBridgeConfig)
    twilio: TwilioBridgeConfig = Field(default_factory=TwilioBridgeConfig)
    matrix: MatrixBridgeConfig = Field(default_factory=MatrixBridgeConfig)
    irc: IRCBridgeConfig = Field(default_factory=IRCBridgeConfig)
    signal: SignalBridgeConfig = Field(default_factory=SignalBridgeConfig)


class BrowserConfig(BaseModel):
    enabled: bool = False
    max_page_load_ms: int = 30000
    max_content_chars: int = 50000
    blocked_domains: list[str] = Field(default_factory=lambda: [
        "*.bank.*", "*.gov", "localhost", "127.0.0.*",
    ])


class UsageConfig(BaseModel):
    enabled: bool = True
    persist_file: str = "data/usage.json"
    persist_interval: int = 60  # seconds


class MCPConfig(BaseModel):
    tools_dir: str = "mcp_tools"
    startup_timeout: int = 10
    enabled_servers: list[str] = Field(
        default_factory=lambda: [
            "youtube_music", "system_control", "notes",
            "local_calendar", "google_calendar", "gmail",
        ]
    )
    voice_servers: list[str] = Field(
        default_factory=lambda: ["system_control"],
    )

    @field_validator("enabled_servers", mode="before")
    @classmethod
    def split_csv(cls, v):
        if isinstance(v, str):
            return [s.strip() for s in v.split(",") if s.strip()]
        return v


class WebhookConfig(BaseModel):
    enabled: bool = False
    secret: str = ""  # HMAC-SHA256 secret for signature verification; empty = no verification
    allowed_events: list[str] = Field(
        default_factory=lambda: ["message", "reminder", "notification"]
    )

    @field_validator("allowed_events", mode="before")
    @classmethod
    def split_csv(cls, v):
        if isinstance(v, str):
            return [s.strip() for s in v.split(",") if s.strip()]
        return v


class ChatConfig(BaseModel):
    save_conversations: bool = False
    conversations_dir: str = "data/conversations"
    session_timeout: int = 86400  # seconds of inactivity before auto-rotating session (default 24h)


class RemoteConfig(BaseModel):
    enabled: bool = False
    audio_output: str = "phone"  # "phone" = audio plays on mobile device; "computer" = plays on server speakers
    wg_interface: str = "wg0"
    wg_subnet: str = "10.10.0"
    wg_port: int = 51820
    wg_endpoint: str = ""  # Public IP or hostname — set after running setup-wireguard.sh

    @field_validator("audio_output", mode="before")
    @classmethod
    def validate_audio_output(cls, v):
        allowed = {"phone", "computer"}
        if v not in allowed:
            raise ValueError(f"audio_output must be one of {allowed}, got {v!r}")
        return v


class AdminConfig(BaseModel):
    host: str = "0.0.0.0"
    port: int = 8080
    log_buffer_size: int = 1000
    log_format: str = "text"  # "text" or "json"
    log_file: str = ""  # path to log file (RotatingFileHandler); empty = stdout only
    auth_enabled: bool = True
    auth_username: str = "admin"


class ClaudeRelayConfig(BaseModel):
    """Claude Code relay — SSH to dev machine for coding via any interface."""
    enabled: bool = False
    host: str = ""              # dev machine IP (e.g. "10.1.92.101")
    user: str = ""              # SSH user on dev machine
    password: str = ""          # SSH password
    project_dir: str = ""       # project directory on dev machine
    skip_permissions: bool = False  # pass --dangerously-skip-permissions to claude
    timeout_initial: float = 300    # seconds for first message (no session yet)
    timeout: float = 180            # seconds for subsequent messages
    max_retries: int = 2            # retries on transient failures (0 = disabled)
    control_persist: int = 300      # SSH ControlPersist seconds (0 = disable)


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
    cloud_llm: CloudLLMConfig = Field(default_factory=CloudLLMConfig)
    compute: ComputeConfig = Field(default_factory=ComputeConfig)
    memory: MemoryConfig = Field(default_factory=MemoryConfig)
    mcp: MCPConfig = Field(default_factory=MCPConfig)
    scheduler: SchedulerConfig = Field(default_factory=SchedulerConfig)
    weather: WeatherConfig = Field(default_factory=WeatherConfig)
    gemini: GeminiConfig = Field(default_factory=GeminiConfig)
    youtube_music: YouTubeMusicConfig = Field(default_factory=YouTubeMusicConfig)
    notes: NotesConfig = Field(default_factory=NotesConfig)
    local_calendar: LocalCalendarConfig = Field(default_factory=LocalCalendarConfig)
    google_auth: GoogleAuthConfig = Field(default_factory=GoogleAuthConfig)
    remote: RemoteConfig = Field(default_factory=RemoteConfig)
    webhook: WebhookConfig = Field(default_factory=WebhookConfig)
    usage: UsageConfig = Field(default_factory=UsageConfig)
    chat: ChatConfig = Field(default_factory=ChatConfig)
    admin: AdminConfig = Field(default_factory=AdminConfig)
    bridges: BridgesConfig = Field(default_factory=BridgesConfig)
    browser: BrowserConfig = Field(default_factory=BrowserConfig)
    channel_profiles: ChannelProfilesConfig = Field(default_factory=ChannelProfilesConfig)
    claude_relay: ClaudeRelayConfig = Field(default_factory=ClaudeRelayConfig)

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
        """Write current settings back to config.yaml atomically with restricted permissions."""
        data = self.model_dump()
        tmp = CONFIG_YAML.with_suffix(".tmp")
        fd = os.open(str(tmp), os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
        with os.fdopen(fd, "w") as f:
            yaml.dump(data, f, default_flow_style=False, sort_keys=False)
        tmp.replace(CONFIG_YAML)
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
