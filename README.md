# The Claw

A local-first voice AI agent with MCP tool calling. Privacy-first: zero cloud audio processing. All speech recognition, wake word detection, and language model inference run entirely on your hardware.

## Features

- **Voice-activated** -- custom "Hey Claw" wake word with openWakeWord (ONNX)
- **Local speech-to-text** -- faster-whisper for on-device transcription
- **Local LLM** -- llama-swap + llama.cpp inference via the OpenAI SDK (any GGUF model: qwen, llama, mistral, etc.)
- **Text-to-speech** -- Piper TTS (ONNX) with optional Fish Speech support
- **MCP tool calling** -- extensible tool servers discovered automatically from `mcp_tools/`
- **Conversational memory** -- ChromaDB vector store with automatic fact extraction
- **Web admin panel** -- htmx + Pico CSS dashboard with live status, chat, settings, and logs
- **Config hot-reload** -- edit `config.yaml` and changes apply immediately (no restart)
- **Systemd service** -- runs as a user service under PipeWire

## Architecture

```
                         +------------------+
                         |   Admin Panel    |
                         | (FastAPI + htmx) |
                         +--------+---------+
                                  |
    +-----------------------------+-----------------------------+
    |                        Orchestrator                       |
    |                        (Claw class)                       |
    +-----+----------+----------+-----------+----------+--------+
          |          |          |           |          |
    +-----+--+ +----+----+ +---+----+ +----+---+ +---+--------+
    | Audio   | | Agent   | | Memory | | MCP    | | TTS        |
    | Pipeline| | Core    | | Engine | | Handler| | Manager    |
    +----+----+ +----+----+ +---+----+ +----+---+ +---+--------+
         |           |          |           |          |
    sounddevice  OpenAI SDK  ChromaDB    stdio     piper-tts
    openWakeWord  llama.cpp  sentence-  sessions   (or Fish
    faster-whisper           transformers           Speech)
                                          |
                              +-----------+-----------+
                              |     MCP Tool Servers  |
                              +-----------------------+
                              | youtube_music         |
                              | weather               |
                              | gmail                 |
                              | google_calendar       |
                              | local_calendar        |
                              | notes                 |
                              | system_control        |
                              +-----------------------+
```

### Pipeline Flow

1. **Wake word detection** -- sounddevice streams audio to openWakeWord, which listens for "Hey Claw" (or other configured wake words)
2. **Audio capture** -- once triggered, records speech until silence is detected
3. **Transcription** -- faster-whisper converts audio to text locally
4. **Agent processing** -- the LLM (via llama-swap/llama.cpp) processes the query, optionally calling MCP tools in a bounded loop (max 5 iterations)
5. **Response** -- the answer is spoken via TTS and displayed in the admin panel
6. **Memory** -- conversations are stored in ChromaDB; facts are extracted in the background

## Quick Start

### Prerequisites

- Ubuntu 24.04 (or compatible Linux with PipeWire)
- Python 3.12+
- llama-swap + llama.cpp (installed by `install.sh`)
- A USB microphone (or any PipeWire-compatible audio input)

### Install

```bash
git clone <repo-url> ~/claw
cd ~/claw
bash install.sh
```

The installer handles system dependencies (portaudio, ffmpeg, mpv), creates a Python virtual environment, installs the project, downloads the default Piper TTS voice model, and sets up a systemd user service.

### Configure

Copy and edit the environment file:

```bash
cp .env.example .env
# Edit .env or config.yaml with your preferences
```

Download a model (GGUF format from Hugging Face):

```bash
huggingface-cli download unsloth/Qwen2.5-14B-Instruct-GGUF --include "Qwen2.5-14B-Instruct-Q4_K_M.gguf" --local-dir ~/models
```

### Run

**As a systemd service:**

```bash
systemctl --user start claw
systemctl --user enable claw    # auto-start on boot
journalctl --user -u claw -f   # follow logs
```

**Directly:**

```bash
source ~/claw/.venv/bin/activate

# Full mode: voice + admin panel
python -m claw --mode both

# Text-only mode (no microphone needed)
python -m claw --mode cli

# Voice-only mode
python -m claw --mode voice
```

The admin panel is available at `http://localhost:8080` in all modes.

## Running Modes

| Mode | Voice Loop | Admin Panel | Use Case |
|------|-----------|-------------|----------|
| `both` (default) | Yes | Yes | Production -- full voice assistant |
| `voice` | Yes | Yes | Same as `both` |
| `cli` | No | Yes | Development and testing without a microphone |

## MCP Tools

Tools are auto-discovered from subdirectories of `mcp_tools/`. Each tool is a standalone FastMCP server launched as a subprocess with stdio transport.

| Tool | Description | Requires |
|------|-------------|----------|
| **system_control** | Time, uptime, disk/memory usage | Nothing (always available) |
| **weather** | Current conditions and forecasts | OpenWeatherMap API key (optional -- falls back to Open-Meteo) |
| **notes** | Create, search, and manage notes and reminders | Nothing (local JSON storage) |
| **local_calendar** | Private on-device calendar | Nothing (local JSON storage) |
| **youtube_music** | Search, play, queue music via mpv | Browser cookie auth + mpv |
| **google_calendar** | Read and manage Google Calendar events | Google OAuth credentials |
| **gmail** | Read, search, and send emails | Google OAuth credentials |

Enable or disable tools in `config.yaml` under `mcp.enabled_servers` or through the admin panel Settings page.

## Configuration

The Claw uses a layered configuration system with this priority (highest first):

1. **Environment variables** -- prefixed with `CLAW_`, nested with `__` (e.g., `CLAW_LLM__MODEL=qwen2.5:14b`)
2. **`.env` file** -- same format as environment variables
3. **`config.yaml`** -- the primary configuration file
4. **Defaults** -- built-in defaults from the Pydantic settings models

### Key Configuration Sections

| Section | What it controls |
|---------|-----------------|
| `audio` | Microphone device, sample rate, silence detection, chime settings |
| `wake` | Wake word model paths, detection thresholds, inference framework |
| `whisper` | STT model size, compute type, language |
| `tts` | TTS engine (piper/fish_speech), voice model, speed, enable/disable |
| `llm` | LLM server URL, model name, temperature, max tokens, context window, system prompt |
| `memory` | ChromaDB path, embedding model, result limits |
| `mcp` | Tools directory, enabled servers, startup timeout |
| `youtube_music` | Auth file, playback defaults, history |
| `weather` | API key, default location |
| `notes` | Storage directory, max notes limit |
| `local_calendar` | Storage file path |
| `google_auth` | OAuth credentials, per-account settings for Gmail and Google Calendar |
| `chat` | Conversation saving, session timeout |
| `admin` | Host, port, log buffer size |

### Hot-Reload

The Claw watches `config.yaml` for changes using `watchfiles`. When the file is modified (via the admin panel Settings page or a text editor), settings are automatically reloaded without restarting the service. This applies to most settings including wake word thresholds, LLM parameters, and TTS options.

Note: MCP tool server processes cache their config at startup. To apply config changes to MCP tools, restart the service with `systemctl --user restart claw`.

## Development

### Setup

```bash
pip install -e ".[dev]"
```

### Tests

```bash
pytest
```

### Linting

```bash
ruff check src/ mcp_tools/
ruff format src/ mcp_tools/
```

### Project Structure

```
claw/
  src/claw/
    __main__.py          # CLI entry point
    main.py              # Orchestrator (Claw class)
    config.py            # Settings with YAML + env loading
    admin/               # FastAPI admin panel (htmx + Pico CSS + SSE)
    agent_core/          # LLM client and agent loop
    audio_pipeline/      # Capture, wake word, transcriber, chime, TTS
    mcp_handler/         # MCP client, registry, and tool router
    memory_engine/       # ChromaDB store and retriever
  mcp_tools/
    system_control/      # Time, uptime, disk, memory
    weather/             # OpenWeatherMap + Open-Meteo fallback
    notes/               # Notes and reminders (local JSON)
    local_calendar/      # Private calendar (local JSON)
    youtube_music/       # Music playback via ytmusicapi + mpv
    google_calendar/     # Google Calendar API
    gmail/               # Gmail API
  training/              # Wake word training scripts and config
  data/                  # Runtime data (ChromaDB, models, notes, etc.)
  config.yaml            # Primary configuration file
  install.sh             # Installation script
  claw.service           # Systemd user service unit
```

## Further Documentation

- [Deployment Guide](docs/DEPLOYMENT.md) -- system requirements, install walkthrough, systemd management
- [Troubleshooting](docs/TROUBLESHOOTING.md) -- common issues and solutions
- [Wake Word Training](docs/TRAINING.md) -- training a custom "Hey Claw" wake word model

## License

TBD
