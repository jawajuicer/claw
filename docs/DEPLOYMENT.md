# Deployment Guide

This guide covers deploying The Claw on a dedicated machine as a always-on voice assistant.

## System Requirements

| Component | Requirement |
|-----------|-------------|
| **OS** | Ubuntu 24.04 LTS (or compatible Linux with PipeWire) |
| **Python** | 3.12+ |
| **RAM** | 8 GB minimum, 16+ GB recommended (LLM + embeddings + ChromaDB) |
| **CPU** | Modern multi-core (AMD Ryzen or Intel, 4+ cores recommended) |
| **Audio** | USB microphone or any PipeWire-compatible audio input |
| **Audio server** | PipeWire (default on Ubuntu 24.04) |
| **Network** | Internet for weather, Google services, YouTube Music |

### Software Prerequisites

- **llama-swap + llama.cpp** -- installed automatically by `install.sh`; provides the LLM inference backend
- **PipeWire** -- ships with Ubuntu 24.04 by default; runs as a user service

## Installation

### 1. Clone the Repository

```bash
git clone <repo-url> ~/claw
cd ~/claw
```

### 2. Run the Installer

```bash
bash install.sh
```

The installer performs these steps:

1. **System dependencies** -- installs `python3.12-venv`, `python3-dev`, `libportaudio2`, `portaudio19-dev`, `ffmpeg`, `mpv`, and `libmpv-dev` via apt
2. **Python virtual environment** -- creates `.venv/` in the project directory
3. **Core install** -- `pip install -e .` (editable install of the claw package)
4. **TTS support** (optional) -- installs `piper-tts` and downloads the default Piper voice model (`en_US-lessac-medium`)
5. **YouTube Music support** (optional) -- installs `ytmusicapi` and `yt-dlp`
6. **Google services support** (optional) -- installs `google-api-python-client`, `google-auth-oauthlib`, `google-auth`
7. **Data directories** -- creates `data/` subdirectories for ChromaDB, models, logs, notes, calendar, etc.
8. **Environment file** -- copies `.env.example` to `.env` if it does not exist
9. **Systemd service** -- installs `claw.service` to `~/.config/systemd/user/` and enables linger for boot-time start

### 3. Download a Model

Download a GGUF model from Hugging Face into `~/models/`:

```bash
pip install huggingface-hub
huggingface-cli download unsloth/Qwen2.5-14B-Instruct-GGUF \
    --include "Qwen2.5-14B-Instruct-Q4_K_M.gguf" --local-dir ~/models
```

Then add the model to `~/claw/llama-swap-config.yaml` and restart llama-swap. Smaller models (0.5B-4B) respond faster but are less capable; larger models (14B+) are more capable but slower.

### 4. Configure

Edit `config.yaml` or `.env` to match your setup. At minimum:

- `llm.model` -- the model name as defined in `llama-swap-config.yaml`
- `audio.device_index` -- leave as `null` for system default, or set to a specific device index (list devices with `python -c "import sounddevice; print(sounddevice.query_devices())"`)

## Systemd Service Management

The Claw runs as a **user** systemd service (not root). This is required because PipeWire runs per-user.

### Basic Commands

```bash
# Start the service
systemctl --user start claw

# Stop the service
systemctl --user stop claw

# Restart (required after MCP config changes)
systemctl --user restart claw

# Check status
systemctl --user status claw

# Enable auto-start on boot
systemctl --user enable claw

# Disable auto-start
systemctl --user disable claw

# Follow logs in real-time
journalctl --user -u claw -f

# View recent logs
journalctl --user -u claw --since "1 hour ago"
```

### Linger

The installer enables linger for your user account so the service starts at boot without requiring a login session:

```bash
sudo loginctl enable-linger $(whoami)
```

### Service Configuration

The systemd unit file (`claw.service`) is configured with:

- **Restart policy** -- `on-failure` with a 5-second delay
- **Memory limit** -- 8 GB (`MemoryMax=8G`)
- **CPU quota** -- 200% (2 cores)
- **Working directory** -- `~/claw`
- **Dependencies** -- starts after `pipewire.service` and `llama-swap.service`

To adjust resource limits, edit `~/.config/systemd/user/claw.service` and run `systemctl --user daemon-reload`.

## Optional Features Setup

### TTS (Text-to-Speech)

TTS is installed by default via `install.sh`. The default engine is Piper with the `en_US-lessac-medium` voice model.

To disable TTS, set `tts.enabled: false` in `config.yaml` or toggle it in the admin panel Settings.

**Using a different Piper voice:**

1. Download a voice model from [Piper voices](https://huggingface.co/rhasspy/piper-voices)
2. Place the `.onnx` and `.onnx.json` files in `data/models/tts/piper/`
3. Update `tts.piper_model` in `config.yaml`

**Using Fish Speech (alternative TTS engine):**

1. Run a Fish Speech server locally (default port 8090)
2. Set `tts.engine: fish_speech` and `tts.fish_speech_url` in `config.yaml`
3. Optionally configure reference audio for voice cloning

### YouTube Music

YouTube Music uses browser cookie-based authentication (not OAuth).

**Setup:**

1. Ensure `youtube-music` optional dependencies are installed (handled by `install.sh`)
2. Run the auth setup script:
   ```bash
   source ~/claw/.venv/bin/activate
   python mcp_tools/youtube_music/setup_auth.py
   ```
3. Follow the prompts to extract cookies from your browser
4. Enable in config: set `youtube_music.enabled: true` in `config.yaml` or toggle in Settings
5. Restart the service: `systemctl --user restart claw`

**Cookie refresh:** Browser cookies expire periodically. If music searches start failing, re-run the auth setup script to extract fresh cookies.

### Google Services (Gmail and Google Calendar)

Google services require OAuth 2.0 credentials from the Google Cloud Console.

**Setup:**

1. Create a project at [Google Cloud Console](https://console.cloud.google.com)
2. Enable the Gmail API and Google Calendar API
3. Create OAuth 2.0 credentials (Desktop application type)
4. Download the credentials JSON and save it as `data/google/credentials.json`
5. Run the account setup script:
   ```bash
   source ~/claw/.venv/bin/activate
   python mcp_tools/google_auth/setup_auth.py
   ```
6. Follow the prompts to authorize your Google account
7. Enable services per-account in the admin panel Settings or in `config.yaml` under `google_auth.accounts`
8. Restart the service: `systemctl --user restart claw`

**Multiple accounts:** The Claw supports multiple Google accounts. Each account gets its own token file and can independently enable/disable Gmail and Calendar.

### Weather

Weather works out of the box using the free Open-Meteo API (no key required). For better accuracy, you can optionally configure an OpenWeatherMap API key:

1. Sign up at [OpenWeatherMap](https://openweathermap.org/api) (free tier)
2. Set `weather.api_key` in `config.yaml`
3. Optionally set `weather.default_location` (e.g., `"Akron, OH"`) -- if not set, location is auto-detected from your IP address

### Cloud LLM Providers

The Claw can optionally use cloud LLM backends (Claude, Gemini) instead of or alongside local inference. Cloud providers use the same OpenAI-compatible protocol, so tool calling and all agent features work identically.

**Configuration in `config.yaml`:**

```yaml
cloud_llm:
  active_provider: "local"  # "local", "claude", or "gemini"
  failover_to_local: true   # fall back to local if cloud fails
  failover_chain: []        # ordered list, e.g. ["claude", "gemini", "local"]
  providers:
    claude:
      enabled: true
      model: "claude-sonnet-4-20250514"
      api_key_secret: "claude_api_key"  # looked up in SecretStore
      max_tokens: 4096
      timeout: 60
    gemini:
      enabled: true
      model: "gemini-2.5-flash"
      api_key_secret: "gemini_api_key"
      max_tokens: 4096
      timeout: 60
```

API keys are stored in the SecretStore (not in config.yaml). Set them via the admin panel Settings or by placing them in `data/secrets/`.

The `failover_chain` lets you define an ordered list of providers to try. For example, `["claude", "local"]` will attempt Claude first, then fall back to local inference on timeout or server errors. If `failover_chain` is empty, the `failover_to_local` flag controls simple local fallback.

The active provider can also be switched at runtime through the admin panel without restarting.

### Scheduler (Reminders)

The scheduler runs in the background and fires reminders at their scheduled time. Reminders are created via the notes MCP tool (voice or chat) and stored in `data/notes/reminders.json`.

When a reminder fires, it:
- Broadcasts an SSE event to the admin panel
- Speaks the reminder via TTS (pausing music if playing, then resuming)
- Removes the fired reminder from the file
- On startup, checks for and fires any reminders missed during downtime

**Configuration in `config.yaml`:**

```yaml
scheduler:
  enabled: true
  poll_interval: 30   # seconds between checks
  announce_tts: true   # speak reminders aloud
```

The scheduler is enabled by default. No additional setup is required.

### Webhooks

Webhooks allow external systems (e.g., Home Assistant, IFTTT, custom scripts) to send events to The Claw. The endpoint is `POST /api/webhook` on the admin panel.

Supported event types:
- **`message`** -- feeds text to the agent and returns the response (like typing in chat)
- **`notification`** -- broadcasts a message via SSE and optionally speaks it via TTS
- **`reminder`** -- creates a scheduled reminder

**Configuration in `config.yaml`:**

```yaml
webhook:
  enabled: true
  secret: "your-hmac-secret"       # HMAC-SHA256 verification; empty = no verification
  allowed_events:
    - message
    - reminder
    - notification
```

**Example request:**

```bash
# Send a message (returns agent response)
curl -X POST http://<device-ip>:8080/api/webhook \
  -H "Content-Type: application/json" \
  -H "X-Signature: sha256=<hmac-hex>" \
  -d '{"type": "message", "payload": {"text": "What is the weather?"}, "source": "home-assistant"}'

# Send a notification (spoken + displayed)
curl -X POST http://<device-ip>:8080/api/webhook \
  -H "Content-Type: application/json" \
  -d '{"type": "notification", "payload": {"title": "Front Door", "message": "Someone is at the door"}}'
```

If `secret` is set, requests must include an `X-Signature` header with `sha256=<hex>` where the hex value is the HMAC-SHA256 of the request body using the configured secret.

### Remote Access and Android App

The Claw includes a remote access API and an Android companion app for interacting from outside the local network over a WireGuard VPN tunnel.

**Remote API endpoints** (all under `/api/remote/`):

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/ping` | GET | Health check |
| `/chat` | POST | Text chat |
| `/tts` | POST | Text-to-speech (returns WAV) |
| `/status` | GET | System status |
| `/events` | GET | SSE real-time updates |
| `/audio` | WS | Voice interaction (WebSocket) |

The WebSocket voice endpoint accepts 16kHz 16-bit mono PCM audio and supports push-to-talk, client-side wake word, and server-side wake word modes. Authentication uses per-device API keys via `X-API-Key` header (or `?key=` query parameter for WebSocket).

**Configuration in `config.yaml`:**

```yaml
remote:
  enabled: true
  audio_output: "phone"   # "phone" or "computer" — where TTS audio plays
  wg_interface: "wg0"
  wg_subnet: "10.10.0"
  wg_port: 51820
  wg_endpoint: ""          # public IP or hostname for the server
```

**WireGuard setup:**

```bash
sudo bash scripts/setup-wireguard.sh
```

This creates the WireGuard interface and generates server/client key pairs. Set `remote.wg_endpoint` to your server's public IP after setup.

**Android app** (`claw-android/`):

The companion app is a Kotlin/Jetpack Compose application (Android 8.0+, API 26) that provides:
- On-device "Hey Claw" wake word detection (ONNX Runtime)
- Voice streaming to the server over WireGuard VPN
- Text chat interface
- Music playback with YouTube audio proxy
- QR code scanning for quick server pairing
- Android Auto support
- Embedded WireGuard tunnel (no separate WireGuard app required)

Build the APK with Android Studio or Gradle:

```bash
cd claw-android
./gradlew assembleDebug
# Output: claw-android/app/build/outputs/apk/debug/claw-debug.apk
```

A pre-built debug APK is available at `claw-android/claw-debug.apk`. The APK filename is always `claw.apk` or `claw-debug.apk` (never version-suffixed).

### Usage Tracking

The Claw tracks token usage per provider (local, Claude, Gemini) with daily breakdowns and cost estimates. Usage data is persisted to disk and survives restarts.

**Configuration in `config.yaml`:**

```yaml
usage:
  enabled: true
  persist_file: "data/usage.json"
  persist_interval: 60   # seconds between disk writes
```

Usage stats are visible in the admin panel. Cost estimates use per-provider pricing (local inference is free).

## Updating

To update The Claw to the latest version:

```bash
cd ~/claw
git pull
source .venv/bin/activate
pip install -e .
systemctl --user restart claw
```

If optional dependencies have changed:

```bash
pip install -e ".[tts,youtube-music,google]"
systemctl --user restart claw
```

## Admin Panel

The admin panel runs on port 8080 by default and provides:

- **Status dashboard** -- current state (idle/listening/processing/speaking), last transcription, last response
- **Chat interface** -- text-based conversation with The Claw (with optional TTS playback)
- **Settings** -- live configuration editor for all settings sections
- **Logs** -- real-time log viewer via SSE

Access it at `http://<device-ip>:8080` from any browser on the local network.

To change the port, set `admin.port` in `config.yaml` or `CLAW_ADMIN__PORT` in `.env`.
