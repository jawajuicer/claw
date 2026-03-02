#!/usr/bin/env bash
set -euo pipefail

# The Claw — Installation Script
# Target: Ubuntu 24.04 with Python 3.12 and PipeWire

INSTALL_DIR="${HOME}/claw"
VENV_DIR="${INSTALL_DIR}/.venv"
DATA_DIR="${INSTALL_DIR}/data"
SERVICE_NAME="claw"

echo "=== The Claw — Installer ==="
echo "Install directory: ${INSTALL_DIR}"

# ── System dependencies ─────────────────────────────────────────────────────

echo ""
echo "--- Installing system dependencies ---"
sudo apt-get update -qq
sudo apt-get install -y -qq \
    python3.12-venv \
    python3-dev \
    libportaudio2 \
    portaudio19-dev \
    ffmpeg \
    mpv \
    libmpv-dev

# ── Python virtual environment ──────────────────────────────────────────────

echo ""
echo "--- Setting up Python venv ---"
if [ ! -d "${VENV_DIR}" ]; then
    python3 -m venv "${VENV_DIR}"
    echo "Created venv at ${VENV_DIR}"
else
    echo "Venv already exists at ${VENV_DIR}"
fi

source "${VENV_DIR}/bin/activate"
pip install --upgrade pip setuptools wheel -q

# ── Install the project ─────────────────────────────────────────────────────

echo ""
echo "--- Installing The Claw ---"
pip install -e "${INSTALL_DIR}" -q

# ── Optional: TTS (Piper) ──────────────────────────────────────────────
echo ""
echo "--- Installing TTS support ---"
pip install -e "${INSTALL_DIR}[tts]" -q || echo "TTS deps failed (optional, can install later)"

# Download default Piper voice model
PIPER_MODEL_DIR="${DATA_DIR}/models/tts/piper"
mkdir -p "${PIPER_MODEL_DIR}"
PIPER_MODEL="${PIPER_MODEL_DIR}/en_US-lessac-medium.onnx"
if [ ! -f "${PIPER_MODEL}" ]; then
    echo "--- Downloading default Piper voice model ---"
    curl -sL "https://huggingface.co/rhasspy/piper-voices/resolve/main/en/en_US/lessac/medium/en_US-lessac-medium.onnx" \
        -o "${PIPER_MODEL}" || echo "Piper model download failed (can download later)"
    curl -sL "https://huggingface.co/rhasspy/piper-voices/resolve/main/en/en_US/lessac/medium/en_US-lessac-medium.onnx.json" \
        -o "${PIPER_MODEL}.json" || echo "Piper model config download failed"
else
    echo "Piper voice model already exists"
fi

# ── Optional: YouTube Music ───────────────────────────────────────────────
echo ""
echo "--- Installing YouTube Music support ---"
pip install -e "${INSTALL_DIR}[youtube-music]" -q || echo "YouTube Music deps failed (optional, can install later)"

# ── Optional: Google services (Gmail, Calendar) ─────────────────────────
echo ""
echo "--- Installing Google services support ---"
pip install -e "${INSTALL_DIR}[google]" -q || echo "Google deps failed (optional, can install later)"

# ── Data directories ────────────────────────────────────────────────────────

echo ""
echo "--- Creating data directories ---"
mkdir -p "${DATA_DIR}"/{chromadb,models,logs,youtube_music,notes,calendar,google,google/tokens,models/wake,models/tts/piper}

# ── Environment file ────────────────────────────────────────────────────────

if [ ! -f "${INSTALL_DIR}/.env" ]; then
    cp "${INSTALL_DIR}/.env.example" "${INSTALL_DIR}/.env"
    echo "Created .env from .env.example — edit as needed"
else
    echo ".env already exists"
fi

# ── systemd user service ────────────────────────────────────────────────────

echo ""
echo "--- Setting up systemd user service ---"
SYSTEMD_DIR="${HOME}/.config/systemd/user"
mkdir -p "${SYSTEMD_DIR}"
cp "${INSTALL_DIR}/claw.service" "${SYSTEMD_DIR}/${SERVICE_NAME}.service"
systemctl --user daemon-reload
echo "Service installed at ${SYSTEMD_DIR}/${SERVICE_NAME}.service"

# Enable linger so user services start at boot
echo ""
echo "--- Enabling linger for boot-time start ---"
sudo loginctl enable-linger "$(whoami)"

echo ""
echo "=== Installation complete ==="
echo ""
echo "YouTube Music setup (optional):"
echo "  python mcp_tools/youtube_music/setup_auth.py   # One-time auth"
echo "  Then enable in Settings or set youtube_music.enabled: true in config.yaml"
echo ""
echo "Google services setup (Gmail, Calendar — optional):"
echo "  1. Create OAuth credentials at https://console.cloud.google.com/apis/credentials"
echo "  2. Save as data/google/credentials.json"
echo "  3. python mcp_tools/google_auth/setup_auth.py   # Add accounts interactively"
echo "  4. Enable services per-account in Settings"
echo ""
echo "Commands:"
echo "  systemctl --user start ${SERVICE_NAME}     # Start"
echo "  systemctl --user stop ${SERVICE_NAME}      # Stop"
echo "  systemctl --user status ${SERVICE_NAME}    # Status"
echo "  systemctl --user enable ${SERVICE_NAME}    # Auto-start on boot"
echo "  journalctl --user -u ${SERVICE_NAME} -f    # Follow logs"
echo ""
echo "Or run directly:"
echo "  source ${VENV_DIR}/bin/activate"
echo "  python -m claw --mode cli     # Text mode"
echo "  python -m claw --mode voice   # Voice mode"
echo "  python -m claw --mode both    # Full mode (default)"
