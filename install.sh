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
    libmpv-dev \
    cmake \
    build-essential

# ── Detect GPU hardware & build llama.cpp ─────────────────────────────────

echo ""
echo "--- Detecting GPU hardware ---"
CMAKE_GPU_FLAGS=""
DETECTED_BACKEND="cpu"

if command -v nvidia-smi &>/dev/null && nvidia-smi --query-gpu=name --format=csv,noheader &>/dev/null; then
    GPU_NAME=$(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | head -1)
    echo "NVIDIA GPU detected: ${GPU_NAME}"
    echo "Installing CUDA toolkit..."
    sudo apt-get install -y -qq nvidia-cuda-toolkit
    CMAKE_GPU_FLAGS="-DGGML_CUDA=ON"
    DETECTED_BACKEND="cuda"
elif [ -d /sys/class/kfd/kfd/topology/nodes ]; then
    GFX_VER=$(grep -rh 'gfx_target_version' /sys/class/kfd/kfd/topology/nodes/*/properties 2>/dev/null | awk '{print $2}' | grep -v '^0$' | head -1)
    if [ -n "${GFX_VER}" ]; then
        echo "AMD GPU detected (gfx${GFX_VER})"
        echo "Installing ROCm..."
        sudo apt-get install -y -qq rocm-dev
        CMAKE_GPU_FLAGS="-DGGML_ROCM=ON"
        DETECTED_BACKEND="rocm"
    fi
elif [ -e /dev/dri/renderD128 ]; then
    echo "Vulkan-capable GPU detected"
    echo "Installing Vulkan tools..."
    sudo apt-get install -y -qq vulkan-tools libvulkan-dev
    CMAKE_GPU_FLAGS="-DGGML_VULKAN=ON"
    DETECTED_BACKEND="vulkan"
fi

if [ "${DETECTED_BACKEND}" = "cpu" ]; then
    echo "No GPU detected — building for CPU only"
fi

echo ""
echo "--- Building llama.cpp (${DETECTED_BACKEND}) ---"
if [ ! -f /opt/llama.cpp/build/bin/llama-server ]; then
    if [ ! -d /opt/llama.cpp ]; then
        sudo git clone https://github.com/ggml-org/llama.cpp.git /opt/llama.cpp
    fi
    cd /opt/llama.cpp
    sudo cmake -B build -DCMAKE_BUILD_TYPE=Release ${CMAKE_GPU_FLAGS}
    sudo cmake --build build --config Release -j "$(nproc)"
    cd "${INSTALL_DIR}"
    echo "llama.cpp built at /opt/llama.cpp/build/bin/llama-server (${DETECTED_BACKEND})"
else
    echo "llama.cpp already built"
fi

# ── Download llama-swap ────────────────────────────────────────────────────

echo ""
echo "--- Installing llama-swap ---"
if [ ! -f /opt/llama-swap/llama-swap ]; then
    sudo mkdir -p /opt/llama-swap
    LLAMA_SWAP_URL=$(curl -sL https://api.github.com/repos/mostlygeek/llama-swap/releases/latest \
        | python3 -c "import sys,json; r=json.load(sys.stdin); urls=[a['browser_download_url'] for a in r.get('assets',[]) if 'linux_amd64' in a['name']]; print(urls[0]) if urls else None" \
        | head -1)
    if [ -z "${LLAMA_SWAP_URL}" ]; then
        echo "ERROR: Could not find llama-swap download URL. Check GitHub API rate limits."
        exit 1
    fi
    sudo curl -L -o /tmp/llama-swap.tar.gz "${LLAMA_SWAP_URL}"
    sudo tar xzf /tmp/llama-swap.tar.gz -C /opt/llama-swap
    sudo rm -f /tmp/llama-swap.tar.gz
    echo "llama-swap installed at /opt/llama-swap/llama-swap"
else
    echo "llama-swap already installed"
fi

# ── Models directory ───────────────────────────────────────────────────────

mkdir -p "${HOME}/models"

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

# ── Write detected compute backend to config.yaml ────────────────────────

CONFIG_FILE="${INSTALL_DIR}/config.yaml"
if [ -f "${CONFIG_FILE}" ] && ! grep -q "^compute:" "${CONFIG_FILE}"; then
    echo "" >> "${CONFIG_FILE}"
    echo "compute:" >> "${CONFIG_FILE}"
    echo "  backend: ${DETECTED_BACKEND}" >> "${CONFIG_FILE}"
    echo "  gpu_layers: 99" >> "${CONFIG_FILE}"
    echo "Appended compute backend (${DETECTED_BACKEND}) to config.yaml"
fi

# ── systemd user service ────────────────────────────────────────────────────

echo ""
echo "--- Setting up systemd user service ---"
SYSTEMD_DIR="${HOME}/.config/systemd/user"
mkdir -p "${SYSTEMD_DIR}"
cp "${INSTALL_DIR}/claw.service" "${SYSTEMD_DIR}/${SERVICE_NAME}.service"

# llama-swap service
cat > "${SYSTEMD_DIR}/llama-swap.service" <<LLAMA_SWAP_EOF
[Unit]
Description=llama-swap — LLM inference proxy
After=network.target

[Service]
Type=simple
ExecStart=/opt/llama-swap/llama-swap --config %h/claw/llama-swap-config.yaml --listen :8081
Restart=on-failure
RestartSec=5
Environment=PATH=/opt/llama.cpp/build/bin:/usr/local/bin:/usr/bin:/bin

[Install]
WantedBy=default.target
LLAMA_SWAP_EOF

systemctl --user daemon-reload
echo "Services installed at ${SYSTEMD_DIR}/"

# Enable linger so user services start at boot
echo ""
echo "--- Enabling linger for boot-time start ---"
sudo loginctl enable-linger "$(whoami)"

echo ""
echo "=== Installation complete ==="
echo ""
echo "Next steps:"
echo "  1. Download a model into ~/models/ (GGUF format):"
echo "     pip install huggingface-hub"
echo "     huggingface-cli download unsloth/Qwen2.5-0.5B-Instruct-GGUF --include '*.gguf' --local-dir ~/models"
echo "  2. Create ~/claw/llama-swap-config.yaml (see docs/DEPLOYMENT.md)"
echo "  3. systemctl --user start llama-swap"
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
