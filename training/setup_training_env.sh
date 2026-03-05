#!/usr/bin/env bash
# Setup training environment for custom "hey claw" wake word model.
# Run on the claw device: bash ~/claw/training/setup_training_env.sh
set -euo pipefail

TRAINING_DIR="$HOME/claw/training"
VENV_DIR="$TRAINING_DIR/.venv"
DATA_DIR="$TRAINING_DIR/data"

echo "=== Hey Claw Wake Word Training Setup ==="
echo "Training dir: $TRAINING_DIR"
echo ""

# ── 1. System dependencies ─────────────────────────────────────────────────
echo "[1/8] Installing system dependencies..."
# Pass password via stdin for non-interactive SSH sessions
echo harper | sudo -S apt-get update -qq 2>/dev/null
echo harper | sudo -S apt-get install -y -qq espeak-ng libsndfile1 ffmpeg tmux 2>/dev/null

# ── 2. Create isolated venv ────────────────────────────────────────────────
echo "[2/8] Creating isolated training venv..."
if [ -d "$VENV_DIR" ]; then
    echo "  Venv already exists, skipping creation."
else
    python3 -m venv "$VENV_DIR"
fi
source "$VENV_DIR/bin/activate"

# Pin setuptools to avoid pkg_resources removal breaking speechbrain
pip install --quiet --upgrade pip 'setuptools<82' wheel

# ── 3. Install PyTorch CPU-only ─────────────────────────────────────────────
echo "[3/8] Installing PyTorch (CPU-only)..."
pip install --quiet torch torchaudio --index-url https://download.pytorch.org/whl/cpu

# ── 4. Install Python packages ──────────────────────────────────────────────
echo "[4/8] Installing training dependencies..."
pip install --quiet \
    openwakeword \
    speechbrain \
    audiomentations \
    torch-audiomentations \
    acoustics \
    mutagen \
    pronouncing \
    deep-phonemizer \
    datasets \
    webrtcvad \
    soundfile \
    onnx \
    onnxruntime \
    scipy \
    tqdm

# ── 5. Clone openWakeWord repo (for train.py) ──────────────────────────────
echo "[5/8] Cloning openWakeWord repository..."
OWW_DIR="$TRAINING_DIR/openWakeWord"
if [ -d "$OWW_DIR" ]; then
    echo "  openWakeWord already cloned, pulling latest..."
    cd "$OWW_DIR" && git pull --quiet && cd "$TRAINING_DIR"
else
    git clone --quiet https://github.com/dscripka/openWakeWord.git "$OWW_DIR"
fi

# ── 6. Clone & install piper-sample-generator ───────────────────────────────
echo "[6/8] Setting up piper-sample-generator..."
PSG_DIR="$TRAINING_DIR/piper-sample-generator"
if [ -d "$PSG_DIR" ]; then
    echo "  piper-sample-generator already cloned, pulling latest..."
    cd "$PSG_DIR" && git pull --quiet && cd "$TRAINING_DIR"
else
    git clone --quiet https://github.com/rhasspy/piper-sample-generator.git "$PSG_DIR"
fi
pip install --quiet -e "$PSG_DIR"

# Download Piper voice models for speaker diversity
PIPER_DIR="$DATA_DIR/piper_voices"
mkdir -p "$PIPER_DIR"
echo "  Downloading Piper voice models for speaker diversity..."

declare -A PIPER_MODELS=(
    ["en_US-lessac-medium"]="https://huggingface.co/rhasspy/piper-voices/resolve/main/en/en_US/lessac/medium/en_US-lessac-medium.onnx"
    ["en_US-libritts_r-medium"]="https://huggingface.co/rhasspy/piper-voices/resolve/main/en/en_US/libritts_r/medium/en_US-libritts_r-medium.onnx"
    ["en_US-cori-medium"]="https://huggingface.co/rhasspy/piper-voices/resolve/main/en/en_US/cori/medium/en_US-cori-medium.onnx"
)
declare -A PIPER_CONFIGS=(
    ["en_US-lessac-medium"]="https://huggingface.co/rhasspy/piper-voices/resolve/main/en/en_US/lessac/medium/en_US-lessac-medium.onnx.json"
    ["en_US-libritts_r-medium"]="https://huggingface.co/rhasspy/piper-voices/resolve/main/en/en_US/libritts_r/medium/en_US-libritts_r-medium.onnx.json"
    ["en_US-cori-medium"]="https://huggingface.co/rhasspy/piper-voices/resolve/main/en/en_US/cori/medium/en_US-cori-medium.onnx.json"
)

for name in "${!PIPER_MODELS[@]}"; do
    if [ ! -f "$PIPER_DIR/${name}.onnx" ]; then
        echo "    Downloading ${name}..."
        wget -q -O "$PIPER_DIR/${name}.onnx" "${PIPER_MODELS[$name]}"
        wget -q -O "$PIPER_DIR/${name}.onnx.json" "${PIPER_CONFIGS[$name]}"
    else
        echo "    ${name} already downloaded."
    fi
done

# ── 7. Download ACAV100M negative features ──────────────────────────────────
echo "[7/8] Downloading ACAV100M negative features (~17.3GB)..."
ACAV_DIR="$DATA_DIR/negative_features"
mkdir -p "$ACAV_DIR"

if [ -f "$ACAV_DIR/download_complete" ]; then
    echo "  ACAV100M already downloaded."
else
    cd "$ACAV_DIR"
    # Download the ACAV100M clips features (used as negative examples)
    python3 -c "
from huggingface_hub import snapshot_download
snapshot_download(
    repo_id='davidscripka/openwakeword_features_ACAV100M',
    repo_type='dataset',
    local_dir='.',
    local_dir_use_symlinks=False,
)
"
    touch download_complete
    cd "$TRAINING_DIR"
    echo "  ACAV100M download complete."
fi

# Download validation features
VALIDATION_DIR="$DATA_DIR/validation_features"
mkdir -p "$VALIDATION_DIR"
if [ -f "$VALIDATION_DIR/download_complete" ]; then
    echo "  Validation features already downloaded."
else
    cd "$VALIDATION_DIR"
    python3 -c "
from huggingface_hub import snapshot_download
snapshot_download(
    repo_id='davidscripka/openwakeword_features_mit',
    repo_type='dataset',
    local_dir='.',
    local_dir_use_symlinks=False,
)
"
    touch download_complete
    cd "$TRAINING_DIR"
    echo "  Validation features download complete."
fi

# ── 8. Download MIT impulse responses ────────────────────────────────────────
echo "[8/8] Downloading MIT impulse responses for reverb augmentation..."
RIR_DIR="$DATA_DIR/mit_rirs"
mkdir -p "$RIR_DIR"
if [ -f "$RIR_DIR/download_complete" ]; then
    echo "  MIT RIRs already downloaded."
else
    cd "$RIR_DIR"
    wget -q "http://mcdermottlab.mit.edu/Reverb/IRMAudio/Audio.zip" -O mit_rir.zip
    unzip -qo mit_rir.zip
    rm -f mit_rir.zip
    touch download_complete
    cd "$TRAINING_DIR"
    echo "  MIT RIRs download complete."
fi

echo ""
echo "=== Setup Complete ==="
echo "Venv: $VENV_DIR"
echo "Data: $DATA_DIR"
echo "Next: bash ~/claw/training/train.sh"
