#!/usr/bin/env bash
# Deploy trained "hey claw" model to the claw service.
# Run on claw device: bash ~/claw/training/deploy_model.sh
set -euo pipefail

TRAINING_DIR="$HOME/claw/training"
OUTPUT_DIR="$TRAINING_DIR/output"
CLAW_DIR="$HOME/claw"
WAKE_MODEL_DIR="$CLAW_DIR/data/models/wake"
CONFIG_FILE="$CLAW_DIR/config.yaml"

echo "=== Deploying Hey Claw Wake Word Model ==="

# Find the trained model
MODEL_FILE=$(find "$OUTPUT_DIR" -name "hey_claw*.onnx" -type f 2>/dev/null | head -1)
if [ -z "$MODEL_FILE" ]; then
    echo "ERROR: No hey_claw .onnx model found in $OUTPUT_DIR"
    echo "Has training completed? Check: tail -f $TRAINING_DIR/training.log"
    exit 1
fi

SIZE=$(du -h "$MODEL_FILE" | cut -f1)
echo "Found model: $MODEL_FILE ($SIZE)"

# Create wake model directory
mkdir -p "$WAKE_MODEL_DIR"

# Copy model
cp "$MODEL_FILE" "$WAKE_MODEL_DIR/hey_claw.onnx"
echo "Copied to: $WAKE_MODEL_DIR/hey_claw.onnx"

# Backup config
if [ -f "$CONFIG_FILE" ]; then
    cp "$CONFIG_FILE" "${CONFIG_FILE}.bak.$(date +%Y%m%d_%H%M%S)"
    echo "Config backed up."
fi

# Update config.yaml to add hey_claw alongside hey_jarvis
# Uses python for safe YAML manipulation
"$CLAW_DIR/.venv/bin/python3" - <<'PYEOF'
import yaml
from pathlib import Path

config_path = Path.home() / "claw" / "config.yaml"
with open(config_path) as f:
    config = yaml.safe_load(f) or {}

wake = config.setdefault("wake", {})
model_paths = wake.get("model_paths", ["hey_jarvis_v0.1"])

if "hey_claw" not in model_paths:
    # Add hey_claw as the primary wake word (first position)
    model_paths.insert(0, "hey_claw")
    wake["model_paths"] = model_paths

thresholds = wake.setdefault("thresholds", {})
if "hey_claw" not in thresholds:
    thresholds["hey_claw"] = 0.5

wake["thresholds"] = thresholds
config["wake"] = wake

with open(config_path, "w") as f:
    yaml.dump(config, f, default_flow_style=False, sort_keys=False)

print(f"Config updated: model_paths={model_paths}, thresholds={thresholds}")
PYEOF

echo ""
echo "=== Deployment Complete ==="
echo "The config watcher should auto-reload the service."
echo "Check logs: journalctl --user -u claw -f"
echo "Expected log: 'Wake word models loaded: hey_claw, hey_jarvis_v0.1'"
echo ""
echo "Test by saying 'hey claw' near the device mic."
echo "Adjust threshold in Settings > Wake Word if needed (start at 0.5)."
