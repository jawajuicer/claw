#!/usr/bin/env bash
# Launch "hey claw" wake word training in a tmux session.
# Run on claw device: bash ~/claw/training/train.sh
set -euo pipefail

TRAINING_DIR="$HOME/claw/training"
VENV_DIR="$TRAINING_DIR/.venv"
OWW_DIR="$TRAINING_DIR/openWakeWord"
DATA_DIR="$TRAINING_DIR/data"
LOG_FILE="$TRAINING_DIR/training.log"
CONFIG="$TRAINING_DIR/hey_claw_config.yaml"
TMUX_SESSION="claw-train"

# Verify setup completed
if [ ! -d "$VENV_DIR" ]; then
    echo "ERROR: Training venv not found. Run setup_training_env.sh first."
    exit 1
fi
if [ ! -d "$OWW_DIR" ]; then
    echo "ERROR: openWakeWord repo not found. Run setup_training_env.sh first."
    exit 1
fi
if [ ! -f "$DATA_DIR/openwakeword_features_ACAV100M_2000_hrs_16bit.npy" ]; then
    echo "ERROR: ACAV100M features not downloaded. Run setup_training_env.sh first."
    exit 1
fi
if [ ! -f "$DATA_DIR/validation_set_features.npy" ]; then
    echo "ERROR: Validation features not downloaded. Run setup_training_env.sh first."
    exit 1
fi

# Write the pipeline script that runs inside tmux
cat > "$TRAINING_DIR/_run_pipeline.sh" <<'TRAINEOF'
#!/usr/bin/env bash
set -euo pipefail

TRAINING_DIR="$HOME/claw/training"
LOG_FILE="$TRAINING_DIR/training.log"
CONFIG="$TRAINING_DIR/hey_claw_config.yaml"
TRAIN_SCRIPT="$TRAINING_DIR/openWakeWord/openwakeword/train.py"

source "$TRAINING_DIR/.venv/bin/activate"
cd "$TRAINING_DIR"

echo "========================================" | tee -a "$LOG_FILE"
echo "Hey Claw Training Pipeline" | tee -a "$LOG_FILE"
echo "Started: $(date)" | tee -a "$LOG_FILE"
echo "========================================" | tee -a "$LOG_FILE"

# Phase 1: Generate synthetic clips
echo "" | tee -a "$LOG_FILE"
echo "[Phase 1/3] Generating synthetic clips..." | tee -a "$LOG_FILE"
echo "Start: $(date)" | tee -a "$LOG_FILE"

python3 "$TRAIN_SCRIPT" \
    --training_config "$CONFIG" \
    --generate_clips \
    2>&1 | tee -a "$LOG_FILE"

echo "[Phase 1/3] Complete: $(date)" | tee -a "$LOG_FILE"

# Phase 2: Augment clips and compute features
echo "" | tee -a "$LOG_FILE"
echo "[Phase 2/3] Augmenting clips and computing features..." | tee -a "$LOG_FILE"
echo "Start: $(date)" | tee -a "$LOG_FILE"

python3 "$TRAIN_SCRIPT" \
    --training_config "$CONFIG" \
    --augment_clips \
    2>&1 | tee -a "$LOG_FILE"

echo "[Phase 2/3] Complete: $(date)" | tee -a "$LOG_FILE"

# Phase 3: Train model
echo "" | tee -a "$LOG_FILE"
echo "[Phase 3/3] Training wake word model..." | tee -a "$LOG_FILE"
echo "Start: $(date)" | tee -a "$LOG_FILE"

python3 "$TRAIN_SCRIPT" \
    --training_config "$CONFIG" \
    --train_model \
    2>&1 | tee -a "$LOG_FILE"

echo "[Phase 3/3] Complete: $(date)" | tee -a "$LOG_FILE"

echo "" | tee -a "$LOG_FILE"
echo "========================================" | tee -a "$LOG_FILE"
echo "Training Pipeline Complete!" | tee -a "$LOG_FILE"
echo "Finished: $(date)" | tee -a "$LOG_FILE"
echo "========================================" | tee -a "$LOG_FILE"

# Report model location
MODEL_FILE=$(find "$TRAINING_DIR/output" -name "hey_claw*.onnx" -type f 2>/dev/null | head -1)
if [ -n "$MODEL_FILE" ]; then
    SIZE=$(du -h "$MODEL_FILE" | cut -f1)
    echo "Model: $MODEL_FILE ($SIZE)" | tee -a "$LOG_FILE"
    echo "" | tee -a "$LOG_FILE"
    echo "Deploy with: bash ~/claw/training/deploy_model.sh" | tee -a "$LOG_FILE"
else
    echo "WARNING: No hey_claw .onnx model found in output/" | tee -a "$LOG_FILE"
    echo "Check training log for errors." | tee -a "$LOG_FILE"
fi
TRAINEOF

chmod +x "$TRAINING_DIR/_run_pipeline.sh"

# Kill existing tmux session if present
tmux kill-session -t "$TMUX_SESSION" 2>/dev/null || true

echo "Launching training in tmux session '$TMUX_SESSION'..."
echo "Log file: $LOG_FILE"
echo ""

# Start tmux session running the pipeline
tmux new-session -d -s "$TMUX_SESSION" "bash $TRAINING_DIR/_run_pipeline.sh"

echo "Training started! Monitor with:"
echo "  tmux attach -t $TMUX_SESSION"
echo "  tail -f $LOG_FILE"
echo ""
echo "Expected duration: 12-24 hours"
echo "The session will survive SSH disconnection."
