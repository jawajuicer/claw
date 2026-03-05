# Wake Word Training

This guide covers training a custom "Hey Claw" wake word model using [openWakeWord](https://github.com/dscripka/openWakeWord).

## Overview

The Claw uses openWakeWord for wake word detection with ONNX runtime. A pre-built `hey_jarvis_v0.1` model is available out of the box, but for the best experience you should train a custom "Hey Claw" model.

The training pipeline:

1. **Generate synthetic clips** -- uses Piper TTS to generate thousands of audio samples of "hey claw" with diverse speakers and prosody
2. **Augment clips** -- applies room impulse responses (reverb), noise, and other augmentations to simulate real-world conditions
3. **Train model** -- trains a small DNN classifier to distinguish "hey claw" from background audio and phonetically similar phrases

The result is a compact ONNX model (under 100 KB) that runs in real-time on CPU.

## Prerequisites

- The Claw deployed on a machine with at least 16 GB RAM and 20+ GB free disk space
- Internet access for downloading training data (one-time, approximately 18 GB)
- Approximately 12-24 hours for the full training pipeline (CPU-only)

## Setup

The `training/` directory contains all scripts needed. Run the setup script on the target machine:

```bash
cd ~/claw
bash training/setup_training_env.sh
```

This creates an isolated virtual environment at `training/.venv/` (separate from the main Claw venv) and installs all training dependencies:

- **PyTorch** (CPU-only) and speechbrain
- **openWakeWord** training scripts (cloned from GitHub)
- **piper-sample-generator** for synthetic speech generation
- **Piper voice models** (3 English voices for speaker diversity)
- **ACAV100M features** -- approximately 2000 hours of diverse audio used as negative examples (approximately 17 GB download)
- **MIT impulse responses** -- room reverb recordings for augmentation
- **Validation features** -- approximately 11 hours of speech/noise/music for false-positive testing

The setup is idempotent -- running it again will skip already-completed steps.

## Training Configuration

The training config file is `training/hey_claw_config.yaml`. Key parameters:

### Target Phrase

```yaml
target_phrase:
  - "hey claw"
```

### Negative Phrases

Phonetically similar phrases that the model should learn to reject:

```yaml
custom_negative_phrases:
  - "hey claude"
  - "hey call"
  - "he called"
  - "hey clock"
  - "the claw"
  - "hey class"
  - "hey claws"
  - "hey clause"
  - "hey clog"
  - "hay bale"
  - "a claw"
  - "hey clay"
```

### Sample Generation

```yaml
n_samples: 50000          # number of positive training samples
n_samples_val: 5000       # number of validation samples
augmentation_rounds: 2    # augmentation passes per sample
```

### Model Architecture

```yaml
model_type: "dnn"         # lightweight DNN classifier
layer_size: 32            # hidden layer width
```

### Training Parameters

```yaml
steps: 50000                          # training steps
max_negative_weight: 1500             # weight for negative examples
target_false_positives_per_hour: 0.2  # target FP rate
```

The default configuration is tuned for a good balance of accuracy and false-positive rate. You generally do not need to change these values unless you are experimenting.

## Running Training

Launch the training pipeline:

```bash
bash training/train.sh
```

This starts a `tmux` session named `claw-train` that runs all three phases sequentially. The tmux session survives SSH disconnection, so you can safely close your terminal.

### Monitoring

```bash
# Attach to the tmux session to see live output
tmux attach -t claw-train

# Or follow the log file
tail -f ~/claw/training/training.log
```

### Training Phases

| Phase | Description | Approximate Duration |
|-------|-------------|---------------------|
| 1/3 | Generate synthetic clips (50,000 samples via Piper TTS) | 2-4 hours |
| 2/3 | Augment clips and compute audio features | 4-8 hours |
| 3/3 | Train the DNN wake word model | 4-8 hours |

Total expected duration: 12-24 hours depending on CPU speed.

### Output

The trained model is saved to `training/output/` as an ONNX file (e.g., `hey_claw.onnx`).

## Deploying the Trained Model

Once training is complete, deploy the model to The Claw:

```bash
bash training/deploy_model.sh
```

This script:

1. Copies the trained `hey_claw.onnx` model to `data/models/wake/`
2. Updates `config.yaml` to add `hey_claw` to `wake.model_paths` (alongside any existing models)
3. Sets a default detection threshold of 0.5 for the new model

The config watcher automatically reloads the settings, so the new wake word model should activate within a few seconds. Check the logs for confirmation:

```bash
journalctl --user -u claw -f
# Look for: "Wake word models loaded: hey_claw, hey_jarvis_v0.1"
```

### Testing

Say "Hey Claw" near the device microphone. If detection is too sensitive (false triggers) or not sensitive enough (missed triggers), adjust the threshold:

- In `config.yaml`: set `wake.thresholds.hey_claw` (lower = more sensitive, higher = fewer false positives)
- In the admin panel: navigate to Settings and adjust the wake word threshold
- Recommended starting value: 0.5. Adjust in increments of 0.05.

### Running Multiple Wake Words

The Claw supports multiple wake word models simultaneously. After deploying, both "Hey Claw" and "Hey Jarvis" (or any other configured models) will be active. To use only "Hey Claw", remove other models from `wake.model_paths` in `config.yaml`:

```yaml
wake:
  model_paths:
    - hey_claw
  thresholds:
    hey_claw: 0.5
```

## Retraining

If you want to retrain with different parameters:

1. Edit `training/hey_claw_config.yaml` (adjust samples, steps, negative phrases, etc.)
2. Optionally delete `training/output/` to start fresh
3. Run `bash training/train.sh` again
4. Deploy with `bash training/deploy_model.sh`

The training data downloads (ACAV100M features, MIT impulse responses, Piper voices) are cached and will not be re-downloaded.

## Troubleshooting

- **Setup fails at ACAV100M download** -- this is a 17 GB download. Ensure you have sufficient disk space and a stable internet connection. The download is resumable.
- **Training runs out of memory** -- reduce `n_samples` or `batch_n_per_class` values in the config. 16 GB RAM should be sufficient with default settings.
- **Model has too many false positives** -- increase `max_negative_weight` or add more entries to `custom_negative_phrases`. You can also raise the detection threshold after deployment.
- **Model misses "hey claw" too often** -- increase `n_samples` or `augmentation_rounds`. Lower the detection threshold after deployment.
- **tmux session not found** -- if the training finished or crashed, check `training/training.log` for the outcome.
