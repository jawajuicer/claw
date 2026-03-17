# openWakeWord ONNX Models

The Claw uses openWakeWord for on-device wake word detection. The ONNX model files
are too large to bundle in the source tree and must be downloaded separately.

## Required Models

You need three ONNX models that form the openWakeWord inference pipeline:

1. **melspectrogram.onnx** - Converts raw audio to mel-spectrogram features
2. **embedding_model.onnx** - Converts mel-spectrograms to audio embeddings
3. **hey_claw.onnx** - Classifier that detects the "Hey Claw" wake word

## How to Obtain

### Option A: Use pre-trained openWakeWord models (recommended for testing)

1. Install openWakeWord on your desktop:
   ```
   pip install openwakeword
   ```

2. Download the base models:
   ```python
   import openwakeword
   openwakeword.utils.download_models()
   ```

3. The melspectrogram and embedding models will be at:
   - `~/.local/lib/python3.x/site-packages/openwakeword/resources/models/melspectrogram.onnx`
   - `~/.local/lib/python3.x/site-packages/openwakeword/resources/models/embedding_model.onnx`

4. For the wake word classifier, you can either:
   - Train a custom "Hey Claw" model using openwakeword's training tools
   - Use the pre-trained "hey_jarvis" model renamed (NOT recommended for production)
   - Use the Claw server's trained model if available at `data/models/hey_claw.onnx`

### Option B: Train a custom wake word

Follow the openWakeWord training guide:
https://github.com/dscripka/openWakeWord#training-new-models

Use "Hey Claw" as the target phrase.

## Installation

Copy all three `.onnx` files into this directory:

```
app/src/main/assets/models/
    melspectrogram.onnx
    embedding_model.onnx
    hey_claw.onnx
```

Then rebuild the app. The WakeWordEngine will automatically detect and load them.

## Without Models

If the models are not present, the app will still function but wake word detection
will be disabled. You can still use push-to-talk voice input and text chat.
