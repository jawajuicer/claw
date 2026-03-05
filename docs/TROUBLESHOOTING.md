# Troubleshooting

Common issues and their solutions when running The Claw.

## Audio Device Not Found

**Symptom:** The service fails to start or logs errors about no audio device.

**Diagnosis:**

```bash
# List available audio devices
source ~/claw/.venv/bin/activate
python -c "import sounddevice; print(sounddevice.query_devices())"
```

**Solutions:**

- Verify your USB microphone is plugged in and recognized by the system: `lsusb` and `arecord -l`
- Ensure PipeWire is running: `systemctl --user status pipewire`
- If you see multiple devices, set `audio.device_index` in `config.yaml` to the correct index from the device list
- Leave `audio.device_index` as `null` to use the system default input device
- Restart PipeWire if it is in a bad state: `systemctl --user restart pipewire`

## Wake Word Not Triggering

**Symptom:** The Claw stays in "idle" state and never responds to the wake word.

**Solutions:**

- **Threshold too high** -- lower `wake.default_threshold` in `config.yaml` (default is 0.5, try 0.3-0.4). You can also set per-model thresholds under `wake.thresholds`
- **Microphone too quiet** -- check input volume with `pavucontrol` or `wpctl status`. Speak closer to the microphone
- **Wrong model** -- verify `wake.model_paths` lists valid model names. Built-in models (like `hey_jarvis_v0.1`) are downloaded automatically. Custom models must exist as `.onnx` files in the `data/models/wake/` directory
- **Audio format mismatch** -- ensure `audio.sample_rate` is 16000 and `audio.channels` is 1. openWakeWord requires 16kHz mono audio

**Testing wake word detection:**

Run in CLI mode and check the logs for wake word scores:

```bash
python -m claw --mode voice --log-level DEBUG
```

Look for log lines from `claw.audio_pipeline.wake_word` showing detection scores.

## Ollama Connection Issues

**Symptom:** Errors about connection refused or timeouts when processing queries.

**Diagnosis:**

```bash
# Check if Ollama is running
curl http://localhost:11434/v1/models

# Check if the configured model exists
ollama list
```

**Solutions:**

- Start Ollama if it is not running: `ollama serve` or `systemctl start ollama`
- Verify `llm.base_url` in `config.yaml` points to the correct Ollama address (default: `http://localhost:11434/v1`)
- Pull the configured model: `ollama pull <model-name>` (check `llm.model` in config)
- Increase `llm.timeout` if the model is slow to respond (default is 45 seconds)
- If Ollama runs on a different machine, update `llm.base_url` to that machine's address

## MCP Tool Timeouts

**Symptom:** Tool calls fail with timeout errors or the agent gets stuck in the tool-calling loop.

**Solutions:**

- Increase `mcp.startup_timeout` in `config.yaml` (default is 10 seconds)
- Check that the MCP tool server script exists: `ls mcp_tools/<tool-name>/server.py`
- Check the logs for specific tool errors: `journalctl --user -u claw | grep "MCP"`
- Disable problematic tools by removing them from `mcp.enabled_servers` in `config.yaml`
- Restart the service to re-initialize MCP connections: `systemctl --user restart claw`

**Note:** MCP servers run as subprocesses with stdio transport. If a tool server crashes, the registry logs a warning and continues without that tool.

## YouTube Music Auth Expiry

**Symptom:** Music searches return errors or empty results. Playback commands fail.

**Cause:** Browser cookies used for YouTube Music authentication expire periodically.

**Solution:**

1. Re-run the auth setup to extract fresh cookies:
   ```bash
   source ~/claw/.venv/bin/activate
   python mcp_tools/youtube_music/setup_auth.py
   ```
2. Restart the service: `systemctl --user restart claw`

**Other YouTube Music issues:**

- **mpv not found** -- ensure mpv is installed: `sudo apt install mpv`
- **No audio output** -- check PipeWire is routing audio to the correct output device
- **Playback stops** -- the mpv process may have crashed. Restart the service to re-initialize the player

## Google OAuth Errors

**Symptom:** Gmail or Google Calendar tools fail with authentication errors.

**Diagnosis:**

Check the logs for specific Google API errors:

```bash
journalctl --user -u claw | grep -i "google\|gmail\|calendar"
```

**Common issues:**

- **Token expired** -- tokens refresh automatically, but if the refresh token is revoked, re-run the auth setup:
  ```bash
  source ~/claw/.venv/bin/activate
  python mcp_tools/google_auth/setup_auth.py
  ```
- **Missing credentials.json** -- ensure `data/google/credentials.json` exists and contains valid OAuth 2.0 credentials
- **Incorrect scopes** -- if you added new API scopes, delete the token file (e.g., `data/google/tokens/personal.json`) and re-authorize
- **API not enabled** -- ensure the Gmail API and Google Calendar API are enabled in the Google Cloud Console for your project
- **Account not configured** -- check that the account is properly defined under `google_auth.accounts` in `config.yaml` with the correct `email` and `token_file`

## Config Hot-Reload Not Working

**Symptom:** Changes to `config.yaml` are not picked up without a restart.

**Solutions:**

- **watchfiles not installed** -- ensure it is installed: `pip install watchfiles`
- **File not saved** -- the watcher triggers on filesystem write events. Ensure your editor actually writes the file (some editors write to a temp file and rename)
- **MCP tools** -- MCP tool servers cache their config at startup and do NOT hot-reload. Restart the service for MCP config changes: `systemctl --user restart claw`
- **Check logs** -- look for "Config file changed, reloading..." in the logs:
  ```bash
  journalctl --user -u claw | grep -i "reload"
  ```

**What hot-reloads and what does not:**

| Reloads automatically | Requires restart |
|----------------------|-----------------|
| Wake word thresholds | MCP enabled_servers list |
| LLM model, temperature, etc. | MCP tool-specific config (youtube_music, weather keys) |
| TTS settings | New MCP tool servers |
| Audio silence thresholds | |
| Admin panel settings | |

## TTS Not Speaking

**Symptom:** The Claw processes queries and shows responses in the admin panel, but does not speak.

**Solutions:**

- **TTS disabled** -- check `tts.enabled` and `tts.voice_loop_enabled` in `config.yaml`. Both must be `true` for voice responses
- **Piper model missing** -- verify the model file exists at the path specified in `tts.piper_model` (default: `data/models/tts/piper/en_US-lessac-medium.onnx`). Re-run `install.sh` to re-download the model
- **No audio output device** -- ensure PipeWire has a valid output sink: `wpctl status`
- **Piper not installed** -- install TTS dependencies: `pip install -e ".[tts]"`
- **Fish Speech server down** -- if using Fish Speech as the TTS engine, ensure the server is running at the configured URL

**Testing TTS independently:**

```bash
source ~/claw/.venv/bin/activate
python -c "
from claw.audio_pipeline.tts.manager import TTSManager
mgr = TTSManager()
mgr.initialize()
import asyncio
asyncio.run(mgr.speak('Hello, this is a test.'))
mgr.shutdown()
"
```

## General Debugging

### Enable Debug Logging

```bash
# When running directly
python -m claw --mode cli --log-level DEBUG

# For the systemd service, edit the unit file
systemctl --user edit claw
# Add under [Service]:
#   Environment=CLAW_LOG_LEVEL=DEBUG
# Then restart
systemctl --user restart claw
```

### Check Service Logs

```bash
# Follow logs live
journalctl --user -u claw -f

# Last 100 lines
journalctl --user -u claw -n 100

# Since last boot
journalctl --user -u claw -b

# Errors only
journalctl --user -u claw -p err
```

### Admin Panel Logs

The admin panel at `http://localhost:8080` includes a real-time log viewer accessible from the navigation menu. This can be easier to scan than raw journal output.

### Check Component Status

The admin panel Status page shows the current state of all subsystems:
- Audio pipeline state (idle, listening, processing, speaking)
- Connected MCP servers and available tools
- Memory store statistics (conversation count, fact count)
- Last transcription and response
