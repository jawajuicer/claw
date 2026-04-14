# Image Handling — Feature Summary

## What Was Added

Claw can now receive and understand images sent through any channel — admin chat, Signal, Discord, and the Android app. When a user sends an image, Claw downloads it, validates and resizes it, then passes it to the Gemma 4 E4B vision model for analysis. The model was already loaded with its multimodal projector (`--mmproj`), so this feature connects the existing vision capability to every input channel.

## How It Works

```
User sends image (Signal / Discord / Admin UI / Android)
    |
    v
Bridge adapter downloads raw image bytes
    |
    v
Bridge Manager validates, resizes (max 1024px), limits to 4 images
    |
    v
Agent builds OpenAI-format multimodal message with base64 image data
    |
    v
LLM (Gemma 4 E4B + mmproj) analyzes the image and responds
```

## New Files

| File | Purpose |
|------|---------|
| `src/claw/agent_core/image_utils.py` | Image validation (Pillow), proportional resize to 1024px max, JPEG/PNG re-encoding, 20MB size limit, max 4 images per message |

## Changed Files

| File | What Changed | Why |
|------|-------------|-----|
| `pyproject.toml` | Added `Pillow>=10.0` dependency | Needed for image validation, format detection, and resizing |
| `src/claw/bridge/base.py` | Added `images: list[bytes]` field to `InboundMessage` | Gives every bridge adapter a standard way to pass image data alongside text |
| `src/claw/agent_core/conversation.py` | Added `add_user_multimodal()` method, `_extract_text()` helper, updated `estimate_tokens()` and `compact()` | Builds the OpenAI multimodal content-parts format (`image_url` with base64 data URI). Helper methods needed updating because they assumed message content was always a string — now it can be a list of text and image parts |
| `src/claw/agent_core/agent.py` | Added `images` parameter to `process_utterance()` and `process_utterance_stream()`, updated fast-model routing | Threads images from the input channels all the way to the conversation session. Images always use the full model (not the fast/lightweight model) since vision requires it |
| `src/claw/bridge/adapters/signal.py` | Added `_download_image_attachments()`, fixed early return for image-only messages | Downloads image attachments from the signal-cli REST API. Previously, messages with no text were silently dropped — now an image-only message (no caption) is processed |
| `src/claw/bridge/adapters/discord.py` | Extract image attachments from `message.attachments` | Downloads image data from Discord CDN, skips files over 10MB |
| `src/claw/bridge/manager.py` | Calls `prepare_images_for_llm()` and passes results to agent | Central place where raw images from any bridge are validated/resized before reaching the agent. Runs in a thread pool to avoid blocking the event loop |
| `src/claw/admin/routes.py` | `/api/chat` and `/api/chat/stream` accept optional `image` field (base64) | Lets the admin web UI send images alongside text messages |
| `src/claw/admin/remote.py` | `/api/remote/chat` accepts optional `image` field (base64) | Lets the Android app send images |
| `src/claw/admin/templates/chat.html` | Added image attach button (paperclip icon), file picker, thumbnail preview, base64 encoding in JS | Users can click the paperclip, select an image, see a preview, and send it with their message |
| `src/claw/admin/static/style.css` | Styles for the attach button and image preview | Visual styling for the new UI elements |

## What QA and Code Review Found

Two independent review agents examined the implementation and found several issues. All were fixed before completion.

### Bugs Found and Fixed

| Bug | Found By | What Was Wrong | Fix |
|-----|----------|---------------|-----|
| **Conversation search crash** | QA | Saved conversations with images store content as a list of parts (not a string). The search endpoint called `.lower()` on this list, which crashes with `AttributeError` | Used the new `_extract_text()` helper to safely extract text from any content format before searching |
| **Corrupt conversation titles** | QA | Auto-generated titles did `content[:50]` — when content is a list, this slices the list instead of a string, producing a title like `[{"type": "text", ...}]` | Extract text first with `_extract_text()`, then slice, with a fallback title of "Image conversation" |
| **Invalid base64 crashes** | Both | Sending malformed base64 in the `image` field caused an unhandled `binascii.Error`, returning a 500 error | Added try/except around `b64decode()` returning a clean 400 error in all three endpoints (chat, stream, remote) |

### Hardening Applied

| Concern | Found By | What Was Done |
|---------|----------|--------------|
| **No raw image size limit** | Code Review | Added a 20MB ceiling in `prepare_images_for_llm()` — images larger than this are skipped before Pillow even tries to decode them, preventing memory spikes |
| **Signal attachment ID field** | QA | signal-cli versions may use `id` or `filename` for the attachment identifier. Now checks both fields, and logs a warning if neither is found |
| **Event loop blocking** | QA | Image resize is CPU-intensive (Pillow decode + resize + re-encode). Wrapped the call in `asyncio.to_thread()` in the bridge manager so it doesn't block polling, message sending, or other async work |

### Noted but Not Changed

| Item | Why Left As-Is |
|------|---------------|
| **Single image per admin chat message** | The backend supports up to 4 images, but the UI sends one at a time. Multi-image upload can be added later without backend changes |
| **Claude Code relay ignores images** | The relay protocol sends only text to the Claude Code subprocess. Images are not supported in relay mode. This is a known limitation, not a bug |
| **Animated GIF/WebP loses animation** | Only the first frame is extracted. This is correct for LLM vision input — models can't process animation |
| **Conversation JSON stores base64 image data** | Saved conversations include the full base64 images, which can make files large. Acceptable for local storage; could be optimized later if needed |
