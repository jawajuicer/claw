"""Main orchestrator — wires all components and runs the event loop."""

from __future__ import annotations

import asyncio
import logging
import re
import signal

import sounddevice as sd
import uvicorn

from claw.admin.app import create_admin_app
from claw.admin.sse import StatusBroadcaster
from claw.agent_core.agent import Agent
from claw.agent_core.llm_client import LLMClient
from claw.audio_pipeline.chime import play_listening_chime
from claw.config import get_settings, watch_config
from claw.mcp_handler.registry import MCPRegistry
from claw.mcp_handler.router import ToolRouter
from claw.memory_engine.retriever import MemoryRetriever
from claw.memory_engine.store import MemoryStore

log = logging.getLogger(__name__)

# Pattern to strip wake word prefixes from transcription.
# The audio buffer may capture speech overlapping the wake word,
# so STT can produce "hey claw play something" instead of "play something".
_WAKE_PREFIX_RE = re.compile(
    r"^(?:hey\s+claw|claw)[,;:.\s!]*",
    re.IGNORECASE,
)


def _strip_wake_prefix(text: str) -> str:
    """Remove wake word prefix from transcribed text."""
    stripped = _WAKE_PREFIX_RE.sub("", text).strip()
    return stripped if stripped else text


class Claw:
    """Top-level orchestrator that owns all subsystems."""

    def __init__(self) -> None:
        self.settings = get_settings()
        self.broadcaster = StatusBroadcaster()

        # Memory
        self.store = MemoryStore()
        self.retriever = MemoryRetriever(self.store)

        # LLM + Agent
        self.llm = LLMClient()
        self.agent = Agent(self.llm, self.retriever)

        # MCP
        self.registry = MCPRegistry()
        self.router = ToolRouter(self.registry)
        self.agent.tool_router = self.router

        # TTS (initialized in initialize())
        self.tts = None

        # Admin
        self.admin_app = create_admin_app(self.broadcaster, self.agent, self.registry)

        self._shutdown_event = asyncio.Event()
        self._bg_tasks: set[asyncio.Task] = set()

    async def initialize(self) -> None:
        """Initialize all subsystems."""
        log.info("Initializing The Claw...")

        # Memory store (loads embedding model + opens ChromaDB)
        self.store.initialize()
        await self.broadcaster.update_memory_stats(self.retriever.get_stats())

        # MCP servers
        try:
            await self.registry.initialize()
            servers = self.registry.list_servers()
            await self.broadcaster.update_mcp_servers(servers)
            log.info("MCP servers: %s", servers)
        except Exception:
            log.exception("MCP initialization failed (non-fatal, continuing)")

        # TTS
        try:
            from claw.audio_pipeline.tts.manager import TTSManager

            self.tts = TTSManager()
            self.tts.initialize()
            self.admin_app.state.tts = self.tts
        except Exception:
            log.exception("TTS initialization failed (non-fatal, continuing)")
            self.tts = None
            self.admin_app.state.tts = None

        log.info("The Claw initialized")

    async def run_voice_loop(self) -> None:
        """Main voice interaction loop: wake word → record → transcribe → agent.

        Self-healing: if audio capture fails, retries with exponential backoff
        (up to 60s). Transient hardware errors (USB disconnect/reconnect) are
        recovered automatically without restarting the service.
        """
        from claw.audio_pipeline.capture import AudioCapture
        from claw.audio_pipeline.transcriber import Transcriber
        from claw.audio_pipeline.wake_word import WakeWordDetector

        wake = WakeWordDetector()
        transcriber = Transcriber()
        wake.load()
        transcriber.load()

        retry_delay = 2
        max_retry_delay = 60

        while not self._shutdown_event.is_set():
            capture = AudioCapture()
            try:
                capture.start()
            except Exception:
                log.exception(
                    "Audio capture failed to start, retrying in %ds", retry_delay
                )
                await self.broadcaster.update_state("error")
                await asyncio.sleep(retry_delay)
                retry_delay = min(retry_delay * 2, max_retry_delay)
                continue

            # Successfully started — reset backoff
            retry_delay = 2
            await self.broadcaster.update_state("idle")
            log.info("Voice loop started — listening for wake word")

            try:
                await self._voice_loop_inner(capture, wake, transcriber)
            except sd.PortAudioError:
                log.exception("Audio device error, restarting capture")
            except Exception:
                log.exception("Unexpected voice loop error, restarting capture")
            finally:
                capture.stop()

            if not self._shutdown_event.is_set():
                await asyncio.sleep(retry_delay)

    async def _voice_loop_inner(
        self,
        capture: "AudioCapture",
        wake: "WakeWordDetector",
        transcriber: "Transcriber",
    ) -> None:
        """Inner voice loop — separated so the outer loop can restart on errors."""
        while not self._shutdown_event.is_set():
            chunk = capture.read_chunk()
            if chunk is None:
                await asyncio.sleep(0.01)
                continue

            # Wake word detection
            detected = wake.process_chunk(chunk)
            if detected is None:
                continue

            # Wake word triggered
            log.info("Wake word detected: %s", detected)
            await self.broadcaster.update_state("listening", last_wake_word=detected)
            await asyncio.to_thread(play_listening_chime)

            # Conversation turn loop: handles follow-up questions
            # from Claw without requiring the wake word again.
            # All tools are passed — the agent's tool router selects
            # only what's needed via a lightweight manifest call.
            tools = self.registry.get_openai_tools() or None
            response = ""

            while not self._shutdown_event.is_set():
                # Record until silence — buffer already contains any speech
                # the user said after the wake word (continuous utterances
                # like "hey jarvis tell me a joke" are preserved)
                audio = await capture.record_until_silence()
                if len(audio) < self.settings.audio.sample_rate * 0.5:
                    log.info("Recording too short, ignoring")
                    break

                # Transcribe
                await self.broadcaster.update_state("processing")
                text = await transcriber.transcribe(audio)
                if not text.strip():
                    log.info("Empty transcription, ignoring")
                    break

                # Strip wake word prefix from transcription — the audio
                # buffer may include speech overlapping the wake word.
                text = _strip_wake_prefix(text)

                await self.broadcaster.update_transcription(text)
                log.info("User said: %s", text)

                # Agent processing (with tools so voice can use get_time etc.)
                response = await self.agent.process_utterance(text, tools=tools)
                await self.broadcaster.update_response(response)
                log.info("Claw responds: %s", response)

                # TTS playback
                if self.tts and self.settings.tts.voice_loop_enabled:
                    log.info("Starting TTS playback...")
                    await self.broadcaster.update_state("speaking")
                    wake.pause()
                    try:
                        await self.tts.speak(response)
                        log.info("TTS playback complete")
                    except Exception:
                        log.exception("TTS playback failed")
                    finally:
                        wake.resume()
                        capture.flush()
                else:
                    log.info("TTS skipped (tts=%s, voice_loop=%s)",
                             self.tts is not None, self.settings.tts.voice_loop_enabled)

                # If Claw asked a follow-up question, listen again.
                # Check last 10 chars to handle trailing emojis after "?"
                if "?" in response.rstrip()[-10:]:
                    log.info("Follow-up question detected, listening for response")
                    await self.broadcaster.update_state("listening")
                    await asyncio.to_thread(play_listening_chime)
                    continue

                # Otherwise, conversation turn is done
                break

            # Post-conversation fact extraction (background, gated)
            if response and not (self.agent.last_usage and self.agent.last_usage.timed_out):
                self._spawn_bg(self._gated_fact_extraction())

            await self.broadcaster.update_state("idle")

    async def run_cli_loop(self) -> None:
        """Text input loop for development/testing without audio."""
        log.info("CLI mode started — type your messages (Ctrl+C to quit)")
        await self.broadcaster.update_state("idle")

        while not self._shutdown_event.is_set():
            try:
                text = await asyncio.to_thread(input, "\nYou: ")
            except (EOFError, KeyboardInterrupt):
                break

            text = text.strip()
            if not text:
                continue
            if text.lower() in ("quit", "exit"):
                break
            if text.lower() == "/new":
                self.agent.new_session()
                print("[New conversation session started]")
                continue

            await self.broadcaster.update_state("processing")
            await self.broadcaster.update_transcription(text)

            tools = self.registry.get_openai_tools()
            response = await self.agent.process_utterance(text, tools=tools or None)

            print(f"\nClaw: {response}")
            await self.broadcaster.update_response(response)

            # TTS playback in CLI mode
            if self.tts and self.settings.tts.voice_loop_enabled:
                try:
                    await self.tts.speak(response)
                except Exception:
                    log.exception("TTS playback failed")

            await self.broadcaster.update_state("idle")

            # Background fact extraction (gated)
            if not (self.agent.last_usage and self.agent.last_usage.timed_out):
                self._spawn_bg(self._gated_fact_extraction())

    async def run_admin_server(self) -> None:
        """Run the admin panel web server."""
        cfg = self.settings.admin
        config = uvicorn.Config(
            self.admin_app,
            host=cfg.host,
            port=cfg.port,
            log_level="warning",
        )
        server = uvicorn.Server(config)
        log.info("Admin panel running at http://%s:%d", cfg.host, cfg.port)
        await server.serve()

    async def run(self, mode: str = "both") -> None:
        """Run The Claw with the specified mode.

        Args:
            mode: "voice" (audio only), "cli" (text only), or "both" (voice + admin).
        """
        await self.initialize()

        tasks: list[asyncio.Task] = []

        # Always watch config for changes
        tasks.append(asyncio.create_task(watch_config()))

        if mode in ("voice", "both"):
            tasks.append(asyncio.create_task(self.run_voice_loop()))
            tasks.append(asyncio.create_task(self.run_admin_server()))
        elif mode == "cli":
            tasks.append(asyncio.create_task(self.run_cli_loop()))
            tasks.append(asyncio.create_task(self.run_admin_server()))

        # Handle shutdown signals
        loop = asyncio.get_running_loop()
        for sig in (signal.SIGINT, signal.SIGTERM):
            loop.add_signal_handler(sig, self._shutdown_event.set)

        try:
            # Wait for shutdown or any task to complete
            done, pending = await asyncio.wait(tasks, return_when=asyncio.FIRST_COMPLETED)
            for task in done:
                if task.exception():
                    log.error("Task failed: %s", task.exception())
        finally:
            log.info("Shutting down The Claw...")
            self._shutdown_event.set()
            for task in tasks:
                task.cancel()
            await asyncio.gather(*tasks, return_exceptions=True)
            await self.registry.shutdown()
            if self.tts:
                self.tts.shutdown()
            log.info("The Claw shut down")

    def _spawn_bg(self, coro) -> None:
        """Create a tracked background task."""
        task = asyncio.create_task(coro)
        self._bg_tasks.add(task)
        task.add_done_callback(self._bg_tasks.discard)

    async def _gated_fact_extraction(self) -> None:
        """Wait briefly then extract facts, but only if the LLM is free."""
        await asyncio.sleep(2)
        if self.llm.busy:
            log.info("Skipping fact extraction — LLM is busy")
            return
        await self._background_fact_extraction()

    async def _background_fact_extraction(self) -> None:
        """Extract facts from the latest conversation in the background."""
        try:
            facts = await self.agent.extract_facts()
            if facts:
                await self.broadcaster.update_memory_stats(self.retriever.get_stats())
                log.info("Extracted %d facts from conversation", len(facts))
        except Exception:
            log.exception("Background fact extraction failed")
