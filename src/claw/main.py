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
from claw.mcp_handler.stats import ToolStats
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

        # Usage tracking
        from claw.agent_core.usage_tracker import UsageTracker
        self.usage_tracker = UsageTracker()
        self.agent.usage_tracker = self.usage_tracker

        # MCP
        self.registry = MCPRegistry()
        self.tool_stats = ToolStats()
        self.router = ToolRouter(self.registry, stats=self.tool_stats)
        self.agent.tool_router = self.router

        # TTS (initialized in initialize())
        self.tts = None

        # Scheduler (initialized in initialize())
        self.scheduler = None

        # Shared transcriber (initialized in initialize() when remote enabled)
        self.transcriber = None

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

        # Share tool router, stats, and usage tracker with admin app
        self.admin_app.state.tool_router = self.router
        self.admin_app.state.tool_stats = self.tool_stats
        self.admin_app.state.usage_tracker = self.usage_tracker

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

        # Scheduler
        try:
            from claw.scheduler.scheduler import Scheduler

            self.scheduler = Scheduler(
                broadcaster=self.broadcaster,
                tts=self.tts,
                router=self.router,
            )
            self.admin_app.state.scheduler = self.scheduler
        except Exception:
            log.exception("Scheduler initialization failed (non-fatal, continuing)")
            self.scheduler = None
            self.admin_app.state.scheduler = None

        # Shared transcriber for remote audio (loaded when remote is enabled)
        if self.settings.remote.enabled:
            try:
                from claw.audio_pipeline.transcriber import Transcriber

                self.transcriber = Transcriber()
                self.transcriber.load()
                self.admin_app.state.remote_transcriber = self.transcriber
                log.info("Remote transcriber loaded")
            except Exception:
                log.exception("Remote transcriber init failed (non-fatal)")
                self.transcriber = None

        # Pre-warm the LLM so first interaction has no cold start
        try:
            log.info("Pre-warming LLM model...")
            await self.llm.chat_simple("hi")
            log.info("LLM model warm and ready")
        except Exception:
            log.warning("LLM pre-warm failed (non-fatal, will load on first request)")

        log.info("The Claw initialized")

    async def run_voice_loop(self) -> None:
        """Main voice interaction loop: wake word → record → transcribe → agent.

        Self-healing: if audio capture fails, retries with exponential backoff
        (up to 60s). Transient hardware errors (USB disconnect/reconnect) are
        recovered automatically without restarting the service.
        """
        from claw.audio_pipeline.capture import AudioCapture
        from claw.audio_pipeline.transcriber import Transcriber
        from claw.audio_pipeline.vad import StreamingVAD
        from claw.audio_pipeline.wake_word import WakeWordDetector

        wake = WakeWordDetector()
        wake.load()

        # Load streaming VAD for speech-based end-of-utterance detection
        vad: StreamingVAD | None = None
        try:
            vad = StreamingVAD(threshold=self.settings.audio.vad_threshold)
            vad.load()
        except Exception:
            log.warning("Streaming VAD failed to load, falling back to RMS silence detection")

        # Reuse shared transcriber if already loaded (e.g. for remote access),
        # otherwise create a dedicated one for the voice loop.
        if self.transcriber is not None:
            transcriber = self.transcriber
        else:
            transcriber = Transcriber()
            transcriber.load()

        retry_delay = 2
        max_retry_delay = 60

        while not self._shutdown_event.is_set():
            capture = AudioCapture(vad=vad)
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
            # Expose capture and VAD to admin for audio diagnostics page
            self.admin_app.state.audio_capture = capture
            self.admin_app.state.vad = vad
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
        _audio_stats_counter = 0
        _AUDIO_STATS_INTERVAL = 25  # broadcast every ~25 chunks (~2s at 80ms/chunk)

        while not self._shutdown_event.is_set():
            chunk = capture.read_chunk()
            if chunk is None:
                await asyncio.sleep(0.01)
                continue

            # Periodically broadcast audio diagnostics
            _audio_stats_counter += 1
            if _audio_stats_counter >= _AUDIO_STATS_INTERVAL:
                _audio_stats_counter = 0
                vad_obj = getattr(self.admin_app.state, "vad", None)
                stats = capture.get_metrics()
                if vad_obj:
                    stats["vad_speech_prob"] = round(vad_obj.last_speech_prob, 4)
                    stats["vad_threshold"] = vad_obj.threshold
                self._spawn_bg(self.broadcaster.update_audio_stats(stats))

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
                # like "hey claw tell me a joke" are preserved)
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
                    music_played = self._music_was_played(response)
                    music_mode = self.settings.tts.music_announcement

                    if music_played and music_mode == "none":
                        log.info("TTS skipped for music announcement (mode=none)")
                    else:
                        if music_played and music_mode == "before":
                            # Pause music, speak announcement, then resume
                            log.info("Pausing music for TTS announcement")
                            try:
                                await self.router.call_tool("pause", {})
                            except Exception:
                                log.warning("Failed to pause music for announcement")

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
                            if music_played and music_mode == "before":
                                log.info("Resuming music after TTS announcement")
                                try:
                                    await self.router.call_tool("resume", {})
                                except Exception:
                                    log.warning("Failed to resume music after announcement")
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

            # Handle slash commands
            if text.startswith("/"):
                from claw.agent_core.commands import dispatch_command
                result = await dispatch_command(text, self.agent)
                if result:
                    print(f"\n{result['content']}")
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
        tasks.append(asyncio.create_task(watch_config(), name="watch_config"))

        # Scheduler for reminders
        if self.scheduler:
            tasks.append(asyncio.create_task(self.scheduler.run(), name="scheduler"))

        # Usage persistence
        tasks.append(asyncio.create_task(self._persist_usage_loop(), name="persist_usage"))

        if mode in ("voice", "both"):
            tasks.append(asyncio.create_task(self.run_voice_loop(), name="run_voice_loop"))
            tasks.append(asyncio.create_task(self.run_admin_server(), name="run_admin_server"))
        elif mode == "cli":
            tasks.append(asyncio.create_task(self.run_cli_loop(), name="run_cli_loop"))
            tasks.append(asyncio.create_task(self.run_admin_server(), name="run_admin_server"))

        # Handle shutdown signals
        loop = asyncio.get_running_loop()
        for sig in (signal.SIGINT, signal.SIGTERM):
            loop.add_signal_handler(sig, self._shutdown_event.set)

        # Define critical tasks — if these fail, we must shut down
        critical_names = {"run_voice_loop", "run_cli_loop", "run_admin_server"}

        try:
            # Wait for shutdown or task failures; restart non-critical tasks
            remaining = set(tasks)
            while remaining:
                done, remaining = await asyncio.wait(
                    remaining, return_when=asyncio.FIRST_COMPLETED,
                )
                should_shutdown = False
                for task in done:
                    name = task.get_name()
                    if task.exception():
                        log.error("Task %s failed", name, exc_info=task.exception())
                        if name in critical_names:
                            should_shutdown = True
                    else:
                        # Task completed without exception (e.g. clean exit)
                        if name in critical_names:
                            should_shutdown = True
                if should_shutdown or self._shutdown_event.is_set():
                    break
        finally:
            log.info("Shutting down The Claw...")
            self._shutdown_event.set()
            # Cancel background tasks (fact extraction, etc.)
            for bg_task in list(self._bg_tasks):
                bg_task.cancel()
            if self._bg_tasks:
                await asyncio.gather(*self._bg_tasks, return_exceptions=True)
            for task in tasks:
                task.cancel()
            await asyncio.gather(*tasks, return_exceptions=True)
            await self.registry.shutdown()
            if self.scheduler:
                self.scheduler.stop()
            if self.tts:
                self.tts.shutdown()
            await self.usage_tracker.persist()
            log.info("The Claw shut down")

    @staticmethod
    def _music_was_played(response: str) -> bool:
        """Check if the agent response indicates music playback started."""
        return response.lstrip().startswith("Now playing:")

    def _spawn_bg(self, coro) -> None:
        """Create a tracked background task."""
        task = asyncio.create_task(coro)
        self._bg_tasks.add(task)
        task.add_done_callback(self._bg_tasks.discard)

    async def _persist_usage_loop(self) -> None:
        """Periodically save usage data to disk."""
        interval = self.settings.usage.persist_interval
        while not self._shutdown_event.is_set():
            await asyncio.sleep(interval)
            try:
                await self.usage_tracker.persist()
            except Exception:
                log.exception("Usage persistence failed")

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
