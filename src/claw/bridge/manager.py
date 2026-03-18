"""Bridge manager -- orchestrates adapter lifecycle and message routing."""

from __future__ import annotations

import asyncio
import logging
from typing import TYPE_CHECKING

from claw.bridge.base import BridgeAdapter, InboundMessage
from claw.bridge.formatter import format_response
from claw.bridge.session_store import BridgeSessionStore
from claw.config import get_settings

if TYPE_CHECKING:
    from claw.admin.sse import StatusBroadcaster
    from claw.agent_core.agent import Agent
    from claw.mcp_handler.registry import MCPRegistry
    from claw.memory_engine.store import MemoryStore

log = logging.getLogger(__name__)


class BridgeManager:
    """Manages messaging bridge adapters and routes messages to the agent.

    Lifecycle: initialized in Claw.initialize(), run() added as asyncio task,
    stop() called during shutdown. Same pattern as Scheduler.

    Session isolation: each (platform, user_id) gets its own ConversationSession.
    The manager swaps sessions under chat_lock before calling agent.process_utterance().
    """

    def __init__(
        self,
        agent: Agent,
        registry: MCPRegistry,
        broadcaster: StatusBroadcaster,
        chat_lock: asyncio.Lock,
        memory_store: MemoryStore | None = None,
    ) -> None:
        self._agent = agent
        self._registry = registry
        self._broadcaster = broadcaster
        self._chat_lock = chat_lock
        self._memory_store = memory_store
        self._session_store = BridgeSessionStore()
        self._adapters: dict[str, BridgeAdapter] = {}
        self._tasks: dict[str, asyncio.Task] = {}
        self._running = False

    @property
    def running(self) -> bool:
        return self._running

    @property
    def adapters(self) -> dict[str, BridgeAdapter]:
        return dict(self._adapters)

    def _load_adapters(self) -> None:
        """Load enabled bridge adapters based on configuration."""
        settings = get_settings()
        if not hasattr(settings, "bridges"):
            return

        bridges_cfg = settings.bridges
        adapter_map = {
            "telegram": ("claw.bridge.adapters.telegram", "TelegramAdapter"),
            "discord": ("claw.bridge.adapters.discord", "DiscordAdapter"),
            "slack": ("claw.bridge.adapters.slack", "SlackAdapter"),
            "twilio": ("claw.bridge.adapters.twilio", "TwilioAdapter"),
            "matrix": ("claw.bridge.adapters.matrix", "MatrixAdapter"),
            "irc": ("claw.bridge.adapters.irc", "IRCAdapter"),
            "signal": ("claw.bridge.adapters.signal", "SignalAdapter"),
        }

        for platform, (module_path, class_name) in adapter_map.items():
            platform_cfg = getattr(bridges_cfg, platform, None)
            if platform_cfg is None or not platform_cfg.enabled:
                continue

            try:
                import importlib
                module = importlib.import_module(module_path)
                adapter_cls = getattr(module, class_name)

                # Signal adapter accepts memory_store for passive group ingestion
                kwargs: dict = {
                    "config": platform_cfg.model_dump(),
                    "manager": self,
                }
                if platform == "signal" and self._memory_store is not None:
                    kwargs["memory_store"] = self._memory_store

                adapter = adapter_cls(**kwargs)
                self._adapters[platform] = adapter
                log.info("Loaded bridge adapter: %s", platform)
            except ImportError as e:
                log.warning(
                    "Bridge adapter '%s' not available (missing dependency: %s)",
                    platform, e,
                )
            except Exception:
                log.exception("Failed to load bridge adapter: %s", platform)

    async def run(self) -> None:
        """Start all enabled bridge adapters."""
        self._running = True
        self._load_adapters()

        if not self._adapters:
            log.info("No bridge adapters enabled")
            self._running = False
            return

        log.info("Starting %d bridge adapter(s): %s",
                 len(self._adapters), list(self._adapters.keys()))

        for platform, adapter in self._adapters.items():
            task = asyncio.create_task(
                adapter.run_with_reconnect(),
                name=f"bridge_{platform}",
            )
            self._tasks[platform] = task

        # Wait for all tasks (they run until stopped)
        try:
            await asyncio.gather(*self._tasks.values(), return_exceptions=True)
        except asyncio.CancelledError:
            pass

    async def stop(self) -> None:
        """Stop all bridge adapters gracefully."""
        self._running = False
        log.info("Stopping bridge adapters...")

        for platform, adapter in self._adapters.items():
            try:
                await adapter.stop()
            except Exception:
                log.exception("Error stopping %s adapter", platform)

        for platform, task in self._tasks.items():
            task.cancel()

        if self._tasks:
            await asyncio.gather(*self._tasks.values(), return_exceptions=True)

        self._tasks.clear()
        self._adapters.clear()
        log.info("All bridge adapters stopped")

    async def dispatch(self, msg: InboundMessage) -> str | None:
        """Route an inbound message through the agent and return the response.

        Uses session swapping under chat_lock for conversation isolation.
        Each (platform, user_id) gets their own ConversationSession.
        """
        log.info(
            "[%s] Message from %s (%s): %s",
            msg.platform, msg.user_name, msg.user_id, msg.text[:100],
        )

        # Get or create user's session
        user_session = self._session_store.get_or_create(msg.platform, msg.user_id)

        # Get available tools — only for admin users
        if msg.is_admin:
            tools = self._registry.get_openai_tools() or None
        else:
            tools = None
            log.info(
                "[%s] Non-admin user %s (%s) — tools disabled",
                msg.platform, msg.user_name, msg.user_id,
            )

        # Build context for the agent: platform info + fresh memory retrieval.
        # Bridge sessions persist, so session-init memory may be stale.
        context_parts: list[str] = []

        # Platform context — tells the agent where the message came from
        if msg.is_direct:
            context_parts.append(
                f"[{msg.platform.title()} DM from {msg.user_name}]"
            )
        else:
            context_parts.append(
                f"[{msg.platform.title()} group chat — {msg.user_name} mentioned you]"
            )

        # Admin-only context: group members, memory, account info
        if msg.is_admin:
            # Group member list — so the agent knows who's in the chat
            if not msg.is_direct:
                adapter = self._adapters.get(msg.platform)
                if adapter and hasattr(adapter, "get_group_members"):
                    try:
                        members = await adapter.get_group_members(msg.channel_id)
                        if members:
                            names = [m["name"] for m in members]
                            context_parts.append(
                                f"[Group members: {', '.join(names)}]"
                            )
                    except Exception:
                        log.debug("Failed to fetch group members", exc_info=True)

            # Fresh memory retrieval (group observations, past conversations, facts)
            try:
                memory_ctx = await asyncio.wait_for(
                    asyncio.to_thread(
                        self._agent.retriever.retrieve_context,
                        msg.text, 8, 1200,
                    ),
                    timeout=5.0,
                )
                if memory_ctx:
                    context_parts.append(memory_ctx)
            except Exception:
                log.debug(
                    "Memory context retrieval failed for bridge message",
                    exc_info=True,
                )
        else:
            # Non-admin: add a note so the LLM knows not to offer tools
            context_parts.append(
                "[This user does not have admin access. "
                "You can chat and answer general questions, but do NOT "
                "offer to send emails, access calendars, play music, "
                "or use any tools on their behalf.]"
            )

        context = "\n".join(context_parts) if context_parts else None

        async with self._chat_lock:
            # Swap in user's session
            original_session = self._agent._session
            self._agent._session = user_session

            try:
                response = await self._agent.process_utterance(
                    msg.text, tools=tools, context=context,
                )
            except Exception:
                log.exception("[%s] Agent processing failed for %s", msg.platform, msg.user_id)
                response = None
            finally:
                # Save updated session and restore original
                self._session_store.update(
                    msg.platform, msg.user_id, self._agent._session
                )
                self._agent._session = original_session

        if response:
            # Broadcast the interaction via SSE
            await self._broadcaster.broadcast("bridge_message", {
                "platform": msg.platform,
                "user": msg.user_name,
                "text": msg.text[:200],
                "response": response[:200],
            })

            # Format response for the platform
            adapter = self._adapters.get(msg.platform)
            if adapter:
                chunks = format_response(response, adapter.LIMITS)
                # Return first chunk; send remaining chunks directly
                if len(chunks) > 1:
                    for chunk in chunks[1:]:
                        try:
                            await adapter.send_message(
                                msg.channel_id, chunk, msg.reply_context
                            )
                        except Exception:
                            log.exception("[%s] Failed to send continuation chunk", msg.platform)
                return chunks[0] if chunks else response
            return response

        return None

    def get_status(self) -> dict:
        """Return status of all bridge adapters."""
        return {
            platform: {
                "running": adapter.running,
                "platform": adapter.PLATFORM,
            }
            for platform, adapter in self._adapters.items()
        }
