"""Bridge manager -- orchestrates adapter lifecycle and message routing."""

from __future__ import annotations

import asyncio
import logging
import time
from typing import TYPE_CHECKING

from claw.bridge.base import BridgeAdapter, InboundMessage
from claw.bridge.formatter import format_response
from claw.bridge.profiles import resolve_profile
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

    Session isolation: DMs get per-user sessions, groups get shared sessions.
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
        DMs get per-user sessions, groups get shared sessions.
        """
        log.info(
            "[%s] Message from %s (%s): %s",
            msg.platform, msg.user_name, msg.user_id, msg.text[:100],
        )

        # Resolve channel profile for memory scoping and behavior customization
        profile = resolve_profile(msg.platform, msg.channel_id)
        memory_scope = profile.memory_scope if profile.memory_scope != "shared" else None

        # Determine session key: group vs DM
        adapter = self._adapters.get(msg.platform)
        use_group_session = (
            not msg.is_direct
            and adapter is not None
            and adapter._config.get("group_sessions", True)
        )
        session_key = BridgeSessionStore.make_key(
            msg.platform, msg.user_id, msg.channel_id,
            is_direct=not use_group_session if not msg.is_direct else True,
        )

        # Get or create session
        user_session = self._session_store.get_or_create(session_key)
        log.info(
            "[%s] Session %s: %d messages (sessions_active=%d)",
            msg.platform, session_key,
            len(user_session.messages), self._session_store.count,
        )

        # Get available tools — only for admin users, respecting profile
        if msg.is_admin and profile.tools_enabled:
            tools = self._registry.get_openai_tools() or None
        elif msg.is_admin and not profile.tools_enabled:
            tools = None
            log.info("[%s] Tools disabled by channel profile", msg.platform)
        else:
            tools = None
            log.info(
                "[%s] Non-admin user %s (%s) — tools disabled",
                msg.platform, msg.user_name, msg.user_id,
            )

        # Build context for the agent: platform info + fresh memory retrieval.
        # Bridge sessions persist, so session-init memory may be stale.
        t_dispatch = time.monotonic()
        context_parts: list[str] = []

        # Platform context — tells the agent where the message came from
        if msg.is_direct:
            context_parts.append(
                f"[{msg.platform.title()} DM from {msg.user_name}]"
            )
        else:
            # Enhanced group context with sender attribution guidance
            context_parts.append(
                f"[{msg.platform.title()} group chat — responding to {msg.user_name}.\n"
                f" This is a shared group conversation. Messages from different users\n"
                f" appear with [Name]: prefix. Address users by name.]"
            )

        # Append profile system prompt addon if set
        if profile.system_prompt_addon:
            context_parts.append(profile.system_prompt_addon)

        # Admin-only context: group members + memory retrieval in parallel
        if msg.is_admin:
            # Only fetch memory for new sessions — ongoing conversations
            # already have full context in message history.  Injecting stale
            # ChromaDB results on every turn confuses small models by mixing
            # unrelated past events/emails into the current conversation.
            is_new_session = not user_session.messages

            async def _fetch_group_members() -> str | None:
                if msg.is_direct:
                    return None
                adpt = self._adapters.get(msg.platform)
                if not adpt or not hasattr(adpt, "get_group_members"):
                    return None
                try:
                    members = await adpt.get_group_members(msg.channel_id)
                    if members:
                        names = [m["name"] for m in members]
                        return f"[Group members: {', '.join(names)}]"
                except Exception:
                    log.debug("Failed to fetch group members", exc_info=True)
                return None

            async def _fetch_memory() -> str | None:
                try:
                    return await asyncio.wait_for(
                        asyncio.to_thread(
                            self._agent.retriever.retrieve_context,
                            msg.text, 8, 1200, memory_scope,
                        ),
                        timeout=3.0,
                    )
                except Exception:
                    log.debug(
                        "Memory context retrieval failed for bridge message",
                        exc_info=True,
                    )
                return None

            # Run context assembly — memory only on first message in session
            t_ctx = time.monotonic()
            if is_new_session:
                group_ctx, memory_ctx = await asyncio.gather(
                    _fetch_group_members(), _fetch_memory(),
                )
            else:
                group_ctx = await _fetch_group_members()
                memory_ctx = None
            log.info(
                "[%s] Context assembly: %.1fms (new_session=%s)",
                msg.platform, (time.monotonic() - t_ctx) * 1000, is_new_session,
            )
            if group_ctx:
                context_parts.append(group_ctx)
            if memory_ctx:
                context_parts.append(memory_ctx)
        else:
            # Non-admin: add a note so the LLM knows not to offer tools
            context_parts.append(
                "[This user does not have admin access. "
                "You can chat and answer general questions, but do NOT "
                "offer to send emails, access calendars, play music, "
                "or use any tools on their behalf.]"
            )

        context = "\n".join(context_parts) if context_parts else None

        # Sender attribution for group messages
        if not msg.is_direct:
            agent_text = f"[{msg.user_name}]: {msg.text}"
        else:
            agent_text = msg.text

        # Check if Claude relay is active — relay doesn't use session state,
        # so we can skip the lock to avoid blocking other messages for ~3 min.
        is_relay = (
            hasattr(self._agent, '_claude_relay')
            and self._agent._claude_relay.active
        )

        if is_relay:
            try:
                t_agent = time.monotonic()
                response = await self._agent.process_utterance(
                    agent_text, tools=tools, context=context,
                    _skip_session_memory=True,
                    memory_scope=memory_scope,
                    interactive=True,
                )
                log.info(
                    "[%s] Relay processing (no lock): %.1fms",
                    msg.platform, (time.monotonic() - t_agent) * 1000,
                )
            except Exception:
                log.exception("[%s] Relay processing failed for %s", msg.platform, msg.user_id)
                response = None
        else:
            async with self._chat_lock:
                # Swap in user's session
                original_session = self._agent._session
                self._agent._session = user_session

                try:
                    t_agent = time.monotonic()
                    response = await self._agent.process_utterance(
                        agent_text, tools=tools, context=context,
                        _skip_session_memory=True,
                        memory_scope=memory_scope,
                        interactive=True,
                    )
                    log.info(
                        "[%s] Agent processing: %.1fms",
                        msg.platform, (time.monotonic() - t_agent) * 1000,
                    )
                except Exception:
                    log.exception("[%s] Agent processing failed for %s", msg.platform, msg.user_id)
                    response = None
                finally:
                    # Save updated session and restore original
                    self._session_store.update(
                        session_key, self._agent._session
                    )
                    self._agent._session = original_session

        log.info(
            "[%s] Total dispatch: %.1fms",
            msg.platform, (time.monotonic() - t_dispatch) * 1000,
        )

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
                    for i, chunk in enumerate(chunks[1:]):
                        try:
                            await adapter.send_message(
                                msg.channel_id, chunk, msg.reply_context
                            )
                        except Exception:
                            log.error(
                                "[%s] Failed to send chunk %d/%d after retries",
                                msg.platform, i + 2, len(chunks),
                            )
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
