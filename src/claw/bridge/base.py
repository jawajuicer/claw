"""Base types and abstract adapter for messaging bridges."""

from __future__ import annotations

import asyncio
import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from claw.bridge.manager import BridgeManager

log = logging.getLogger(__name__)


@dataclass
class PlatformLimits:
    """Platform-specific constraints for message formatting."""
    max_message_length: int = 4096
    markdown_flavor: str = "standard"  # "standard", "telegram", "slack", "html", "irc"
    supports_typing: bool = True
    supports_reactions: bool = False
    supports_threads: bool = False
    supports_media: bool = False


@dataclass
class InboundMessage:
    """Normalized inbound message from any platform."""
    platform: str
    user_id: str
    user_name: str
    channel_id: str
    text: str
    is_direct: bool = True  # True for DMs, False for group messages
    is_mention: bool = False  # True if bot was @mentioned in group
    is_admin: bool = True  # True if user can execute tools (default: all users)
    reply_context: dict = field(default_factory=dict)  # platform-specific reply data
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))


class BridgeAdapter(ABC):
    """Abstract base class for platform-specific message adapters.

    Subclasses implement platform connection logic. Messages are routed
    through BridgeManager for agent processing.
    """

    PLATFORM: str = ""
    LIMITS: PlatformLimits = PlatformLimits()

    def __init__(self, config: dict, manager: BridgeManager) -> None:
        self._config = config
        self._manager = manager
        self._running = False
        self._reconnect_delay = 2
        self._max_reconnect_delay = 300

    @property
    def running(self) -> bool:
        return self._running

    @abstractmethod
    async def start(self) -> None:
        """Connect to the platform and begin receiving messages."""

    @abstractmethod
    async def stop(self) -> None:
        """Disconnect from the platform gracefully."""

    @abstractmethod
    async def send_message(self, channel_id: str, text: str, reply_context: dict | None = None) -> None:
        """Send a message to a specific channel/user."""

    async def send_typing(self, channel_id: str) -> None:
        """Send typing indicator. Override if platform supports it."""

    async def on_message(self, msg: InboundMessage) -> None:
        """Route an inbound message through the bridge manager.

        Called by adapter subclasses when a message is received.
        Handles group chat filtering (only respond to DMs or @mentions).
        """
        # Group chat policy: only respond to DMs or @mentions
        if not msg.is_direct and not msg.is_mention:
            return

        # Send typing indicator if supported
        if self.LIMITS.supports_typing:
            try:
                await self.send_typing(msg.channel_id)
            except Exception:
                pass

        # Route through manager for agent processing
        response = await self._manager.dispatch(msg)
        if response:
            await self.send_message(msg.channel_id, response, msg.reply_context)

    async def run_with_reconnect(self) -> None:
        """Run the adapter with exponential backoff reconnection.

        Wraps start() in a retry loop. On disconnect, waits with
        exponential backoff before reconnecting.
        """
        self._running = True
        delay = self._reconnect_delay

        while self._running:
            try:
                log.info("[%s] Connecting...", self.PLATFORM)
                await self.start()
                # If start() returns normally, reset delay
                delay = self._reconnect_delay
            except asyncio.CancelledError:
                break
            except Exception:
                log.exception("[%s] Connection error, retrying in %ds", self.PLATFORM, delay)
                await asyncio.sleep(delay)
                delay = min(delay * 2, self._max_reconnect_delay)
            else:
                if self._running:
                    log.warning("[%s] Disconnected, retrying in %ds", self.PLATFORM, delay)
                    await asyncio.sleep(delay)
                    delay = min(delay * 2, self._max_reconnect_delay)

        log.info("[%s] Adapter stopped", self.PLATFORM)
