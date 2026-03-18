"""IRC bridge adapter using a lightweight asyncio client."""

from __future__ import annotations

import asyncio
import logging
import re
from typing import TYPE_CHECKING

from claw.bridge.base import BridgeAdapter, InboundMessage, PlatformLimits

if TYPE_CHECKING:
    from claw.bridge.manager import BridgeManager

log = logging.getLogger(__name__)


class IRCAdapter(BridgeAdapter):
    """IRC messaging bridge using a lightweight asyncio client.

    No external dependencies required — implements the subset of the IRC
    protocol needed for messaging (NICK, USER, JOIN, PRIVMSG, PING/PONG).

    Config keys:
        server         -- IRC server hostname (default: "irc.libera.chat")
        port           -- port number (default: 6697)
        nickname       -- bot nick (default: "claw-bot")
        channels       -- list of channels to join, e.g. ["#mychannel"]
        password_secret-- encrypted secret name for server/NickServ password
        use_tls        -- enable TLS (default: True)
        allowed_users  -- list of nicks allowed to interact (empty = all)
    """

    PLATFORM = "irc"
    LIMITS = PlatformLimits(
        max_message_length=450,  # 512 - protocol overhead
        markdown_flavor="irc",
        supports_typing=False,
        supports_reactions=False,
        supports_threads=False,
        supports_media=False,
    )

    def __init__(self, config: dict, manager: BridgeManager) -> None:
        super().__init__(config, manager)
        self._server: str = config.get("server", "irc.libera.chat")
        self._port: int = config.get("port", 6697)
        self._nickname: str = config.get("nickname", "claw-bot")
        self._channels: list[str] = config.get("channels", [])
        self._use_tls: bool = config.get("use_tls", True)
        self._allowed_users: set[str] = set(config.get("allowed_users", []))
        self._reader: asyncio.StreamReader | None = None
        self._writer: asyncio.StreamWriter | None = None

    # ------------------------------------------------------------------
    # helpers
    # ------------------------------------------------------------------

    def _get_password(self) -> str | None:
        secret_name = self._config.get("password_secret", "")
        if not secret_name:
            return None
        from claw.secret_store import load as secret_load

        return secret_load(secret_name)

    async def _send_raw(self, message: str) -> None:
        """Send a raw IRC protocol line and flush the buffer."""
        if self._writer is not None:
            self._writer.write(f"{message}\r\n".encode("utf-8"))
            try:
                await self._writer.drain()
            except (ConnectionError, OSError):
                pass  # Will be caught by the read loop

    # ------------------------------------------------------------------
    # lifecycle
    # ------------------------------------------------------------------

    async def start(self) -> None:
        if self._use_tls:
            import ssl

            ssl_ctx = ssl.create_default_context()
            self._reader, self._writer = await asyncio.open_connection(
                self._server, self._port, ssl=ssl_ctx
            )
        else:
            self._reader, self._writer = await asyncio.open_connection(
                self._server, self._port
            )

        # Registration sequence
        password = self._get_password()
        if password:
            await self._send_raw(f"PASS {password}")
        await self._send_raw(f"NICK {self._nickname}")
        await self._send_raw(f"USER {self._nickname} 0 * :The Claw Bot")

        log.info(
            "[irc] Connecting to %s:%d as %s (TLS=%s)",
            self._server,
            self._port,
            self._nickname,
            self._use_tls,
        )

        # Read loop
        while self._running:
            try:
                line = await asyncio.wait_for(
                    self._reader.readline(), timeout=300
                )
            except asyncio.TimeoutError:
                # Keep-alive: send a PING if idle too long
                await self._send_raw(f"PING :{self._server}")
                continue
            except asyncio.CancelledError:
                break

            if not line:
                break  # Connection closed by server

            decoded = line.decode("utf-8", errors="replace").strip()
            if not decoded:
                continue

            await self._handle_line(decoded)

    async def stop(self) -> None:
        self._running = False
        if self._writer is not None:
            try:
                await self._send_raw("QUIT :Shutting down")
                self._writer.close()
                await self._writer.wait_closed()
            except Exception:
                pass
            self._writer = None
            self._reader = None

    # ------------------------------------------------------------------
    # messaging
    # ------------------------------------------------------------------

    async def send_message(
        self,
        channel_id: str,
        text: str,
        reply_context: dict | None = None,
    ) -> None:
        """Send a PRIVMSG to a channel or user.

        Long messages are split by newline so each IRC line stays within
        protocol limits.
        """
        if self._writer is None:
            return
        for line in text.split("\n"):
            stripped = line.strip()
            if stripped:
                await self._send_raw(f"PRIVMSG {channel_id} :{stripped}")

    # ------------------------------------------------------------------
    # IRC protocol handling
    # ------------------------------------------------------------------

    async def _handle_line(self, line: str) -> None:
        """Parse and dispatch an IRC protocol line."""
        # PING from server — must reply immediately
        if line.startswith("PING"):
            pong_target = line.split(":", 1)[1] if ":" in line else ""
            await self._send_raw(f"PONG :{pong_target}")
            return

        # General format: :prefix COMMAND params :trailing
        match = re.match(r"^(?::(\S+)\s)?(\S+)\s?(.*)", line)
        if not match:
            return

        prefix, command, params = match.groups()

        if command == "001":
            # RPL_WELCOME — server accepted our registration
            log.info("[irc] Registered with server")
            for channel in self._channels:
                await self._send_raw(f"JOIN {channel}")
                log.info("[irc] Joining %s", channel)

        elif command == "PRIVMSG":
            await self._handle_privmsg(prefix or "", params)

    async def _handle_privmsg(self, prefix: str, params: str) -> None:
        """Handle an incoming PRIVMSG."""
        if not prefix:
            return

        # Extract nick from nick!user@host
        nick = prefix.split("!")[0]

        if self._allowed_users and nick not in self._allowed_users:
            return

        # Split "target :message"
        parts = params.split(" :", 1)
        if len(parts) < 2:
            return

        target = parts[0].strip()
        text = parts[1].strip()

        # DM if target is our nick (not a channel)
        is_direct = not target.startswith("#") and not target.startswith("&")
        is_mention = False

        # Check for "claw-bot: some text" style mention in channels
        if not is_direct:
            mention_re = re.compile(
                rf"^{re.escape(self._nickname)}[,:]\s*", re.IGNORECASE
            )
            if mention_re.match(text):
                is_mention = True
                text = mention_re.sub("", text).strip()

        # For DMs, reply back to the sender; for channels, reply in the channel
        channel_id = nick if is_direct else target

        msg = InboundMessage(
            platform=self.PLATFORM,
            user_id=nick,
            user_name=nick,
            channel_id=channel_id,
            text=text,
            is_direct=is_direct,
            is_mention=is_mention,
            reply_context={},
        )
        await self.on_message(msg)
