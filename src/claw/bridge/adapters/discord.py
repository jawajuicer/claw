"""Discord bridge adapter using discord.py."""

from __future__ import annotations

import asyncio
import logging
from typing import TYPE_CHECKING

try:
    import discord

    _HAS_DISCORD = True
except ImportError:
    _HAS_DISCORD = False

from claw.bridge.base import BridgeAdapter, InboundMessage, PlatformLimits

if TYPE_CHECKING:
    from claw.bridge.manager import BridgeManager

log = logging.getLogger(__name__)


class DiscordAdapter(BridgeAdapter):
    """Discord messaging bridge using discord.py Gateway WebSocket.

    Config keys:
        token_secret     -- name of the encrypted secret holding the bot token
        allowed_channels -- list of channel IDs to listen on (empty = all)
        allowed_users    -- list of user IDs to accept messages from (empty = all)
    """

    PLATFORM = "discord"
    LIMITS = PlatformLimits(
        max_message_length=2000,
        markdown_flavor="standard",
        supports_typing=True,
        supports_reactions=True,
        supports_threads=True,
        supports_media=True,
    )

    def __init__(self, config: dict, manager: BridgeManager) -> None:
        if not _HAS_DISCORD:
            raise ImportError(
                "discord.py is required for the Discord adapter "
                "(pip install 'discord.py>=2')"
            )
        super().__init__(config, manager)
        self._client: discord.Client | None = None
        self._allowed_channels: set[str] = set(
            str(c) for c in config.get("allowed_channels", [])
        )
        self._allowed_users: set[str] = set(
            str(u) for u in config.get("allowed_users", [])
        )
        self._group_sessions: bool = config.get("group_sessions", True)

    # ------------------------------------------------------------------
    # helpers
    # ------------------------------------------------------------------

    def _get_token(self) -> str:
        from claw.secret_store import load as secret_load

        secret_name = self._config.get("token_secret", "discord_bot_token")
        token = secret_load(secret_name)
        if not token:
            raise RuntimeError(
                f"Discord bot token not found in secret store (key: {secret_name})"
            )
        return token

    def _is_allowed_channel(self, channel_id: str) -> bool:
        if not self._allowed_channels:
            return True
        return str(channel_id) in self._allowed_channels

    def _is_allowed_user(self, user_id: str) -> bool:
        if not self._allowed_users:
            return True
        return str(user_id) in self._allowed_users

    async def get_group_members(self, channel_id: str) -> list[dict]:
        """Fetch members of a Discord channel (guild text channel).

        Returns a list of dicts: [{"uuid": ..., "name": ...}, ...]
        Excludes the bot itself.
        """
        if self._client is None or self._client.user is None:
            return []
        try:
            channel = self._client.get_channel(int(channel_id))
            if channel is None:
                channel = await self._client.fetch_channel(int(channel_id))
            if channel is None or not hasattr(channel, "members"):
                return []
            members = []
            for member in channel.members:  # type: ignore[union-attr]
                if member == self._client.user:
                    continue
                members.append({
                    "uuid": str(member.id),
                    "name": member.display_name,
                })
            return members
        except Exception:
            log.exception("[discord] Failed to fetch group members for %s", channel_id)
            return []

    # ------------------------------------------------------------------
    # lifecycle
    # ------------------------------------------------------------------

    async def start(self) -> None:
        intents = discord.Intents.default()
        intents.message_content = True

        self._client = discord.Client(intents=intents)
        adapter = self

        @self._client.event
        async def on_ready() -> None:
            log.info("[discord] Connected as %s", adapter._client.user)

        @self._client.event
        async def on_message(message: discord.Message) -> None:
            client = adapter._client
            if client is None or client.user is None:
                return
            # Ignore own messages
            if message.author == client.user:
                return
            if not adapter._is_allowed_user(str(message.author.id)):
                return
            if not adapter._is_allowed_channel(str(message.channel.id)):
                return

            is_direct = isinstance(message.channel, discord.DMChannel)
            is_mention = client.user in message.mentions if not is_direct else False

            text = message.content
            # Strip bot mention from text
            if is_mention:
                text = text.replace(f"<@{client.user.id}>", "").strip()
                text = text.replace(f"<@!{client.user.id}>", "").strip()

            msg = InboundMessage(
                platform=adapter.PLATFORM,
                user_id=str(message.author.id),
                user_name=message.author.display_name,
                channel_id=str(message.channel.id),
                text=text,
                is_direct=is_direct,
                is_mention=is_mention,
                reply_context={
                    "message_id": message.id,
                    "channel_id": message.channel.id,
                },
            )
            await adapter.on_message(msg)

        token = self._get_token()
        # client.start() blocks until the client disconnects
        await self._client.start(token)

    async def stop(self) -> None:
        self._running = False
        if self._client is not None:
            await self._client.close()
            self._client = None

    # ------------------------------------------------------------------
    # messaging
    # ------------------------------------------------------------------

    async def send_message(
        self,
        channel_id: str,
        text: str,
        reply_context: dict | None = None,
    ) -> None:
        if self._client is None:
            return
        channel = self._client.get_channel(int(channel_id))
        if channel is None:
            channel = await self._client.fetch_channel(int(channel_id))
        if channel is None:
            log.warning("[discord] Could not resolve channel %s", channel_id)
            return

        kwargs: dict = {"content": text}
        if reply_context and reply_context.get("message_id"):
            try:
                ref_msg = await channel.fetch_message(reply_context["message_id"])  # type: ignore[union-attr]
                kwargs["reference"] = ref_msg
            except Exception:
                pass  # Fall through and send without reference

        await channel.send(**kwargs)  # type: ignore[union-attr]

    async def send_typing(self, channel_id: str) -> None:
        if self._client is None:
            return
        channel = self._client.get_channel(int(channel_id))
        if channel is not None and hasattr(channel, "typing"):
            await channel.typing()  # type: ignore[union-attr]
