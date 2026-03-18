"""Slack bridge adapter using slack-bolt Socket Mode."""

from __future__ import annotations

import asyncio
import logging
from typing import TYPE_CHECKING

try:
    from slack_bolt.adapter.socket_mode.async_handler import AsyncSocketModeHandler
    from slack_bolt.async_app import AsyncApp

    _HAS_SLACK = True
except ImportError:
    _HAS_SLACK = False

from claw.bridge.base import BridgeAdapter, InboundMessage, PlatformLimits

if TYPE_CHECKING:
    from claw.bridge.manager import BridgeManager

log = logging.getLogger(__name__)


class SlackAdapter(BridgeAdapter):
    """Slack messaging bridge using Socket Mode (no public URL needed).

    Config keys:
        bot_token_secret -- encrypted secret name for the Bot User OAuth Token
        app_token_secret -- encrypted secret name for the App-Level Token
        allowed_channels -- list of channel IDs (empty = all)
        allowed_users    -- list of user IDs (empty = all)
    """

    PLATFORM = "slack"
    LIMITS = PlatformLimits(
        max_message_length=40_000,
        markdown_flavor="slack",
        supports_typing=True,
        supports_reactions=True,
        supports_threads=True,
        supports_media=True,
    )

    def __init__(self, config: dict, manager: BridgeManager) -> None:
        if not _HAS_SLACK:
            raise ImportError(
                "slack-bolt is required for the Slack adapter "
                "(pip install 'slack-bolt>=1' 'slack-sdk>=3')"
            )
        super().__init__(config, manager)
        self._app: AsyncApp | None = None
        self._handler: AsyncSocketModeHandler | None = None
        self._bot_user_id: str | None = None
        self._allowed_channels: set[str] = set(
            str(c) for c in config.get("allowed_channels", [])
        )
        self._allowed_users: set[str] = set(
            str(u) for u in config.get("allowed_users", [])
        )

    # ------------------------------------------------------------------
    # helpers
    # ------------------------------------------------------------------

    def _get_tokens(self) -> tuple[str, str]:
        from claw.secret_store import load as secret_load

        bot_secret = self._config.get("bot_token_secret", "slack_bot_token")
        app_secret = self._config.get("app_token_secret", "slack_app_token")
        bot_token = secret_load(bot_secret)
        app_token = secret_load(app_secret)
        if not bot_token:
            raise RuntimeError(
                f"Slack bot token not found in secret store (key: {bot_secret})"
            )
        if not app_token:
            raise RuntimeError(
                f"Slack app token not found in secret store (key: {app_secret})"
            )
        return bot_token, app_token

    # ------------------------------------------------------------------
    # lifecycle
    # ------------------------------------------------------------------

    async def start(self) -> None:  # noqa: C901
        bot_token, app_token = self._get_tokens()

        self._app = AsyncApp(token=bot_token)
        adapter = self

        # Discover bot's own user ID for mention detection
        auth_response = await self._app.client.auth_test()
        self._bot_user_id = auth_response["user_id"]

        @self._app.event("message")
        async def _handle_message(event: dict, say: object) -> None:
            # Skip bot messages and subtypes (edits, joins, etc.)
            if event.get("subtype"):
                return

            user_id = event.get("user", "")
            if not user_id:
                return
            if adapter._allowed_users and user_id not in adapter._allowed_users:
                return

            channel_id = event.get("channel", "")
            if (
                adapter._allowed_channels
                and channel_id not in adapter._allowed_channels
            ):
                return

            text = event.get("text", "")
            channel_type = event.get("channel_type", "")
            is_direct = channel_type in ("im", "mpim")
            is_mention = (
                f"<@{adapter._bot_user_id}>" in text if not is_direct else False
            )

            # Strip bot mention from text
            if is_mention and adapter._bot_user_id:
                text = text.replace(f"<@{adapter._bot_user_id}>", "").strip()

            # Resolve display name
            user_name = user_id
            if adapter._app is not None:
                try:
                    user_info = await adapter._app.client.users_info(user=user_id)
                    profile = user_info["user"]["profile"]
                    user_name = (
                        profile.get("display_name")
                        or profile.get("real_name", user_id)
                    )
                except Exception:
                    pass

            msg = InboundMessage(
                platform=adapter.PLATFORM,
                user_id=user_id,
                user_name=user_name,
                channel_id=channel_id,
                text=text,
                is_direct=is_direct,
                is_mention=is_mention,
                reply_context={
                    "channel": channel_id,
                    "thread_ts": event.get("ts"),
                },
            )
            await adapter.on_message(msg)

        self._handler = AsyncSocketModeHandler(self._app, app_token)
        await self._handler.start_async()
        log.info("[slack] Socket Mode connected (bot user: %s)", self._bot_user_id)

        # Block until the adapter is told to stop
        while self._running:
            await asyncio.sleep(1)

    async def stop(self) -> None:
        self._running = False
        if self._handler is not None:
            await self._handler.close_async()
            self._handler = None

    # ------------------------------------------------------------------
    # messaging
    # ------------------------------------------------------------------

    async def send_message(
        self,
        channel_id: str,
        text: str,
        reply_context: dict | None = None,
    ) -> None:
        if self._app is None:
            return
        kwargs: dict = {"channel": channel_id, "text": text}
        if reply_context and reply_context.get("thread_ts"):
            kwargs["thread_ts"] = reply_context["thread_ts"]
        await self._app.client.chat_postMessage(**kwargs)

    async def send_typing(self, channel_id: str) -> None:
        # Slack's Web API does not expose a typing-indicator endpoint for bots.
        # The typing indicator is only available via the RTM API which is
        # deprecated for new apps.
        pass
