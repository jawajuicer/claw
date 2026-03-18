"""Telegram bridge adapter using python-telegram-bot."""

from __future__ import annotations

import asyncio
import logging
from typing import TYPE_CHECKING

try:
    from telegram import Update
    from telegram.ext import Application, MessageHandler, filters

    _HAS_TELEGRAM = True
except ImportError:
    _HAS_TELEGRAM = False

from claw.bridge.base import BridgeAdapter, InboundMessage, PlatformLimits

if TYPE_CHECKING:
    from claw.bridge.manager import BridgeManager

log = logging.getLogger(__name__)


class TelegramAdapter(BridgeAdapter):
    """Telegram messaging bridge using python-telegram-bot.

    Supports polling mode (works behind NAT) and webhook mode.

    Config keys:
        token_secret   -- name of the encrypted secret holding the bot token
        allowed_users  -- list of Telegram user IDs (empty = allow all)
        mode           -- "polling" (default) or "webhook"
        webhook_secret -- optional webhook verification secret
    """

    PLATFORM = "telegram"
    LIMITS = PlatformLimits(
        max_message_length=4096,
        markdown_flavor="telegram",
        supports_typing=True,
        supports_reactions=True,
        supports_threads=True,
        supports_media=True,
    )

    def __init__(self, config: dict, manager: BridgeManager) -> None:
        if not _HAS_TELEGRAM:
            raise ImportError(
                "python-telegram-bot is required for the Telegram adapter "
                "(pip install 'python-telegram-bot>=21')"
            )
        super().__init__(config, manager)
        self._app: Application | None = None
        self._allowed_users: set[str] = set(
            str(u) for u in config.get("allowed_users", [])
        )

    # ------------------------------------------------------------------
    # helpers
    # ------------------------------------------------------------------

    def _get_token(self) -> str:
        from claw.secret_store import load as secret_load

        secret_name = self._config.get("token_secret", "telegram_bot_token")
        token = secret_load(secret_name)
        if not token:
            raise RuntimeError(
                f"Telegram bot token not found in secret store (key: {secret_name})"
            )
        return token

    def _is_allowed(self, user_id: str) -> bool:
        """Check if user is in the allowed list.  Empty list = allow all."""
        if not self._allowed_users:
            return True
        return str(user_id) in self._allowed_users

    # ------------------------------------------------------------------
    # lifecycle
    # ------------------------------------------------------------------

    async def start(self) -> None:  # noqa: C901 — unavoidable handler setup
        token = self._get_token()
        self._app = Application.builder().token(token).build()
        adapter = self

        # ---- message handler ----------------------------------------
        async def _handle_message(update: Update, context: object) -> None:
            if not update.message or not update.message.text:
                return

            user = update.message.from_user
            if not user:
                return

            user_id = str(user.id)
            if not adapter._is_allowed(user_id):
                log.info("[telegram] Ignoring message from unauthorized user %s", user_id)
                return

            chat = update.message.chat
            is_direct = chat.type == "private"
            is_mention = False
            text = update.message.text

            # In groups, only respond to @mentions or replies to the bot
            if not is_direct and adapter._app is not None:
                bot_info = await adapter._app.bot.get_me()
                bot_username = f"@{bot_info.username}"
                if bot_username in text:
                    is_mention = True
                    text = text.replace(bot_username, "").strip()
                elif (
                    update.message.reply_to_message
                    and update.message.reply_to_message.from_user
                    and update.message.reply_to_message.from_user.id == bot_info.id
                ):
                    is_mention = True

            msg = InboundMessage(
                platform=adapter.PLATFORM,
                user_id=user_id,
                user_name=user.full_name or user.username or user_id,
                channel_id=str(chat.id),
                text=text,
                is_direct=is_direct,
                is_mention=is_mention,
                reply_context={
                    "message_id": update.message.message_id,
                    "chat_id": chat.id,
                },
            )
            await adapter.on_message(msg)

        self._app.add_handler(
            MessageHandler(filters.TEXT & ~filters.COMMAND, _handle_message)
        )

        # ---- initialize & start ------------------------------------
        await self._app.initialize()
        await self._app.start()

        mode = self._config.get("mode", "polling")
        if mode == "polling":
            await self._app.updater.start_polling(drop_pending_updates=True)
            log.info("[telegram] Polling started")
        else:
            log.info(
                "[telegram] Webhook mode — waiting for updates via "
                "/api/bridge/telegram/webhook"
            )

        # Block until the adapter is told to stop
        while self._running:
            await asyncio.sleep(1)

    async def stop(self) -> None:
        self._running = False
        if self._app is not None:
            try:
                if self._app.updater and self._app.updater.running:
                    await self._app.updater.stop()
                await self._app.stop()
                await self._app.shutdown()
            except Exception:
                log.exception("[telegram] Error during shutdown")
            self._app = None

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
        kwargs: dict = {"chat_id": int(channel_id), "text": text}
        if reply_context and reply_context.get("message_id"):
            kwargs["reply_to_message_id"] = reply_context["message_id"]
        await self._app.bot.send_message(**kwargs)

    async def send_typing(self, channel_id: str) -> None:
        if self._app is not None:
            await self._app.bot.send_chat_action(
                chat_id=int(channel_id), action="typing"
            )

    # ------------------------------------------------------------------
    # webhook support
    # ------------------------------------------------------------------

    async def handle_webhook(self, update_data: dict) -> None:
        """Process a webhook update from the FastAPI endpoint."""
        if self._app is not None:
            update = Update.de_json(update_data, self._app.bot)
            await self._app.process_update(update)
