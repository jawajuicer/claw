"""Matrix bridge adapter using matrix-nio."""

from __future__ import annotations

import asyncio
import logging
from typing import TYPE_CHECKING

try:
    from nio import AsyncClient as NioAsyncClient
    from nio import MatrixRoom, RoomMessageText

    _HAS_NIO = True
except ImportError:
    _HAS_NIO = False

from claw.bridge.base import BridgeAdapter, InboundMessage, PlatformLimits

if TYPE_CHECKING:
    from claw.bridge.manager import BridgeManager

log = logging.getLogger(__name__)


class MatrixAdapter(BridgeAdapter):
    """Matrix messaging bridge using matrix-nio.

    Config keys:
        homeserver    -- e.g. "https://matrix.org"
        user_id       -- fully-qualified Matrix ID, e.g. "@claw:matrix.org"
        token_secret  -- encrypted secret name for the access token
        allowed_rooms -- list of room IDs (empty = all joined rooms)
        allowed_users -- list of Matrix user IDs (empty = all)
    """

    PLATFORM = "matrix"
    LIMITS = PlatformLimits(
        max_message_length=65536,
        markdown_flavor="html",
        supports_typing=True,
        supports_reactions=True,
        supports_threads=True,
        supports_media=True,
    )

    def __init__(self, config: dict, manager: BridgeManager) -> None:
        if not _HAS_NIO:
            raise ImportError(
                "matrix-nio is required for the Matrix adapter "
                "(pip install 'matrix-nio>=0.24')"
            )
        super().__init__(config, manager)
        self._client: NioAsyncClient | None = None
        self._homeserver: str = config.get("homeserver", "https://matrix.org")
        self._user_id: str = config.get("user_id", "")
        self._allowed_rooms: set[str] = set(config.get("allowed_rooms", []))
        self._allowed_users: set[str] = set(config.get("allowed_users", []))

    # ------------------------------------------------------------------
    # helpers
    # ------------------------------------------------------------------

    def _get_token(self) -> str:
        from claw.secret_store import load as secret_load

        secret_name = self._config.get("token_secret", "matrix_access_token")
        token = secret_load(secret_name)
        if not token:
            raise RuntimeError(
                f"Matrix access token not found in secret store (key: {secret_name})"
            )
        return token

    # ------------------------------------------------------------------
    # lifecycle
    # ------------------------------------------------------------------

    async def start(self) -> None:
        token = self._get_token()
        self._client = NioAsyncClient(self._homeserver, self._user_id)
        self._client.access_token = token

        adapter = self

        async def _message_callback(room: MatrixRoom, event: RoomMessageText) -> None:
            # Ignore own messages
            if event.sender == adapter._user_id:
                return
            if adapter._allowed_rooms and room.room_id not in adapter._allowed_rooms:
                return
            if adapter._allowed_users and event.sender not in adapter._allowed_users:
                return

            # DM heuristic: room with at most 2 members
            is_direct = room.member_count <= 2
            text = event.body

            # Mention detection in body text or formatted_body
            formatted = getattr(event, "formatted_body", "") or ""
            is_mention = adapter._user_id in text or adapter._user_id in formatted

            if is_mention:
                text = text.replace(adapter._user_id, "").strip()

            user_name = room.user_name(event.sender) or event.sender

            msg = InboundMessage(
                platform=adapter.PLATFORM,
                user_id=event.sender,
                user_name=user_name,
                channel_id=room.room_id,
                text=text,
                is_direct=is_direct,
                is_mention=is_mention,
                reply_context={
                    "room_id": room.room_id,
                    "event_id": event.event_id,
                },
            )
            await adapter.on_message(msg)

        self._client.add_event_callback(_message_callback, RoomMessageText)

        # Initial sync to establish room state
        await self._client.sync(timeout=30_000)
        log.info("[matrix] Connected as %s", self._user_id)

        # Persistent sync loop — matrix-nio uses long-polling
        while self._running:
            try:
                await self._client.sync(timeout=30_000)
            except asyncio.CancelledError:
                break
            except Exception:
                log.exception("[matrix] Sync error")
                await asyncio.sleep(5)

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
        await self._client.room_send(
            room_id=channel_id,
            message_type="m.room.message",
            content={"msgtype": "m.text", "body": text},
        )

    async def send_typing(self, channel_id: str) -> None:
        if self._client is not None:
            await self._client.room_typing(
                channel_id, typing_state=True, timeout=10_000
            )
