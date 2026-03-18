"""Twilio bridge adapter for SMS and WhatsApp messaging."""

from __future__ import annotations

import asyncio
import logging
from typing import TYPE_CHECKING

try:
    from twilio.rest import Client as TwilioClient

    _HAS_TWILIO = True
except ImportError:
    _HAS_TWILIO = False

from claw.bridge.base import BridgeAdapter, InboundMessage, PlatformLimits

if TYPE_CHECKING:
    from claw.bridge.manager import BridgeManager

log = logging.getLogger(__name__)


class TwilioAdapter(BridgeAdapter):
    """Twilio SMS / WhatsApp bridge via webhooks.

    Messages arrive via POST to /api/bridge/twilio/sms (configured externally
    in the Twilio console or via the admin panel).  Replies are sent through
    the Twilio REST API.

    Config keys:
        account_sid_secret -- encrypted secret name for the Account SID
        auth_token_secret  -- encrypted secret name for the Auth Token
        from_number        -- the Twilio phone number (E.164, e.g. "+15551234567")
        allowed_numbers    -- list of phone numbers allowed to interact (empty = all)
    """

    PLATFORM = "twilio"
    LIMITS = PlatformLimits(
        max_message_length=1600,  # SMS segment limit
        markdown_flavor="standard",  # plain text for SMS
        supports_typing=False,
        supports_reactions=False,
        supports_threads=False,
        supports_media=True,  # MMS / WhatsApp media
    )

    def __init__(self, config: dict, manager: BridgeManager) -> None:
        if not _HAS_TWILIO:
            raise ImportError(
                "twilio is required for the Twilio adapter "
                "(pip install 'twilio>=9')"
            )
        super().__init__(config, manager)
        self._client: TwilioClient | None = None
        self._from_number: str = config.get("from_number", "")
        self._allowed_numbers: set[str] = set(config.get("allowed_numbers", []))

    # ------------------------------------------------------------------
    # helpers
    # ------------------------------------------------------------------

    def _get_credentials(self) -> tuple[str, str]:
        from claw.secret_store import load as secret_load

        sid_secret = self._config.get("account_sid_secret", "twilio_account_sid")
        auth_secret = self._config.get("auth_token_secret", "twilio_auth_token")
        sid = secret_load(sid_secret)
        auth = secret_load(auth_secret)
        if not sid or not auth:
            raise RuntimeError(
                "Twilio credentials not found in secret store "
                f"(sid key: {sid_secret}, auth key: {auth_secret})"
            )
        return sid, auth

    # ------------------------------------------------------------------
    # lifecycle
    # ------------------------------------------------------------------

    async def start(self) -> None:
        sid, auth = self._get_credentials()
        self._client = TwilioClient(sid, auth)
        log.info("[twilio] Client initialized (from: %s)", self._from_number)

        # Twilio is entirely webhook-driven — just idle until stopped
        while self._running:
            await asyncio.sleep(1)

    async def stop(self) -> None:
        self._running = False
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
        """Send an SMS or WhatsApp message via the Twilio REST API.

        ``channel_id`` is the recipient in E.164 format (e.g. "+15551234567")
        or ``"whatsapp:+15551234567"`` for WhatsApp.
        """
        if self._client is None:
            return

        from_num = self._from_number
        # WhatsApp numbers carry the "whatsapp:" prefix
        if channel_id.startswith("whatsapp:"):
            from_num = f"whatsapp:{from_num}"

        # Twilio client is synchronous — run in a thread to avoid blocking
        await asyncio.to_thread(
            self._client.messages.create,
            body=text,
            from_=from_num,
            to=channel_id,
        )

    # ------------------------------------------------------------------
    # webhook support
    # ------------------------------------------------------------------

    async def handle_webhook(self, form_data: dict) -> None:
        """Process an incoming Twilio webhook (called from the FastAPI route).

        Expected form fields (standard Twilio webhook payload):
            From -- sender number, e.g. "+15559876543" or "whatsapp:+15559876543"
            Body -- message text
        """
        from_number = form_data.get("From", "")
        body = form_data.get("Body", "")

        if not from_number or not body:
            return

        # Normalize: strip "whatsapp:" prefix for ACL check
        clean_number = from_number.replace("whatsapp:", "")
        if self._allowed_numbers and clean_number not in self._allowed_numbers:
            log.info(
                "[twilio] Ignoring message from unauthorized number %s",
                clean_number,
            )
            return

        msg = InboundMessage(
            platform=self.PLATFORM,
            user_id=clean_number,
            user_name=clean_number,
            channel_id=from_number,  # full number (with whatsapp: prefix if applicable)
            text=body,
            is_direct=True,
            is_mention=False,
            reply_context={},
        )
        await self.on_message(msg)
