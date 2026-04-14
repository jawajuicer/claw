"""Signal bridge adapter via signal-cli-rest-api (HTTP)."""

from __future__ import annotations

import asyncio
import base64
import logging
import re
import uuid
from datetime import datetime, timezone
from typing import TYPE_CHECKING

from claw.bridge.base import BridgeAdapter, InboundMessage, PlatformLimits
from claw.bridge.profiles import resolve_profile

if TYPE_CHECKING:
    from claw.bridge.manager import BridgeManager
    from claw.memory_engine.store import MemoryStore

log = logging.getLogger(__name__)


try:
    import httpx

    _HAS_HTTPX = True
except ImportError:
    _HAS_HTTPX = False


class SignalAdapter(BridgeAdapter):
    """Signal messaging bridge via signal-cli-rest-api.

    Connects to a local signal-cli-rest-api Docker container over HTTP.
    Messages are received by polling GET /v1/receive/{number} and sent
    via POST /v2/send.

    Config keys:
        api_url         -- signal-cli-rest-api base URL (default: http://localhost:8080)
        phone_number    -- registered Signal phone number in E.164 format
        bot_name        -- name to match for mentions in groups (default: "Claw")
        allowed_groups  -- list of group IDs to listen in (empty = all)
        allowed_users   -- list of phone numbers allowed to DM (empty = all)
        observe_groups  -- passively ingest non-mention group messages into memory
        poll_interval   -- seconds between receive polls (default: 1.0)
    """

    PLATFORM = "signal"
    LIMITS = PlatformLimits(
        max_message_length=65000,
        markdown_flavor="standard",
        supports_typing=True,
        supports_reactions=True,
        supports_threads=False,
        supports_media=True,
    )

    def __init__(
        self,
        config: dict,
        manager: BridgeManager,
        memory_store: MemoryStore | None = None,
    ) -> None:
        if not _HAS_HTTPX:
            raise ImportError(
                "httpx is required for the Signal adapter (pip install httpx)"
            )
        super().__init__(config, manager)
        self._api_url: str = config.get("api_url", "http://localhost:8080").rstrip("/")
        self._phone_number: str = config.get("phone_number", "")
        self._bot_name: str = config.get("bot_name", "Claw")
        self._mention_re = re.compile(
            rf"\b{re.escape(self._bot_name)}\b", re.IGNORECASE
        )
        self._allowed_groups: set[str] = set(config.get("allowed_groups", []))
        self._allowed_users: set[str] = set(config.get("allowed_users", []))
        self._admin_users: set[str] = set(config.get("admin_users", []))
        self._observe_groups: bool = config.get("observe_groups", True)
        self._poll_interval: float = config.get("poll_interval", 1.0)
        self._memory_store: MemoryStore | None = memory_store
        self._http_client = None  # lazy-initialized httpx.AsyncClient

    # ------------------------------------------------------------------
    # helpers
    # ------------------------------------------------------------------

    def _is_allowed_user(self, phone: str) -> bool:
        """Check if a user is in the allowed list.  Empty list = allow all."""
        if not self._allowed_users:
            return True
        return phone in self._allowed_users

    def _is_allowed_group(self, group_id: str) -> bool:
        """Check if a group is in the allowed list.  Empty list = allow all."""
        if not self._allowed_groups:
            return True
        return group_id in self._allowed_groups

    def _is_admin(self, source_number: str, source_uuid: str = "") -> bool:
        """Check if a user is an admin (can execute tools).  Empty list = all are admins."""
        if not self._admin_users:
            return True
        return source_number in self._admin_users or source_uuid in self._admin_users

    def _is_mention(self, text: str, data_msg: dict | None = None) -> bool:
        """Check if the message mentions the bot.

        Checks both:
        1. Signal's structured mentions array (when user uses @mention UI)
        2. Text regex fallback (when user types the bot name literally)
        """
        # Check Signal's structured mentions array first
        if data_msg:
            for mention in data_msg.get("mentions", []):
                # Match by phone number (our registered number)
                if mention.get("number") == self._phone_number:
                    return True
                # Match by UUID (signal-cli may use either)
                if mention.get("uuid") and mention.get("number") == self._phone_number:
                    return True
        # Fall back to text regex
        return bool(self._mention_re.search(text))

    async def _ensure_http_client(self):
        """Lazy-initialize the httpx async client."""
        if self._http_client is None:
            self._http_client = httpx.AsyncClient(
                base_url=self._api_url,
                timeout=30.0,
            )
        return self._http_client

    async def _close_http_client(self) -> None:
        """Close the httpx client if open."""
        if self._http_client is not None:
            try:
                await self._http_client.aclose()
            except Exception:
                pass
            self._http_client = None

    async def _download_image_attachments(self, data_msg: dict) -> list[bytes]:
        """Download image attachments from signal-cli REST API."""
        attachments = data_msg.get("attachments", [])
        images = []
        for att in attachments:
            content_type = att.get("contentType", "")
            if not content_type.startswith("image/"):
                continue
            att_id = att.get("id") or att.get("filename")
            if not att_id:
                log.warning("[signal] Attachment missing id/filename: %s", att)
                continue
            try:
                client = await self._ensure_http_client()
                resp = await client.get(f"/v1/attachments/{att_id}")
                if resp.status_code == 200:
                    images.append(resp.content)
                else:
                    log.warning("[signal] Attachment %s returned %d", att_id, resp.status_code)
            except Exception:
                log.warning("[signal] Failed to download attachment %s", att_id, exc_info=True)
        return images

    def _extract_sender_name(self, envelope: dict) -> str:
        """Extract a human-readable sender name from an envelope.

        signal-cli-rest-api provides sourceName in the envelope when
        the contact has a profile name.  Falls back to the phone number.
        """
        return (
            envelope.get("sourceName")
            or envelope.get("sourceNumber")
            or "unknown"
        )

    def _extract_group_info(self, data_msg: dict) -> tuple[str, str] | None:
        """Extract (group_id, group_name) from a data message, or None for DMs."""
        group_info = data_msg.get("groupInfo")
        if not group_info:
            return None
        group_id = group_info.get("groupId", "")
        group_name = group_info.get("groupName", group_id)
        return group_id, group_name

    async def get_group_members(self, group_id: str) -> list[dict]:
        """Fetch group member names by cross-referencing groups and contacts.

        Returns a list of dicts: [{"uuid": ..., "name": ...}, ...]
        Excludes the bot itself from the list.
        """
        client = await self._ensure_http_client()
        try:
            # Get group details (member UUIDs)
            resp = await client.get(f"/v1/groups/{self._phone_number}")
            if resp.status_code != 200:
                return []
            groups = resp.json()
            # Find matching group
            member_uuids: list[str] = []
            for g in groups:
                gid = g.get("id", "")
                internal = g.get("internal_id", "")
                if group_id in (gid, internal):
                    member_uuids = g.get("members", [])
                    break
            if not member_uuids:
                return []

            # Get contacts to resolve UUIDs to names
            resp = await client.get(f"/v1/contacts/{self._phone_number}")
            if resp.status_code != 200:
                return []
            contacts = resp.json()
            uuid_to_name: dict[str, str] = {}
            for c in contacts:
                uid = c.get("uuid", "")
                p = c.get("profile", {})
                name = f"{p.get('given_name', '')} {p.get('lastname', '')}".strip()
                if uid and name:
                    uuid_to_name[uid] = name

            # Build member list, excluding the bot
            members = []
            for uid in member_uuids:
                if uid == self._phone_number:
                    continue
                name = uuid_to_name.get(uid, uid)
                # Skip the bot's own UUID
                bot_name = uuid_to_name.get(uid, "")
                if bot_name.lower() == self._bot_name.lower():
                    continue
                members.append({"uuid": uid, "name": name})
            return members
        except Exception:
            log.exception("[signal] Failed to fetch group members")
            return []

    # ------------------------------------------------------------------
    # passive memory ingestion
    # ------------------------------------------------------------------

    async def _store_observation(
        self,
        text: str,
        sender_name: str,
        sender_number: str,
        group_id: str,
        group_name: str,
        *,
        is_mention: bool = False,
        msg_timestamp: int | None = None,
    ) -> None:
        """Store a group message as an observation in ChromaDB memory.

        These passive observations let the agent recall group chat context
        when asked about it later, without having been directly addressed.
        Mention messages are also stored so all group activity is captured.
        Runs the sync ChromaDB write in a thread to avoid blocking the loop.
        """
        if self._memory_store is None:
            return

        try:
            # Resolve profile to determine memory scope for this group
            profile = resolve_profile("signal", group_id)
            scope = profile.memory_scope if profile.memory_scope != "shared" else None

            prefix = "[mention] " if is_mention else ""
            doc = f'[Signal group "{group_name}"] {prefix}{sender_name}: {text}'
            obs_id = f"signal_obs_{uuid.uuid4().hex[:16]}"
            metadata = {
                "platform": "signal",
                "category": "bridge_observation",
                "subcategory": "mention" if is_mention else "passive",
                "sender_name": sender_name or "unknown",
                "sender_number": sender_number or "unknown",
                "group_id": group_id or "unknown",
                "group_name": group_name or "unknown",
                "timestamp": datetime.now(timezone.utc).isoformat(),
            }
            if msg_timestamp is not None:
                metadata["signal_timestamp"] = str(msg_timestamp)
            await asyncio.to_thread(
                self._memory_store.add_conversation, obs_id, doc, metadata, scope,
            )
        except Exception:
            log.exception("[signal] Failed to store group observation")

    async def _store_delete_notice(
        self,
        target_timestamp: int,
        sender_name: str,
        sender_number: str,
        group_id: str = "",
        group_name: str = "",
    ) -> None:
        """Store a record that a message was deleted (delete-for-everyone).

        The original message text is already stored; this adds a deletion
        marker referencing the original by its Signal timestamp.
        """
        if self._memory_store is None:
            return

        try:
            profile = resolve_profile("signal", group_id or "")
            scope = profile.memory_scope if profile.memory_scope != "shared" else None

            context = f'group "{group_name}"' if group_name else "DM"
            doc = (
                f"[Signal {context}] [DELETED] {sender_name} deleted their message "
                f"(original timestamp: {target_timestamp})"
            )
            obs_id = f"signal_del_{uuid.uuid4().hex[:16]}"
            metadata = {
                "platform": "signal",
                "category": "bridge_observation",
                "subcategory": "deleted",
                "sender_name": sender_name or "unknown",
                "sender_number": sender_number or "unknown",
                "group_id": group_id or "unknown",
                "group_name": group_name or "unknown",
                "target_timestamp": str(target_timestamp),
                "timestamp": datetime.now(timezone.utc).isoformat(),
            }
            await asyncio.to_thread(
                self._memory_store.add_conversation, obs_id, doc, metadata, scope,
            )
            log.info("[signal] Stored delete notice for message ts=%d from %s", target_timestamp, sender_name)
        except Exception:
            log.exception("[signal] Failed to store delete notice")

    async def _store_edit_notice(
        self,
        new_text: str,
        original_timestamp: int,
        sender_name: str,
        sender_number: str,
        group_id: str = "",
        group_name: str = "",
    ) -> None:
        """Store a record that a message was edited, preserving the new version.

        The original message text is already stored; this adds an edit marker
        with the new text, referencing the original by its Signal timestamp.
        """
        if self._memory_store is None:
            return

        try:
            profile = resolve_profile("signal", group_id or "")
            scope = profile.memory_scope if profile.memory_scope != "shared" else None

            context = f'group "{group_name}"' if group_name else "DM"
            doc = (
                f'[Signal {context}] [EDITED] {sender_name} edited their message '
                f'(original timestamp: {original_timestamp}): {new_text}'
            )
            obs_id = f"signal_edit_{uuid.uuid4().hex[:16]}"
            metadata = {
                "platform": "signal",
                "category": "bridge_observation",
                "subcategory": "edited",
                "sender_name": sender_name or "unknown",
                "sender_number": sender_number or "unknown",
                "group_id": group_id or "unknown",
                "group_name": group_name or "unknown",
                "original_timestamp": str(original_timestamp),
                "timestamp": datetime.now(timezone.utc).isoformat(),
            }
            await asyncio.to_thread(
                self._memory_store.add_conversation, obs_id, doc, metadata, scope,
            )
            log.info("[signal] Stored edit notice for message ts=%d from %s", original_timestamp, sender_name)
        except Exception:
            log.exception("[signal] Failed to store edit notice")

    # ------------------------------------------------------------------
    # lifecycle
    # ------------------------------------------------------------------

    async def start(self) -> None:
        if not self._phone_number:
            raise RuntimeError(
                "Signal adapter requires phone_number in config "
                "(E.164 format, e.g. '+15551234567')"
            )

        client = await self._ensure_http_client()

        # Verify connectivity to signal-cli-rest-api
        try:
            resp = await client.get("/v1/about")
            resp.raise_for_status()
            log.info("[signal] Connected to signal-cli-rest-api at %s", self._api_url)
        except Exception:
            log.exception(
                "[signal] Cannot reach signal-cli-rest-api at %s", self._api_url,
            )
            raise

        log.info("[signal] Polling for messages on %s", self._phone_number)

        # Poll loop
        while self._running:
            try:
                await self._poll_messages(client)
            except asyncio.CancelledError:
                break
            except Exception:
                log.exception("[signal] Error polling messages")

            await asyncio.sleep(self._poll_interval)

    async def _poll_messages(self, client) -> None:
        """Fetch and process pending messages from signal-cli-rest-api."""
        resp = await client.get(f"/v1/receive/{self._phone_number}")
        if resp.status_code != 200:
            log.warning(
                "[signal] Receive returned %d: %s",
                resp.status_code,
                resp.text[:200],
            )
            return

        messages = resp.json()
        if not isinstance(messages, list):
            return

        for envelope_wrapper in messages:
            envelope = envelope_wrapper.get("envelope", {})
            if envelope:
                await self._process_envelope(envelope)

    async def _process_envelope(self, envelope: dict) -> None:
        """Process a single Signal envelope (shared by poll and webhook paths).

        Handles data messages (text), remote deletes, and edits.
        """
        source_number = envelope.get("sourceNumber") or envelope.get("sourceUuid") or ""
        source_uuid = envelope.get("sourceUuid") or ""
        sender_name = self._extract_sender_name(envelope)

        # --- Handle remote delete (delete-for-everyone) ---
        # signal-cli sends this as dataMessage.remoteDelete.timestamp
        data_msg = envelope.get("dataMessage")
        if data_msg:
            remote_delete = data_msg.get("remoteDelete")
            if remote_delete:
                target_ts = remote_delete.get("timestamp", 0)
                group_info = self._extract_group_info(data_msg)
                gid, gname = group_info if group_info else ("", "")
                log.info("[signal] Delete-for-everyone from %s, target ts=%d", sender_name, target_ts)
                await self._store_delete_notice(target_ts, sender_name, source_number, gid, gname)
                return

        # --- Handle edited messages ---
        # signal-cli sends edits with dataMessage containing the new text
        # and an editMessage wrapper with the original timestamp
        edit_msg = envelope.get("editMessage")
        if edit_msg:
            inner_data = edit_msg.get("dataMessage", {})
            target_ts = edit_msg.get("targetSentTimestamp", 0)
            new_text = inner_data.get("message", "")
            if new_text and target_ts:
                group_info = self._extract_group_info(inner_data)
                gid, gname = group_info if group_info else ("", "")
                log.info("[signal] Edit from %s (original ts=%d): %s", sender_name, target_ts, new_text[:80])
                await self._store_edit_notice(new_text, target_ts, sender_name, source_number, gid, gname)
                # Also store the edited version as a regular observation
                if group_info and self._observe_groups:
                    await self._store_observation(
                        new_text, sender_name, source_number, gid, gname,
                        msg_timestamp=inner_data.get("timestamp"),
                    )
            return

        # --- Handle regular data messages ---
        if not data_msg:
            return

        text = data_msg.get("message", "")
        images = await self._download_image_attachments(data_msg)
        if not text and not images:
            return

        group_info = self._extract_group_info(data_msg)
        msg_timestamp = data_msg.get("timestamp")

        log.debug("[signal] Envelope: sourceNumber=%s sourceUuid=%s dataMessage keys=%s groupInfo=%s mentions=%s",
                  envelope.get("sourceNumber"), envelope.get("sourceUuid"),
                  list(data_msg.keys()), data_msg.get("groupInfo"), data_msg.get("mentions"))

        if group_info:
            group_id, group_name = group_info
            log.debug("[signal] Group message: group_id=%r group_name=%r", group_id, group_name)
            if not self._is_allowed_group(group_id):
                return

            is_mention = self._is_mention(text, data_msg)

            msg = InboundMessage(
                platform=self.PLATFORM,
                user_id=source_number,
                user_name=sender_name,
                channel_id=group_id,
                text=text,
                is_direct=False,
                is_mention=is_mention,
                is_admin=self._is_admin(source_number, source_uuid),
                reply_context={
                    "group_id": group_id,
                    "is_group": True,
                    "timestamp": msg_timestamp,
                },
                images=images,
            )

            if is_mention:
                # Store mention as observation AND route to agent
                if self._observe_groups:
                    await self._store_observation(
                        text, sender_name, source_number, group_id, group_name,
                        is_mention=True, msg_timestamp=msg_timestamp,
                    )
                await self.on_message(msg)
            elif self._observe_groups:
                await self._store_observation(
                    text, sender_name, source_number, group_id, group_name,
                    msg_timestamp=msg_timestamp,
                )
        else:
            if not self._is_allowed_user(source_number):
                log.info(
                    "[signal] Ignoring DM from unauthorized user %s",
                    source_number,
                )
                return

            msg = InboundMessage(
                platform=self.PLATFORM,
                user_id=source_number,
                user_name=sender_name,
                channel_id=source_number,
                text=text,
                is_direct=True,
                is_mention=False,
                is_admin=self._is_admin(source_number, source_uuid),
                reply_context={
                    "is_group": False,
                    "timestamp": msg_timestamp,
                },
                images=images,
            )
            await self.on_message(msg)

    async def stop(self) -> None:
        self._running = False
        await self._close_http_client()

    # ------------------------------------------------------------------
    # messaging
    # ------------------------------------------------------------------

    async def send_message(
        self,
        channel_id: str,
        text: str,
        reply_context: dict | None = None,
    ) -> None:
        """Send a message to a user (phone number/UUID) or group (group ID).

        Uses reply_context["is_group"] to distinguish DMs from groups.
        Falls back to checking if channel_id starts with "+" for DMs.
        """
        if not channel_id:
            log.warning("[signal] Cannot send message: no channel_id")
            return

        client = await self._ensure_http_client()

        payload: dict = {
            "message": text,
            "number": self._phone_number,
            "text_mode": "normal",
        }

        # Use reply_context to determine DM vs group
        is_group = (reply_context or {}).get("is_group")
        if is_group is None:
            # Fallback: phone numbers start with "+", group IDs don't
            is_group = not channel_id.startswith("+")

        if is_group:
            # signal-cli-rest-api v2/send needs group IDs as recipients
            # in the format "group.<base64(internal_id)>"
            if channel_id.startswith("group."):
                group_send_id = channel_id
            else:
                group_send_id = "group." + base64.b64encode(channel_id.encode()).decode()
            payload["recipients"] = [group_send_id]
        else:
            payload["recipients"] = [channel_id]

        log.debug("[signal] send_message payload: %s", {k: v for k, v in payload.items() if k != "message"})

        max_retries = 3
        last_exc: Exception | None = None
        for attempt in range(1, max_retries + 1):
            try:
                resp = await client.post("/v2/send", json=payload)
                if resp.status_code in (200, 201):
                    return  # success
                log.warning(
                    "[signal] Send attempt %d/%d failed (%d): %s",
                    attempt, max_retries, resp.status_code, resp.text[:200],
                )
                last_exc = RuntimeError(f"Signal API returned {resp.status_code}")
            except Exception as exc:
                log.warning(
                    "[signal] Send attempt %d/%d error: %s",
                    attempt, max_retries, exc,
                )
                last_exc = exc

            if attempt < max_retries:
                await asyncio.sleep(1.0 * attempt)  # 1s, 2s backoff

        log.error("[signal] All %d send attempts failed for %s", max_retries, channel_id)
        raise last_exc  # type: ignore[misc]  # propagate so callers know

    async def send_typing(self, channel_id: str) -> None:
        """Send a typing indicator via the signal-cli-rest-api."""
        client = await self._ensure_http_client()
        try:
            payload: dict = {
                "recipient": channel_id,
            }
            await client.put(
                f"/v1/typing-indicator/{self._phone_number}",
                json=payload,
            )
        except Exception:
            pass  # typing indicators are best-effort

    async def send_reaction(
        self,
        channel_id: str,
        emoji: str,
        target_author: str,
        target_timestamp: int,
    ) -> None:
        """Send a reaction to a specific message."""
        client = await self._ensure_http_client()
        try:
            payload = {
                "recipient": channel_id,
                "emoji": emoji,
                "target_author": target_author,
                "target_timestamp": target_timestamp,
            }
            await client.post(
                f"/v1/reactions/{self._phone_number}",
                json=payload,
            )
        except Exception:
            log.exception("[signal] Failed to send reaction")

    # ------------------------------------------------------------------
    # webhook support
    # ------------------------------------------------------------------

    async def handle_webhook(self, payload: dict) -> None:
        """Process a webhook callback from signal-cli-rest-api.

        signal-cli-rest-api can be configured to POST incoming messages
        to a callback URL. The envelope format is the same as the receive
        endpoint, so we delegate to the shared _process_envelope method.
        """
        envelope = payload.get("envelope", {})
        if envelope:
            await self._process_envelope(envelope)
