"""Conversation session storage for bridge messaging with group support."""

from __future__ import annotations

import logging

from claw.agent_core.conversation import ConversationSession

log = logging.getLogger(__name__)


class BridgeSessionStore:
    """Stores conversation sessions keyed by string session keys.

    Supports both per-user DM sessions and shared group sessions:
    - DMs: "{platform}:dm:{user_id}"
    - Groups: "{platform}:group:{channel_id}"

    Sessions are created lazily on first message.
    """

    def __init__(self, max_sessions: int = 100) -> None:
        self._sessions: dict[str, ConversationSession] = {}
        self._max_sessions = max_sessions

    @staticmethod
    def make_key(platform: str, user_id: str, channel_id: str, is_direct: bool) -> str:
        """Build a session key based on message type.

        DMs get per-user sessions, groups get shared sessions.
        """
        if is_direct:
            return f"{platform}:dm:{user_id}"
        return f"{platform}:group:{channel_id}"

    def get_or_create(self, key: str) -> ConversationSession:
        """Get existing session or create a new one."""
        if key not in self._sessions:
            if len(self._sessions) >= self._max_sessions:
                self._evict_oldest()
            self._sessions[key] = ConversationSession()
            log.debug("Created new session for %s", key)
        return self._sessions[key]

    def update(self, key: str, session: ConversationSession) -> None:
        """Update the stored session after agent processing."""
        self._sessions[key] = session

    def remove(self, key: str) -> None:
        """Remove a session."""
        self._sessions.pop(key, None)

    def clear(self) -> None:
        """Remove all sessions."""
        self._sessions.clear()

    def _evict_oldest(self) -> None:
        """Remove the oldest session when at capacity (FIFO)."""
        if self._sessions:
            oldest_key = next(iter(self._sessions))
            del self._sessions[oldest_key]
            log.debug("Evicted oldest session: %s", oldest_key)

    @property
    def count(self) -> int:
        return len(self._sessions)
