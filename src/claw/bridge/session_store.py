"""Per-user conversation session storage for bridge messaging."""

from __future__ import annotations

import logging

from claw.agent_core.conversation import ConversationSession

log = logging.getLogger(__name__)


class BridgeSessionStore:
    """Stores conversation sessions keyed by (platform, user_id).

    Each remote user gets their own ConversationSession for conversation
    isolation. Sessions are created lazily on first message.
    """

    def __init__(self, max_sessions: int = 100) -> None:
        self._sessions: dict[tuple[str, str], ConversationSession] = {}
        self._max_sessions = max_sessions

    def get_or_create(self, platform: str, user_id: str) -> ConversationSession:
        """Get existing session or create a new one."""
        key = (platform, user_id)
        if key not in self._sessions:
            if len(self._sessions) >= self._max_sessions:
                self._evict_oldest()
            self._sessions[key] = ConversationSession()
            log.debug("Created new session for %s/%s", platform, user_id)
        return self._sessions[key]

    def update(self, platform: str, user_id: str, session: ConversationSession) -> None:
        """Update the stored session after agent processing."""
        self._sessions[(platform, user_id)] = session

    def remove(self, platform: str, user_id: str) -> None:
        """Remove a user's session."""
        self._sessions.pop((platform, user_id), None)

    def clear(self) -> None:
        """Remove all sessions."""
        self._sessions.clear()

    def _evict_oldest(self) -> None:
        """Remove the oldest session when at capacity (FIFO)."""
        if self._sessions:
            oldest_key = next(iter(self._sessions))
            del self._sessions[oldest_key]
            log.debug("Evicted oldest session: %s/%s", *oldest_key)

    @property
    def count(self) -> int:
        return len(self._sessions)
