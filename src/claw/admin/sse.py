"""Server-Sent Events broadcaster for real-time dashboard updates."""

from __future__ import annotations

import asyncio
import json
import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone

log = logging.getLogger(__name__)


@dataclass
class StatusBroadcaster:
    """Broadcasts status updates to connected SSE clients."""

    _subscribers: list[asyncio.Queue] = field(default_factory=list)
    _status: dict = field(default_factory=lambda: {
        "state": "idle",
        "last_wake_word": None,
        "last_transcription": None,
        "last_response": None,
        "mcp_servers": {},
        "memory_stats": {},
        "uptime_start": datetime.now(timezone.utc).isoformat(),
    })

    def subscribe(self) -> asyncio.Queue:
        """Create a new subscriber queue."""
        queue: asyncio.Queue = asyncio.Queue(maxsize=50)
        self._subscribers.append(queue)
        log.debug("SSE subscriber added (total: %d)", len(self._subscribers))
        return queue

    def unsubscribe(self, queue: asyncio.Queue) -> None:
        """Remove a subscriber queue."""
        if queue in self._subscribers:
            self._subscribers.remove(queue)
            log.debug("SSE subscriber removed (total: %d)", len(self._subscribers))

    # Events that should be persisted in the status snapshot
    _PERSISTENT_EVENTS = {"status", "transcription", "response", "memory", "mcp", "llm_provider"}

    async def broadcast(self, event: str, data: dict) -> None:
        """Send an event to all subscribers."""
        if event in self._PERSISTENT_EVENTS:
            self._status.update(data)
        message = json.dumps({"event": event, **data})
        alive: list[asyncio.Queue] = []
        for queue in self._subscribers:
            try:
                queue.put_nowait(message)
                alive.append(queue)
            except asyncio.QueueFull:
                pass  # drop slow subscriber
        self._subscribers = alive

    async def update_state(self, state: str, **extra) -> None:
        """Update the system state and broadcast."""
        await self.broadcast("status", {"state": state, **extra})

    async def update_transcription(self, text: str) -> None:
        await self.broadcast("transcription", {"last_transcription": text})

    async def update_response(self, text: str) -> None:
        await self.broadcast("response", {"last_response": text})

    async def update_memory_stats(self, stats: dict) -> None:
        await self.broadcast("memory", {"memory_stats": stats})

    async def update_mcp_servers(self, servers: dict) -> None:
        await self.broadcast("mcp", {"mcp_servers": servers})

    async def update_audio_stats(self, stats: dict) -> None:
        await self.broadcast("audio_stats", {"audio_stats": stats})

    async def broadcast_token(self, token: str, done: bool = False) -> None:
        """Broadcast a streaming token to SSE subscribers (for Android, etc.)."""
        await self.broadcast("token", {"token": token, "done": done})

    def get_status(self) -> dict:
        return dict(self._status)

    async def event_generator(self, queue: asyncio.Queue):
        """Async generator that yields SSE-formatted events."""
        # Send initial status
        yield json.dumps({"event": "status", **self._status})
        try:
            while True:
                data = await queue.get()
                yield data
        except asyncio.CancelledError:
            pass
