"""Tests for claw.admin.sse — StatusBroadcaster."""

from __future__ import annotations

import asyncio
import json


from claw.admin.sse import StatusBroadcaster


class TestSubscribeUnsubscribe:
    """Test subscriber management."""

    def test_subscribe_returns_queue(self):
        b = StatusBroadcaster()
        q = b.subscribe()
        assert isinstance(q, asyncio.Queue)
        assert len(b._subscribers) == 1

    def test_unsubscribe_removes_queue(self):
        b = StatusBroadcaster()
        q = b.subscribe()
        b.unsubscribe(q)
        assert len(b._subscribers) == 0

    def test_unsubscribe_nonexistent_queue(self):
        b = StatusBroadcaster()
        q = asyncio.Queue()
        # Should not raise
        b.unsubscribe(q)

    def test_multiple_subscribers(self):
        b = StatusBroadcaster()
        b.subscribe()
        b.subscribe()
        assert len(b._subscribers) == 2


class TestBroadcast:
    """Test event broadcasting."""

    async def test_broadcast_delivers_to_subscriber(self):
        b = StatusBroadcaster()
        q = b.subscribe()
        await b.broadcast("status", {"state": "processing"})
        msg = q.get_nowait()
        parsed = json.loads(msg)
        assert parsed["event"] == "status"
        assert parsed["state"] == "processing"

    async def test_broadcast_delivers_to_all_subscribers(self):
        b = StatusBroadcaster()
        q1 = b.subscribe()
        q2 = b.subscribe()
        await b.broadcast("test", {"data": "hello"})
        assert not q1.empty()
        assert not q2.empty()

    async def test_broadcast_removes_full_queues(self):
        b = StatusBroadcaster()
        q = b.subscribe()
        # Fill the queue to capacity (maxsize=50)
        for i in range(50):
            q.put_nowait(f"msg_{i}")
        assert q.full()

        await b.broadcast("overflow", {"x": 1})
        # Full queue should have been removed
        assert len(b._subscribers) == 0

    async def test_broadcast_updates_internal_status(self):
        b = StatusBroadcaster()
        await b.broadcast("status", {"state": "listening"})
        assert b._status["state"] == "listening"


class TestUpdateMethods:
    """Test convenience update methods."""

    async def test_update_state(self):
        b = StatusBroadcaster()
        q = b.subscribe()
        await b.update_state("processing")
        msg = json.loads(q.get_nowait())
        assert msg["state"] == "processing"
        assert msg["event"] == "status"

    async def test_update_transcription(self):
        b = StatusBroadcaster()
        q = b.subscribe()
        await b.update_transcription("Hello world")
        msg = json.loads(q.get_nowait())
        assert msg["last_transcription"] == "Hello world"

    async def test_update_response(self):
        b = StatusBroadcaster()
        q = b.subscribe()
        await b.update_response("Answer text")
        msg = json.loads(q.get_nowait())
        assert msg["last_response"] == "Answer text"

    async def test_update_memory_stats(self):
        b = StatusBroadcaster()
        q = b.subscribe()
        stats = {"conversations": 10, "facts": 5}
        await b.update_memory_stats(stats)
        msg = json.loads(q.get_nowait())
        assert msg["memory_stats"] == stats

    async def test_update_mcp_servers(self):
        b = StatusBroadcaster()
        q = b.subscribe()
        servers = {"weather": ["get_weather"]}
        await b.update_mcp_servers(servers)
        msg = json.loads(q.get_nowait())
        assert msg["mcp_servers"] == servers


class TestGetStatus:
    """Test get_status snapshot."""

    def test_initial_status(self):
        b = StatusBroadcaster()
        status = b.get_status()
        assert status["state"] == "idle"
        assert "uptime_start" in status

    async def test_status_reflects_updates(self):
        b = StatusBroadcaster()
        await b.update_state("listening")
        status = b.get_status()
        assert status["state"] == "listening"

    def test_get_status_returns_copy(self):
        b = StatusBroadcaster()
        s1 = b.get_status()
        s1["state"] = "modified"
        s2 = b.get_status()
        assert s2["state"] == "idle"


class TestEventGenerator:
    """Test the async event generator."""

    async def test_yields_initial_status(self):
        b = StatusBroadcaster()
        q = b.subscribe()

        gen = b.event_generator(q)
        first = await gen.__anext__()
        parsed = json.loads(first)
        assert parsed["event"] == "status"
        assert parsed["state"] == "idle"

    async def test_yields_broadcast_events(self):
        b = StatusBroadcaster()
        q = b.subscribe()

        gen = b.event_generator(q)
        # Consume initial status
        await gen.__anext__()

        # Broadcast something
        await b.broadcast("test", {"value": 42})
        data = await gen.__anext__()
        parsed = json.loads(data)
        assert parsed["value"] == 42
