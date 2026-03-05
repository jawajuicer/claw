"""Tests for claw.admin.routes — FastAPI admin API endpoints."""

from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from claw.admin.routes import LogBuffer, router, _valid_convo_id


def _create_test_app(settings, tmp_path):
    """Build a minimal FastAPI app with mocked state for route testing."""

    from claw.admin.sse import StatusBroadcaster

    app = FastAPI()
    app.include_router(router)

    # Set up mock app state
    app.state.broadcaster = StatusBroadcaster()
    app.state.log_buffer = LogBuffer(maxlen=100)

    # Minimal templates stub
    templates = MagicMock()
    templates.TemplateResponse = MagicMock(return_value="<html>test</html>")
    app.state.templates = templates

    # Mock agent and registry
    mock_agent = MagicMock()
    mock_agent.session = None
    mock_agent.last_usage = None
    mock_agent.process_utterance = AsyncMock(return_value="Agent says hello")
    mock_agent.extract_facts = AsyncMock(return_value=[])
    mock_agent.new_session = MagicMock()
    app.state.agent = mock_agent

    mock_registry = MagicMock()
    mock_registry.get_openai_tools.return_value = []
    app.state.registry = mock_registry

    app.state.chat_lock = asyncio.Lock()
    app.state.bg_tasks = set()
    app.state.tts = None

    return app


@pytest.fixture()
def test_client(settings, tmp_path, tmp_config):
    app = _create_test_app(settings, tmp_path)
    with TestClient(app) as client:
        yield client, app


class TestAPIStatus:
    """Test the /api/status endpoint."""

    def test_status_returns_json(self, test_client):
        client, app = test_client
        resp = client.get("/api/status")
        assert resp.status_code == 200
        data = resp.json()
        assert data["state"] == "idle"


class TestAPILogs:
    """Test the /api/logs endpoint."""

    def test_empty_logs(self, test_client):
        client, app = test_client
        resp = client.get("/api/logs")
        assert resp.status_code == 200
        assert resp.json()["logs"] == []


class TestAPIChatNew:
    """Test the /api/chat/new endpoint."""

    def test_new_session(self, test_client):
        client, app = test_client
        resp = client.post("/api/chat/new")
        assert resp.status_code == 200
        assert resp.json()["status"] == "ok"
        app.state.agent.new_session.assert_called_once()


class TestAPIChat:
    """Test the /api/chat endpoint."""

    def test_empty_message_returns_400(self, test_client):
        client, app = test_client
        resp = client.post("/api/chat", json={"message": ""})
        assert resp.status_code == 400

    def test_agent_unavailable(self, test_client):
        client, app = test_client
        app.state.agent = None
        resp = client.post("/api/chat", json={"message": "Hello"})
        data = resp.json()
        assert "not available" in data["content"].lower()


class TestAPItts:
    """Test the /api/tts endpoint."""

    def test_empty_text_returns_400(self, test_client):
        client, app = test_client
        resp = client.post("/api/tts", json={"text": ""})
        assert resp.status_code == 400

    def test_tts_unavailable_returns_503(self, test_client):
        client, app = test_client
        app.state.tts = None
        resp = client.post("/api/tts", json={"text": "Hello"})
        assert resp.status_code == 503


class TestConversationPersistence:
    """Test conversation save/load/delete endpoints."""

    def test_save_and_list(self, test_client, tmp_config):
        client, app = test_client
        messages = [
            {"role": "user", "content": "Hello"},
            {"role": "assistant", "content": "Hi there!"},
        ]
        with patch("claw.admin.routes._conversations_dir") as mock_dir:
            from pathlib import Path
            import tempfile

            with tempfile.TemporaryDirectory() as td:
                mock_dir.return_value = Path(td)

                # Save
                resp = client.post("/api/conversations/save", json={
                    "id": "test123",
                    "messages": messages,
                })
                assert resp.status_code == 200
                assert resp.json()["id"] == "test123"

                # List
                resp = client.get("/api/conversations")
                assert resp.status_code == 200
                convos = resp.json()
                assert len(convos) == 1
                assert convos[0]["id"] == "test123"

                # Get
                resp = client.get("/api/conversations/test123")
                assert resp.status_code == 200
                data = resp.json()
                assert len(data["messages"]) == 2

                # Delete
                resp = client.delete("/api/conversations/test123")
                assert resp.status_code == 200

    def test_invalid_convo_id_rejected(self, test_client, tmp_config):
        client, app = test_client
        with patch("claw.admin.routes._conversations_dir"):
            resp = client.post("/api/conversations/save", json={
                "id": "../../../etc/passwd",
                "messages": [],
            })
            assert resp.status_code == 400

    def test_get_nonexistent_returns_404(self, test_client, tmp_config):
        client, app = test_client
        with patch("claw.admin.routes._conversations_dir") as mock_dir:
            from pathlib import Path
            import tempfile

            with tempfile.TemporaryDirectory() as td:
                mock_dir.return_value = Path(td)
                resp = client.get("/api/conversations/nonexistent")
                assert resp.status_code == 404


class TestValidConvoId:
    """Test the _valid_convo_id helper."""

    def test_valid_ids(self):
        assert _valid_convo_id("abc123") is True
        assert _valid_convo_id("test-id_01") is True
        assert _valid_convo_id("a") is True

    def test_invalid_ids(self):
        assert _valid_convo_id("") is False
        assert _valid_convo_id("../../../etc") is False
        assert _valid_convo_id("a" * 65) is False
        assert _valid_convo_id("has spaces") is False
        assert _valid_convo_id("has.dot") is False


class TestLogBuffer:
    """Test the LogBuffer logging handler."""

    def test_ring_buffer(self):
        buf = LogBuffer(maxlen=3)
        import logging

        for i in range(5):
            record = logging.LogRecord(
                name="test", level=logging.INFO, pathname="", lineno=0,
                msg=f"msg {i}", args=(), exc_info=None,
            )
            buf.emit(record)
        assert len(buf.records) == 3

    def test_format_applied(self):
        buf = LogBuffer(maxlen=10)
        import logging

        record = logging.LogRecord(
            name="claw.test", level=logging.WARNING, pathname="", lineno=0,
            msg="something happened", args=(), exc_info=None,
        )
        buf.emit(record)
        assert "WARNING" in buf.records[0]
        assert "something happened" in buf.records[0]
