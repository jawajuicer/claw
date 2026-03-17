"""Tests for claw.admin.routes — FastAPI admin API endpoints."""

from __future__ import annotations

import asyncio
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from claw.admin.routes import LogBuffer, router, _valid_convo_id, _redact_settings
from claw.mcp_handler.stats import ToolStats


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


class TestRedactSettings:
    """Test the _redact_settings helper."""

    def test_masks_weather_api_key(self):
        data = {"weather": {"api_key": "abc123def456ghij", "default_location": "Akron, OH"}}
        result = _redact_settings(data)
        assert result["weather"]["api_key"] == "abc1***ghij"
        assert result["weather"]["default_location"] == "Akron, OH"

    def test_masks_llm_api_key(self):
        data = {"llm": {"api_key": "sk-1234567890abcdef", "model": "qwen3.5:4b"}}
        result = _redact_settings(data)
        assert "***" in result["llm"]["api_key"]
        assert result["llm"]["model"] == "qwen3.5:4b"

    def test_skips_no_key_sentinel(self):
        data = {"llm": {"api_key": "no-key"}}
        result = _redact_settings(data)
        assert result["llm"]["api_key"] == "no-key"

    def test_skips_empty_key(self):
        data = {"weather": {"api_key": ""}}
        result = _redact_settings(data)
        assert result["weather"]["api_key"] == ""

    def test_masks_short_key(self):
        data = {"weather": {"api_key": "abcd1234"}}
        result = _redact_settings(data)
        assert result["weather"]["api_key"] == "ab***"

    def test_get_settings_returns_redacted(self, test_client, tmp_config):
        """GET /api/settings returns masked api_key values."""
        client, app = test_client
        from claw.config import Settings
        s = Settings()
        s.weather.api_key = "abc123def456ghij"
        with patch("claw.config.get_settings", return_value=s):
            resp = client.get("/api/settings")
            assert resp.status_code == 200
            data = resp.json()
            assert "***" in data["weather"]["api_key"]
            assert data["weather"]["api_key"] == "abc1***ghij"

    def test_post_settings_masked_value_preserves_original(self, test_client, tmp_config):
        """POST /api/settings with masked value does not overwrite the real key."""
        client, app = test_client
        from claw.config import Settings
        s = Settings()
        s.weather.api_key = "abc123def456ghij"
        with (
            patch("claw.config.get_settings", return_value=s),
            patch("claw.config.reload_settings"),
        ):
            # Simulate frontend sending back the masked value
            resp = client.post("/api/settings", json={
                "weather": {"api_key": "abc1***ghij", "default_location": "Columbus, OH"},
            })
            assert resp.status_code == 200
            # The real key should be preserved (not overwritten with masked value)
            assert s.weather.api_key == "abc123def456ghij"


class TestConversationSearch:
    """Test the /api/conversations/search endpoint."""

    def test_search_finds_matching(self, test_client, tmp_config):
        client, app = test_client
        with patch("claw.admin.routes._conversations_dir") as mock_dir:
            import tempfile
            with tempfile.TemporaryDirectory() as td:
                mock_dir.return_value = Path(td)
                # Save two conversations
                client.post("/api/conversations/save", json={
                    "id": "c1",
                    "messages": [{"role": "user", "content": "Tell me about weather"}],
                })
                client.post("/api/conversations/save", json={
                    "id": "c2",
                    "messages": [{"role": "user", "content": "Play some music"}],
                })

                resp = client.get("/api/conversations/search?q=weather")
                assert resp.status_code == 200
                results = resp.json()
                assert len(results) == 1
                assert results[0]["id"] == "c1"

    def test_search_empty_query_returns_empty(self, test_client, tmp_config):
        client, app = test_client
        resp = client.get("/api/conversations/search?q=")
        assert resp.status_code == 200
        assert resp.json() == []


class TestConversationExport:
    """Test the /api/conversations/{id}/export endpoint."""

    def test_export_json(self, test_client, tmp_config):
        client, app = test_client
        with patch("claw.admin.routes._conversations_dir") as mock_dir:
            import tempfile
            with tempfile.TemporaryDirectory() as td:
                mock_dir.return_value = Path(td)
                client.post("/api/conversations/save", json={
                    "id": "exp1",
                    "messages": [{"role": "user", "content": "Hello"}],
                })

                resp = client.get("/api/conversations/exp1/export?format=json")
                assert resp.status_code == 200
                assert "application/json" in resp.headers["content-type"]
                assert resp.headers["content-disposition"] == 'attachment; filename="exp1.json"'

    def test_export_markdown(self, test_client, tmp_config):
        client, app = test_client
        with patch("claw.admin.routes._conversations_dir") as mock_dir:
            import tempfile
            with tempfile.TemporaryDirectory() as td:
                mock_dir.return_value = Path(td)
                client.post("/api/conversations/save", json={
                    "id": "exp2",
                    "messages": [
                        {"role": "user", "content": "Hello"},
                        {"role": "assistant", "content": "Hi there!"},
                    ],
                })

                resp = client.get("/api/conversations/exp2/export?format=md")
                assert resp.status_code == 200
                assert "text/markdown" in resp.headers["content-type"]
                body = resp.text
                assert "**User:**" in body
                assert "**Claw:**" in body

    def test_export_not_found(self, test_client, tmp_config):
        client, app = test_client
        with patch("claw.admin.routes._conversations_dir") as mock_dir:
            import tempfile
            with tempfile.TemporaryDirectory() as td:
                mock_dir.return_value = Path(td)
                resp = client.get("/api/conversations/nope/export?format=json")
                assert resp.status_code == 404


class TestBulkDelete:
    """Test the /api/conversations/bulk-delete endpoint."""

    def test_bulk_delete_multiple(self, test_client, tmp_config):
        client, app = test_client
        with patch("claw.admin.routes._conversations_dir") as mock_dir:
            import tempfile
            with tempfile.TemporaryDirectory() as td:
                mock_dir.return_value = Path(td)
                for cid in ["a1", "a2", "a3"]:
                    client.post("/api/conversations/save", json={
                        "id": cid, "messages": [{"role": "user", "content": "hi"}],
                    })

                resp = client.post("/api/conversations/bulk-delete", json={"ids": ["a1", "a3"]})
                assert resp.status_code == 200
                assert resp.json()["deleted"] == 2

                # Only a2 should remain
                remaining = client.get("/api/conversations").json()
                assert len(remaining) == 1
                assert remaining[0]["id"] == "a2"

    def test_bulk_delete_empty(self, test_client, tmp_config):
        client, app = test_client
        resp = client.post("/api/conversations/bulk-delete", json={"ids": []})
        assert resp.status_code == 400


class TestToolStats:
    """Test the ToolStats class and /api/tools/stats endpoint."""

    def test_record_and_summary(self):
        stats = ToolStats(maxlen=10)
        stats.record("get_time", "system_control", 0.05, True)
        stats.record("get_time", "system_control", 0.03, True)
        stats.record("search_music", "youtube_music", 0.5, True)
        stats.record("search_music", "youtube_music", 0.0, False)

        summary = stats.summary()
        assert summary["get_time"]["count"] == 2
        assert summary["get_time"]["errors"] == 0
        assert summary["search_music"]["count"] == 2
        assert summary["search_music"]["errors"] == 1
        assert summary["search_music"]["error_rate"] == 0.5

    def test_ring_buffer_eviction(self):
        stats = ToolStats(maxlen=3)
        for i in range(5):
            stats.record(f"tool_{i}", "server", 0.1, True)

        recent = stats.recent(10)
        assert len(recent) == 3
        # Cumulative counts survive eviction
        summary = stats.summary()
        assert len(summary) == 5

    def test_api_endpoint(self, test_client, tmp_config):
        client, app = test_client
        stats = ToolStats()
        stats.record("test_tool", "test_server", 0.1, True)
        app.state.tool_stats = stats

        resp = client.get("/api/tools/stats")
        assert resp.status_code == 200
        data = resp.json()
        assert "test_tool" in data["tools"]

    def test_api_no_stats(self, test_client, tmp_config):
        client, app = test_client
        app.state.tool_stats = None
        resp = client.get("/api/tools/stats")
        assert resp.status_code == 200
        assert resp.json() == {"tools": {}, "recent": []}


class TestHealthCheck:
    """Test the /api/health endpoint."""

    def test_health_returns_json(self, test_client, tmp_config):
        client, app = test_client
        with patch("httpx.AsyncClient") as mock_httpx:
            mock_response = MagicMock()
            mock_response.status_code = 200
            mock_client_instance = AsyncMock()
            mock_client_instance.get = AsyncMock(return_value=mock_response)
            mock_client_instance.__aenter__ = AsyncMock(return_value=mock_client_instance)
            mock_client_instance.__aexit__ = AsyncMock(return_value=None)
            mock_httpx.return_value = mock_client_instance

            resp = client.get("/api/health")
            data = resp.json()
            assert "status" in data
            assert "checks" in data
            assert "llm" in data["checks"]
            assert "audio" in data["checks"]
            assert "mcp" in data["checks"]


class TestAudioStats:
    """Test the /api/audio/stats endpoint."""

    def test_no_capture_returns_inactive(self, test_client, tmp_config):
        client, app = test_client
        app.state.audio_capture = None
        resp = client.get("/api/audio/stats")
        assert resp.status_code == 200
        assert resp.json()["active"] is False

    def test_with_capture_returns_metrics(self, test_client, tmp_config):
        client, app = test_client
        mock_capture = MagicMock()
        mock_capture.get_metrics.return_value = {
            "rms": 0.05, "agc_gain": 1.2, "noise_floor": 0.01,
            "sample_rate": 16000, "block_size": 1280, "device_index": None,
        }
        app.state.audio_capture = mock_capture
        app.state.vad = None

        resp = client.get("/api/audio/stats")
        assert resp.status_code == 200
        data = resp.json()
        assert data["active"] is True
        assert data["rms"] == 0.05


class TestLogBufferJSON:
    """Test the LogBuffer JSON format mode."""

    def test_json_format_output(self):
        import logging as _logging
        buf = LogBuffer(maxlen=10, fmt="json")
        record = _logging.LogRecord(
            name="claw.test", level=_logging.INFO, pathname="", lineno=0,
            msg="test message", args=(), exc_info=None,
        )
        buf.emit(record)
        import json as _json
        entry = _json.loads(buf.records[0])
        assert entry["level"] == "INFO"
        assert entry["logger"] == "claw.test"
        assert entry["message"] == "test message"
        assert "timestamp" in entry

    def test_json_exc_info_includes_traceback(self):
        import logging as _logging
        import sys
        import json as _json
        buf = LogBuffer(maxlen=10, fmt="json")
        try:
            raise ValueError("test error")
        except ValueError:
            record = _logging.LogRecord(
                name="test", level=_logging.ERROR, pathname="", lineno=0,
                msg="Error", args=(), exc_info=sys.exc_info(),
            )
            buf.emit(record)
        entry = _json.loads(buf.records[0])
        assert "Traceback" in entry.get("exc_info", "")
        assert "ValueError" in entry.get("exc_info", "")


class TestSearchToolCallMessages:
    """Regression test: search must handle content=None (tool-call messages)."""

    def test_search_with_none_content_no_crash(self, test_client, tmp_config):
        client, app = test_client
        with patch("claw.admin.routes._conversations_dir") as mock_dir:
            import tempfile
            with tempfile.TemporaryDirectory() as td:
                mock_dir.return_value = Path(td)
                client.post("/api/conversations/save", json={
                    "id": "tc1",
                    "messages": [
                        {"role": "user", "content": "Play music"},
                        {"role": "assistant", "content": None},
                        {"role": "tool", "content": "Now playing: test song"},
                    ],
                })
                resp = client.get("/api/conversations/search?q=music")
                assert resp.status_code == 200
                assert len(resp.json()) == 1


class TestExportNoneContent:
    """Regression test: markdown export must skip None-content messages."""

    def test_markdown_no_literal_none(self, test_client, tmp_config):
        client, app = test_client
        with patch("claw.admin.routes._conversations_dir") as mock_dir:
            import tempfile
            with tempfile.TemporaryDirectory() as td:
                mock_dir.return_value = Path(td)
                client.post("/api/conversations/save", json={
                    "id": "exp_none",
                    "messages": [
                        {"role": "user", "content": "Play music"},
                        {"role": "assistant", "content": None},
                        {"role": "tool", "content": "Now playing"},
                        {"role": "assistant", "content": "Playing your song"},
                    ],
                })
                resp = client.get("/api/conversations/exp_none/export?format=md")
                assert resp.status_code == 200
                body = resp.text
                assert "None" not in body
                assert "**User:**" in body
                assert "**Claw:**" in body
