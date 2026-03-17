"""Tests for the remote API endpoints."""

from __future__ import annotations

import asyncio
import json
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi.testclient import TestClient

from claw.admin.app import create_admin_app
from claw.admin.sse import StatusBroadcaster


@pytest.fixture
def _enable_remote(monkeypatch):
    """Enable remote access in settings."""
    import claw.config as cfg

    original_get = cfg.get_settings

    def patched_get():
        s = original_get()
        object.__setattr__(s.remote, "enabled", True)
        object.__setattr__(s.admin, "auth_enabled", True)
        return s

    monkeypatch.setattr(cfg, "get_settings", patched_get)
    monkeypatch.setattr(cfg, "_settings", None)  # force reload
    yield
    monkeypatch.setattr(cfg, "_settings", None)


@pytest.fixture
def mock_agent():
    agent = MagicMock()
    agent.session = MagicMock()
    agent.session.messages = []
    agent.last_usage = None
    agent.process_utterance = AsyncMock(return_value="Hello from Claw!")
    agent.extract_facts = AsyncMock(return_value=[])
    agent.new_session = MagicMock()
    return agent


@pytest.fixture
def mock_registry():
    reg = MagicMock()
    reg.get_openai_tools = MagicMock(return_value=[])
    return reg


@pytest.fixture
def mock_tts():
    tts = MagicMock()
    tts.synthesize_wav = AsyncMock(return_value=b"RIFF" + b"\x00" * 100)
    return tts


@pytest.fixture
def app(mock_agent, mock_registry, mock_tts):
    broadcaster = StatusBroadcaster()
    application = create_admin_app(broadcaster, agent=mock_agent, registry=mock_registry)
    application.state.tts = mock_tts
    application.state.remote_transcriber = MagicMock()
    application.state.tool_router = MagicMock()
    return application


@pytest.fixture
def api_key_value(monkeypatch):
    """Create a test device and return its API key."""
    import claw.admin.api_key as ak
    import claw.secret_store as ss
    import tempfile
    from pathlib import Path

    tmp = Path(tempfile.mkdtemp())
    devices_dir = tmp / "remote"
    secrets_dir = tmp / "secrets"
    devices_dir.mkdir()
    secrets_dir.mkdir()

    monkeypatch.setattr(ak, "_DEVICES_DIR", devices_dir)
    monkeypatch.setattr(ak, "_REGISTRY_FILE", devices_dir / "devices.json")

    def _test_secrets_dir():
        secrets_dir.mkdir(parents=True, exist_ok=True)
        return secrets_dir

    monkeypatch.setattr(ss, "_secrets_dir", _test_secrets_dir)

    result = ak.create_device("test-phone")
    return result["api_key"]


class TestRemotePing:
    def test_ping_with_valid_key(self, app, _enable_remote, api_key_value):
        client = TestClient(app)
        resp = client.get("/api/remote/ping", headers={"X-API-Key": api_key_value})
        assert resp.status_code == 200
        assert resp.json()["status"] == "ok"

    def test_ping_without_key_returns_401(self, app, _enable_remote):
        client = TestClient(app)
        resp = client.get("/api/remote/ping")
        assert resp.status_code == 401

    def test_ping_with_invalid_key_returns_401(self, app, _enable_remote):
        client = TestClient(app)
        resp = client.get("/api/remote/ping", headers={"X-API-Key": "0" * 64})
        assert resp.status_code == 401

    def test_ping_when_remote_disabled(self, app):
        """When remote.enabled=False, all remote endpoints return 503."""
        import claw.config as cfg
        s = cfg.get_settings()
        # remote.enabled defaults to False
        client = TestClient(app)
        resp = client.get("/api/remote/ping", headers={"X-API-Key": "0" * 64})
        assert resp.status_code == 503


class TestRemoteChat:
    def test_chat_returns_response(self, app, _enable_remote, api_key_value, mock_agent):
        client = TestClient(app)
        resp = client.post(
            "/api/remote/chat",
            json={"message": "Hello"},
            headers={"X-API-Key": api_key_value},
        )
        assert resp.status_code == 200
        data = resp.json()
        assert data["content"] == "Hello from Claw!"
        assert data["role"] == "assistant"
        assert data["device"] == "test-phone"
        mock_agent.process_utterance.assert_awaited_once()

    def test_chat_empty_message_returns_400(self, app, _enable_remote, api_key_value):
        client = TestClient(app)
        resp = client.post(
            "/api/remote/chat",
            json={"message": ""},
            headers={"X-API-Key": api_key_value},
        )
        assert resp.status_code == 400

    def test_chat_with_query_param_key(self, app, _enable_remote, api_key_value, mock_agent):
        """API key can also be passed as ?key= query param."""
        client = TestClient(app)
        resp = client.post(
            f"/api/remote/chat?key={api_key_value}",
            json={"message": "Hello"},
        )
        assert resp.status_code == 200


class TestRemoteTTS:
    def test_tts_returns_wav(self, app, _enable_remote, api_key_value, mock_tts):
        client = TestClient(app)
        resp = client.post(
            "/api/remote/tts",
            json={"text": "Hello world"},
            headers={"X-API-Key": api_key_value},
        )
        assert resp.status_code == 200
        assert resp.headers["content-type"] == "audio/wav"
        assert resp.content.startswith(b"RIFF")

    def test_tts_empty_text_returns_400(self, app, _enable_remote, api_key_value):
        client = TestClient(app)
        resp = client.post(
            "/api/remote/tts",
            json={"text": ""},
            headers={"X-API-Key": api_key_value},
        )
        assert resp.status_code == 400


class TestRemoteStatus:
    def test_status_returns_state(self, app, _enable_remote, api_key_value):
        client = TestClient(app)
        resp = client.get("/api/remote/status", headers={"X-API-Key": api_key_value})
        assert resp.status_code == 200
        data = resp.json()
        assert "state" in data
        assert data["state"] == "idle"


class TestRemoteStreamValidation:
    def test_invalid_video_id_returns_400(self, app, _enable_remote, api_key_value):
        client = TestClient(app)
        resp = client.get(
            "/api/remote/stream/invalid!id",
            headers={"X-API-Key": api_key_value},
        )
        assert resp.status_code == 400

    def test_valid_video_id_format(self, app, _enable_remote, api_key_value):
        """Valid video ID format accepted (will fail at yt-dlp, but format is valid)."""
        client = TestClient(app)
        with patch("subprocess.run") as mock_run:
            mock_run.return_value = MagicMock(returncode=1, stderr="test error")
            resp = client.get(
                "/api/remote/stream/dQw4w9WgXcQ",
                headers={"X-API-Key": api_key_value},
            )
            # Should reach yt-dlp (not rejected for format), then fail gracefully
            assert resp.status_code == 502


class TestDeviceManagement:
    """Device management endpoints (admin-authenticated, not remote API key)."""

    def test_create_and_list_devices(self, app, monkeypatch):
        """Test device CRUD via admin panel endpoints."""
        import claw.admin.api_key as ak
        import claw.secret_store as ss
        import tempfile
        from pathlib import Path

        tmp = Path(tempfile.mkdtemp())
        devices_dir = tmp / "remote"
        secrets_dir = tmp / "secrets"
        devices_dir.mkdir()
        secrets_dir.mkdir()
        monkeypatch.setattr(ak, "_DEVICES_DIR", devices_dir)
        monkeypatch.setattr(ak, "_REGISTRY_FILE", devices_dir / "devices.json")
        monkeypatch.setattr(ss, "_secrets_dir", lambda: secrets_dir)

        # Disable auth for admin endpoints in test
        import claw.config as cfg
        original_get = cfg.get_settings

        def patched_get():
            s = original_get()
            object.__setattr__(s.admin, "auth_enabled", False)
            return s

        monkeypatch.setattr(cfg, "get_settings", patched_get)

        client = TestClient(app)

        # Create device
        resp = client.post("/api/devices", json={"name": "my-phone"})
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "ok"
        assert data["name"] == "my-phone"
        assert len(data["api_key"]) == 64

        # List devices
        resp = client.get("/api/devices")
        assert resp.status_code == 200
        devices = resp.json()["devices"]
        assert len(devices) == 1
        assert devices[0]["name"] == "my-phone"

        # Delete device
        resp = client.delete("/api/devices/my-phone")
        assert resp.status_code == 200

        # Verify deleted
        resp = client.get("/api/devices")
        assert len(resp.json()["devices"]) == 0


class TestMusicInfoExtraction:
    def test_extract_video_id_from_tool_result(self):
        from claw.admin.remote import _extract_music_from_session

        agent = MagicMock()
        agent.session.messages = [
            {"role": "user", "content": "play some music"},
            {"role": "assistant", "tool_calls": [{"function": {"name": "play_song"}}]},
            {"role": "tool", "content": "Now playing: Bohemian Rhapsody by Queen [video:dQw4w9WgXcQ]"},
            {"role": "assistant", "content": "Now playing Bohemian Rhapsody by Queen!"},
        ]

        result = _extract_music_from_session(agent, 0)
        assert result is not None
        assert result["video_id"] == "dQw4w9WgXcQ"
        assert result["title"] == "Bohemian Rhapsody"
        assert result["artist"] == "Queen"
        assert result["stream_url"] == "/api/remote/stream/dQw4w9WgXcQ"

    def test_no_video_tag_returns_none(self):
        from claw.admin.remote import _extract_music_from_session

        agent = MagicMock()
        agent.session.messages = [
            {"role": "user", "content": "tell me a joke"},
            {"role": "assistant", "content": "Why did the chicken..."},
        ]

        result = _extract_music_from_session(agent, 0)
        assert result is None

    def test_no_session_returns_none(self):
        from claw.admin.remote import _extract_music_from_session

        agent = MagicMock()
        agent.session = None
        assert _extract_music_from_session(agent, 0) is None
