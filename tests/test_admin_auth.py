"""Tests for admin panel Basic Auth middleware."""

from __future__ import annotations

import asyncio
import base64
from unittest.mock import MagicMock, patch

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from claw.admin.auth import BasicAuthMiddleware
from claw.admin.routes import LogBuffer, router


def _make_app():
    """Build a minimal test app with auth middleware."""
    from claw.admin.sse import StatusBroadcaster

    app = FastAPI()
    app.add_middleware(BasicAuthMiddleware)
    app.include_router(router)

    app.state.broadcaster = StatusBroadcaster()
    app.state.log_buffer = LogBuffer(maxlen=100)
    templates = MagicMock()
    templates.TemplateResponse = MagicMock(return_value="<html>test</html>")
    app.state.templates = templates
    app.state.agent = None
    app.state.registry = None
    app.state.chat_lock = asyncio.Lock()
    app.state.bg_tasks = set()
    app.state.tts = None

    return app


def _basic_auth(username: str, password: str) -> dict:
    """Build Basic Auth header."""
    creds = base64.b64encode(f"{username}:{password}".encode()).decode()
    return {"Authorization": f"Basic {creds}"}


def _mock_admin_cfg(auth_enabled=True, auth_username="admin"):
    """Create a mock settings object with admin config."""
    mock = MagicMock()
    mock.admin.auth_enabled = auth_enabled
    mock.admin.auth_username = auth_username
    return mock


class TestBasicAuthMiddleware:
    """Test HTTP Basic Auth middleware."""

    def test_unauthenticated_returns_401(self, tmp_config):
        """Request without auth header gets 401 with WWW-Authenticate."""
        app = _make_app()
        with (
            patch("claw.config.get_settings", return_value=_mock_admin_cfg()),
            patch("claw.secret_store.load", return_value="testpassword123"),
        ):
            with TestClient(app, raise_server_exceptions=False) as client:
                resp = client.get("/api/status")
                assert resp.status_code == 401
                assert "WWW-Authenticate" in resp.headers
                assert "Basic" in resp.headers["WWW-Authenticate"]

    def test_wrong_password_returns_401(self, tmp_config):
        """Wrong credentials get 401."""
        app = _make_app()
        with (
            patch("claw.config.get_settings", return_value=_mock_admin_cfg()),
            patch("claw.secret_store.load", return_value="correctpassword"),
        ):
            with TestClient(app, raise_server_exceptions=False) as client:
                resp = client.get("/api/status", headers=_basic_auth("admin", "wrongpassword"))
                assert resp.status_code == 401

    def test_correct_credentials_pass_through(self, tmp_config):
        """Correct credentials allow access."""
        app = _make_app()
        with (
            patch("claw.config.get_settings", return_value=_mock_admin_cfg()),
            patch("claw.secret_store.load", return_value="correctpassword"),
        ):
            with TestClient(app, raise_server_exceptions=False) as client:
                resp = client.get("/api/status", headers=_basic_auth("admin", "correctpassword"))
                assert resp.status_code == 200

    def test_static_files_bypass_auth(self, tmp_config):
        """Static file paths bypass authentication."""
        app = _make_app()
        with (
            patch("claw.config.get_settings", return_value=_mock_admin_cfg()),
            patch("claw.secret_store.load", return_value="testpassword123"),
        ):
            with TestClient(app, raise_server_exceptions=False) as client:
                resp = client.get("/static/nonexistent.css")
                # Bypasses auth — will be 404/405 but NOT 401
                assert resp.status_code != 401

    def test_oauth_callback_bypasses_auth(self, tmp_config):
        """OAuth callback path bypasses authentication."""
        app = _make_app()
        with (
            patch("claw.config.get_settings", return_value=_mock_admin_cfg()),
            patch("claw.secret_store.load", return_value="testpassword123"),
        ):
            with TestClient(app, raise_server_exceptions=False, follow_redirects=False) as client:
                resp = client.get("/auth/google/callback?code=test&state=test")
                # Should be a redirect (302), not auth failure (401)
                assert resp.status_code != 401
                assert resp.status_code == 302

    def test_auth_disabled_allows_all(self, tmp_config):
        """With auth_enabled=False, all requests pass through."""
        app = _make_app()
        with patch("claw.config.get_settings", return_value=_mock_admin_cfg(auth_enabled=False)):
            with TestClient(app, raise_server_exceptions=False) as client:
                resp = client.get("/api/status")
                assert resp.status_code == 200

    def test_no_password_configured_returns_503(self, tmp_config):
        """If no admin password exists in secret store, return 503."""
        app = _make_app()
        with (
            patch("claw.config.get_settings", return_value=_mock_admin_cfg()),
            patch("claw.secret_store.load", return_value=None),
        ):
            with TestClient(app, raise_server_exceptions=False) as client:
                resp = client.get("/api/status")
                assert resp.status_code == 503
                assert "setup" in resp.json().get("setup", "").lower()

    def test_setup_endpoint_stores_password(self, tmp_config):
        """POST /api/admin/setup stores the initial password."""
        app = _make_app()
        stored = {}

        def mock_store(name, value):
            stored[name] = value

        with (
            patch("claw.config.get_settings", return_value=_mock_admin_cfg()),
            patch("claw.secret_store.load", return_value=None),
            patch("claw.secret_store.exists", return_value=False),
            patch("claw.secret_store.store", side_effect=mock_store),
        ):
            with TestClient(app, raise_server_exceptions=False) as client:
                resp = client.post("/api/admin/setup", json={"password": "mysecurepassword"})
                assert resp.status_code == 200
                assert stored.get("admin_password") == "mysecurepassword"

    def test_setup_rejects_if_already_configured(self, tmp_config):
        """POST /api/admin/setup fails if a password already exists."""
        app = _make_app()
        with (
            patch("claw.config.get_settings", return_value=_mock_admin_cfg()),
            patch("claw.secret_store.load", return_value=None),
            patch("claw.secret_store.exists", return_value=True),
        ):
            with TestClient(app, raise_server_exceptions=False) as client:
                resp = client.post("/api/admin/setup", json={"password": "newpassword1"})
                assert resp.status_code == 409

    def test_setup_rejects_short_password(self, tmp_config):
        """POST /api/admin/setup rejects passwords under 8 characters."""
        app = _make_app()
        with (
            patch("claw.config.get_settings", return_value=_mock_admin_cfg()),
            patch("claw.secret_store.load", return_value=None),
            patch("claw.secret_store.exists", return_value=False),
        ):
            with TestClient(app, raise_server_exceptions=False) as client:
                resp = client.post("/api/admin/setup", json={"password": "short"})
                assert resp.status_code == 400
