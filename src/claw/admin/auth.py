"""HTTP Basic Auth + API Key middleware for the admin panel.

Admin panel routes use HTTP Basic Auth. Remote API routes (/api/remote/*)
use device API keys via the X-API-Key header. WebSocket connections bypass
this middleware and handle auth in the endpoint handler.
"""

from __future__ import annotations

import logging
import secrets

from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import JSONResponse, Response

log = logging.getLogger(__name__)

# Paths that bypass authentication entirely
_PUBLIC_PREFIXES = ("/static/", "/auth/google/", "/app", "/api/remote/app/", "/api/webhook")


class BasicAuthMiddleware(BaseHTTPMiddleware):
    """Require HTTP Basic Auth for admin routes, API Key for remote routes.

    - Skips static files and OAuth callbacks.
    - If auth is disabled in config, passes everything through.
    - Remote API paths (/api/remote/) require a valid device API key.
    - WebSocket upgrades pass through (auth handled in endpoint).
    """

    async def dispatch(self, request: Request, call_next) -> Response:
        from claw.config import get_settings

        cfg = get_settings().admin

        # Auth disabled — pass through
        if not cfg.auth_enabled:
            return await call_next(request)

        path = request.url.path

        # Public paths bypass auth
        if any(path.startswith(prefix) for prefix in _PUBLIC_PREFIXES):
            return await call_next(request)

        # Setup endpoint is accessible without auth (but only works when no password set)
        if path == "/api/admin/setup" and request.method == "POST":
            return await call_next(request)

        # Remote API: authenticate via device API key
        if path.startswith("/api/remote/"):
            return await self._check_api_key(request, call_next)

        # Admin panel: accept device API key as an alternative to Basic Auth
        # (allows the Android app's WebView to access admin panel)
        api_key = (
            request.headers.get("X-API-Key", "")
            or request.query_params.get("key", "")
        )
        if api_key:
            from claw.admin.api_key import verify_key

            device = verify_key(api_key)
            if device is not None:
                request.state.device_name = device
                return await call_next(request)

        # Admin panel: authenticate via HTTP Basic Auth
        return await self._check_basic_auth(request, call_next)

    async def _check_api_key(self, request: Request, call_next) -> Response:
        """Verify X-API-Key header for remote API routes."""
        from claw.config import get_settings

        if not get_settings().remote.enabled:
            return JSONResponse(
                {"error": "Remote access is not enabled"},
                status_code=503,
            )

        api_key = (
            request.headers.get("X-API-Key", "")
            or request.query_params.get("key", "")
        )
        if not api_key:
            return JSONResponse({"error": "API key required"}, status_code=401)

        from claw.admin.api_key import verify_key

        device = verify_key(api_key)
        if device is None:
            return JSONResponse({"error": "Invalid API key"}, status_code=401)

        # Attach device name for downstream handlers
        request.state.device_name = device
        return await call_next(request)

    async def _check_basic_auth(self, request: Request, call_next) -> Response:
        """Verify HTTP Basic Auth for admin panel routes."""
        from claw.config import get_settings
        from claw.secret_store import load as secret_load

        cfg = get_settings().admin
        stored_password = secret_load("admin_password")

        if stored_password is None:
            return JSONResponse(
                {
                    "error": "Admin password not configured",
                    "setup": "POST /api/admin/setup with {\"password\": \"...\"}",
                },
                status_code=503,
            )

        auth = request.headers.get("Authorization", "")
        if not auth.startswith("Basic "):
            return _unauthorized()

        import base64

        try:
            decoded = base64.b64decode(auth[6:]).decode("utf-8")
            username, password = decoded.split(":", 1)
        except Exception:
            return _unauthorized()

        expected_user = cfg.auth_username
        if not (
            secrets.compare_digest(username, expected_user)
            and secrets.compare_digest(password, stored_password)
        ):
            return _unauthorized()

        return await call_next(request)


def _unauthorized() -> Response:
    return JSONResponse(
        {"error": "Authentication required"},
        status_code=401,
        headers={"WWW-Authenticate": 'Basic realm="The Claw Admin"'},
    )
