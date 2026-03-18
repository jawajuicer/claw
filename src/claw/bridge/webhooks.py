"""FastAPI sub-router for webhook-based bridge platforms (Twilio, Telegram)."""

from __future__ import annotations

import hashlib
import hmac
import logging

from fastapi import APIRouter, Request
from fastapi.responses import JSONResponse, PlainTextResponse

log = logging.getLogger(__name__)

bridge_webhook_router = APIRouter(prefix="/api/bridge", tags=["bridge"])


@bridge_webhook_router.post("/twilio/sms")
async def twilio_sms_webhook(request: Request) -> PlainTextResponse:
    """Handle incoming Twilio SMS/WhatsApp messages."""
    manager = getattr(request.app.state, "bridge_manager", None)
    if not manager:
        return PlainTextResponse("Bridge not configured", status_code=503)

    adapter = manager.adapters.get("twilio")
    if not adapter:
        return PlainTextResponse("Twilio adapter not enabled", status_code=503)

    # Validate Twilio signature
    form = await request.form()
    if not await _validate_twilio_signature(request, form):
        return PlainTextResponse("Invalid signature", status_code=403)

    try:
        await adapter.handle_webhook(dict(form))
    except Exception:
        log.exception("Twilio webhook processing failed")

    # Twilio expects TwiML response; empty response = no reply via webhook
    # (we send replies asynchronously via the API)
    return PlainTextResponse('<?xml version="1.0" encoding="UTF-8"?><Response></Response>',
                             media_type="application/xml")


@bridge_webhook_router.post("/telegram/webhook")
async def telegram_webhook(request: Request) -> JSONResponse:
    """Handle incoming Telegram webhook updates."""
    manager = getattr(request.app.state, "bridge_manager", None)
    if not manager:
        return JSONResponse({"error": "Bridge not configured"}, status_code=503)

    adapter = manager.adapters.get("telegram")
    if not adapter:
        return JSONResponse({"error": "Telegram adapter not enabled"}, status_code=503)

    # Validate Telegram secret token if configured
    secret_token = request.headers.get("X-Telegram-Bot-Api-Secret-Token", "")
    expected_token = adapter._config.get("webhook_secret", "")
    if expected_token and secret_token != expected_token:
        return JSONResponse({"error": "Invalid token"}, status_code=403)

    try:
        body = await request.json()
        await adapter.handle_webhook(body)
    except Exception:
        log.exception("Telegram webhook processing failed")

    return JSONResponse({"ok": True})


async def _validate_twilio_signature(request: Request, form: dict) -> bool:
    """Validate Twilio request signature using auth token."""
    from claw.config import get_settings

    settings = get_settings()
    if not hasattr(settings, "bridges"):
        return False

    twilio_cfg = getattr(settings.bridges, "twilio", None)
    if twilio_cfg is None:
        return False

    auth_token_secret = getattr(twilio_cfg, "auth_token_secret", "")
    if not auth_token_secret:
        return True  # No token configured = skip validation

    try:
        from claw.secret_store import load as secret_load
        token = secret_load(auth_token_secret)
        if not token:
            log.warning("Twilio auth token not found in secret store")
            return True  # Allow if not configured

        # Twilio signature validation
        sig = request.headers.get("X-Twilio-Signature", "")
        url = str(request.url)
        # Sort form params and concatenate
        param_str = "".join(f"{k}{form[k]}" for k in sorted(form.keys()))
        data = url + param_str
        expected = hmac.HMAC(
            token.encode("utf-8"),
            data.encode("utf-8"),
            hashlib.sha1,
        ).digest()

        import base64
        expected_sig = base64.b64encode(expected).decode("utf-8")
        return hmac.compare_digest(sig, expected_sig)
    except ImportError:
        return True  # Skip if dependencies missing
    except Exception:
        log.exception("Twilio signature validation failed")
        return False
