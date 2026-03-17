"""Shared Google OAuth credentials for Gmail and Google Calendar."""

from __future__ import annotations

import json
import logging
import os
from pathlib import Path

log = logging.getLogger(__name__)

_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent


def _resolve_path(rel: str) -> Path:
    p = Path(rel)
    if p.is_absolute():
        return p
    return _PROJECT_ROOT / rel


def get_credentials(credentials_file: str, token_file: str, scopes: list[str]):
    """Load or refresh Google OAuth credentials.

    Returns a google.oauth2.credentials.Credentials object, or None if not authenticated.
    """
    from google.auth.transport.requests import Request
    from google.oauth2.credentials import Credentials

    token_path = _resolve_path(token_file)
    creds = None

    if token_path.exists():
        creds = Credentials.from_authorized_user_file(str(token_path), scopes)

    if creds and creds.expired and creds.refresh_token:
        try:
            creds.refresh(Request())
            fd = os.open(str(token_path), os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
            with os.fdopen(fd, "w") as f:
                f.write(creds.to_json())
            log.info("Google OAuth token refreshed")
        except Exception:
            log.exception("Failed to refresh Google OAuth token — re-authorize in Settings")
            creds = None

    return creds


def resolve_account(
    accounts: dict, service: str, requested: str
) -> tuple[str, str, dict | None]:
    """Pick the right Google account for a service.

    Args:
        accounts: The google_auth.accounts dict from config (label → account dict).
        service: "calendar", "gmail", or "youtube_music".
        requested: Account label from the tool call ("" = auto-select).

    Returns:
        (account_label, error_message, account_config_dict).
        On success error_message is "" and account_config_dict is the account dict.
        On failure account_config_dict is None.
    """
    if not accounts:
        return ("", "No Google accounts configured. Link one in Settings > Google Accounts.", None)

    # Build list of accounts that have this service enabled
    def _service_enabled(acct: dict) -> bool:
        if service == "youtube_music":
            return bool(acct.get("youtube_music", False))
        svc_cfg = acct.get(service, {})
        if isinstance(svc_cfg, dict):
            return svc_cfg.get("enabled", False)
        return False

    if requested:
        if requested not in accounts:
            available = ", ".join(accounts.keys())
            return ("", f"Account '{requested}' not found. Available accounts: {available}", None)
        acct = accounts[requested]
        if not _service_enabled(acct):
            return ("", f"Account '{requested}' does not have {service} enabled.", None)
        return (requested, "", acct)

    # Auto-select: find accounts with the service enabled
    enabled = [(label, acct) for label, acct in accounts.items() if _service_enabled(acct)]

    if len(enabled) == 0:
        available = ", ".join(accounts.keys())
        return ("", f"No accounts have {service} enabled. Available accounts: {available}", None)

    if len(enabled) == 1:
        return (enabled[0][0], "", enabled[0][1])

    labels = ", ".join(label for label, _ in enabled)
    return ("", f"Multiple accounts have {service}: {labels}. Please specify which account.", None)


def is_authenticated(token_file: str) -> bool:
    """Check if a valid token file exists."""
    token_path = _resolve_path(token_file)
    if not token_path.exists():
        return False
    try:
        data = json.loads(token_path.read_text())
        return bool(data.get("refresh_token"))
    except (json.JSONDecodeError, KeyError):
        return False
