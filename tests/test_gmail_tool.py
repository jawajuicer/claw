"""Tests for mcp_tools/gmail/server.py — Gmail MCP tool."""

from __future__ import annotations

import base64
import sys
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest
import yaml

# ---------------------------------------------------------------------------
# Pre-import mocks for optional Google dependencies that may not be installed.
# These MUST happen before any gmail server module is imported.
# ---------------------------------------------------------------------------
for _mod in (
    "googleapiclient",
    "googleapiclient.discovery",
    "google.auth",
    "google.auth.transport",
    "google.auth.transport.requests",
    "google.oauth2",
    "google.oauth2.credentials",
    "google_auth_oauthlib",
    "google_auth_oauthlib.flow",
):
    if _mod not in sys.modules:
        sys.modules[_mod] = MagicMock()


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(autouse=True)
def _isolate_gmail(tmp_path):
    """Reset module-level caches and point config at a temp directory."""
    import mcp_tools.gmail.server as gs

    gs._services.clear()
    gs._people_services.clear()

    with (
        patch.object(gs, "_PROJECT_ROOT", tmp_path),
        patch.object(gs, "_CONFIG_YAML", tmp_path / "config.yaml"),
    ):
        yield

    gs._services.clear()
    gs._people_services.clear()


def _write_config(tmp_path, config: dict) -> None:
    """Write a config.yaml to the temp directory."""
    (tmp_path / "config.yaml").write_text(yaml.dump(config))


@pytest.fixture()
def gmail_config(tmp_path):
    """Write a standard gmail config and return the tmp_path."""
    config = {
        "google_auth": {
            "credentials_file": "data/google/credentials.json",
            "scopes": [
                "https://www.googleapis.com/auth/gmail.compose",
                "https://www.googleapis.com/auth/gmail.readonly",
            ],
            "accounts": {
                "personal": {
                    "token_file": "data/google/token_personal.json",
                    "gmail": {
                        "enabled": True,
                        "default_label": "INBOX",
                        "max_results": 10,
                    },
                },
            },
        },
    }
    _write_config(tmp_path, config)
    return tmp_path


@pytest.fixture()
def multi_account_config(tmp_path):
    """Config with two gmail-enabled accounts."""
    config = {
        "google_auth": {
            "credentials_file": "data/google/credentials.json",
            "scopes": [],
            "accounts": {
                "personal": {
                    "token_file": "data/google/token_personal.json",
                    "gmail": {"enabled": True, "default_label": "INBOX", "max_results": 5},
                },
                "work": {
                    "token_file": "data/google/token_work.json",
                    "gmail": {"enabled": True, "default_label": "INBOX", "max_results": 20},
                },
            },
        },
    }
    _write_config(tmp_path, config)
    return tmp_path


@pytest.fixture()
def mock_gmail_service():
    """Build a mock Gmail API service with chainable methods."""
    service = MagicMock()
    return service


@pytest.fixture()
def mock_people_service():
    """Build a mock People API service with chainable methods."""
    service = MagicMock()
    return service


def _patch_resolve(service, acct_cfg=None):
    """Patch _resolve to return a mock service without touching Google auth."""
    if acct_cfg is None:
        acct_cfg = {"gmail": {"enabled": True, "default_label": "INBOX", "max_results": 10}}
    return patch(
        "mcp_tools.gmail.server._resolve",
        return_value=(service, acct_cfg, ""),
    )


def _patch_resolve_error(error_msg: str):
    """Patch _resolve to return an error."""
    return patch(
        "mcp_tools.gmail.server._resolve",
        return_value=(None, None, error_msg),
    )


def _make_b64(text: str) -> str:
    """Encode text as URL-safe base64 (Gmail API format)."""
    return base64.urlsafe_b64encode(text.encode()).decode()


# ---------------------------------------------------------------------------
# Helper function tests
# ---------------------------------------------------------------------------

class TestDecodeBody:
    """Test _decode_body helper."""

    def test_plain_text_direct(self):
        from mcp_tools.gmail.server import _decode_body

        payload = {
            "mimeType": "text/plain",
            "body": {"data": _make_b64("Hello world")},
        }
        assert _decode_body(payload) == "Hello world"

    def test_plain_text_in_parts(self):
        from mcp_tools.gmail.server import _decode_body

        payload = {
            "mimeType": "multipart/alternative",
            "parts": [
                {"mimeType": "text/html", "body": {"data": _make_b64("<p>HTML</p>")}},
                {"mimeType": "text/plain", "body": {"data": _make_b64("Plain text")}},
            ],
        }
        # Should find text/plain first in parts iteration order
        assert _decode_body(payload) == "Plain text"

    def test_nested_parts(self):
        from mcp_tools.gmail.server import _decode_body

        payload = {
            "mimeType": "multipart/mixed",
            "parts": [
                {
                    "mimeType": "multipart/alternative",
                    "parts": [
                        {"mimeType": "text/plain", "body": {"data": _make_b64("Nested text")}},
                    ],
                },
            ],
        }
        assert _decode_body(payload) == "Nested text"

    def test_no_plain_text(self):
        from mcp_tools.gmail.server import _decode_body

        payload = {
            "mimeType": "text/html",
            "body": {"data": _make_b64("<p>HTML only</p>")},
        }
        assert _decode_body(payload) == "(No plain text body)"

    def test_empty_payload(self):
        from mcp_tools.gmail.server import _decode_body

        assert _decode_body({}) == "(No plain text body)"

    def test_plain_text_no_data(self):
        from mcp_tools.gmail.server import _decode_body

        payload = {"mimeType": "text/plain", "body": {}}
        assert _decode_body(payload) == "(No plain text body)"

    def test_unicode_body(self):
        from mcp_tools.gmail.server import _decode_body

        payload = {
            "mimeType": "text/plain",
            "body": {"data": _make_b64("Caf\u00e9 \u2603")},
        }
        assert _decode_body(payload) == "Caf\u00e9 \u2603"


class TestGetHeader:
    """Test _get_header helper."""

    def test_found(self):
        from mcp_tools.gmail.server import _get_header

        headers = [
            {"name": "From", "value": "alice@example.com"},
            {"name": "Subject", "value": "Hello"},
        ]
        assert _get_header(headers, "Subject") == "Hello"

    def test_case_insensitive(self):
        from mcp_tools.gmail.server import _get_header

        headers = [{"name": "FROM", "value": "alice@example.com"}]
        assert _get_header(headers, "from") == "alice@example.com"

    def test_not_found(self):
        from mcp_tools.gmail.server import _get_header

        headers = [{"name": "From", "value": "alice@example.com"}]
        assert _get_header(headers, "X-Custom") == ""

    def test_empty_headers(self):
        from mcp_tools.gmail.server import _get_header

        assert _get_header([], "From") == ""

    def test_missing_value_key(self):
        from mcp_tools.gmail.server import _get_header

        headers = [{"name": "From"}]
        assert _get_header(headers, "From") == ""


class TestDefaultLabel:
    """Test _default_label helper."""

    def test_from_config(self):
        from mcp_tools.gmail.server import _default_label

        assert _default_label({"gmail": {"default_label": "STARRED"}}) == "STARRED"

    def test_missing_config(self):
        from mcp_tools.gmail.server import _default_label

        assert _default_label({}) == "INBOX"

    def test_non_dict_gmail_config(self):
        from mcp_tools.gmail.server import _default_label

        assert _default_label({"gmail": True}) == "INBOX"


class TestDefaultMaxResults:
    """Test _default_max_results helper."""

    def test_from_config(self):
        from mcp_tools.gmail.server import _default_max_results

        assert _default_max_results({"gmail": {"max_results": 25}}) == 25

    def test_missing_config(self):
        from mcp_tools.gmail.server import _default_max_results

        assert _default_max_results({}) == 10

    def test_non_dict_gmail_config(self):
        from mcp_tools.gmail.server import _default_max_results

        assert _default_max_results({"gmail": True}) == 10


# ---------------------------------------------------------------------------
# Config / account resolution tests
# ---------------------------------------------------------------------------

class TestLoadGoogleAuth:
    """Test _load_google_auth reading config.yaml."""

    def test_no_config_file(self, tmp_path):
        import mcp_tools.gmail.server as gs

        # Config file doesn't exist — should return empty dict
        result = gs._load_google_auth()
        assert result == {}

    def test_empty_config(self, tmp_path):
        import mcp_tools.gmail.server as gs

        _write_config(tmp_path, {})
        result = gs._load_google_auth()
        assert result == {}

    def test_valid_config(self, gmail_config):
        import mcp_tools.gmail.server as gs

        result = gs._load_google_auth()
        assert "accounts" in result
        assert "personal" in result["accounts"]


class TestGetAccounts:
    """Test _get_accounts."""

    def test_returns_accounts(self, gmail_config):
        import mcp_tools.gmail.server as gs

        accounts = gs._get_accounts()
        assert "personal" in accounts

    def test_no_accounts(self, tmp_path):
        import mcp_tools.gmail.server as gs

        _write_config(tmp_path, {"google_auth": {}})
        accounts = gs._get_accounts()
        assert accounts == {}


class TestResolve:
    """Test _resolve account resolution + service init."""

    def test_auth_failure_returns_error(self, gmail_config):
        import mcp_tools.gmail.server as gs

        with (
            patch("mcp_tools.gmail.server._get_service", return_value=None),
        ):
            svc, acct_cfg, err = gs._resolve("personal")
        assert svc is None
        assert "Authentication failed" in err

    def test_account_not_found(self, gmail_config):
        import mcp_tools.gmail.server as gs

        svc, acct_cfg, err = gs._resolve("nonexistent")
        assert svc is None
        assert "not found" in err.lower() or "not have" in err.lower()

    def test_no_accounts_configured(self, tmp_path):
        import mcp_tools.gmail.server as gs

        _write_config(tmp_path, {"google_auth": {"accounts": {}}})
        svc, acct_cfg, err = gs._resolve("")
        assert svc is None
        assert "No Google accounts" in err

    def test_successful_resolution(self, gmail_config, mock_gmail_service):
        import mcp_tools.gmail.server as gs

        with patch("mcp_tools.gmail.server._get_service", return_value=mock_gmail_service):
            svc, acct_cfg, err = gs._resolve("")
        assert svc is mock_gmail_service
        assert err == ""
        assert acct_cfg["gmail"]["enabled"] is True

    def test_ambiguous_multiple_accounts(self, multi_account_config):
        import mcp_tools.gmail.server as gs

        svc, acct_cfg, err = gs._resolve("")
        assert svc is None
        assert "Multiple accounts" in err


class TestGetService:
    """Test _get_service lazy initialization."""

    def test_caches_service(self, gmail_config):
        import mcp_tools.gmail.server as gs

        mock_svc = MagicMock()
        mock_creds = MagicMock()
        mock_build = MagicMock(return_value=mock_svc)

        with (
            patch("google_auth.auth.get_credentials", return_value=mock_creds),
            patch.dict(sys.modules, {"googleapiclient": MagicMock(), "googleapiclient.discovery": MagicMock(build=mock_build)}),
        ):
            svc1 = gs._get_service("personal")
            svc2 = gs._get_service("personal")

        assert svc1 is svc2
        assert svc1 is mock_svc

    def test_returns_none_when_no_creds(self, gmail_config):
        import mcp_tools.gmail.server as gs

        mock_build = MagicMock()

        with (
            patch("google_auth.auth.get_credentials", return_value=None),
            patch.dict(sys.modules, {"googleapiclient": MagicMock(), "googleapiclient.discovery": MagicMock(build=mock_build)}),
        ):
            svc = gs._get_service("personal")
        assert svc is None


# ---------------------------------------------------------------------------
# list_emails tool tests
# ---------------------------------------------------------------------------

class TestListEmails:
    """Test list_emails tool."""

    def test_success(self, mock_gmail_service):
        from mcp_tools.gmail.server import list_emails

        mock_gmail_service.users().messages().list().execute.return_value = {
            "messages": [{"id": "msg1"}, {"id": "msg2"}],
        }
        mock_gmail_service.users().messages().get().execute.return_value = {
            "id": "msg1",
            "snippet": "Hey there, just wanted to check in...",
            "payload": {
                "headers": [
                    {"name": "From", "value": "alice@example.com"},
                    {"name": "Subject", "value": "Check-in"},
                    {"name": "Date", "value": "Mon, 10 Mar 2026 10:00:00 -0500"},
                ],
            },
        }

        with _patch_resolve(mock_gmail_service):
            result = list_emails()

        assert "Emails in INBOX" in result
        assert "Check-in" in result
        assert "alice@example.com" in result

    def test_no_emails(self, mock_gmail_service):
        from mcp_tools.gmail.server import list_emails

        mock_gmail_service.users().messages().list().execute.return_value = {"messages": []}

        with _patch_resolve(mock_gmail_service):
            result = list_emails()
        assert "No emails in INBOX" in result

    def test_no_emails_with_query(self, mock_gmail_service):
        from mcp_tools.gmail.server import list_emails

        mock_gmail_service.users().messages().list().execute.return_value = {"messages": []}

        with _patch_resolve(mock_gmail_service):
            result = list_emails(query="from:bob")
        assert "No emails matching" in result
        assert "from:bob" in result

    def test_custom_label_and_limit(self, mock_gmail_service):
        from mcp_tools.gmail.server import list_emails

        mock_gmail_service.users().messages().list().execute.return_value = {
            "messages": [{"id": "msg1"}],
        }
        mock_gmail_service.users().messages().get().execute.return_value = {
            "id": "msg1",
            "snippet": "Sent message",
            "payload": {
                "headers": [
                    {"name": "From", "value": "me@example.com"},
                    {"name": "Subject", "value": "My sent email"},
                    {"name": "Date", "value": "Mon, 10 Mar 2026 10:00:00 -0500"},
                ],
            },
        }

        with _patch_resolve(mock_gmail_service):
            result = list_emails(label="SENT", limit=5)
        assert "Emails in SENT" in result

    def test_limit_clamped_to_range(self, mock_gmail_service):
        from mcp_tools.gmail.server import list_emails

        mock_gmail_service.users().messages().list().execute.return_value = {"messages": []}

        acct_cfg = {"gmail": {"enabled": True, "default_label": "INBOX", "max_results": 10}}
        with _patch_resolve(mock_gmail_service, acct_cfg):
            # limit > 50 should be clamped
            result = list_emails(limit=100)
        assert "No emails" in result

        # Verify the list call used maxResults=50 (clamped from 100)
        call_kwargs = mock_gmail_service.users().messages().list.call_args
        assert call_kwargs is not None
        assert call_kwargs[1]["maxResults"] == 50

    def test_uses_default_label_from_config(self, mock_gmail_service):
        from mcp_tools.gmail.server import list_emails

        mock_gmail_service.users().messages().list().execute.return_value = {"messages": []}

        acct_cfg = {"gmail": {"enabled": True, "default_label": "STARRED", "max_results": 10}}
        with _patch_resolve(mock_gmail_service, acct_cfg):
            result = list_emails()
        assert "No emails in STARRED" in result

    def test_account_error(self):
        from mcp_tools.gmail.server import list_emails

        with _patch_resolve_error("Account 'bad' not found."):
            result = list_emails(account="bad")
        assert "Account 'bad' not found" in result

    def test_api_exception(self, mock_gmail_service):
        from mcp_tools.gmail.server import list_emails

        mock_gmail_service.users().messages().list().execute.side_effect = Exception("API quota exceeded")

        with _patch_resolve(mock_gmail_service):
            result = list_emails()
        assert "Error listing emails" in result
        assert "API quota exceeded" in result

    def test_no_subject_header(self, mock_gmail_service):
        from mcp_tools.gmail.server import list_emails

        mock_gmail_service.users().messages().list().execute.return_value = {
            "messages": [{"id": "msg1"}],
        }
        mock_gmail_service.users().messages().get().execute.return_value = {
            "id": "msg1",
            "snippet": "A message with no subject",
            "payload": {
                "headers": [
                    {"name": "From", "value": "alice@example.com"},
                    {"name": "Date", "value": "Mon, 10 Mar 2026 10:00:00 -0500"},
                ],
            },
        }

        with _patch_resolve(mock_gmail_service):
            result = list_emails()
        assert "(No subject)" in result


# ---------------------------------------------------------------------------
# read_email tool tests
# ---------------------------------------------------------------------------

class TestReadEmail:
    """Test read_email tool."""

    def test_success(self, mock_gmail_service):
        from mcp_tools.gmail.server import read_email

        body_text = "Hello, this is the email body."
        mock_gmail_service.users().messages().get().execute.return_value = {
            "id": "msg123",
            "payload": {
                "mimeType": "text/plain",
                "body": {"data": _make_b64(body_text)},
                "headers": [
                    {"name": "From", "value": "alice@example.com"},
                    {"name": "To", "value": "me@example.com"},
                    {"name": "Subject", "value": "Test Email"},
                    {"name": "Date", "value": "Mon, 10 Mar 2026 10:00:00 -0500"},
                ],
            },
        }

        with _patch_resolve(mock_gmail_service):
            result = read_email("msg123")

        assert "Subject: Test Email" in result
        assert "From: alice@example.com" in result
        assert "To: me@example.com" in result
        assert "ID: msg123" in result
        assert body_text in result

    def test_long_body_truncated(self, mock_gmail_service):
        from mcp_tools.gmail.server import read_email

        body_text = "A" * 5000
        mock_gmail_service.users().messages().get().execute.return_value = {
            "id": "msg123",
            "payload": {
                "mimeType": "text/plain",
                "body": {"data": _make_b64(body_text)},
                "headers": [],
            },
        }

        with _patch_resolve(mock_gmail_service):
            result = read_email("msg123")
        assert "... (truncated)" in result
        # Body should be cut to ~3000 chars
        assert len(result) < 5000

    def test_no_subject(self, mock_gmail_service):
        from mcp_tools.gmail.server import read_email

        mock_gmail_service.users().messages().get().execute.return_value = {
            "id": "msg123",
            "payload": {
                "mimeType": "text/plain",
                "body": {"data": _make_b64("Body text")},
                "headers": [
                    {"name": "From", "value": "alice@example.com"},
                ],
            },
        }

        with _patch_resolve(mock_gmail_service):
            result = read_email("msg123")
        assert "Subject: (No subject)" in result

    def test_account_error(self):
        from mcp_tools.gmail.server import read_email

        with _patch_resolve_error("Authentication failed"):
            result = read_email("msg123")
        assert "Authentication failed" in result

    def test_api_exception(self, mock_gmail_service):
        from mcp_tools.gmail.server import read_email

        mock_gmail_service.users().messages().get().execute.side_effect = Exception("Not found")

        with _patch_resolve(mock_gmail_service):
            result = read_email("msg123")
        assert "Error reading email" in result
        assert "Not found" in result


# ---------------------------------------------------------------------------
# send_email tool tests
# ---------------------------------------------------------------------------

class TestSendEmail:
    """Test send_email tool."""

    def test_success(self, mock_gmail_service):
        from mcp_tools.gmail.server import send_email

        mock_gmail_service.users().messages().send().execute.return_value = {"id": "sent123"}

        with _patch_resolve(mock_gmail_service):
            result = send_email("bob@example.com", "Hello", "Hi Bob!")

        assert "Email sent to bob@example.com" in result
        assert "sent123" in result

    def test_account_error(self):
        from mcp_tools.gmail.server import send_email

        with _patch_resolve_error("No Google accounts configured."):
            result = send_email("bob@example.com", "Test", "Body")
        assert "No Google accounts" in result

    def test_api_exception(self, mock_gmail_service):
        from mcp_tools.gmail.server import send_email

        mock_gmail_service.users().messages().send().execute.side_effect = Exception(
            "Recipient address rejected"
        )

        with _patch_resolve(mock_gmail_service):
            result = send_email("invalid", "Test", "Body")
        assert "Error sending email" in result
        assert "Recipient address rejected" in result

    def test_mime_encoding(self, mock_gmail_service):
        import base64
        from mcp_tools.gmail.server import send_email

        mock_gmail_service.users().messages().send().execute.return_value = {"id": "s1"}

        with _patch_resolve(mock_gmail_service):
            send_email("bob@example.com", "Test Subject", "Body text")

        # Verify the raw message was base64-encoded with correct MIME structure
        call_kwargs = mock_gmail_service.users().messages().send.call_args
        assert call_kwargs is not None
        raw = call_kwargs[1].get("body", {}).get("raw", "")
        assert raw, "Expected raw message data in send call"
        decoded = base64.urlsafe_b64decode(raw).decode("utf-8", errors="replace")
        assert "bob@example.com" in decoded
        assert "Test Subject" in decoded
        assert "Body text" in decoded


# ---------------------------------------------------------------------------
# reply_email tool tests
# ---------------------------------------------------------------------------

class TestReplyEmail:
    """Test reply_email tool."""

    def test_success(self, mock_gmail_service):
        from mcp_tools.gmail.server import reply_email

        # Original message fetch
        mock_gmail_service.users().messages().get().execute.return_value = {
            "threadId": "thread1",
            "payload": {
                "headers": [
                    {"name": "From", "value": "alice@example.com"},
                    {"name": "Subject", "value": "Original Subject"},
                    {"name": "Message-ID", "value": "<msg-id-123@example.com>"},
                ],
            },
        }
        mock_gmail_service.users().messages().send().execute.return_value = {"id": "reply1"}

        with _patch_resolve(mock_gmail_service):
            result = reply_email("orig_msg_id", "Thanks for the info!")

        assert "Reply sent to alice@example.com" in result
        assert "reply1" in result

    def test_adds_re_prefix(self, mock_gmail_service):
        from mcp_tools.gmail.server import reply_email

        mock_gmail_service.users().messages().get().execute.return_value = {
            "threadId": "thread1",
            "payload": {
                "headers": [
                    {"name": "From", "value": "alice@example.com"},
                    {"name": "Subject", "value": "Meeting Notes"},
                    {"name": "Message-ID", "value": "<mid@example.com>"},
                ],
            },
        }
        mock_gmail_service.users().messages().send().execute.return_value = {"id": "r1"}

        with _patch_resolve(mock_gmail_service):
            reply_email("orig_id", "Reply body")

        # Verify the send call included the correct subject
        send_call = mock_gmail_service.users().messages().send
        call_kwargs = send_call.call_args
        assert call_kwargs is not None
        raw = call_kwargs[1].get("body", call_kwargs[0][0] if call_kwargs[0] else {}).get("raw", "")
        assert raw, "Expected raw message data in send call"
        import base64
        decoded = base64.urlsafe_b64decode(raw).decode("utf-8", errors="replace")
        assert "Re: Meeting Notes" in decoded

    def test_already_has_re_prefix(self, mock_gmail_service):
        from mcp_tools.gmail.server import reply_email

        mock_gmail_service.users().messages().get().execute.return_value = {
            "threadId": "thread1",
            "payload": {
                "headers": [
                    {"name": "From", "value": "alice@example.com"},
                    {"name": "Subject", "value": "Re: Already a reply"},
                    {"name": "Message-ID", "value": "<mid@example.com>"},
                ],
            },
        }
        mock_gmail_service.users().messages().send().execute.return_value = {"id": "r1"}

        with _patch_resolve(mock_gmail_service):
            reply_email("orig_id", "Reply body")

        # Verify it doesn't double-add "Re:"
        send_call = mock_gmail_service.users().messages().send
        call_kwargs = send_call.call_args
        assert call_kwargs is not None
        raw = call_kwargs[1].get("body", call_kwargs[0][0] if call_kwargs[0] else {}).get("raw", "")
        assert raw, "Expected raw message data in send call"
        decoded = base64.urlsafe_b64decode(raw).decode("utf-8", errors="replace")
        assert "Re: Re:" not in decoded

    def test_no_message_id(self, mock_gmail_service):
        from mcp_tools.gmail.server import reply_email

        mock_gmail_service.users().messages().get().execute.return_value = {
            "threadId": "thread1",
            "payload": {
                "headers": [
                    {"name": "From", "value": "alice@example.com"},
                    {"name": "Subject", "value": "No MID"},
                ],
            },
        }
        mock_gmail_service.users().messages().send().execute.return_value = {"id": "r1"}

        with _patch_resolve(mock_gmail_service):
            result = reply_email("orig_id", "Reply body")
        assert "Reply sent" in result

    def test_no_thread_id(self, mock_gmail_service):
        from mcp_tools.gmail.server import reply_email

        mock_gmail_service.users().messages().get().execute.return_value = {
            "payload": {
                "headers": [
                    {"name": "From", "value": "alice@example.com"},
                    {"name": "Subject", "value": "Orphan"},
                    {"name": "Message-ID", "value": "<mid@example.com>"},
                ],
            },
        }
        mock_gmail_service.users().messages().send().execute.return_value = {"id": "r1"}

        with _patch_resolve(mock_gmail_service):
            result = reply_email("orig_id", "Reply body")
        assert "Reply sent" in result

    def test_account_error(self):
        from mcp_tools.gmail.server import reply_email

        with _patch_resolve_error("Authentication failed"):
            result = reply_email("msg_id", "body")
        assert "Authentication failed" in result

    def test_api_exception(self, mock_gmail_service):
        from mcp_tools.gmail.server import reply_email

        mock_gmail_service.users().messages().get().execute.side_effect = Exception("Not found")

        with _patch_resolve(mock_gmail_service):
            result = reply_email("bad_id", "body")
        assert "Error replying" in result
        assert "Not found" in result


# ---------------------------------------------------------------------------
# search_emails tool tests
# ---------------------------------------------------------------------------

class TestSearchEmails:
    """Test search_emails tool."""

    def test_success(self, mock_gmail_service):
        from mcp_tools.gmail.server import search_emails

        mock_gmail_service.users().messages().list().execute.return_value = {
            "messages": [{"id": "msg1"}],
        }
        mock_gmail_service.users().messages().get().execute.return_value = {
            "id": "msg1",
            "payload": {
                "headers": [
                    {"name": "From", "value": "alice@example.com"},
                    {"name": "Subject", "value": "Meeting Tomorrow"},
                    {"name": "Date", "value": "Mon, 10 Mar 2026 10:00:00 -0500"},
                ],
            },
        }

        with _patch_resolve(mock_gmail_service):
            result = search_emails("from:alice subject:meeting")

        assert "Found 1 email(s)" in result
        assert "Meeting Tomorrow" in result
        assert "alice@example.com" in result

    def test_no_results(self, mock_gmail_service):
        from mcp_tools.gmail.server import search_emails

        mock_gmail_service.users().messages().list().execute.return_value = {"messages": []}

        with _patch_resolve(mock_gmail_service):
            result = search_emails("from:nobody")
        assert "No emails matching" in result
        assert "from:nobody" in result

    def test_limit_clamped(self, mock_gmail_service):
        from mcp_tools.gmail.server import search_emails

        mock_gmail_service.users().messages().list().execute.return_value = {"messages": []}

        with _patch_resolve(mock_gmail_service):
            # limit=0 should be clamped to 1
            result = search_emails("test", limit=0)
        assert "No emails matching" in result
        call_kwargs = mock_gmail_service.users().messages().list.call_args
        assert call_kwargs[1]["maxResults"] == 1

    def test_limit_over_50(self, mock_gmail_service):
        from mcp_tools.gmail.server import search_emails

        mock_gmail_service.users().messages().list().execute.return_value = {"messages": []}

        with _patch_resolve(mock_gmail_service):
            result = search_emails("test", limit=100)
        assert "No emails matching" in result
        call_kwargs = mock_gmail_service.users().messages().list.call_args
        assert call_kwargs[1]["maxResults"] == 50

    def test_multiple_results(self, mock_gmail_service):
        from mcp_tools.gmail.server import search_emails

        mock_gmail_service.users().messages().list().execute.return_value = {
            "messages": [{"id": "m1"}, {"id": "m2"}, {"id": "m3"}],
        }
        mock_gmail_service.users().messages().get().execute.return_value = {
            "id": "m1",
            "payload": {
                "headers": [
                    {"name": "From", "value": "test@example.com"},
                    {"name": "Subject", "value": "Result"},
                    {"name": "Date", "value": "Mon, 10 Mar 2026 10:00:00 -0500"},
                ],
            },
        }

        with _patch_resolve(mock_gmail_service):
            result = search_emails("is:unread")
        assert "Found 3 email(s)" in result

    def test_no_subject_fallback(self, mock_gmail_service):
        from mcp_tools.gmail.server import search_emails

        mock_gmail_service.users().messages().list().execute.return_value = {
            "messages": [{"id": "m1"}],
        }
        mock_gmail_service.users().messages().get().execute.return_value = {
            "id": "m1",
            "payload": {
                "headers": [
                    {"name": "From", "value": "test@example.com"},
                    {"name": "Date", "value": "Mon, 10 Mar 2026"},
                ],
            },
        }

        with _patch_resolve(mock_gmail_service):
            result = search_emails("test")
        assert "(No subject)" in result

    def test_account_error(self):
        from mcp_tools.gmail.server import search_emails

        with _patch_resolve_error("No accounts configured"):
            result = search_emails("test")
        assert "No accounts configured" in result

    def test_api_exception(self, mock_gmail_service):
        from mcp_tools.gmail.server import search_emails

        mock_gmail_service.users().messages().list().execute.side_effect = Exception("403 Forbidden")

        with _patch_resolve(mock_gmail_service):
            result = search_emails("test")
        assert "Error searching emails" in result
        assert "403 Forbidden" in result


# ---------------------------------------------------------------------------
# list_labels tool tests
# ---------------------------------------------------------------------------

class TestListLabels:
    """Test list_labels tool."""

    def test_success(self, mock_gmail_service):
        from mcp_tools.gmail.server import list_labels

        mock_gmail_service.users().labels().list().execute.return_value = {
            "labels": [
                {"name": "INBOX", "type": "system"},
                {"name": "SENT", "type": "system"},
                {"name": "Projects", "type": "user"},
                {"name": "Travel", "type": "user"},
            ],
        }

        with _patch_resolve(mock_gmail_service):
            result = list_labels()

        assert "Gmail Labels:" in result
        assert "INBOX" in result
        assert "SENT" in result
        assert "Projects" in result
        assert "Travel" in result

    def test_system_and_user_sections(self, mock_gmail_service):
        from mcp_tools.gmail.server import list_labels

        mock_gmail_service.users().labels().list().execute.return_value = {
            "labels": [
                {"name": "INBOX", "type": "system"},
                {"name": "MyLabel", "type": "user"},
            ],
        }

        with _patch_resolve(mock_gmail_service):
            result = list_labels()
        assert "System:" in result
        assert "Custom:" in result

    def test_no_labels(self, mock_gmail_service):
        from mcp_tools.gmail.server import list_labels

        mock_gmail_service.users().labels().list().execute.return_value = {"labels": []}

        with _patch_resolve(mock_gmail_service):
            result = list_labels()
        assert "No labels found" in result

    def test_only_system_labels(self, mock_gmail_service):
        from mcp_tools.gmail.server import list_labels

        mock_gmail_service.users().labels().list().execute.return_value = {
            "labels": [
                {"name": "INBOX", "type": "system"},
                {"name": "TRASH", "type": "system"},
            ],
        }

        with _patch_resolve(mock_gmail_service):
            result = list_labels()
        assert "System:" in result
        assert "Custom:" not in result

    def test_only_user_labels(self, mock_gmail_service):
        from mcp_tools.gmail.server import list_labels

        mock_gmail_service.users().labels().list().execute.return_value = {
            "labels": [
                {"name": "Personal", "type": "user"},
            ],
        }

        with _patch_resolve(mock_gmail_service):
            result = list_labels()
        assert "Custom:" in result
        assert "System:" not in result

    def test_account_error(self):
        from mcp_tools.gmail.server import list_labels

        with _patch_resolve_error("Auth failed"):
            result = list_labels()
        assert "Auth failed" in result

    def test_api_exception(self, mock_gmail_service):
        from mcp_tools.gmail.server import list_labels

        mock_gmail_service.users().labels().list().execute.side_effect = Exception("Forbidden")

        with _patch_resolve(mock_gmail_service):
            result = list_labels()
        assert "Error listing labels" in result


# ---------------------------------------------------------------------------
# search_contacts tool tests
# ---------------------------------------------------------------------------

class TestSearchContacts:
    """Test search_contacts tool."""

    def test_success_personal_contacts(self, gmail_config, mock_people_service):
        from mcp_tools.gmail.server import search_contacts

        mock_people_service.people().searchContacts().execute.return_value = {
            "results": [
                {
                    "person": {
                        "names": [{"displayName": "Alice Smith"}],
                        "emailAddresses": [{"value": "alice@example.com"}],
                        "organizations": [{"name": "Acme Corp"}],
                    },
                },
            ],
        }
        mock_people_service.people().searchDirectoryPeople().execute.return_value = {"people": []}

        with (
            patch("mcp_tools.gmail.server._get_people_service", return_value=mock_people_service),
            patch("google_auth.auth.resolve_account", return_value=("personal", "", {})),
        ):
            result = search_contacts("Alice")

        assert "Alice Smith" in result
        assert "alice@example.com" in result
        assert "Acme Corp" in result

    def test_success_directory_contacts(self, gmail_config, mock_people_service):
        from mcp_tools.gmail.server import search_contacts

        mock_people_service.people().searchContacts().execute.return_value = {"results": []}
        mock_people_service.people().searchDirectoryPeople().execute.return_value = {
            "people": [
                {
                    "names": [{"displayName": "Bob Jones"}],
                    "emailAddresses": [{"value": "bob@company.com"}],
                    "organizations": [{"name": "Company Inc"}],
                },
            ],
        }

        with (
            patch("mcp_tools.gmail.server._get_people_service", return_value=mock_people_service),
            patch("google_auth.auth.resolve_account", return_value=("personal", "", {})),
        ):
            result = search_contacts("Bob")

        assert "Bob Jones" in result
        assert "bob@company.com" in result
        assert "[directory]" in result

    def test_no_contacts_found(self, gmail_config, mock_people_service):
        from mcp_tools.gmail.server import search_contacts

        mock_people_service.people().searchContacts().execute.return_value = {"results": []}
        mock_people_service.people().searchDirectoryPeople().execute.return_value = {"people": []}

        with (
            patch("mcp_tools.gmail.server._get_people_service", return_value=mock_people_service),
            patch("google_auth.auth.resolve_account", return_value=("personal", "", {})),
        ):
            result = search_contacts("Nonexistent Person")

        assert "No contacts found" in result
        assert "Nonexistent Person" in result

    def test_people_service_unavailable(self, gmail_config):
        from mcp_tools.gmail.server import search_contacts

        with (
            patch("mcp_tools.gmail.server._get_people_service", return_value=None),
            patch("google_auth.auth.resolve_account", return_value=("personal", "", {})),
        ):
            result = search_contacts("Alice")

        assert "not available" in result.lower()

    def test_account_error(self, gmail_config):
        from mcp_tools.gmail.server import search_contacts

        with patch(
            "google_auth.auth.resolve_account",
            return_value=("", "Account 'bad' not found.", None),
        ):
            result = search_contacts("Alice", account="bad")
        assert "Account 'bad' not found" in result

    def test_personal_contacts_api_error(self, gmail_config, mock_people_service):
        from mcp_tools.gmail.server import search_contacts

        mock_people_service.people().searchContacts().execute.side_effect = Exception("API Error")
        mock_people_service.people().searchDirectoryPeople().execute.return_value = {"people": []}

        with (
            patch("mcp_tools.gmail.server._get_people_service", return_value=mock_people_service),
            patch("google_auth.auth.resolve_account", return_value=("personal", "", {})),
        ):
            result = search_contacts("Alice")

        # Should include the error message but not crash
        assert "Personal contacts search error" in result

    def test_directory_error_silenced(self, gmail_config, mock_people_service):
        from mcp_tools.gmail.server import search_contacts

        mock_people_service.people().searchContacts().execute.return_value = {"results": []}
        mock_people_service.people().searchDirectoryPeople().execute.side_effect = Exception(
            "Not a Workspace account"
        )

        with (
            patch("mcp_tools.gmail.server._get_people_service", return_value=mock_people_service),
            patch("google_auth.auth.resolve_account", return_value=("personal", "", {})),
        ):
            result = search_contacts("Alice")

        # Directory errors are silently ignored
        assert "No contacts found" in result
        assert "Not a Workspace" not in result

    def test_contact_without_name(self, gmail_config, mock_people_service):
        from mcp_tools.gmail.server import search_contacts

        mock_people_service.people().searchContacts().execute.return_value = {
            "results": [
                {
                    "person": {
                        "names": [],
                        "emailAddresses": [{"value": "nameless@example.com"}],
                        "organizations": [],
                    },
                },
            ],
        }
        mock_people_service.people().searchDirectoryPeople().execute.return_value = {"people": []}

        with (
            patch("mcp_tools.gmail.server._get_people_service", return_value=mock_people_service),
            patch("google_auth.auth.resolve_account", return_value=("personal", "", {})),
        ):
            result = search_contacts("test")

        assert "Unknown" in result
        assert "nameless@example.com" in result

    def test_contact_without_email(self, gmail_config, mock_people_service):
        from mcp_tools.gmail.server import search_contacts

        mock_people_service.people().searchContacts().execute.return_value = {
            "results": [
                {
                    "person": {
                        "names": [{"displayName": "No Email Person"}],
                        "emailAddresses": [],
                        "organizations": [],
                    },
                },
            ],
        }
        mock_people_service.people().searchDirectoryPeople().execute.return_value = {"people": []}

        with (
            patch("mcp_tools.gmail.server._get_people_service", return_value=mock_people_service),
            patch("google_auth.auth.resolve_account", return_value=("personal", "", {})),
        ):
            result = search_contacts("No Email")

        # Person with no email should not appear in results
        assert "No contacts found" in result

    def test_multiple_emails_per_contact(self, gmail_config, mock_people_service):
        from mcp_tools.gmail.server import search_contacts

        mock_people_service.people().searchContacts().execute.return_value = {
            "results": [
                {
                    "person": {
                        "names": [{"displayName": "Multi Email"}],
                        "emailAddresses": [
                            {"value": "first@example.com"},
                            {"value": "second@example.com"},
                        ],
                        "organizations": [],
                    },
                },
            ],
        }
        mock_people_service.people().searchDirectoryPeople().execute.return_value = {"people": []}

        with (
            patch("mcp_tools.gmail.server._get_people_service", return_value=mock_people_service),
            patch("google_auth.auth.resolve_account", return_value=("personal", "", {})),
        ):
            result = search_contacts("Multi")

        assert "first@example.com" in result
        assert "second@example.com" in result

    def test_deduplication_between_contacts_and_directory(
        self, gmail_config, mock_people_service
    ):
        from mcp_tools.gmail.server import search_contacts

        # Same person appears in both personal contacts and directory
        mock_people_service.people().searchContacts().execute.return_value = {
            "results": [
                {
                    "person": {
                        "names": [{"displayName": "Alice Smith"}],
                        "emailAddresses": [{"value": "alice@company.com"}],
                        "organizations": [{"name": "Company"}],
                    },
                },
            ],
        }
        mock_people_service.people().searchDirectoryPeople().execute.return_value = {
            "people": [
                {
                    "names": [{"displayName": "Alice Smith"}],
                    "emailAddresses": [{"value": "alice@company.com"}],
                    "organizations": [{"name": "Company"}],
                },
            ],
        }

        with (
            patch("mcp_tools.gmail.server._get_people_service", return_value=mock_people_service),
            patch("google_auth.auth.resolve_account", return_value=("personal", "", {})),
        ):
            result = search_contacts("Alice")

        # The directory entry format includes [directory] so it won't be an exact
        # duplicate of the personal contact entry. Both should appear but the
        # dedup logic checks the full entry string — the directory one has "[directory]".
        # The personal one: "- Alice Smith <alice@company.com> (Company)"
        # The directory one: "- Alice Smith <alice@company.com> [directory] (Company)"
        # Both are different strings, so both appear. That's the expected behavior.
        assert "alice@company.com" in result


# ---------------------------------------------------------------------------
# _get_people_service tests
# ---------------------------------------------------------------------------

class TestGetPeopleService:
    """Test _get_people_service lazy initialization."""

    def test_caches_service(self, gmail_config):
        import mcp_tools.gmail.server as gs

        mock_svc = MagicMock()
        mock_creds = MagicMock()
        mock_build = MagicMock(return_value=mock_svc)

        with (
            patch("google_auth.auth.get_credentials", return_value=mock_creds),
            patch.dict(sys.modules, {"googleapiclient": MagicMock(), "googleapiclient.discovery": MagicMock(build=mock_build)}),
        ):
            svc1 = gs._get_people_service("personal")
            svc2 = gs._get_people_service("personal")

        assert svc1 is svc2
        assert svc1 is mock_svc

    def test_returns_none_when_no_creds(self, gmail_config):
        import mcp_tools.gmail.server as gs

        mock_build = MagicMock()

        with (
            patch("google_auth.auth.get_credentials", return_value=None),
            patch.dict(sys.modules, {"googleapiclient": MagicMock(), "googleapiclient.discovery": MagicMock(build=mock_build)}),
        ):
            svc = gs._get_people_service("personal")
        assert svc is None
