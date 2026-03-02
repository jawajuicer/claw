"""Gmail MCP server — read, search, and send emails via Google Gmail API."""

from __future__ import annotations

import base64
import sys
from email.mime.text import MIMEText
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import yaml
from mcp.server.fastmcp import FastMCP

mcp = FastMCP("Gmail")

_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
_CONFIG_YAML = _PROJECT_ROOT / "config.yaml"
_services: dict[str, object] = {}


def _load_google_auth() -> dict:
    """Read google_auth from config.yaml fresh each time (no caching)."""
    if _CONFIG_YAML.exists():
        with open(_CONFIG_YAML) as f:
            data = yaml.safe_load(f) or {}
        return data.get("google_auth", {})
    return {}


def _get_accounts() -> dict:
    auth = _load_google_auth()
    return auth.get("accounts", {})


def _get_service(account_label: str):
    """Lazy-init a Gmail API service for a specific account."""
    if account_label in _services:
        return _services[account_label]

    from google_auth.auth import get_credentials
    from googleapiclient.discovery import build

    auth = _load_google_auth()
    accounts = auth.get("accounts", {})
    acct = accounts.get(account_label, {})

    creds = get_credentials(
        credentials_file=auth.get("credentials_file", "data/google/credentials.json"),
        token_file=acct.get("token_file", ""),
        scopes=auth.get("scopes", [
            "https://www.googleapis.com/auth/gmail.compose",
            "https://www.googleapis.com/auth/gmail.readonly",
        ]),
    )
    if creds is None:
        return None

    service = build("gmail", "v1", credentials=creds)
    _services[account_label] = service
    return service


def _resolve(account: str):
    """Common account resolution + service init. Returns (service, acct_cfg, error)."""
    from google_auth.auth import resolve_account

    accounts = _get_accounts()
    label, err, acct_cfg = resolve_account(accounts, "gmail", account)
    if err:
        return None, None, err
    svc = _get_service(label)
    if svc is None:
        return None, None, (
            "Gmail is not configured. "
            "Run: python mcp_tools/google_auth/setup_auth.py "
            "and enable Gmail in Settings."
        )
    return svc, acct_cfg, ""


def _decode_body(payload: dict) -> str:
    """Extract plain-text body from a Gmail message payload."""
    if payload.get("mimeType") == "text/plain" and payload.get("body", {}).get("data"):
        return base64.urlsafe_b64decode(payload["body"]["data"]).decode("utf-8", errors="replace")

    for part in payload.get("parts", []):
        if part.get("mimeType") == "text/plain" and part.get("body", {}).get("data"):
            return base64.urlsafe_b64decode(part["body"]["data"]).decode("utf-8", errors="replace")
        if part.get("parts"):
            result = _decode_body(part)
            if result:
                return result

    return "(No plain text body)"


def _get_header(headers: list[dict], name: str) -> str:
    """Get a header value by name."""
    for h in headers:
        if h.get("name", "").lower() == name.lower():
            return h.get("value", "")
    return ""


def _default_label(acct_cfg: dict) -> str:
    gmail_cfg = acct_cfg.get("gmail", {})
    if isinstance(gmail_cfg, dict):
        return gmail_cfg.get("default_label", "INBOX")
    return "INBOX"


def _default_max_results(acct_cfg: dict) -> int:
    gmail_cfg = acct_cfg.get("gmail", {})
    if isinstance(gmail_cfg, dict):
        return gmail_cfg.get("max_results", 10)
    return 10


@mcp.tool()
def list_emails(label: str = "", limit: int = 0, query: str = "", account: str = "") -> str:
    """List recent emails.

    Args:
        label: Gmail label to list from (e.g., "INBOX", "SENT", "STARRED"). Defaults to account setting.
        limit: Number of emails to return (1-50). 0 uses account default.
        query: Optional Gmail search query to filter results.
        account: Google account label (e.g., "personal", "work"). Leave empty to auto-select.
    """
    service, acct_cfg, err = _resolve(account)
    if err:
        return err

    if not label:
        label = _default_label(acct_cfg)
    if limit <= 0:
        limit = _default_max_results(acct_cfg)
    limit = max(1, min(50, limit))

    try:
        kwargs = {"userId": "me", "maxResults": limit, "labelIds": [label]}
        if query:
            kwargs["q"] = query
        result = service.users().messages().list(**kwargs).execute()
        messages = result.get("messages", [])

        if not messages:
            return f"No emails in {label}." if not query else f"No emails matching '{query}'."

        lines = [f"Emails in {label} ({len(messages)}):"]
        for msg_ref in messages:
            msg = service.users().messages().get(
                userId="me", id=msg_ref["id"], format="metadata",
                metadataHeaders=["From", "Subject", "Date"],
            ).execute()
            headers = msg.get("payload", {}).get("headers", [])
            frm = _get_header(headers, "From")
            subject = _get_header(headers, "Subject") or "(No subject)"
            date = _get_header(headers, "Date")
            snippet = msg.get("snippet", "")[:60]
            lines.append(f"- {subject}\n  From: {frm} | {date}\n  {snippet}... [id: {msg['id']}]")
        return "\n".join(lines)
    except Exception as e:
        return f"Error listing emails: {e}"


@mcp.tool()
def read_email(email_id: str, account: str = "") -> str:
    """Read the full content of an email.

    Args:
        email_id: The email's ID.
        account: Google account label (e.g., "personal", "work"). Leave empty to auto-select.
    """
    service, _, err = _resolve(account)
    if err:
        return err

    try:
        msg = service.users().messages().get(userId="me", id=email_id, format="full").execute()
        headers = msg.get("payload", {}).get("headers", [])
        frm = _get_header(headers, "From")
        to = _get_header(headers, "To")
        subject = _get_header(headers, "Subject") or "(No subject)"
        date = _get_header(headers, "Date")
        body = _decode_body(msg.get("payload", {}))

        if len(body) > 3000:
            body = body[:3000] + "\n\n... (truncated)"

        parts = [
            f"Subject: {subject}",
            f"From: {frm}",
            f"To: {to}",
            f"Date: {date}",
            f"ID: {email_id}",
            f"\n{body}",
        ]
        return "\n".join(parts)
    except Exception as e:
        return f"Error reading email: {e}"


@mcp.tool()
def send_email(to: str, subject: str, body: str, account: str = "") -> str:
    """Send an email.

    Args:
        to: Recipient email address.
        subject: Email subject line.
        body: Email body text.
        account: Google account label (e.g., "personal", "work"). Leave empty to auto-select.
    """
    service, _, err = _resolve(account)
    if err:
        return err

    try:
        message = MIMEText(body)
        message["to"] = to
        message["subject"] = subject
        raw = base64.urlsafe_b64encode(message.as_bytes()).decode()
        sent = service.users().messages().send(userId="me", body={"raw": raw}).execute()
        return f"Email sent to {to} (id: {sent['id']})"
    except Exception as e:
        return f"Error sending email: {e}"


@mcp.tool()
def reply_email(email_id: str, body: str, account: str = "") -> str:
    """Reply to an email.

    Args:
        email_id: The email ID to reply to.
        body: Reply body text.
        account: Google account label (e.g., "personal", "work"). Leave empty to auto-select.
    """
    service, _, err = _resolve(account)
    if err:
        return err

    try:
        original = service.users().messages().get(userId="me", id=email_id, format="metadata",
                                                   metadataHeaders=["From", "Subject", "Message-ID"]).execute()
        headers = original.get("payload", {}).get("headers", [])
        reply_to = _get_header(headers, "From")
        subject = _get_header(headers, "Subject")
        message_id = _get_header(headers, "Message-ID")
        thread_id = original.get("threadId", "")

        if not subject.lower().startswith("re:"):
            subject = f"Re: {subject}"

        message = MIMEText(body)
        message["to"] = reply_to
        message["subject"] = subject
        if message_id:
            message["In-Reply-To"] = message_id
            message["References"] = message_id

        raw = base64.urlsafe_b64encode(message.as_bytes()).decode()
        send_body = {"raw": raw}
        if thread_id:
            send_body["threadId"] = thread_id

        sent = service.users().messages().send(userId="me", body=send_body).execute()
        return f"Reply sent to {reply_to} (id: {sent['id']})"
    except Exception as e:
        return f"Error replying: {e}"


@mcp.tool()
def search_emails(query: str, limit: int = 10, account: str = "") -> str:
    """Search emails using Gmail search syntax.

    Args:
        query: Gmail search query (e.g., "from:alice subject:meeting", "is:unread").
        limit: Maximum results (1-50).
        account: Google account label (e.g., "personal", "work"). Leave empty to auto-select.
    """
    service, _, err = _resolve(account)
    if err:
        return err

    limit = max(1, min(50, limit))

    try:
        result = service.users().messages().list(userId="me", q=query, maxResults=limit).execute()
        messages = result.get("messages", [])

        if not messages:
            return f"No emails matching '{query}'."

        lines = [f"Found {len(messages)} email(s) matching '{query}':"]
        for msg_ref in messages:
            msg = service.users().messages().get(
                userId="me", id=msg_ref["id"], format="metadata",
                metadataHeaders=["From", "Subject", "Date"],
            ).execute()
            headers = msg.get("payload", {}).get("headers", [])
            frm = _get_header(headers, "From")
            subject = _get_header(headers, "Subject") or "(No subject)"
            date = _get_header(headers, "Date")
            lines.append(f"- {subject}\n  From: {frm} | {date} [id: {msg['id']}]")
        return "\n".join(lines)
    except Exception as e:
        return f"Error searching emails: {e}"


@mcp.tool()
def list_labels(account: str = "") -> str:
    """List all Gmail labels/folders.

    Args:
        account: Google account label (e.g., "personal", "work"). Leave empty to auto-select.
    """
    service, _, err = _resolve(account)
    if err:
        return err

    try:
        result = service.users().labels().list(userId="me").execute()
        labels = result.get("labels", [])
        if not labels:
            return "No labels found."

        system_labels = []
        user_labels = []
        for label in labels:
            if label.get("type") == "system":
                system_labels.append(label["name"])
            else:
                user_labels.append(label["name"])

        lines = ["Gmail Labels:"]
        if system_labels:
            lines.append(f"System: {', '.join(sorted(system_labels))}")
        if user_labels:
            lines.append(f"Custom: {', '.join(sorted(user_labels))}")
        return "\n".join(lines)
    except Exception as e:
        return f"Error listing labels: {e}"


if __name__ == "__main__":
    mcp.run()
