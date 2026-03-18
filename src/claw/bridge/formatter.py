"""Response formatting and splitting for platform-specific message limits."""

from __future__ import annotations

import logging
import re

from claw.bridge.base import PlatformLimits

log = logging.getLogger(__name__)


def format_response(text: str, limits: PlatformLimits) -> list[str]:
    """Format and split a response for a specific platform.

    Handles:
    - Markdown flavor conversion
    - Message length splitting at sentence boundaries

    Returns a list of message chunks ready to send.
    """
    text = convert_markdown(text, limits.markdown_flavor)
    return split_message(text, limits.max_message_length)


def split_message(text: str, max_length: int) -> list[str]:
    """Split text into chunks that fit within the platform's message limit.

    Tries to split at paragraph boundaries first, then sentence boundaries,
    then word boundaries as a last resort.
    """
    if max_length <= 0 or len(text) <= max_length:
        return [text]

    chunks: list[str] = []
    remaining = text

    while remaining:
        if len(remaining) <= max_length:
            chunks.append(remaining)
            break

        # Try paragraph break
        split_pos = _find_split_point(remaining, max_length, "\n\n")
        if split_pos == -1:
            # Try sentence break
            split_pos = _find_split_point(remaining, max_length, ". ")
        if split_pos == -1:
            # Try newline
            split_pos = _find_split_point(remaining, max_length, "\n")
        if split_pos == -1:
            # Try word break
            split_pos = _find_split_point(remaining, max_length, " ")
        if split_pos == -1:
            # Hard break as last resort
            split_pos = max_length

        chunk = remaining[:split_pos].rstrip()
        if chunk:
            chunks.append(chunk)
        remaining = remaining[split_pos:].lstrip()

    return chunks or [text]


def _find_split_point(text: str, max_length: int, delimiter: str) -> int:
    """Find the last occurrence of delimiter within max_length."""
    # Search in the last 25% of the allowed length for a natural break
    search_start = max(0, max_length - max_length // 4)
    pos = text.rfind(delimiter, search_start, max_length)
    if pos != -1:
        return pos + len(delimiter)
    # Also try from the beginning
    pos = text.rfind(delimiter, 0, max_length)
    if pos != -1:
        return pos + len(delimiter)
    return -1


def convert_markdown(text: str, flavor: str) -> str:
    """Convert standard markdown to platform-specific flavor."""
    if flavor == "standard":
        return text

    if flavor == "telegram":
        return _to_telegram_markdown(text)
    elif flavor == "slack":
        return _to_slack_mrkdwn(text)
    elif flavor == "html":
        return _to_html(text)
    elif flavor == "irc":
        return _to_irc(text)

    return text


def _to_telegram_markdown(text: str) -> str:
    """Convert to Telegram MarkdownV2 format.

    Telegram MarkdownV2 requires ALL special characters to be escaped
    with a backslash when used outside of formatting entities. The safest
    approach is to escape everything that isn't already part of bold/italic
    formatting markers, since Telegram's parser is extremely strict.
    """
    # All chars Telegram MarkdownV2 considers special
    _TG_SPECIAL = set(r"_*[]()~`>#+-=|{}.!\\" )

    # Simple approach: escape all special chars. This means bold (**)
    # and italic (*) markers will also be escaped, effectively sending
    # plain text. This is safer than trying to parse markdown structure
    # and avoids Telegram 400 errors from mismatched formatting.
    result = []
    for ch in text:
        if ch in _TG_SPECIAL:
            result.append(f"\\{ch}")
        else:
            result.append(ch)
    return "".join(result)


def _to_slack_mrkdwn(text: str) -> str:
    """Convert standard markdown to Slack mrkdwn format."""
    # Italic FIRST (single *), before bold conversion removes the ** markers
    text = re.sub(r"(?<!\*)\*(?!\*)(.+?)(?<!\*)\*(?!\*)", r"_\1_", text)
    # Bold: **text** -> *text*
    text = re.sub(r"\*\*(.+?)\*\*", r"*\1*", text)
    # Strikethrough: ~~text~~ -> ~text~
    text = re.sub(r"~~(.+?)~~", r"~\1~", text)
    # Links: [text](url) -> <url|text>
    text = re.sub(r"\[(.+?)\]\((.+?)\)", r"<\2|\1>", text)
    return text


def _to_html(text: str) -> str:
    """Convert markdown to basic HTML (for platforms that use HTML formatting)."""
    # Bold
    text = re.sub(r"\*\*(.+?)\*\*", r"<b>\1</b>", text)
    # Italic
    text = re.sub(r"(?<!\*)\*(?!\*)(.+?)(?<!\*)\*(?!\*)", r"<i>\1</i>", text)
    # Code blocks
    text = re.sub(r"```(.+?)```", r"<pre>\1</pre>", text, flags=re.DOTALL)
    # Inline code
    text = re.sub(r"`(.+?)`", r"<code>\1</code>", text)
    # Newlines
    text = text.replace("\n", "<br>")
    return text


def _to_irc(text: str) -> str:
    """Convert markdown to IRC formatting codes."""
    # Bold: **text** -> \x02text\x02
    text = re.sub(r"\*\*(.+?)\*\*", "\x02\\1\x02", text)
    # Italic: *text* -> \x1dtext\x1d
    text = re.sub(r"(?<!\*)\*(?!\*)(.+?)(?<!\*)\*(?!\*)", "\x1d\\1\x1d", text)
    # Underline: __text__ -> \x1ftext\x1f
    text = re.sub(r"__(.+?)__", "\x1f\\1\x1f", text)
    # Code: `text` -> text (no IRC equivalent, just strip backticks)
    text = re.sub(r"`(.+?)`", r"\1", text)
    return text
