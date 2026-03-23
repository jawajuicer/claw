"""TTS text sanitizer — cleans text for natural speech output.

Strips markdown formatting, code blocks, symbols, URLs, and other
artifacts that TTS engines would otherwise read literally.
Applied to ALL Claw voice responses.
"""

from __future__ import annotations

import re

# Compiled patterns for performance (applied in order)
_CODE_BLOCK = re.compile(r"```[\s\S]*?```")
_INLINE_CODE = re.compile(r"`([^`]+)`")
_HTML_TAGS = re.compile(r"<[^>]+>")
_URLS = re.compile(r"https?://\S+")
_FILE_PATHS = re.compile(r"(?:^|\s)(?:~/|/)[\w.\-]+(?:/[\w.\-]+)+", re.MULTILINE)
_MARKDOWN_BOLD = re.compile(r"\*\*(.+?)\*\*")
_MARKDOWN_ITALIC = re.compile(r"(?<!\*)\*(?!\*)(.+?)(?<!\*)\*(?!\*)")
_MARKDOWN_STRIKETHROUGH = re.compile(r"~~(.+?)~~")
_MARKDOWN_HEADERS = re.compile(r"^#{1,6}\s*", re.MULTILINE)
_MARKDOWN_HORIZONTAL_RULE = re.compile(r"^[-*_]{3,}\s*$", re.MULTILINE)
_MARKDOWN_LINK = re.compile(r"\[([^\]]+)\]\([^)]+\)")
_MARKDOWN_IMAGE = re.compile(r"!\[([^\]]*)\]\([^)]+\)")
_BULLET_POINTS = re.compile(r"^\s*[-*+]\s+", re.MULTILINE)
_NUMBERED_LIST = re.compile(r"^\s*\d+[.)]\s+", re.MULTILINE)
_BLOCKQUOTE = re.compile(r"^\s*>\s*", re.MULTILINE)
_MULTIPLE_NEWLINES = re.compile(r"\n{3,}")
_MULTIPLE_SPACES = re.compile(r"  +")
_EMOJI = re.compile(
    "["
    "\U0001f600-\U0001f64f"  # emoticons
    "\U0001f300-\U0001f5ff"  # symbols & pictographs
    "\U0001f680-\U0001f6ff"  # transport & map
    "\U0001f1e0-\U0001f1ff"  # flags
    "\U00002702-\U000027b0"
    "\U000024c2-\U0001f251"
    "\U0001f900-\U0001f9ff"  # supplemental
    "\U0001fa00-\U0001fa6f"  # chess, extended-A
    "\U0001fa70-\U0001faff"  # extended-B
    "\U00002600-\U000026ff"  # misc symbols
    "]+",
)


def sanitize_for_speech(text: str) -> str:
    """Clean text for natural TTS output.

    Strips code, markdown, symbols, and formatting so the TTS engine
    speaks only natural words. Safe to call on any text — plain text
    passes through with minimal changes.
    """
    if not text:
        return text

    # Remove code blocks entirely (they'd be gibberish spoken aloud)
    text = _CODE_BLOCK.sub("", text)

    # Extract inline code content (keep the word, drop the backticks)
    text = _INLINE_CODE.sub(r"\1", text)

    # Remove any remaining XML/HTML tags
    text = _HTML_TAGS.sub("", text)

    # Remove URLs
    text = _URLS.sub("", text)

    # Remove file paths (e.g. /home/user/file.py, ~/config.yaml)
    text = _FILE_PATHS.sub("", text)

    # Remove markdown images (keep alt text if present)
    text = _MARKDOWN_IMAGE.sub(r"\1", text)

    # Convert markdown links to just the link text
    text = _MARKDOWN_LINK.sub(r"\1", text)

    # Strip markdown formatting (keep the text)
    text = _MARKDOWN_BOLD.sub(r"\1", text)
    text = _MARKDOWN_ITALIC.sub(r"\1", text)
    text = _MARKDOWN_STRIKETHROUGH.sub(r"\1", text)
    text = _MARKDOWN_HEADERS.sub("", text)
    text = _MARKDOWN_HORIZONTAL_RULE.sub("", text)
    text = _BLOCKQUOTE.sub("", text)

    # Clean up list markers
    text = _BULLET_POINTS.sub("", text)
    text = _NUMBERED_LIST.sub("", text)

    # Symbol replacements
    text = text.replace("&", " and ")
    text = text.replace("→", " to ")
    text = text.replace("←", " from ")
    text = text.replace("=>", " to ")
    text = text.replace("->", " to ")
    text = text.replace("...", ". ")
    text = text.replace("—", ", ")
    text = text.replace("–", ", ")

    # Remove brackets/braces/pipes that TTS would read literally
    for ch in "{}[]|\\<>^":
        text = text.replace(ch, "")

    # Remove parentheses but keep content
    text = text.replace("(", ", ").replace(")", ", ")

    # Convert # to "number" only when followed by a digit, otherwise remove
    text = re.sub(r"#(\d)", r"number \1", text)
    text = text.replace("#", "")

    # Remove emojis
    text = _EMOJI.sub("", text)

    # Normalize whitespace
    text = _MULTIPLE_NEWLINES.sub("\n\n", text)
    text = _MULTIPLE_SPACES.sub(" ", text)

    # Clean up comma/period artifacts from removals
    text = re.sub(r"\s*,\s*,\s*", ", ", text)
    text = re.sub(r"\s*\.\s*\.\s*", ". ", text)
    text = re.sub(r"^\s*[,.:]\s*", "", text, flags=re.MULTILINE)

    return text.strip()
