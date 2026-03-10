"""Gemini MCP server — Google AI tools with rate limiting and audit logging."""

from __future__ import annotations

import json
import logging
import mimetypes
import os
from datetime import date, datetime
from pathlib import Path

from mcp.server.fastmcp import FastMCP

log = logging.getLogger(__name__)

mcp = FastMCP("Gemini")

# ---------------------------------------------------------------------------
# Config — loaded lazily from config.yaml, cached for process lifetime
# ---------------------------------------------------------------------------
_config: dict | None = None
_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
_CONFIG_YAML = _PROJECT_ROOT / "config.yaml"


def _load_config() -> dict:
    """Read gemini config from config.yaml."""
    global _config
    if _config is not None:
        return _config
    if _CONFIG_YAML.exists():
        import yaml
        with open(_CONFIG_YAML) as f:
            data = yaml.safe_load(f) or {}
        _config = data.get("gemini", {})
    else:
        _config = {}
    return _config


def _get_api_key() -> str | None:
    """Load API key from SecretStore, falling back to env var."""
    # Try env var first (CI / override)
    key = os.environ.get("GEMINI_API_KEY")
    if key:
        return key
    # Try encrypted store
    try:
        import sys
        sys.path.insert(0, str(_PROJECT_ROOT / "src"))
        from claw.secret_store import load
        key = load("gemini_api_key")
        if key:
            return key
    except Exception:
        log.debug("SecretStore not available, checking env only")
    return None


def _is_enabled(capability: str) -> bool:
    """Check if a specific capability is enabled in config."""
    cfg = _load_config()
    if not cfg.get("enabled", False):
        return False
    return cfg.get(capability, True)


# ---------------------------------------------------------------------------
# Rate limiting — in-memory counters, reset at midnight
# ---------------------------------------------------------------------------
_rate_state = {
    "date": "",
    "requests": 0,
    "grounding_requests": 0,
}


def _check_rate_limit(grounding: bool = False) -> str | None:
    """Check rate limits. Returns error message if exceeded, None if OK."""
    cfg = _load_config()
    today = date.today().isoformat()

    if _rate_state["date"] != today:
        _rate_state["date"] = today
        _rate_state["requests"] = 0
        _rate_state["grounding_requests"] = 0

    req_limit = cfg.get("daily_request_limit", 200)
    grounding_limit = cfg.get("grounding_daily_limit", 400)

    if _rate_state["requests"] >= req_limit:
        return f"Daily request limit reached ({req_limit}). Resets at midnight."

    if grounding and _rate_state["grounding_requests"] >= grounding_limit:
        return f"Daily grounding limit reached ({grounding_limit}). Resets at midnight."

    return None


def _record_request(grounding: bool = False) -> None:
    """Record a request for rate limiting."""
    today = date.today().isoformat()
    if _rate_state["date"] != today:
        _rate_state["date"] = today
        _rate_state["requests"] = 0
        _rate_state["grounding_requests"] = 0

    _rate_state["requests"] += 1
    if grounding:
        _rate_state["grounding_requests"] += 1


# ---------------------------------------------------------------------------
# Audit logging — JSONL files in data/gemini/logs/
# ---------------------------------------------------------------------------

def _log_request(tool: str, prompt_preview: str, response_preview: str, model: str) -> None:
    """Log an API request to the daily JSONL file."""
    cfg = _load_config()
    if not cfg.get("log_requests", True):
        return

    log_dir = _PROJECT_ROOT / cfg.get("log_dir", "data/gemini/logs")
    log_dir.mkdir(parents=True, exist_ok=True)

    today = date.today().isoformat()
    log_file = log_dir / f"{today}.jsonl"

    entry = {
        "timestamp": datetime.now().isoformat(),
        "tool": tool,
        "model": model,
        "prompt_preview": prompt_preview[:500],
        "response_preview": response_preview[:500],
    }

    with open(log_file, "a") as f:
        f.write(json.dumps(entry) + "\n")


def _get_log_entries(limit: int = 100) -> list[dict]:
    """Read recent log entries across all JSONL files."""
    cfg = _load_config()
    log_dir = _PROJECT_ROOT / cfg.get("log_dir", "data/gemini/logs")
    if not log_dir.exists():
        return []

    entries = []
    for log_file in sorted(log_dir.glob("*.jsonl"), reverse=True):
        for line in reversed(log_file.read_text().splitlines()):
            line = line.strip()
            if not line:
                continue
            try:
                entries.append(json.loads(line))
            except json.JSONDecodeError:
                continue
            if len(entries) >= limit:
                return entries
    return entries


def _wipe_logs() -> int:
    """Delete all JSONL log files. Returns count of files deleted."""
    cfg = _load_config()
    log_dir = _PROJECT_ROOT / cfg.get("log_dir", "data/gemini/logs")
    if not log_dir.exists():
        return 0
    count = 0
    for log_file in log_dir.glob("*.jsonl"):
        log_file.unlink()
        count += 1
    return count


# ---------------------------------------------------------------------------
# Gemini API helpers
# ---------------------------------------------------------------------------

def _get_genai():
    """Import and configure the google.generativeai module."""
    import google.generativeai as genai

    key = _get_api_key()
    if not key:
        return None
    genai.configure(api_key=key)
    return genai


def _not_configured_msg() -> str:
    return (
        "Gemini API key not configured. "
        "Set it in the admin panel under Settings > Gemini API, "
        "or set the GEMINI_API_KEY environment variable."
    )


# ---------------------------------------------------------------------------
# MCP tool endpoints
# ---------------------------------------------------------------------------

@mcp.tool()
def gemini_web_search(query: str) -> str:
    """Search the web using Google's Gemini API with grounding.

    Args:
        query: The search query, e.g. "latest Python 3.13 features"
    """
    if not _is_enabled("web_search"):
        return "Web search capability is disabled in settings."

    rate_err = _check_rate_limit(grounding=True)
    if rate_err:
        return rate_err

    genai = _get_genai()
    if genai is None:
        return _not_configured_msg()

    cfg = _load_config()
    model_name = cfg.get("model", "gemini-2.5-flash")

    try:
        model = genai.GenerativeModel(model_name)
        response = model.generate_content(
            f"Search the web and answer: {query}",
            tools="google_search",
        )
        result = response.text
        _record_request(grounding=True)
        _log_request("gemini_web_search", query, result, model_name)
        return result
    except Exception as e:
        log.exception("Gemini web search failed")
        return f"Web search failed: {e}"


@mcp.tool()
def gemini_analyze_document(file_path: str, instruction: str = "Summarize this document.") -> str:
    """Analyze or summarize a document using Gemini's large context window.

    Args:
        file_path: Path to the document file (PDF, TXT, MD, etc.)
        instruction: What to do with the document, e.g. "Summarize the key points"
    """
    if not _is_enabled("document_analysis"):
        return "Document analysis capability is disabled in settings."

    rate_err = _check_rate_limit()
    if rate_err:
        return rate_err

    genai = _get_genai()
    if genai is None:
        return _not_configured_msg()

    # Resolve path relative to project root
    path = Path(file_path)
    if not path.is_absolute():
        path = _PROJECT_ROOT / path
    if not path.exists():
        return f"File not found: {file_path}"

    cfg = _load_config()
    model_name = cfg.get("model", "gemini-2.5-flash")

    try:
        content = path.read_text(errors="replace")
        model = genai.GenerativeModel(model_name)
        response = model.generate_content(
            f"{instruction}\n\n---\n\n{content}",
        )
        result = response.text
        _record_request()
        _log_request("gemini_analyze_document", f"{file_path}: {instruction}", result, model_name)
        return result
    except Exception as e:
        log.exception("Gemini document analysis failed")
        return f"Document analysis failed: {e}"


@mcp.tool()
def gemini_describe_image(image_path: str, question: str = "Describe this image.") -> str:
    """Analyze an image using Gemini's vision capabilities.

    Args:
        image_path: Path to the image file (PNG, JPG, WEBP, etc.)
        question: Question about the image, e.g. "What objects are in this image?"
    """
    if not _is_enabled("image_understanding"):
        return "Image understanding capability is disabled in settings."

    rate_err = _check_rate_limit()
    if rate_err:
        return rate_err

    genai = _get_genai()
    if genai is None:
        return _not_configured_msg()

    path = Path(image_path)
    if not path.is_absolute():
        path = _PROJECT_ROOT / path
    if not path.exists():
        return f"Image not found: {image_path}"

    cfg = _load_config()
    model_name = cfg.get("model", "gemini-2.5-flash")

    try:
        mime_type = mimetypes.guess_type(str(path))[0] or "image/png"
        image_data = path.read_bytes()

        model = genai.GenerativeModel(model_name)
        response = model.generate_content([
            question,
            {"mime_type": mime_type, "data": image_data},
        ])
        result = response.text
        _record_request()
        _log_request("gemini_describe_image", f"{image_path}: {question}", result, model_name)
        return result
    except Exception as e:
        log.exception("Gemini image description failed")
        return f"Image description failed: {e}"


@mcp.tool()
def gemini_ask(question: str) -> str:
    """Ask Google Gemini a question directly. General-purpose query.

    Args:
        question: The question to ask Gemini
    """
    cfg = _load_config()
    if not cfg.get("enabled", False):
        return "Gemini is disabled in settings."

    rate_err = _check_rate_limit()
    if rate_err:
        return rate_err

    genai = _get_genai()
    if genai is None:
        return _not_configured_msg()

    model_name = cfg.get("model", "gemini-2.5-flash")

    try:
        model = genai.GenerativeModel(model_name)
        response = model.generate_content(question)
        result = response.text
        _record_request()
        _log_request("gemini_ask", question, result, model_name)
        return result
    except Exception as e:
        log.exception("Gemini ask failed")
        return f"Gemini query failed: {e}"


@mcp.tool()
def gemini_reason(question: str) -> str:
    """Use Gemini Pro for complex reasoning tasks that require deeper analysis.

    Args:
        question: The complex question or reasoning task
    """
    if not _is_enabled("reasoning_fallback"):
        return "Complex reasoning capability is disabled in settings."

    rate_err = _check_rate_limit()
    if rate_err:
        return rate_err

    genai = _get_genai()
    if genai is None:
        return _not_configured_msg()

    cfg = _load_config()
    model_name = cfg.get("pro_model", "gemini-2.5-pro")

    try:
        model = genai.GenerativeModel(model_name)
        response = model.generate_content(question)
        result = response.text
        _record_request()
        _log_request("gemini_reason", question, result, model_name)
        return result
    except Exception as e:
        log.exception("Gemini reasoning failed")
        return f"Reasoning failed: {e}"


@mcp.tool()
def gemini_usage() -> str:
    """Show today's Gemini API usage statistics and remaining quota."""
    cfg = _load_config()
    today = date.today().isoformat()

    if _rate_state["date"] != today:
        requests = 0
        grounding = 0
    else:
        requests = _rate_state["requests"]
        grounding = _rate_state["grounding_requests"]

    req_limit = cfg.get("daily_request_limit", 200)
    grounding_limit = cfg.get("grounding_daily_limit", 400)

    lines = [
        f"Gemini API Usage for {today}:",
        f"  Requests: {requests}/{req_limit} ({req_limit - requests} remaining)",
        f"  Grounding: {grounding}/{grounding_limit} ({grounding_limit - grounding} remaining)",
        f"  Model: {cfg.get('model', 'gemini-2.5-flash')}",
        f"  Pro Model: {cfg.get('pro_model', 'gemini-2.5-pro')}",
    ]

    key = _get_api_key()
    if key:
        lines.append(f"  API Key: {key[:4]}***{key[-4:]}")
    else:
        lines.append("  API Key: Not configured")

    return "\n".join(lines)


if __name__ == "__main__":
    mcp.run()
