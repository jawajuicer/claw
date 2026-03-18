"""MCP tool server for web browsing via Playwright."""

from __future__ import annotations

import atexit
import json
import os
import re
import threading
import urllib.parse
from datetime import datetime
from pathlib import Path
from urllib.parse import urlparse

from mcp.server.fastmcp import FastMCP

mcp = FastMCP("browser")

_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
_SCREENSHOT_DIR = _PROJECT_ROOT / "data" / "browser" / "screenshots"

# Security: blocked URL schemes
_BLOCKED_SCHEMES = {"file", "javascript", "data", "vbscript"}

# Security: blocked domains (banks, government, private networks)
_BLOCKED_DOMAIN_PATTERNS = [
    r".*\.bank\..*",
    r".*\.gov$",
    r".*\.gov\..*",
    r"localhost",
    r"127\.0\.0\.\d+",
    r"10\.\d+\.\d+\.\d+",
    r"192\.168\.\d+\.\d+",
    r"172\.(1[6-9]|2\d|3[01])\.\d+\.\d+",
    r"169\.254\.\d+\.\d+",  # link-local (includes cloud metadata 169.254.169.254)
    r"0\.0\.0\.0",
]

_MAX_CONTENT_CHARS = 50000
_MAX_PAGE_LOAD_MS = 30000

# Lazy-init browser globals (thread-safe)
_browser = None
_playwright_instance = None
_browser_lock = threading.Lock()


def _validate_url(url: str) -> str | None:
    """Validate URL for security. Returns error message or None if safe."""
    try:
        parsed = urlparse(url)
    except Exception:
        return "Invalid URL format"

    if parsed.scheme.lower() in _BLOCKED_SCHEMES:
        return f"Blocked URL scheme: {parsed.scheme}"

    if not parsed.scheme:
        return "URL must include scheme (http:// or https://)"

    hostname = parsed.hostname or ""
    for pattern in _BLOCKED_DOMAIN_PATTERNS:
        if re.match(pattern, hostname, re.I):
            return f"Blocked domain: {hostname}"

    return None


def _get_browser():
    """Get or create a shared Playwright browser instance (thread-safe)."""
    global _browser, _playwright_instance
    with _browser_lock:
        if _browser is None:
            from playwright.sync_api import sync_playwright

            _playwright_instance = sync_playwright().start()
            _browser = _playwright_instance.chromium.launch(headless=True)
    return _browser


@mcp.tool()
def browse_url(url: str) -> str:
    """Fetch and return the text content of a webpage.

    Args:
        url: The URL to browse (must be http:// or https://)
    """
    error = _validate_url(url)
    if error:
        return f"Error: {error}"

    try:
        browser = _get_browser()
        page = browser.new_page()
        try:
            page.goto(url, timeout=_MAX_PAGE_LOAD_MS, wait_until="domcontentloaded")
            text = page.inner_text("body")
            if len(text) > _MAX_CONTENT_CHARS:
                text = text[:_MAX_CONTENT_CHARS] + "\n\n[Content truncated]"
            title = page.title()
            return json.dumps({
                "title": title,
                "url": url,
                "content": text,
            })
        finally:
            page.close()
    except Exception as e:
        return f"Error browsing {url}: {e}"


@mcp.tool()
def screenshot(url: str, filename: str = "") -> str:
    """Take a screenshot of a webpage.

    Args:
        url: The URL to screenshot
        filename: Optional filename (defaults to domain_timestamp.png)
    """
    error = _validate_url(url)
    if error:
        return f"Error: {error}"

    try:
        browser = _get_browser()
        page = browser.new_page(viewport={"width": 1280, "height": 720})
        try:
            page.goto(url, timeout=_MAX_PAGE_LOAD_MS, wait_until="domcontentloaded")

            _SCREENSHOT_DIR.mkdir(parents=True, exist_ok=True)

            if not filename:
                parsed = urlparse(url)
                domain = parsed.hostname or "page"
                domain = re.sub(r"[^\w.-]", "_", domain)
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                filename = f"{domain}_{timestamp}.png"

            # Sanitize filename
            filename = re.sub(r"[^\w.-]", "_", filename)
            if not filename.endswith(".png"):
                filename += ".png"

            filepath = _SCREENSHOT_DIR / filename
            page.screenshot(path=str(filepath), full_page=False)

            return json.dumps({
                "status": "saved",
                "path": str(filepath),
                "url": url,
            })
        finally:
            page.close()
    except Exception as e:
        return f"Error taking screenshot of {url}: {e}"


@mcp.tool()
def search_web(query: str) -> str:
    """Search the web using DuckDuckGo and return results.

    Args:
        query: The search query
    """
    # Use DuckDuckGo HTML (no API key needed)
    search_url = f"https://html.duckduckgo.com/html/?q={urllib.parse.quote_plus(query)}"

    try:
        browser = _get_browser()
        page = browser.new_page()
        try:
            page.goto(search_url, timeout=_MAX_PAGE_LOAD_MS, wait_until="domcontentloaded")

            results = []
            # DuckDuckGo HTML results are in .result elements
            for result_elem in page.query_selector_all(".result"):
                title_elem = result_elem.query_selector(".result__a")
                snippet_elem = result_elem.query_selector(".result__snippet")
                url_elem = result_elem.query_selector(".result__url")

                if title_elem:
                    results.append({
                        "title": title_elem.inner_text(),
                        "url": (url_elem.inner_text().strip() if url_elem else ""),
                        "snippet": (snippet_elem.inner_text() if snippet_elem else ""),
                    })

                if len(results) >= 5:
                    break

            return json.dumps({
                "query": query,
                "results": results,
            }, indent=2)
        finally:
            page.close()
    except Exception as e:
        return f"Error searching for '{query}': {e}"


# Cleanup on exit
def _cleanup():
    global _browser, _playwright_instance
    if _browser is not None:
        try:
            _browser.close()
        except Exception:
            pass
    if _playwright_instance is not None:
        try:
            _playwright_instance.stop()
        except Exception:
            pass


atexit.register(_cleanup)


if __name__ == "__main__":
    mcp.run()
