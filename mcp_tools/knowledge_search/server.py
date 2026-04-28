"""Knowledge search MCP server — semantic search over Claw's ChromaDB facts collection."""

from __future__ import annotations

import logging
import sys
from pathlib import Path

from mcp.server.fastmcp import FastMCP

log = logging.getLogger(__name__)

mcp = FastMCP("KnowledgeSearch")

# ---------------------------------------------------------------------------
# Path setup — make claw.* importable from src/
# ---------------------------------------------------------------------------
_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(_PROJECT_ROOT / "src"))

# ---------------------------------------------------------------------------
# Lazy MemoryStore singleton (cached for process lifetime)
# ---------------------------------------------------------------------------
_store = None  # type: ignore[var-annotated]


def _get_store():
    """Initialize MemoryStore on first call and cache it."""
    global _store
    if _store is not None:
        return _store
    # Imports deferred to first call so the MCP subprocess starts quickly
    # and so a misconfigured environment doesn't crash schema discovery.
    from claw.memory_engine.store import MemoryStore

    s = MemoryStore()
    s.initialize()
    _store = s
    return _store


_SNIPPET_LEN = 400


def _format_results(results: list[dict]) -> str:
    """Render top-K chunks as a readable string."""
    if not results:
        return "No matching information found."

    blocks: list[str] = []
    for r in results:
        meta = r.get("metadata") or {}
        title = meta.get("title") or "(untitled)"
        source = meta.get("source") or meta.get("file") or "(unknown source)"
        scope = meta.get("scope")
        document = (r.get("document") or "").strip()
        snippet = document[:_SNIPPET_LEN]
        if len(document) > _SNIPPET_LEN:
            snippet += "..."

        header = f"Title: {title}\nSource: {source}"
        if scope:
            header += f"\nScope: {scope}"
        blocks.append(f"{header}\n{snippet}")

    return "\n\n".join(blocks)


@mcp.tool()
def search_knowledge(query: str, scope: str | None = None, top_k: int = 5) -> str:
    """Search Claw's local knowledge base of ingested documents (Dencar product info, manuals, etc.). Use this when the user asks about Dencar, Sealevel I/O, Petit, wiring, configuration, pay stations, or other ingested topics.

    Args:
        query: Natural-language search query (e.g. "Sealevel wiring for pay station").
        scope: Optional scope filter (e.g. "dencar"). When set, also returns shared-scope facts.
        top_k: Number of top results to return (1-10, default 5).
    """
    if not query or not query.strip():
        return "No matching information found."

    # Clamp top_k to [1, 10]
    try:
        k = int(top_k)
    except (TypeError, ValueError):
        k = 5
    k = max(1, min(10, k))

    # Normalize scope: empty string -> None (no filter)
    s = scope.strip() if isinstance(scope, str) else None
    if s == "":
        s = None

    try:
        store = _get_store()
        results = store.query_facts(query.strip(), n_results=k, scope=s)
    except Exception as e:
        log.exception("knowledge_search query failed")
        return f"Knowledge search failed: {e}"

    return _format_results(results)


if __name__ == "__main__":
    mcp.run()
