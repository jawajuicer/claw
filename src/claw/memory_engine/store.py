"""ChromaDB-backed persistent vector memory store."""

from __future__ import annotations

import logging
from pathlib import Path

import chromadb
from chromadb.utils.embedding_functions import SentenceTransformerEmbeddingFunction

from claw.config import PROJECT_ROOT, get_settings

log = logging.getLogger(__name__)


class MemoryStore:
    """Manages three ChromaDB collections: conversations, facts, categories."""

    def __init__(self) -> None:
        cfg = get_settings().memory
        chroma_path = Path(cfg.chroma_path)
        self._chroma_path = chroma_path if chroma_path.is_absolute() else PROJECT_ROOT / chroma_path
        self._embedding_model = cfg.embedding_model
        self._client: chromadb.ClientAPI | None = None
        self._embed_fn: SentenceTransformerEmbeddingFunction | None = None
        self.conversations: chromadb.Collection | None = None
        self.facts: chromadb.Collection | None = None
        self.categories: chromadb.Collection | None = None

    def initialize(self) -> None:
        """Create or open the ChromaDB persistent client and collections."""
        self._chroma_path.mkdir(parents=True, exist_ok=True)

        log.info("Loading embedding model '%s'...", self._embedding_model)
        self._embed_fn = SentenceTransformerEmbeddingFunction(
            model_name=self._embedding_model,
        )

        log.info("Opening ChromaDB at %s", self._chroma_path)
        self._client = chromadb.PersistentClient(path=str(self._chroma_path))

        cfg = get_settings().memory
        self.conversations = self._client.get_or_create_collection(
            name=cfg.conversation_collection,
            embedding_function=self._embed_fn,
        )
        self.facts = self._client.get_or_create_collection(
            name=cfg.facts_collection,
            embedding_function=self._embed_fn,
        )
        self.categories = self._client.get_or_create_collection(
            name=cfg.categories_collection,
            embedding_function=self._embed_fn,
        )
        log.info(
            "Memory store ready (conversations=%d, facts=%d, categories=%d)",
            self.conversations.count(),
            self.facts.count(),
            self.categories.count(),
        )

    def _require_init(self) -> None:
        if self.conversations is None:
            raise RuntimeError("MemoryStore not initialized — call initialize() first")

    def add_conversation(
        self, conversation_id: str, text: str, metadata: dict | None = None, scope: str | None = None,
    ) -> None:
        """Store a conversation turn."""
        self._require_init()
        meta = metadata or {}
        if scope is not None:
            meta["scope"] = scope
        self.conversations.add(
            ids=[conversation_id],
            documents=[text],
            metadatas=[meta],
        )

    def add_fact(
        self, fact_id: str, text: str, metadata: dict | None = None, scope: str | None = None,
    ) -> None:
        """Store an extracted fact."""
        self._require_init()
        meta = metadata or {}
        if scope is not None:
            meta["scope"] = scope
        self.facts.add(
            ids=[fact_id],
            documents=[text],
            metadatas=[meta],
        )

    def add_category(self, category_id: str, text: str, metadata: dict | None = None) -> None:
        """Store a category label."""
        self._require_init()
        meta = metadata or {}
        self.categories.add(
            ids=[category_id],
            documents=[text],
            metadatas=[meta],
        )

    def _scope_where(self, scope: str | None) -> dict | None:
        """Build a ChromaDB where clause for scope filtering.

        When scope is provided, returns documents matching that scope OR
        the "shared" scope (universal facts accessible everywhere).
        When scope is None, returns None (no filtering — backward compatible).
        """
        if scope is None:
            return None
        return {"$or": [{"scope": scope}, {"scope": "shared"}]}

    def query_conversations(
        self, query: str, n_results: int | None = None, scope: str | None = None,
    ) -> list[dict]:
        """Search conversations by semantic similarity."""
        self._require_init()
        n = n_results or get_settings().memory.max_results
        where = self._scope_where(scope)
        kwargs: dict = {"query_texts": [query], "n_results": n}
        if where is not None:
            kwargs["where"] = where
        results = self.conversations.query(**kwargs)
        return self._unpack_results(results)

    def query_facts(
        self, query: str, n_results: int | None = None, scope: str | None = None,
    ) -> list[dict]:
        """Search facts by semantic similarity."""
        self._require_init()
        n = n_results or get_settings().memory.max_results
        where = self._scope_where(scope)
        kwargs: dict = {"query_texts": [query], "n_results": n}
        if where is not None:
            kwargs["where"] = where
        results = self.facts.query(**kwargs)
        return self._unpack_results(results)

    def query_categories(self, query: str, n_results: int | None = None) -> list[dict]:
        """Search categories by semantic similarity."""
        self._require_init()
        n = n_results or get_settings().memory.max_results
        results = self.categories.query(query_texts=[query], n_results=n)
        return self._unpack_results(results)

    def stats(self) -> dict[str, int]:
        """Return counts for each collection."""
        self._require_init()
        return {
            "conversations": self.conversations.count() if self.conversations else 0,
            "facts": self.facts.count() if self.facts else 0,
            "categories": self.categories.count() if self.categories else 0,
        }

    @staticmethod
    def _unpack_results(results: dict) -> list[dict]:
        """Convert ChromaDB query results into a flat list of dicts."""
        items = []
        if not results or not results.get("ids"):
            return items
        for i, doc_id in enumerate(results["ids"][0]):
            item = {
                "id": doc_id,
                "document": results["documents"][0][i] if results.get("documents") else None,
                "distance": results["distances"][0][i] if results.get("distances") else None,
                "metadata": results["metadatas"][0][i] if results.get("metadatas") else None,
            }
            items.append(item)
        return items
