"""Tests for claw.memory_engine.store — MemoryStore (ChromaDB wrapper)."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest


class TestMemoryStoreInit:
    """Test initialization and _require_init guard."""

    def test_not_initialized_raises_on_add(self, settings):
        from claw.memory_engine.store import MemoryStore

        with patch("claw.memory_engine.store.get_settings", return_value=settings):
            store = MemoryStore()
        with pytest.raises(RuntimeError, match="not initialized"):
            store.add_conversation("id1", "hello")

    def test_not_initialized_raises_on_query(self, settings):
        from claw.memory_engine.store import MemoryStore

        with patch("claw.memory_engine.store.get_settings", return_value=settings):
            store = MemoryStore()
        with pytest.raises(RuntimeError, match="not initialized"):
            store.query_conversations("hello")

    def test_initialize_creates_collections(self, settings, tmp_path):
        from claw.memory_engine.store import MemoryStore

        settings.memory.chroma_path = str(tmp_path / "chromadb")

        mock_client = MagicMock()
        mock_collection = MagicMock()
        mock_collection.count.return_value = 0
        mock_client.get_or_create_collection.return_value = mock_collection

        with (
            patch("claw.memory_engine.store.get_settings", return_value=settings),
            patch("claw.memory_engine.store.SentenceTransformerEmbeddingFunction"),
            patch("claw.memory_engine.store.chromadb.PersistentClient", return_value=mock_client),
        ):
            store = MemoryStore()
            store.initialize()
            assert mock_client.get_or_create_collection.call_count == 3
            assert store.conversations is not None
            assert store.facts is not None
            assert store.categories is not None


class TestAddOperations:
    """Test adding documents to collections."""

    @pytest.fixture()
    def store(self, settings):
        from claw.memory_engine.store import MemoryStore

        with patch("claw.memory_engine.store.get_settings", return_value=settings):
            s = MemoryStore()
        s.conversations = MagicMock()
        s.facts = MagicMock()
        s.categories = MagicMock()
        return s

    def test_add_conversation(self, store):
        store.add_conversation("conv1", "hello world", {"role": "user"})
        store.conversations.add.assert_called_once_with(
            ids=["conv1"],
            documents=["hello world"],
            metadatas=[{"role": "user"}],
        )

    def test_add_fact(self, store):
        store.add_fact("fact1", "User likes pizza")
        store.facts.add.assert_called_once_with(
            ids=["fact1"],
            documents=["User likes pizza"],
            metadatas=[{}],
        )

    def test_add_category(self, store):
        store.add_category("cat1", "preferences")
        store.categories.add.assert_called_once_with(
            ids=["cat1"],
            documents=["preferences"],
            metadatas=[{}],
        )

    def test_add_conversation_none_metadata_becomes_empty_dict(self, store):
        store.add_conversation("conv1", "hello")
        store.conversations.add.assert_called_once_with(
            ids=["conv1"],
            documents=["hello"],
            metadatas=[{}],
        )


class TestQueryOperations:
    """Test querying collections."""

    @pytest.fixture()
    def store(self, settings):
        from claw.memory_engine.store import MemoryStore

        with patch("claw.memory_engine.store.get_settings", return_value=settings):
            s = MemoryStore()
        mock_collection = MagicMock()
        mock_collection.query.return_value = {
            "ids": [["id1", "id2"]],
            "documents": [["doc1", "doc2"]],
            "distances": [[0.1, 0.5]],
            "metadatas": [[{"k": "v1"}, {"k": "v2"}]],
        }
        s.conversations = mock_collection
        s.facts = mock_collection
        s.categories = mock_collection
        return s

    def test_query_conversations_returns_list(self, store):
        results = store.query_conversations("hello")
        assert len(results) == 2
        assert results[0]["id"] == "id1"
        assert results[0]["document"] == "doc1"
        assert results[0]["distance"] == 0.1

    def test_query_facts(self, store):
        results = store.query_facts("user preferences")
        assert len(results) == 2

    def test_query_with_custom_n_results(self, store):
        store.query_conversations("test", n_results=10)
        call_kwargs = store.conversations.query.call_args
        assert call_kwargs.kwargs.get("n_results") == 10 or call_kwargs[1].get("n_results") == 10


class TestUnpackResults:
    """Test the static _unpack_results helper."""

    def test_empty_results(self):
        from claw.memory_engine.store import MemoryStore

        assert MemoryStore._unpack_results({}) == []
        assert MemoryStore._unpack_results(None) == []
        assert MemoryStore._unpack_results({"ids": []}) == []

    def test_missing_optional_fields(self):
        from claw.memory_engine.store import MemoryStore

        results = {
            "ids": [["id1"]],
            "documents": None,
            "distances": None,
            "metadatas": None,
        }
        items = MemoryStore._unpack_results(results)
        assert len(items) == 1
        assert items[0]["document"] is None
        assert items[0]["distance"] is None

    def test_full_results(self):
        from claw.memory_engine.store import MemoryStore

        results = {
            "ids": [["id1"]],
            "documents": [["hello"]],
            "distances": [[0.2]],
            "metadatas": [[{"role": "user"}]],
        }
        items = MemoryStore._unpack_results(results)
        assert items[0]["id"] == "id1"
        assert items[0]["document"] == "hello"
        assert items[0]["distance"] == 0.2
        assert items[0]["metadata"] == {"role": "user"}


class TestStats:
    """Test the stats() method."""

    def test_stats_with_initialized_collections(self, settings):
        from claw.memory_engine.store import MemoryStore

        with patch("claw.memory_engine.store.get_settings", return_value=settings):
            store = MemoryStore()
        mock_coll = MagicMock()
        mock_coll.count.return_value = 42
        store.conversations = mock_coll
        store.facts = mock_coll
        store.categories = mock_coll
        stats = store.stats()
        assert stats == {"conversations": 42, "facts": 42, "categories": 42}

    def test_stats_uninitialized(self, settings):
        import pytest
        from claw.memory_engine.store import MemoryStore

        with patch("claw.memory_engine.store.get_settings", return_value=settings):
            store = MemoryStore()
        with pytest.raises(RuntimeError, match="not initialized"):
            store.stats()
