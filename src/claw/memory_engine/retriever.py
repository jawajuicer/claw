"""Semantic retrieval and fact extraction from memory."""

from __future__ import annotations

import logging
import uuid
from datetime import datetime, timezone

from claw.memory_engine.store import MemoryStore

log = logging.getLogger(__name__)


class MemoryRetriever:
    """High-level memory operations: context retrieval and fact extraction."""

    def __init__(self, store: MemoryStore) -> None:
        self.store = store

    def retrieve_context(self, query: str, max_results: int | None = None, max_chars: int = 600) -> str:
        """Retrieve relevant context from all memory collections.

        Returns a formatted string suitable for injection into the LLM system prompt.
        Capped at *max_chars* to keep prompt tokens manageable on CPU inference.
        """
        from claw.config import get_settings

        if max_results is None:
            max_results = get_settings().memory.max_results

        sections: list[str] = []

        # Search facts first (highest value)
        facts = self.store.query_facts(query, n_results=max_results)
        if facts:
            lines = [f"- {f['document']}" for f in facts]
            sections.append("Known facts:\n" + "\n".join(lines))

        # Search past conversations
        convos = self.store.query_conversations(query, n_results=max_results)
        if convos:
            lines = [f"- {c['document']}" for c in convos]
            sections.append("Relevant past conversations:\n" + "\n".join(lines))

        if not sections:
            return ""

        context = "--- Memory Context ---\n" + "\n\n".join(sections) + "\n--- End Memory ---"
        if len(context) > max_chars:
            context = context[:max_chars].rsplit("\n", 1)[0] + "\n--- End Memory ---"
        return context

    def store_conversation_turn(self, role: str, content: str, session_id: str) -> None:
        """Store a single conversation turn in memory."""
        turn_id = f"turn-{uuid.uuid4().hex[:12]}"
        self.store.add_conversation(
            conversation_id=turn_id,
            text=f"[{role}] {content}",
            metadata={
                "role": role,
                "session_id": session_id,
                "timestamp": datetime.now(timezone.utc).isoformat(),
            },
        )

    async def extract_and_store_facts(self, conversation: str, llm_client) -> list[str]:
        """Use the LLM to extract factual information from a conversation.

        This runs post-conversation to build long-term memory.
        """
        extraction_prompt = (
            "Extract any factual information about the user from this conversation. "
            "Return each fact on its own line. If there are no facts, reply with 'NONE'.\n\n"
            f"Conversation:\n{conversation}"
        )

        try:
            response = await llm_client.chat_simple(extraction_prompt)
        except Exception:
            log.exception("Fact extraction failed")
            return []

        if not response or response.strip().upper() == "NONE":
            return []

        facts = [line.strip().lstrip("- ") for line in response.strip().split("\n") if line.strip()]
        now = datetime.now(timezone.utc).isoformat()

        for fact_text in facts:
            if not fact_text:
                continue
            fact_id = f"fact-{uuid.uuid4().hex[:12]}"
            self.store.add_fact(
                fact_id=fact_id,
                text=fact_text,
                metadata={"extracted_at": now},
            )
            log.info("Stored fact: %s", fact_text)

        return facts

    def get_stats(self) -> dict[str, int]:
        return self.store.stats()
