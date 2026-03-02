"""Bounded-iteration agent loop with tool calling."""

from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass, field

from claw.agent_core.conversation import ConversationSession
from claw.agent_core.llm_client import LLMClient
from claw.config import get_settings
from claw.memory_engine.retriever import MemoryRetriever

log = logging.getLogger(__name__)


@dataclass
class UsageStats:
    """Token usage and timing stats for a single process_utterance call."""
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0
    last_prompt_tokens: int = 0  # prompt tokens from most recent LLM call (= context size)
    llm_calls: int = 0
    elapsed_s: float = 0.0

    @property
    def tokens_per_sec(self) -> float:
        if self.elapsed_s > 0 and self.completion_tokens > 0:
            return self.completion_tokens / self.elapsed_s
        return 0.0

    def accumulate(self, response) -> None:
        usage = getattr(response, "usage", None)
        if usage:
            self.prompt_tokens += usage.prompt_tokens or 0
            self.completion_tokens += usage.completion_tokens or 0
            self.total_tokens += usage.total_tokens or 0
            self.last_prompt_tokens = usage.prompt_tokens or 0
        self.llm_calls += 1

    def to_dict(self) -> dict:
        return {
            "prompt_tokens": self.prompt_tokens,
            "completion_tokens": self.completion_tokens,
            "total_tokens": self.total_tokens,
            "last_prompt_tokens": self.last_prompt_tokens,
            "llm_calls": self.llm_calls,
            "elapsed_s": round(self.elapsed_s, 2),
            "tokens_per_sec": round(self.tokens_per_sec, 1),
        }


class Agent:
    """The core agent: LLM call → check tool calls → dispatch → repeat.

    Runs a bounded iteration loop (max N rounds) until the LLM produces
    a content-only response (no tool calls).
    """

    def __init__(
        self,
        llm: LLMClient,
        retriever: MemoryRetriever,
        tool_router=None,
    ) -> None:
        self.llm = llm
        self.retriever = retriever
        self.tool_router = tool_router  # set after MCP init
        self._session: ConversationSession | None = None
        self._last_interaction: float = 0.0  # monotonic timestamp of last utterance
        self.last_usage: UsageStats | None = None

    @property
    def session(self) -> ConversationSession | None:
        return self._session

    async def process_utterance(self, text: str, tools: list[dict] | None = None) -> str:
        """Process a user utterance through the agent loop.

        Args:
            text: The user's transcribed speech or typed input.
            tools: OpenAI-format tool definitions from MCP registry.

        Returns:
            The final assistant response text.
        """
        cfg = get_settings().llm
        max_iterations = cfg.max_iterations
        stats = UsageStats()
        t0 = time.monotonic()

        # Auto-rotate session if idle too long
        session_timeout = get_settings().chat.session_timeout
        if (
            self._session is not None
            and self._last_interaction > 0
            and (t0 - self._last_interaction) > session_timeout
        ):
            log.info(
                "Session expired after %ds idle (timeout=%ds), starting fresh",
                int(t0 - self._last_interaction), session_timeout,
            )
            self._session = None

        # Start or continue a session
        if self._session is None:
            self._session = ConversationSession()
            memory_ctx = self.retriever.retrieve_context(text)
            self._session.initialize(memory_context=memory_ctx)

        self._last_interaction = t0
        self._session.add_user(text)
        self._session.trim_to_fit()

        # Store user turn in memory
        self.retriever.store_conversation_turn("user", text, self._session.session_id)

        for iteration in range(max_iterations):
            log.info("Agent iteration %d/%d", iteration + 1, max_iterations)

            response = await self.llm.chat(
                messages=self._session.get_messages(),
                tools=tools if tools else None,
            )
            stats.accumulate(response)
            choice = response.choices[0]
            message = choice.message

            # If no tool calls, we have a final response
            if not message.tool_calls:
                content = message.content or ""
                self._session.add_assistant(content)
                self.retriever.store_conversation_turn("assistant", content, self._session.session_id)
                stats.elapsed_s = time.monotonic() - t0
                self.last_usage = stats
                log.info("Agent response (iteration %d): %s", iteration + 1, content[:100])
                return content

            # Process tool calls
            log.info("Agent requesting %d tool call(s)", len(message.tool_calls))
            self._session.add_tool_call(message.model_dump())

            for tool_call in message.tool_calls:
                name = tool_call.function.name
                args_str = tool_call.function.arguments
                try:
                    args = json.loads(args_str) if args_str else {}
                except json.JSONDecodeError:
                    args = {}

                log.info("Tool call: %s(%s)", name, json.dumps(args)[:200])

                if self.tool_router:
                    try:
                        result = await self.tool_router.call_tool(name, args)
                    except Exception as e:
                        result = f"Error calling tool '{name}': {e}"
                        log.exception("Tool call failed: %s", name)
                else:
                    result = f"Tool '{name}' is not available (no MCP router configured)"

                result_str = str(result) if not isinstance(result, str) else result
                self._session.add_tool_result(tool_call.id, result_str)
                log.info("Tool result for %s: %s", name, result_str[:200])

        # Exhausted iterations — ask LLM for final answer without tools
        log.warning("Agent hit max iterations (%d), forcing final response", max_iterations)
        response = await self.llm.chat(messages=self._session.get_messages())
        stats.accumulate(response)
        content = response.choices[0].message.content or "I wasn't able to complete that request."
        self._session.add_assistant(content)
        stats.elapsed_s = time.monotonic() - t0
        self.last_usage = stats
        return content

    async def extract_facts(self) -> list[str]:
        """Run post-conversation fact extraction."""
        if self._session is None:
            return []
        text = self._session.get_user_assistant_text()
        if not text:
            return []
        return await self.retriever.extract_and_store_facts(text, self.llm)

    def new_session(self) -> None:
        """Start a fresh conversation session."""
        self._session = None
