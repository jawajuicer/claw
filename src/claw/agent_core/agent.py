"""Bounded-iteration agent loop with tool calling."""

from __future__ import annotations

import asyncio
import json
import logging
import re
import time
from dataclasses import dataclass

from claw.agent_core.conversation import ConversationSession
from claw.agent_core.llm_client import LLMClient
from claw.config import get_settings
from claw.memory_engine.retriever import MemoryRetriever

log = logging.getLogger(__name__)

# Fast keyword routing — bypasses LLM routing for high-confidence patterns.
# Each entry maps a regex pattern to the set of tool names it should select.
_KEYWORD_ROUTES: list[tuple[re.Pattern, set[str]]] = [
    (re.compile(r"\b(?:play|listen\s+to|put\s+on|queue\s+up)\b", re.I),
     {"play_song"}),
    (re.compile(r"\bpause\b", re.I), {"pause"}),
    (re.compile(r"\b(?:resume|unpause)\b", re.I), {"resume"}),
    (re.compile(r"\b(?:skip|next\s+(?:song|track))\b", re.I), {"skip"}),
    (re.compile(r"\bstop\s+(?:playing|music|the\s+music)\b", re.I), {"stop"}),
    (re.compile(r"\b(?:volume|louder|quieter|turn\s+(?:\S+\s+)?(?:up|down)|mute)\b", re.I),
     {"set_volume"}),
    (re.compile(r"\b(?:now\s+playing|what(?:'s|s|\s+is)\s+playing|current\s+(?:song|track))\b", re.I),
     {"now_playing"}),
    (re.compile(r"\b(?:queue|up\s+next|upcoming)\b", re.I), {"get_queue"}),
    (re.compile(r"\b(?:listen(?:ing)?\s+history|recently\s+played)\b", re.I),
     {"listen_history"}),
    (re.compile(r"\b(?:search|find|look\s+up)\s+(?:for\s+)?(?:a\s+)?(?:song|music|track)\b", re.I),
     {"search_music"}),
    # Weather
    (re.compile(r"\b(?:weather|forecast|temperature|rain|snow|sunny|cloudy|how\s+(?:hot|cold|warm))\b", re.I),
     {"get_weather"}),
    # System control
    (re.compile(r"\b(?:what\s+time|current\s+time|time\s+is\s+it)\b", re.I),
     {"get_time"}),
    (re.compile(r"\b(?:uptime|how\s+long.*running)\b", re.I), {"get_uptime"}),
    (re.compile(r"\b(?:disk\s+(?:space|usage)|storage)\b", re.I), {"get_disk_usage"}),
    (re.compile(r"\b(?:memory\s+usage|ram\s+usage|how\s+much\s+(?:memory|ram))\b", re.I),
     {"get_memory_usage"}),
    (re.compile(r"\b(?:system\s+info|system\s+status)\b", re.I), {"get_system_info"}),
    # Calendar
    (re.compile(r"\b(?:calendar|schedule|events?\s+(?:today|tomorrow|this\s+week)|what.*(?:today|tomorrow).*(?:calendar|schedule))\b", re.I),
     {"list_events"}),
    # Notes / reminders
    (re.compile(r"\b(?:create|make|add|write|new)\s+(?:a\s+)?note\b", re.I), {"create_note"}),
    (re.compile(r"\b(?:my\s+notes|list\s+notes|show\s+notes)\b", re.I), {"list_notes"}),
    (re.compile(r"\b(?:remind|reminder|set\s+(?:a\s+)?reminder)\b", re.I), {"set_reminder"}),
    # Email
    (re.compile(r"\b(?:email|inbox|mail|unread)\b", re.I), {"list_emails"}),
]

# Patterns that should NEVER route to tools — skip LLM routing entirely.
_NO_TOOL_PATTERNS: list[re.Pattern] = [
    re.compile(r"\b(?:joke|funny|laugh)\b", re.I),
    re.compile(r"\b(?:how\s+are\s+you|who\s+are\s+you|what\s+are\s+you|your\s+name)\b", re.I),
    re.compile(r"\b(?:thank|thanks|goodbye|bye|good\s+night|see\s+you)\b", re.I),
    re.compile(r"^\s*(?:yes|no|yeah|nah|sure|okay|ok|yep|nope)\s*[.!?]*\s*$", re.I),
    re.compile(r"\b(?:help|what\s+can\s+you\s+do|capabilities)\b", re.I),
    re.compile(r"\b(?:tell\s+me\s+(?:about|a)|explain|describe|define|what\s+is\s+(?:a\s+)?)\b", re.I),
    re.compile(r"\b(?:news|headline|current\s+events)\b", re.I),
    re.compile(r"\b(?:meaning\s+of\s+life|how\s+old|where\s+(?:am|are|is))\b", re.I),
]


@dataclass
class UsageStats:
    """Token usage and timing stats for a single process_utterance call."""
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0
    last_prompt_tokens: int = 0  # prompt tokens from most recent LLM call (= context size)
    llm_calls: int = 0
    elapsed_s: float = 0.0
    timed_out: bool = False

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

    async def process_utterance(
        self,
        text: str,
        tools: list[dict] | None = None,
        model: str | None = None,
    ) -> str:
        """Process a user utterance through the agent loop.

        Args:
            text: The user's transcribed speech or typed input.
            tools: OpenAI-format tool definitions from MCP registry.
            model: Override the default model for this request.

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
            memory_ctx = await asyncio.to_thread(self.retriever.retrieve_context, text)
            self._session.initialize(memory_context=memory_ctx)

        self._last_interaction = t0
        self._session.add_user(text)
        self._session.trim_to_fit()

        # Store user turn in memory
        self.retriever.store_conversation_turn("user", text, self._session.session_id)

        # Fast path: directly dispatch simple tool commands without LLM
        direct = await self._try_direct_dispatch(text)
        if direct is not None:
            self._session.add_assistant(direct)
            self.retriever.store_conversation_turn("assistant", direct, self._session.session_id)
            stats.elapsed_s = time.monotonic() - t0
            self.last_usage = stats
            log.info("Direct dispatch: %s", direct[:100])
            return direct

        # Tool routing: let the LLM pick which tools (if any) it needs
        # from a compact manifest instead of sending all full schemas
        if tools:
            tools = await self._route_tools(text, tools)

        for iteration in range(max_iterations):
            log.info("Agent iteration %d/%d", iteration + 1, max_iterations)

            try:
                response = await self.llm.chat(
                    messages=self._session.get_messages(),
                    tools=tools if tools else None,
                    model=model,
                )
            except asyncio.TimeoutError:
                content = "Sorry, I took too long to respond. Please try again."
                self._session.add_assistant(content)
                stats.elapsed_s = time.monotonic() - t0
                stats.timed_out = True
                self.last_usage = stats
                return content
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
        response = await self.llm.chat(messages=self._session.get_messages(), model=model)
        stats.accumulate(response)
        content = response.choices[0].message.content or "I wasn't able to complete that request."
        self._session.add_assistant(content)
        stats.elapsed_s = time.monotonic() - t0
        self.last_usage = stats
        return content

    async def _try_direct_dispatch(self, text: str) -> str | None:
        """Bypass LLM for simple tool commands that can be parsed directly.

        Returns the tool result string, or None to fall through to LLM.
        """
        if not self.tool_router:
            return None

        text_lower = text.lower().strip()

        # play_song: "play X" / "listen to X" / "put on X"
        for prefix in ["play me ", "play ", "listen to ", "put on ", "queue up "]:
            if text_lower.startswith(prefix):
                query = text[len(prefix):].strip()
                if not query:
                    return None
                by_match = re.match(r"(.+?)\s+by\s+(.+)", query, re.I)
                if by_match:
                    args = {"title": by_match.group(1).strip(), "artist": by_match.group(2).strip()}
                else:
                    args = {"title": query}
                try:
                    return str(await self.tool_router.call_tool("play_song", args))
                except Exception:
                    log.exception("Direct dispatch play_song failed")
                    return None

        # No-arg music controls
        _simple = [
            # Pause: "pause", "pause the music", "pause it"
            (r"^\s*pause\b", "pause"),
            # Resume: "resume", "unpause", "continue playing", "keep playing"
            (r"^\s*(?:resume|unpause)\b", "resume"),
            (r"^\s*(?:continue|keep)\s+playing\b", "resume"),
            # Skip/next: "skip", "next", "next song", "skip this", "skip song"
            (r"^\s*skip\b", "skip"),
            (r"^\s*next\s*$", "skip"),
            (r"^\s*next\s+(?:song|track|one)\b", "skip"),
            # Stop: "stop", "stop playing", "stop the music", "stop music"
            (r"\bstop\b(?:\s+(?:playing|music|the\s+music|it))?$", "stop"),
            # Now playing: "what's playing", "now playing", "current song", "what song is this"
            (r"(?:now\s+playing|what.s\s+playing|current\s+(?:song|track)|what\s+(?:song|track)\s+is\s+this)", "now_playing"),
            # Queue: "queue", "show queue", "what's next", "up next", "what's in the queue"
            (r"(?:show\s+)?(?:the\s+)?queue\b", "get_queue"),
            (r"(?:what.s|up)\s+next\b", "get_queue"),
            # History: "listen history", "recently played", "what did I listen to"
            (r"(?:listen(?:ing)?\s+history|recently\s+played|what\s+did\s+i\s+(?:listen|play))", "listen_history"),
        ]
        for pattern, tool_name in _simple:
            if re.search(pattern, text_lower):
                try:
                    return str(await self.tool_router.call_tool(tool_name, {}))
                except Exception:
                    log.exception("Direct dispatch %s failed", tool_name)
                    return None

        # Restart / go back: "start over", "restart", "go back", "from the top",
        # "play it again", "replay", "start this song over"
        if re.search(
            r"(?:start\s+over|restart|go\s+back|from\s+the\s+(?:top|beginning)|play\s+it\s+again|replay\b|beginning\s+of\s+(?:the\s+)?song)",
            text_lower,
        ):
            try:
                return str(await self.tool_router.call_tool("seek", {"position": 0.0}))
            except Exception:
                log.exception("Direct dispatch seek(0) failed")
                return None

        # set_volume: "volume 50" / "set volume to 80"
        vol_match = re.search(r"(?:volume|set\s+volume)\s*(?:to\s+)?(\d+)", text_lower)
        if vol_match:
            try:
                return str(await self.tool_router.call_tool("set_volume", {"level": int(vol_match.group(1))}))
            except Exception:
                log.exception("Direct dispatch set_volume failed")
                return None

        # Relative volume: "louder", "quieter", "turn it up", "turn it down", "mute"
        if re.search(r"\b(?:louder|turn\s+(?:it\s+)?up|raise\s+(?:the\s+)?volume)\b", text_lower):
            try:
                # Get current status to read volume, bump by 15
                status_json = await self.tool_router.call_tool("get_status", {})
                status = json.loads(status_json)
                cur = status.get("volume", 80)
                return str(await self.tool_router.call_tool("set_volume", {"level": min(100, cur + 15)}))
            except Exception:
                log.exception("Direct dispatch volume up failed")
                return None

        if re.search(r"\b(?:quieter|softer|turn\s+(?:it\s+)?down|lower\s+(?:the\s+)?volume)\b", text_lower):
            try:
                status_json = await self.tool_router.call_tool("get_status", {})
                status = json.loads(status_json)
                cur = status.get("volume", 80)
                return str(await self.tool_router.call_tool("set_volume", {"level": max(0, cur - 15)}))
            except Exception:
                log.exception("Direct dispatch volume down failed")
                return None

        if re.search(r"\bmute\b", text_lower):
            try:
                return str(await self.tool_router.call_tool("set_volume", {"level": 0}))
            except Exception:
                log.exception("Direct dispatch mute failed")
                return None

        return None

    def _keyword_route(self, text: str, tools: list[dict]) -> list[dict] | None:
        """Fast keyword-based routing for common patterns. Bypasses LLM call.

        Returns:
            list of tools if matched, empty list [] for no-tool patterns,
            or None to fall through to LLM routing.
        """
        # Check no-tool patterns first — conversational queries that never need tools
        for pattern in _NO_TOOL_PATTERNS:
            if pattern.search(text):
                log.info("Tool routing (keyword): no tools needed (conversational)")
                return []

        tool_names: set[str] = set()
        for pattern, names in _KEYWORD_ROUTES:
            if pattern.search(text):
                tool_names.update(names)

        if not tool_names:
            return None

        available = {t["function"]["name"] for t in tools}
        matched = tool_names & available
        if not matched:
            return None

        selected = [t for t in tools if t["function"]["name"] in matched]
        log.info("Tool routing (keyword): %s", [t["function"]["name"] for t in selected])
        return selected

    async def _route_tools(self, text: str, tools: list[dict]) -> list[dict] | None:
        """Select which tools (if any) are needed for this request.

        Uses fast keyword matching for high-confidence patterns.
        Falls back to passing all tools and letting the LLM decide.
        """
        # Fast path: keyword-based routing for common patterns
        keyword_result = self._keyword_route(text, tools)
        if keyword_result is not None:
            return keyword_result or None  # [] → None (no tools)

        # No keyword match — give the LLM all available tools and let it decide
        log.info("Tool routing: no keyword match, passing all %d tools to LLM", len(tools))
        return tools

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
