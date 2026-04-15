"""Bounded-iteration agent loop with tool calling."""

from __future__ import annotations

import asyncio
import json
import logging
import re
import time
from dataclasses import dataclass

from claw.agent_core.claude_relay import ClaudeRelay
from claw.agent_core.conversation import ConversationSession
from claw.agent_core.llm_client import LLMClient
from claw.config import get_settings
from claw.memory_engine.retriever import MemoryRetriever

log = logging.getLogger(__name__)

# Claude Code mode activation patterns.
# "claude" alone or with flags (e.g. "claude --dangerously-skip-permissions").
_CLAUDE_MODE_ON_RE = re.compile(
    r"^\s*(?:connect(?:\s+me)?\s+to\s+claude|talk\s+to\s+claude"
    r"|claude\s+code\s+mode|switch\s+to\s+claude"
    r"|open\s+(?:a\s+)?claude\s+(?:code\s+)?session"
    r"|start\s+(?:a\s+)?claude\s+(?:code\s+)?session"
    r"|claude\s+code"
    r"|claude)"  # just "claude" (with optional flags handled below)
    r"(?:\s+--[\w-]+)*"  # optional CLI-style flags
    r"(?:\s+(?:please|now|for\s+me))?\s*[.!?]*\s*$",
    re.I,
)

# Claude Code mode deactivation patterns.
# "exit" alone works when in relay mode.
_CLAUDE_MODE_OFF_RE = re.compile(
    r"^\s*(?:disconnect(?:\s+from\s+claude)?"
    r"|exit(?:\s+claude(?:\s+(?:code\s+)?mode)?)?"
    r"|leave\s+claude(?:\s+(?:code\s+)?mode)?"
    r"|back\s+to\s+normal(?:\s+mode)?"
    r"|normal\s+mode"
    r"|stop\s+claude(?:\s+(?:code\s+)?mode)?"
    r"|end\s+claude\s+(?:code\s+)?session"
    r"|close\s+claude(?:\s+session)?)\s*[.!?]*\s*$",
    re.I,
)

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
    (re.compile(r"\b(?:pin|save|remember)\s+this\s+(?:song\s+)?as\b", re.I), {"pin_song"}),
    (re.compile(r"\b(?:unpin|remove\s+pin)\b", re.I), {"unpin_song"}),
    (re.compile(r"\b(?:list|show)\s+(?:my\s+)?pins\b", re.I), {"list_pins"}),
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
    # Calendar — route all Google Calendar tools so the LLM can pick the right one
    (re.compile(r"\b(?:calendar|schedule|events?\s+(?:today|tomorrow|this\s+week)|what.*(?:today|tomorrow).*(?:calendar|schedule))\b", re.I),
     {"list_events", "list_calendars", "search_events", "create_event", "update_event", "delete_event"}),
    (re.compile(r"\b(?:add|create|put|schedule|book)\b.*\b(?:calendar|event|appointment|meeting)\b", re.I),
     {"create_event", "list_events", "list_calendars"}),
    # Notes / reminders
    (re.compile(r"\b(?:create|make|add|write|new)\s+(?:a\s+)?note\b", re.I), {"create_note"}),
    (re.compile(r"\b(?:my\s+notes|list\s+notes|show\s+notes)\b", re.I), {"list_notes"}),
    (re.compile(r"\b(?:remind|reminder|set\s+(?:a\s+)?reminder)\b", re.I), {"set_reminder"}),
    # Email — route all Gmail tools so the LLM can pick the right one
    (re.compile(r"\b(?:email|inbox|mail|unread|(?:check|read)\s+(?:my\s+)?messages)\b", re.I),
     {"list_emails", "send_email", "reply_email", "search_emails", "read_email", "list_labels", "search_contacts"}),
    # Contact / people lookup
    (re.compile(r"\b(?:contact|address\s+(?:for|of)|email\s+(?:for|of|address))\b", re.I),
     {"search_contacts"}),
    (re.compile(r"\b(?:people|person|who\s+is|find\s+(?:a\s+)?(?:person|people)|search\s+contacts?)\b", re.I),
     {"search_contacts"}),
    # Gemini web search
    (re.compile(r"\b(?:search\s+the\s+web|google\s+it|look\s+it\s+up\s+online|web\s+search)\b", re.I),
     {"gemini_web_search"}),
    (re.compile(r"\b(?:current|latest|recent|today'?s)\s+(?:news|events|prices?|scores?)\b", re.I),
     {"gemini_web_search"}),
    # Gemini direct ask / research
    (re.compile(r"\b(?:ask\s+gemini|have\s+gemini|use\s+gemini|gemini\s+(?:look|search|research|find))\b", re.I), {"gemini_ask"}),
    (re.compile(r"^gemini[,:]?\s+", re.I), {"gemini_ask"}),
    (re.compile(r"\b(?:look\s+up|research|search\s+for)\b.*\bgemini\b", re.I), {"gemini_ask"}),
    (re.compile(r"\bgemini\b.*\b(?:look\s+up|research|search)\b", re.I), {"gemini_ask"}),
    # Cron jobs
    (re.compile(r"\b(?:cron\s+job|recurring|schedule\s+(?:a\s+)?(?:recurring|daily|weekly|hourly))\b", re.I),
     {"create_cron_job", "list_cron_jobs", "delete_cron_job"}),
    (re.compile(r"\b(?:list|show|my)\s+cron\b", re.I), {"list_cron_jobs"}),
    # Inter-agent inbox (use specific phrasing to avoid collision with email "inbox"/"messages")
    (re.compile(r"\b(?:agent\s+inbox|system\s+inbox|cron\s+(?:results|messages)|check\s+notifications)\b", re.I),
     {"check_inbox", "send_to_inbox", "clear_inbox"}),
    # Browser / web browsing
    (re.compile(r"\b(?:browse|open\s+(?:the\s+)?(?:url|website|page|link)|visit\s+(?:the\s+)?(?:url|website|page))\b", re.I),
     {"browse_url", "screenshot", "search_web"}),
    (re.compile(r"\b(?:take\s+(?:a\s+)?screenshot|screenshot\s+(?:of|this))\b", re.I),
     {"screenshot"}),
    # Skills
    (re.compile(r"\b(?:install\s+(?:a\s+)?skill|skill\s+install)\b", re.I),
     {"skill_install", "skill_uninstall", "skill_list"}),
    (re.compile(r"\b(?:list|show)\s+skills\b", re.I), {"skill_list"}),
]

# Patterns that should NEVER route to tools — skip LLM routing entirely.
_NO_TOOL_PATTERNS: list[re.Pattern] = [
    re.compile(r"\b(?:joke|funny|laugh)\b", re.I),
    re.compile(r"\b(?:how\s+are\s+you|who\s+are\s+you|what\s+are\s+you|your\s+name)\b", re.I),
    re.compile(r"\b(?:thank|thanks|goodbye|bye|good\s+night|see\s+you)\b", re.I),
    re.compile(r"^\s*(?:yes|no|yeah|nah|sure|okay|ok|yep|nope)\s*[.!?]*\s*$", re.I),
    re.compile(r"\b(?:help|what\s+can\s+you\s+do|capabilities)\b", re.I),
    re.compile(r"\b(?:tell\s+me\s+(?:about|a)|explain|describe|define|what\s+is\s+(?:a\s+)?)\b", re.I),
    re.compile(r"\b(?:headline)\b", re.I),
    re.compile(r"\b(?:meaning\s+of\s+life|how\s+old|where\s+(?:am|are|is))\b", re.I),
]

# Strip hallucinated tool-call markup that small local models sometimes emit
# in their text output instead of using proper function calling.
# Covers: <|tool_call>...<tool_call|>, <tool_call>...</tool_call>,
#          <|tool_call|>..., and similar variants.
_HALLUCINATED_TOOL_CALL_RE = re.compile(
    r"<\|?tool_call\|?>.*?<\|?/?tool_call\|?>",
    re.DOTALL,
)

# No-arg music controls — compiled once at module level.
_SIMPLE_MUSIC_CONTROLS: list[tuple[re.Pattern, str]] = [
    # Pause: "pause", "pause the music", "pause it"
    (re.compile(r"^\s*pause\b"), "pause"),
    # Resume: "resume", "unpause", "continue playing", "keep playing"
    (re.compile(r"^\s*(?:resume|unpause)\b"), "resume"),
    (re.compile(r"^\s*(?:continue|keep)\s+playing\b"), "resume"),
    # Skip/next: "skip", "next", "next song", "skip this", "skip song"
    (re.compile(r"^\s*skip\b"), "skip"),
    (re.compile(r"^\s*next\s*$"), "skip"),
    (re.compile(r"^\s*next\s+(?:song|track|one)\b"), "skip"),
    # Stop: "stop", "stop playing", "stop the music", "stop music"
    (re.compile(r"\bstop\b(?:\s+(?:playing|music|the\s+music|it))?$"), "stop"),
    # Now playing: "what's playing", "now playing", "current song", "what song is this"
    (re.compile(r"(?:now\s+playing|what(?:'?s|\s+is)\s+playing|current\s+(?:song|track)|what\s+(?:song|track)\s+is\s+this)"), "now_playing"),
    # Queue: "queue", "show queue", "what's next", "up next", "what's in the queue"
    (re.compile(r"(?:show\s+)?(?:the\s+)?queue\b"), "get_queue"),
    (re.compile(r"(?:what(?:'?s|\s+is)|up)\s+next\b"), "get_queue"),
    # History: "listen history", "recently played", "what did I listen to"
    (re.compile(r"(?:listen(?:ing)?\s+history|recently\s+played|what\s+did\s+i\s+(?:listen|play))"), "listen_history"),
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
    provider: str = "local"  # which provider actually served
    tier: str = "standard"  # "fast", "standard", "cloud"

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
            "provider": self.provider,
            "tier": self.tier,
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
        self.usage_tracker = None  # set by main.py after creation
        self._session: ConversationSession | None = None
        self._last_interaction: float = 0.0  # monotonic timestamp of last utterance
        self.last_usage: UsageStats | None = None
        self._model_override: str | None = None
        self._pending_escalation: str | None = None
        self._memory_scope: str | None = None
        # Claude Code voice relay
        self._claude_relay = ClaudeRelay()
        self.tts_override: str | None = None  # voice summary for TTS (caller reads this)

    @staticmethod
    def _clean_response(content: str) -> str:
        """Strip hallucinated tool-call markup from LLM text output."""
        cleaned = _HALLUCINATED_TOOL_CALL_RE.sub("", content).strip()
        return cleaned

    @property
    def session(self) -> ConversationSession | None:
        return self._session

    @property
    def claude_code_active(self) -> bool:
        """Whether the agent is in Claude Code relay mode."""
        return self._claude_relay.active

    async def process_utterance(
        self,
        text: str,
        tools: list[dict] | None = None,
        model: str | None = None,
        context: str | None = None,
        _skip_session_memory: bool = False,
        memory_scope: str | None = None,
        interactive: bool = False,
        voice_mode: bool = False,
        images: list[tuple[bytes, str]] | None = None,
    ) -> str:
        """Process a user utterance through the agent loop.

        Args:
            text: The user's transcribed speech or typed input.
            tools: OpenAI-format tool definitions from MCP registry.
            model: Override the default model for this request.
            context: Additional context to inject before the user message
                     (e.g., platform info, memory refresh for bridge sessions).
            _skip_session_memory: If True, skip memory retrieval during session
                     init (caller already injected memory via context).
            memory_scope: Memory scope for reads/writes (e.g. "personal", "work").
                     None = no filtering (backward compatible for voice/CLI).
            interactive: If True, this is a human-facing caller that can
                     activate/use Claude Code relay. Set by voice, CLI, and
                     bridge callers. NOT set by cron, webhooks, or system calls.
            voice_mode: If True, request a voice summary for TTS (voice loop only).

        Returns:
            The final assistant response text.
        """
        if not model and self._model_override:
            model = self._model_override

        # Claude Code relay mode — bypass ALL local processing.
        # Available to any interactive caller (voice, CLI, bridges).
        self.tts_override = None  # reset each call
        if interactive and self._claude_relay.active:
            if _CLAUDE_MODE_OFF_RE.match(text.strip()):
                await self._claude_relay.async_reset()
                return "Disconnected from Claude Code. Back to normal mode."
            full_response, summary = await self._claude_relay.send(
                text, voice_mode=voice_mode,
            )
            if summary:
                self.tts_override = summary  # voice callers use this for TTS
            return full_response  # full response for display

        settings = get_settings()
        cfg = settings.llm
        max_iterations = cfg.max_iterations
        stats = UsageStats()
        t0 = time.monotonic()

        # Auto-rotate session if idle too long
        session_timeout = settings.chat.session_timeout
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
            self._pending_escalation = None

        # Store memory scope for use by extract_facts()
        self._memory_scope = memory_scope

        # Start or continue a session (also catches uninitialized bridge sessions)
        if self._session is None or not self._session.messages:
            if self._session is None:
                self._session = ConversationSession()
            if _skip_session_memory:
                memory_ctx = ""
            else:
                try:
                    memory_ctx = await asyncio.wait_for(
                        asyncio.to_thread(
                            self.retriever.retrieve_context, text,
                            None, 600, memory_scope,
                        ),
                        timeout=3.0,
                    )
                except asyncio.TimeoutError:
                    log.warning("Memory retrieval timed out")
                    memory_ctx = ""
            self._session.initialize(memory_context=memory_ctx)

        self._last_interaction = t0
        # Inject context (platform info, memory refresh) before user message
        combined = f"{context}\n\n{text}" if context else text
        if images:
            self._session.add_user_multimodal(combined, images)
        else:
            self._session.add_user(combined)
        self._session.trim_to_fit()

        # Auto-compact if context is getting large
        if (cfg.compact_threshold > 0
                and self._session.estimate_tokens() > cfg.context_window * cfg.compact_threshold):
            try:
                summary = await self._session.compact(self.llm, keep_recent=cfg.compact_keep_recent)
                if summary:
                    log.info("Auto-compacted context: %s", summary[:100])
            except Exception:
                log.exception("Auto-compact failed, falling back to trim")

        # Store user turn in memory
        self.retriever.store_conversation_turn("user", text, self._session.session_id, scope=memory_scope)

        # Fast path: directly dispatch simple tool commands without LLM
        direct = await self._try_direct_dispatch(text, interactive=interactive)
        if direct is not None:
            self._session.add_assistant(direct)
            self.retriever.store_conversation_turn("assistant", direct, self._session.session_id, scope=memory_scope)
            stats.elapsed_s = time.monotonic() - t0
            self.last_usage = stats
            await self._record_usage(stats)
            log.info("Direct dispatch: %s", direct[:100])
            return direct

        # Tool routing: let the LLM pick which tools (if any) it needs
        # from a compact manifest instead of sending all full schemas
        if tools:
            tools = await self._route_tools(text, tools)

        # Tier routing: fast model for simple queries, standard for everything else
        effective_model = model
        if not effective_model and self._should_use_fast_model(text, bool(tools), bool(images)):
            effective_model = cfg.fast_model
            stats.tier = "fast"
            log.info("Tier 1 (fast model '%s') for simple query", effective_model)

        for iteration in range(max_iterations):
            log.info("Agent iteration %d/%d", iteration + 1, max_iterations)

            try:
                t_llm = time.monotonic()
                response = await self.llm.chat(
                    messages=self._session.get_messages(),
                    tools=tools if tools else None,
                    model=effective_model,
                )
                log.info("LLM call: %.1fms", (time.monotonic() - t_llm) * 1000)
            except asyncio.TimeoutError:
                content = "Sorry, I took too long to respond. Please try again."
                self._session.add_assistant(content)
                stats.elapsed_s = time.monotonic() - t0
                stats.timed_out = True
                self.last_usage = stats
                await self._record_usage(stats)
                return content
            except RuntimeError as e:
                if "LLM providers failed" in str(e):
                    content = "Sorry, I couldn't reach any LLM backend. Please check your configuration."
                    self._session.add_assistant(content)
                    stats.elapsed_s = time.monotonic() - t0
                    stats.timed_out = True
                    self.last_usage = stats
                    await self._record_usage(stats)
                    return content
                raise
            stats.accumulate(response)
            stats.provider = self.llm.last_serving_provider
            choice = response.choices[0]
            message = choice.message

            # If no tool calls, we have a final response
            if not message.tool_calls:
                content = self._clean_response(message.content or "")
                # If cleaning stripped everything (model only emitted hallucinated
                # tool calls), nudge it to answer directly and retry.
                if not content and iteration < max_iterations - 1:
                    log.warning("Empty response after cleaning hallucinated markup, retrying")
                    self._session.add_assistant("")
                    self._session.add_user("Please answer the question directly.")
                    continue
                content = await self._maybe_escalate(text, content, stats, t0)
                self._session.add_assistant(content)
                self.retriever.store_conversation_turn("assistant", content, self._session.session_id, scope=memory_scope)
                stats.elapsed_s = time.monotonic() - t0
                self.last_usage = stats
                await self._record_usage(stats)
                log.info("Agent response (iteration %d, %.1fms total): %s",
                         iteration + 1, stats.elapsed_s * 1000, content[:100])
                return content

            # Process tool calls
            log.info("Agent requesting %d tool call(s)", len(message.tool_calls))
            self._session.add_tool_call(message.model_dump())

            for tool_call in message.tool_calls:
                name = tool_call.function.name
                args_str = tool_call.function.arguments
                try:
                    args = json.loads(args_str) if args_str else {}
                except json.JSONDecodeError as e:
                    log.warning("Invalid JSON in tool args for %s: %s", name, e)
                    result_str = f"Error: malformed tool arguments (invalid JSON): {args_str[:200]}"
                    self._session.add_tool_result(tool_call.id, result_str)
                    continue

                log.info("Tool call: %s(%s)", name, json.dumps(args)[:200])

                if self.tool_router:
                    try:
                        t_tool = time.monotonic()
                        result = await self.tool_router.call_tool(name, args)
                        log.info("Tool %s: %.1fms", name, (time.monotonic() - t_tool) * 1000)
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
        response = await self.llm.chat(messages=self._session.get_messages(), model=effective_model)
        stats.accumulate(response)
        stats.provider = self.llm.last_serving_provider
        content = self._clean_response(response.choices[0].message.content or "I wasn't able to complete that request.")
        self._session.add_assistant(content)
        self.retriever.store_conversation_turn("assistant", content, self._session.session_id, scope=memory_scope)
        stats.elapsed_s = time.monotonic() - t0
        self.last_usage = stats
        await self._record_usage(stats)
        return content

    async def _try_direct_dispatch(self, text: str, *, interactive: bool = False) -> str | None:
        """Bypass LLM for simple tool commands that can be parsed directly.

        Returns the tool result string, or None to fall through to LLM.
        """
        if not self.tool_router:
            return None

        text_lower = text.lower().strip()

        # Handle pending escalation confirmation
        if self._pending_escalation is not None:
            question = self._pending_escalation
            if re.match(
                r"^\s*(?:yes|yeah|sure|do\s+it|go\s+ahead|please|yep|ok|okay)\s*[.!?]*\s*$",
                text_lower,
            ):
                self._pending_escalation = None
                result = await self._escalate_to_cloud(question)
                return result
            elif re.match(
                r"^\s*(?:no|nah|nope|nevermind|no\s+thanks|never\s+mind)\s*[.!?]*\s*$",
                text_lower,
            ):
                self._pending_escalation = None
                return None  # fall through to normal LLM
            else:
                # Any other utterance clears the stale offer
                self._pending_escalation = None

        # Handle dismissive/closing remarks with a brief acknowledgment
        if re.match(
            r"^\s*(?:nevermind|never\s*mind|forget\s+(?:it|about\s+it)|nvm|"
            r"that'?s?\s+(?:ok|okay|fine|all|it)|"
            r"don'?t\s+worry|no\s+worries|all\s+good)\s*[.!?]*\s*$",
            text_lower,
        ):
            return "Okay."

        # Claude Code mode activation (any interactive caller — not cron/webhooks)
        if interactive and _CLAUDE_MODE_ON_RE.match(text.strip()):
            if not self._claude_relay.available:
                return (
                    "Claude Code relay is not configured. "
                    "Set claude_relay settings in config.yaml."
                )
            # Parse optional flags (e.g. "claude --dangerously-skip-permissions")
            if "--dangerously-skip-permissions" in text_lower:
                self._claude_relay.skip_permissions_override = True
                log.info("Claude Code relay: --dangerously-skip-permissions enabled")
            msg = await self._claude_relay.activate()
            log.info("Claude Code relay mode activated")
            return msg

        # "ask gemini/claude/the cloud ..." direct escalation
        cloud_match = re.match(
            r"(?:ask|have|use)\s+(?:gemini|claude|the\s+cloud)\s+"
            r"(?:to\s+|about\s+|if\s+|why\s+|how\s+|what\s+|where\s+|when\s+|whether\s+)?",
            text, re.I,
        )
        if not cloud_match:
            cloud_match = re.match(r"(?:gemini|claude)[,:]?\s+", text, re.I)
        if cloud_match:
            question = text[cloud_match.end():].strip()
            if question:
                # _escalate_to_cloud already falls back to gemini_ask MCP tool
                return await self._escalate_to_cloud(question)

        # pin_song: "pin this as X" / "save this as X" / "remember this as X"
        pin_match = re.match(
            r"(?:pin|save|remember)\s+this\s+(?:song\s+)?as\s+(.+)",
            text, re.I,
        )
        if pin_match:
            phrase = pin_match.group(1).strip()
            if phrase:
                try:
                    return str(await self.tool_router.call_tool("pin_song", {"phrase": phrase}))
                except Exception:
                    log.exception("Direct dispatch pin_song failed")
                    return None

        # unpin_song: "unpin X" / "remove pin X" / "remove pin for X"
        unpin_match = re.match(
            r"(?:unpin|remove\s+pin(?:\s+for)?)\s+(.+)",
            text, re.I,
        )
        if unpin_match:
            phrase = unpin_match.group(1).strip()
            if phrase:
                try:
                    return str(await self.tool_router.call_tool("unpin_song", {"phrase": phrase}))
                except Exception:
                    log.exception("Direct dispatch unpin_song failed")
                    return None

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

        for pattern, tool_name in _SIMPLE_MUSIC_CONTROLS:
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

    # Patterns indicating a query that needs Tier 2 (standard) rather than Tier 1 (fast)
    _COMPLEX_QUERY_RE = re.compile(
        r"\b(?:explain|compare|contrast|analyze|summarize|write|draft|compose"
        r"|translate|implement|debug|refactor|difference\s+between"
        r"|pros?\s+and\s+cons?|step\s+by\s+step"
        r"|how\s+does|how\s+do|why\s+does|why\s+do|why\s+is|why\s+are"
        r"|what\s+are\s+the|tell\s+me\s+about|describe\s+(?:how|the|a))\b",
        re.I,
    )

    def _should_use_fast_model(self, text: str, has_tools: bool, has_images: bool = False) -> bool:
        """Determine if Tier 1 (fast model) is appropriate for this query."""
        cfg = get_settings().llm
        if not cfg.fast_model:
            return False
        if has_tools:
            return False
        if has_images:
            return False
        # Long messages likely need deeper reasoning
        if len(text.split()) > 20:
            return False
        if self._COMPLEX_QUERY_RE.search(text):
            return False
        return True

    # Patterns that suggest the LLM is uncertain about its answer
    _LOW_CONFIDENCE_RE = re.compile(
        r"(?:I don'?t know|I'?m not sure|I don'?t have access to"
        r"|I can'?t browse|as an AI|I don'?t have real[- ]time"
        r"|my training data|I cannot access the internet"
        r"|I'?m unable to verify|I don'?t have current)",
        re.I,
    )

    async def _escalate_to_cloud(self, question: str) -> str | None:
        """Execute escalation to the configured cloud provider.

        Temporarily switches the LLM client to the cloud provider,
        runs the query, then restores the original provider.
        Falls back to the gemini_ask MCP tool if no cloud client is available.
        """
        settings = get_settings()
        cloud_cfg = settings.cloud_llm
        provider = cloud_cfg.escalation_provider

        # "auto" = pick first available cloud provider
        if provider == "auto":
            for name in cloud_cfg.providers:
                if name in self.llm._cloud_clients:
                    provider = name
                    break
            else:
                provider = None

        # Try cloud LLM client directly (no shared state mutation)
        if provider and provider in self.llm._cloud_clients:
            client, model, temperature, max_tokens = self.llm._get_provider_params(provider)
            if client:
                try:
                    response = await client.chat.completions.create(
                        model=model,
                        messages=[
                            {"role": "system", "content": settings.llm.system_prompt},
                            {"role": "user", "content": question},
                        ],
                        temperature=temperature,
                        max_tokens=max_tokens,
                    )
                    result = response.choices[0].message.content
                    log.info("Cloud escalation to %s succeeded", provider)
                    return result or None
                except Exception:
                    log.exception("Cloud escalation to %s failed", provider)
                    return None

        # Fallback: gemini_ask MCP tool (legacy path)
        if self.tool_router and settings.gemini.enabled:
            try:
                return str(await self.tool_router.call_tool(
                    "gemini_ask", {"question": question},
                ))
            except Exception:
                log.exception("Escalation gemini_ask fallback failed")

        return None

    async def _maybe_escalate(
        self,
        original_question: str,
        content: str,
        stats: UsageStats,
        t0: float,
    ) -> str:
        """Check LLM response for low confidence and optionally escalate to cloud."""
        if not self._LOW_CONFIDENCE_RE.search(content):
            return content

        settings = get_settings()
        cloud_cfg = settings.cloud_llm

        # Determine escalation mode: prefer cloud_llm setting, fall back to gemini setting
        mode = cloud_cfg.escalation_mode
        if mode == "off":
            # Check legacy gemini escalation as fallback
            gemini_cfg = settings.gemini
            if gemini_cfg.enabled and gemini_cfg.escalation_mode != "off":
                mode = gemini_cfg.escalation_mode
            else:
                return content

        # Check if any cloud provider is available
        has_cloud = any(name in self.llm._cloud_clients for name in cloud_cfg.providers)
        has_gemini_tool = self.tool_router and settings.gemini.enabled
        if not has_cloud and not has_gemini_tool:
            return content

        if mode == "auto":
            result = await self._escalate_to_cloud(original_question)
            if result:
                stats.tier = "cloud"
                # Resolve actual provider name for cost tracking
                esc_provider = cloud_cfg.escalation_provider
                if esc_provider == "auto":
                    for name in cloud_cfg.providers:
                        if name in self.llm._cloud_clients:
                            esc_provider = name
                            break
                stats.provider = esc_provider
                log.info("Auto-escalated to %s: %s", esc_provider, result[:100])
                return result
            return content

        # mode == "ask"
        self._pending_escalation = original_question
        # Build a friendly provider name for the prompt
        provider = cloud_cfg.escalation_provider
        if provider == "auto":
            for name in cloud_cfg.providers:
                if name in self.llm._cloud_clients:
                    provider = name
                    break
        provider_label = provider.replace("-", " ").replace("_", " ").title() if provider != "auto" else "a cloud model"
        return (
            content
            + f"\n\nI'm not fully confident in that answer. "
            f"Would you like me to ask {provider_label}?"
        )

    def _keyword_route(self, text: str, tools: list[dict]) -> list[dict] | None:
        """Fast keyword-based routing for common patterns. Bypasses LLM call.

        Returns:
            list of tools if matched, empty list [] for no-tool patterns,
            or None to fall through to LLM routing.
        """
        # Check keyword routes first — specific tool patterns take priority
        tool_names: set[str] = set()
        for pattern, names in _KEYWORD_ROUTES:
            if pattern.search(text):
                tool_names.update(names)

        # Strip Gemini tools when Gemini is disabled
        if tool_names and not get_settings().gemini.enabled:
            tool_names -= {"gemini_web_search", "gemini_ask"}

        if tool_names:
            available = {t["function"]["name"] for t in tools}
            matched = tool_names & available
            if matched:
                selected = [t for t in tools if t["function"]["name"] in matched]
                log.info("Tool routing (keyword): %s", [t["function"]["name"] for t in selected])
                return selected

        # No keyword route matched — check no-tool patterns for conversational queries
        for pattern in _NO_TOOL_PATTERNS:
            if pattern.search(text):
                log.info("Tool routing (keyword): no tools needed (conversational)")
                return []

        return None

    async def _route_tools(self, text: str, tools: list[dict]) -> list[dict] | None:
        """Select which tools (if any) are needed for this request.

        Uses fast keyword matching for high-confidence patterns.
        Falls back to passing all tools and letting the LLM decide.
        """
        # Fast path: keyword-based routing for common patterns
        keyword_result = self._keyword_route(text, tools)
        if keyword_result is not None:
            return keyword_result or None  # [] → None (no tools)

        # No keyword match — skip tools rather than bloating the prompt with
        # all 55 schemas.  The keyword routes cover every registered tool
        # pattern, so an unmatched query almost certainly doesn't need tools.
        log.info("Tool routing: no keyword match, skipping tools")
        return None

    async def process_utterance_stream(
        self,
        text: str,
        tools: list[dict] | None = None,
        model: str | None = None,
        context: str | None = None,
        images: list[tuple[bytes, str]] | None = None,
    ):
        """Async generator yielding streaming events as dicts.

        Event types emitted:
        - {"type": "token",       "content": "<partial text>"}
        - {"type": "tool_start",  "name": "<tool>", "id": "<call_id>"}
        - {"type": "tool_result", "name": "<tool>", "result": "<...>"}
        - {"type": "done",        "content": "<full text>",
           "tools_used": [...], "usage": {...}}
        - {"type": "error",       "message": "<...>"}
        """
        # Streaming is not supported in Claude Code relay mode.
        # The relay runs a subprocess and returns a complete response.
        if self._claude_relay.active:
            yield {"type": "error", "message": "Claude Code relay mode is active. Use the non-streaming endpoint, or disconnect from Claude Code first."}
            return

        if not model and self._model_override:
            model = self._model_override

        cfg = get_settings().llm
        max_iterations = cfg.max_iterations
        stats = UsageStats()
        t0 = time.monotonic()
        tools_used: list[str] = []

        try:
            # ── Session management (same as process_utterance) ──────────
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
                self._pending_escalation = None

            if self._session is None or not self._session.messages:
                if self._session is None:
                    self._session = ConversationSession()
                try:
                    memory_ctx = await asyncio.wait_for(
                        asyncio.to_thread(self.retriever.retrieve_context, text),
                        timeout=3.0,
                    )
                except asyncio.TimeoutError:
                    log.warning("Memory retrieval timed out")
                    memory_ctx = ""
                self._session.initialize(memory_context=memory_ctx)

            self._last_interaction = t0
            combined = f"{context}\n\n{text}" if context else text
            if images:
                self._session.add_user_multimodal(combined, images)
            else:
                self._session.add_user(combined)
            self._session.trim_to_fit()

            # Auto-compact if context is getting large
            cfg_compact = get_settings().llm
            if (cfg_compact.compact_threshold > 0
                    and self._session.estimate_tokens() > cfg_compact.context_window * cfg_compact.compact_threshold):
                try:
                    summary = await self._session.compact(self.llm, keep_recent=cfg_compact.compact_keep_recent)
                    if summary:
                        log.info("Auto-compacted context (stream): %s", summary[:100])
                except Exception:
                    log.exception("Auto-compact failed (stream), falling back to trim")

            self.retriever.store_conversation_turn(
                "user", text, self._session.session_id,
                scope=self._memory_scope,
            )

            # ── Direct dispatch (fast path) ────────────────────────────
            direct = await self._try_direct_dispatch(text)
            if direct is not None:
                self._session.add_assistant(direct)
                self.retriever.store_conversation_turn(
                    "assistant", direct, self._session.session_id,
                    scope=self._memory_scope,
                )
                stats.elapsed_s = time.monotonic() - t0
                self.last_usage = stats
                yield {
                    "type": "done",
                    "content": direct,
                    "tools_used": [],
                    "usage": stats.to_dict(),
                }
                return

            # ── Tool routing ───────────────────────────────────────────
            if tools:
                tools = await self._route_tools(text, tools)

            # Tier routing: fast model for simple queries, standard for everything else
            effective_model = model
            if not effective_model and self._should_use_fast_model(text, bool(tools), bool(images)):
                effective_model = cfg.fast_model
                stats.tier = "fast"
                log.info("Tier 1 (fast model '%s') for simple streaming query", effective_model)

            # ── Streaming agent loop ───────────────────────────────────
            for iteration in range(max_iterations):
                log.info("Agent stream iteration %d/%d", iteration + 1, max_iterations)

                content_parts: list[str] = []
                # Accumulator for incremental tool_call deltas.
                # Keyed by index, each value: {"id": ..., "name": ..., "arguments": ...}
                pending_tool_calls: dict[int, dict] = {}
                chunk_usage = None

                try:
                    async for chunk in self.llm.chat_stream(
                        messages=self._session.get_messages(),
                        tools=tools if tools else None,
                        model=effective_model,
                    ):
                        # Capture usage from the final chunk
                        if chunk.usage is not None:
                            chunk_usage = chunk

                        if not chunk.choices:
                            continue

                        delta = chunk.choices[0].delta

                        # Content tokens
                        if delta.content:
                            content_parts.append(delta.content)
                            yield {"type": "token", "content": delta.content}

                        # Tool call deltas (incremental)
                        if delta.tool_calls:
                            for tc_delta in delta.tool_calls:
                                idx = tc_delta.index
                                if idx not in pending_tool_calls:
                                    pending_tool_calls[idx] = {
                                        "id": "",
                                        "name": "",
                                        "arguments": "",
                                    }
                                entry = pending_tool_calls[idx]
                                if tc_delta.id:
                                    entry["id"] = tc_delta.id
                                if tc_delta.function:
                                    if tc_delta.function.name:
                                        entry["name"] += tc_delta.function.name
                                    if tc_delta.function.arguments:
                                        entry["arguments"] += tc_delta.function.arguments

                except asyncio.TimeoutError:
                    content = "Sorry, I took too long to respond. Please try again."
                    self._session.add_assistant(content)
                    stats.elapsed_s = time.monotonic() - t0
                    stats.timed_out = True
                    self.last_usage = stats
                    await self._record_usage(stats)
                    yield {"type": "error", "message": content}
                    return
                except RuntimeError as e:
                    if "LLM providers failed" in str(e):
                        content = "Sorry, I couldn't reach any LLM backend. Please check your configuration."
                        self._session.add_assistant(content)
                        stats.elapsed_s = time.monotonic() - t0
                        stats.timed_out = True
                        self.last_usage = stats
                        await self._record_usage(stats)
                        yield {"type": "error", "message": content}
                        return
                    raise

                # Accumulate usage from the final chunk
                if chunk_usage is not None:
                    stats.accumulate(chunk_usage)
                stats.provider = self.llm.last_serving_provider

                # ── Stream ended: decide what happened ─────────────────
                full_content = "".join(content_parts)

                if not pending_tool_calls:
                    # Content-only response — we're done
                    full_content = self._clean_response(full_content)
                    # If cleaning stripped everything, nudge and retry
                    if not full_content and iteration < max_iterations - 1:
                        log.warning("Empty stream response after cleaning hallucinated markup, retrying")
                        self._session.add_assistant("")
                        self._session.add_user("Please answer the question directly.")
                        continue
                    full_content = await self._maybe_escalate(
                        text, full_content, stats, t0
                    )
                    self._session.add_assistant(full_content)
                    self.retriever.store_conversation_turn(
                        "assistant", full_content, self._session.session_id,
                        scope=self._memory_scope,
                    )
                    stats.elapsed_s = time.monotonic() - t0
                    self.last_usage = stats
                    await self._record_usage(stats)
                    yield {
                        "type": "done",
                        "content": full_content,
                        "tools_used": sorted(tools_used),
                        "usage": stats.to_dict(),
                    }
                    return

                # ── Process accumulated tool calls ─────────────────────
                # Build the assistant message with tool_calls for session
                assembled_tool_calls = []
                for idx in sorted(pending_tool_calls):
                    tc = pending_tool_calls[idx]
                    assembled_tool_calls.append({
                        "id": tc["id"],
                        "type": "function",
                        "function": {
                            "name": tc["name"],
                            "arguments": tc["arguments"],
                        },
                    })

                assistant_msg = {
                    "role": "assistant",
                    "content": full_content or None,
                    "tool_calls": assembled_tool_calls,
                }
                self._session.add_tool_call(assistant_msg)

                log.info(
                    "Agent stream requesting %d tool call(s)",
                    len(assembled_tool_calls),
                )

                for tc in assembled_tool_calls:
                    name = tc["function"]["name"]
                    args_str = tc["function"]["arguments"]
                    try:
                        args = json.loads(args_str) if args_str else {}
                    except json.JSONDecodeError as e:
                        log.warning("Invalid JSON in tool args for %s: %s", name, e)
                        result_str = f"Error: malformed tool arguments (invalid JSON): {args_str[:200]}"
                        yield {"type": "tool_start", "name": name, "id": tc["id"]}
                        self._session.add_tool_result(tc["id"], result_str)
                        yield {
                            "type": "tool_result",
                            "name": name,
                            "result": result_str[:500],
                        }
                        continue

                    log.info("Tool call: %s(%s)", name, json.dumps(args)[:200])
                    yield {"type": "tool_start", "name": name, "id": tc["id"]}

                    if name not in tools_used:
                        tools_used.append(name)

                    if self.tool_router:
                        try:
                            result = await self.tool_router.call_tool(name, args)
                        except Exception as e:
                            result = f"Error calling tool '{name}': {e}"
                            log.exception("Tool call failed: %s", name)
                    else:
                        result = (
                            f"Tool '{name}' is not available "
                            "(no MCP router configured)"
                        )

                    result_str = (
                        str(result) if not isinstance(result, str) else result
                    )
                    self._session.add_tool_result(tc["id"], result_str)
                    log.info("Tool result for %s: %s", name, result_str[:200])
                    yield {
                        "type": "tool_result",
                        "name": name,
                        "result": result_str[:500],
                    }

            # ── Exhausted iterations — final forced response (non-streaming) ──
            log.warning(
                "Agent stream hit max iterations (%d), forcing final response",
                max_iterations,
            )
            response = await self.llm.chat(
                messages=self._session.get_messages(), model=effective_model
            )
            stats.accumulate(response)
            stats.provider = self.llm.last_serving_provider
            content = self._clean_response(
                response.choices[0].message.content
                or "I wasn't able to complete that request."
            )
            self._session.add_assistant(content)
            self.retriever.store_conversation_turn(
                "assistant", content, self._session.session_id,
                scope=self._memory_scope,
            )
            stats.elapsed_s = time.monotonic() - t0
            self.last_usage = stats
            await self._record_usage(stats)
            # Emit the forced content as tokens so the client sees it
            yield {"type": "token", "content": content}
            yield {
                "type": "done",
                "content": content,
                "tools_used": sorted(tools_used),
                "usage": stats.to_dict(),
            }

        except Exception as exc:
            log.exception("Streaming agent error")
            yield {"type": "error", "message": str(exc)}

    async def _record_usage(self, stats: UsageStats) -> None:
        """Record accumulated usage stats to the usage tracker, if available."""
        if self.usage_tracker and stats.total_tokens > 0:
            await self.usage_tracker.record(
                stats.prompt_tokens, stats.completion_tokens, stats.total_tokens,
                provider=stats.provider, tier=stats.tier,
            )

    async def extract_facts(self) -> list[str]:
        """Run post-conversation fact extraction."""
        if self._session is None:
            return []
        text = self._session.get_user_assistant_text()
        if not text:
            return []
        scope = getattr(self, "_memory_scope", None)
        return await self.retriever.extract_and_store_facts(text, self.llm, scope=scope)

    def new_session(self) -> None:
        """Start a fresh conversation session."""
        self._session = None
        self._pending_escalation = None
