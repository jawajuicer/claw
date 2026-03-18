"""Conversation session and message history management."""

from __future__ import annotations

import asyncio
import logging
import uuid
from dataclasses import dataclass, field

from openai.types.chat import ChatCompletionMessageParam

from claw.config import get_settings

log = logging.getLogger(__name__)


def _build_account_context() -> str:
    """Generate system prompt section describing configured Google accounts."""
    accounts = get_settings().google_auth.accounts
    if not accounts:
        return ""

    lines = ["## Google Accounts"]
    for label, acct in accounts.items():
        services = []
        if acct.calendar.enabled:
            services.append("Google Calendar")
        if acct.gmail.enabled:
            services.append("Gmail")
        if acct.youtube_music:
            services.append("YouTube Music")
        if not services:
            continue
        email_str = f" ({acct.email})" if acct.email else ""
        lines.append(f"- **{label}**{email_str}: {', '.join(services)}")

    if len(lines) == 1:
        return ""

    lines.append("")
    lines.append('When the user mentions an account (e.g., "work calendar"), use that account.')
    if len(accounts) > 1:
        lines.append("When ambiguous and multiple accounts have the same service, ASK which account.")
    lines.append("When only one account has the needed service, use it automatically.")
    lines.append("Infer missing details from context. Only ask if truly ambiguous (e.g., date for a calendar event with no hint).")
    return "\n".join(lines)


@dataclass
class ConversationSession:
    """Manages message history for a single conversation session."""

    session_id: str = field(default_factory=lambda: uuid.uuid4().hex[:12])
    messages: list[ChatCompletionMessageParam] = field(default_factory=list)
    _base_system_prompt: str = ""
    _system_prompt: str = ""

    def initialize(self, memory_context: str = "") -> None:
        """Set up the system prompt with optional memory context.

        Safe to call multiple times — always rebuilds from the base prompt.
        """
        cfg = get_settings().llm
        self._base_system_prompt = cfg.system_prompt
        self._system_prompt = self._base_system_prompt

        account_ctx = _build_account_context()
        if account_ctx:
            self._system_prompt += f"\n\n{account_ctx}"

        if memory_context:
            self._system_prompt += f"\n\n{memory_context}"
        self.messages = [{"role": "system", "content": self._system_prompt}]

    def add_user(self, content: str) -> None:
        self.messages.append({"role": "user", "content": content})

    def add_assistant(self, content: str) -> None:
        self.messages.append({"role": "assistant", "content": content})

    def add_tool_call(self, message) -> None:
        """Add the assistant message that contains tool_calls."""
        self.messages.append(message)

    def add_tool_result(self, tool_call_id: str, content: str) -> None:
        self.messages.append({
            "role": "tool",
            "tool_call_id": tool_call_id,
            "content": content,
        })

    def get_messages(self) -> list[ChatCompletionMessageParam]:
        return list(self.messages)

    def get_user_assistant_text(self) -> str:
        """Return conversation text (user + assistant turns only) for fact extraction."""
        lines = []
        for msg in self.messages:
            if msg.get("role") in ("user", "assistant") and msg.get("content"):
                lines.append(f"{msg['role']}: {msg['content']}")
        return "\n".join(lines)

    def trim_to_fit(self, max_messages: int = 40) -> None:
        """Keep the system prompt + last N messages to avoid context overflow.

        Avoids splitting tool_call/tool_result pairs which would cause API errors.
        """
        if len(self.messages) <= max_messages + 1:
            return
        system = self.messages[0]
        tail = self.messages[-(max_messages):]
        # Ensure we don't start with orphaned tool results
        while tail and tail[0].get("role") == "tool":
            tail = tail[1:]
        self.messages = [system] + tail

    def estimate_tokens(self) -> int:
        """Rough token estimate (chars / 4), including tool call arguments."""
        total = 0
        for m in self.messages:
            content = m.get("content", "") or ""
            total += len(str(content)) // 4
            for tc in m.get("tool_calls", []):
                func = tc.get("function", {})
                total += len(func.get("name", "")) // 4
                total += len(str(func.get("arguments", ""))) // 4
        return total

    async def compact(self, llm_client, keep_recent: int = 6) -> str:
        """Summarize older messages into a compact context block.

        Keeps system prompt + last keep_recent messages verbatim.
        Summarizes everything in between using the LLM.
        Returns the summary text.
        """
        keep_recent = max(2, keep_recent)
        # Need at least: system + 1 old message + keep_recent recent messages
        if len(self.messages) < 2 + keep_recent:
            return ""

        system = self.messages[0]
        # Partition: [system] [old...] [recent...]
        # Adjust keep_recent backward to avoid splitting tool_call/tool_result pairs
        recent_start = len(self.messages) - keep_recent
        # Walk backward to find a safe split point: don't start recent with a
        # tool message (orphaned result) and don't split an assistant's tool_calls
        # from their tool results.
        while recent_start > 1:
            msg_at_split = self.messages[recent_start]
            if msg_at_split.get("role") == "tool":
                # This is a tool result — its tool_call assistant message is before it
                recent_start -= 1
                continue
            # Also check: if the message just before the split is an assistant
            # message with tool_calls, its tool results follow — keep them together
            prev = self.messages[recent_start - 1]
            if prev.get("role") == "assistant" and prev.get("tool_calls"):
                recent_start -= 1
                continue
            break

        old_messages = self.messages[1:recent_start]
        recent_messages = self.messages[recent_start:]

        if not old_messages:
            return ""

        # Format old messages as readable text for the summarization prompt
        formatted_lines = []
        for msg in old_messages:
            role = msg.get("role", "unknown")
            content = msg.get("content", "")
            if role == "tool":
                tool_id = msg.get("tool_call_id", "")
                formatted_lines.append(f"Tool result ({tool_id}): {content}")
            elif role == "assistant" and msg.get("tool_calls"):
                # Assistant message requesting tool calls
                calls = msg.get("tool_calls", [])
                call_strs = []
                for tc in calls:
                    if isinstance(tc, dict):
                        fn = tc.get("function", {})
                        call_strs.append(f"{fn.get('name', '?')}({fn.get('arguments', '')})")
                    else:
                        call_strs.append(str(tc))
                formatted_lines.append(f"Assistant [tool calls]: {', '.join(call_strs)}")
                if content:
                    formatted_lines.append(f"Assistant: {content}")
            else:
                formatted_lines.append(f"{role.capitalize()}: {content}")

        conversation_text = "\n".join(formatted_lines)

        prompt = (
            "Summarize this conversation concisely, preserving key facts, decisions, "
            "tool results, and any important context the assistant needs to continue "
            "the conversation. Keep it under 200 words.\n\n"
            f"{conversation_text}"
        )

        summary = await asyncio.wait_for(llm_client.chat_simple(prompt), timeout=30.0)
        summary = summary.strip()

        if not summary:
            log.warning("Compact produced empty summary, skipping")
            return ""

        summary_message: ChatCompletionMessageParam = {
            "role": "assistant",
            "content": f"[Conversation Summary] {summary}",
        }

        self.messages = [system, summary_message] + recent_messages
        log.info(
            "Compacted %d old messages into summary (%d chars), keeping %d recent",
            len(old_messages), len(summary), len(recent_messages),
        )
        return summary
