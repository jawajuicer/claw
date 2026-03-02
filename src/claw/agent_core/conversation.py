"""Conversation session and message history management."""

from __future__ import annotations

import uuid
from dataclasses import dataclass, field

from openai.types.chat import ChatCompletionMessageParam

from claw.config import get_settings


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
    lines.append("Always ask for any missing required information (date, time, etc.) before acting.")
    return "\n".join(lines)


@dataclass
class ConversationSession:
    """Manages message history for a single conversation session."""

    session_id: str = field(default_factory=lambda: uuid.uuid4().hex[:12])
    messages: list[ChatCompletionMessageParam] = field(default_factory=list)
    _system_prompt: str = ""

    def initialize(self, memory_context: str = "") -> None:
        """Set up the system prompt with optional memory context."""
        cfg = get_settings().llm
        self._system_prompt = cfg.system_prompt

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
