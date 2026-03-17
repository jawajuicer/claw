"""Slash command dispatcher for chat and CLI."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from claw.config import get_settings

if TYPE_CHECKING:
    from claw.agent_core.agent import Agent

log = logging.getLogger(__name__)

# Command registry: name -> (handler, description)
_COMMANDS: dict[str, tuple] = {}


def _register(name: str, description: str):
    def decorator(fn):
        _COMMANDS[name] = (fn, description)
        return fn
    return decorator


async def dispatch_command(text: str, agent: "Agent") -> dict | None:
    """Dispatch a slash command. Returns response dict or None if not a command."""
    text = text.strip()
    if not text.startswith("/"):
        return None

    parts = text.split(None, 1)
    cmd = parts[0].lower()
    arg = parts[1].strip() if len(parts) > 1 else ""

    handler_info = _COMMANDS.get(cmd)
    if handler_info is None:
        return None

    handler, _ = handler_info
    try:
        content = await handler(agent, arg)
    except Exception as e:
        log.exception("Command %s failed", cmd)
        content = f"Command failed: {e}"

    return {
        "role": "assistant",
        "content": content,
        "tools_used": [],
        "usage": {},
        "is_command": True,
    }


@_register("/help", "Show available commands")
async def cmd_help(agent: "Agent", arg: str) -> str:
    lines = ["**Available Commands:**"]
    for name, (_, desc) in sorted(_COMMANDS.items()):
        lines.append(f"  `{name}` -- {desc}")
    return "\n".join(lines)


@_register("/status", "Show system status")
async def cmd_status(agent: "Agent", arg: str) -> str:
    cfg = get_settings()
    lines = ["**System Status:**"]

    # LLM
    provider = getattr(agent.llm, 'active_provider', 'local')
    if provider == 'local':
        model = cfg.llm.model
    else:
        cloud_provider = cfg.cloud_llm.providers.get(provider)
        model = cloud_provider.model if cloud_provider else '?'
    lines.append(f"  Provider: {provider}")
    lines.append(f"  Model: {model}")
    lines.append(f"  Thinking: {cfg.llm.thinking}")

    # Session
    if agent.session:
        msg_count = len(agent.session.messages)
        tokens = agent.session.estimate_tokens()
        lines.append(f"  Session: {agent.session.session_id} ({msg_count} msgs, ~{tokens} tokens)")
    else:
        lines.append("  Session: none")

    # Context
    if agent.last_usage:
        pct = round((agent.last_usage.last_prompt_tokens / cfg.llm.context_window) * 100)
        lines.append(f"  Context: {agent.last_usage.last_prompt_tokens}/{cfg.llm.context_window} ({pct}%)")

    return "\n".join(lines)


@_register("/usage", "Show token usage for this session")
async def cmd_usage(agent: "Agent", arg: str) -> str:
    tracker = getattr(agent, 'usage_tracker', None)
    if tracker is None:
        return "Usage tracking is not enabled."

    session = tracker.get_session_summary()
    today = tracker.get_daily_summary()

    lines = ["**Token Usage:**"]
    lines.append(f"  Session: {session.get('total_tokens', 0):,} tokens ({session.get('calls', 0)} calls)")
    lines.append(f"  Today: {today.get('total_tokens', 0):,} tokens ({today.get('calls', 0)} calls)")

    cost = today.get("cost", {}).get("total", 0)
    if cost > 0:
        lines.append(f"  Est. Cost: ${cost:.4f}")
    else:
        lines.append("  Cost: Free (local)")

    return "\n".join(lines)


@_register("/think", "Set thinking level: /think [off|low|medium|high]")
async def cmd_think(agent: "Agent", arg: str) -> str:
    valid = ("off", "low", "medium", "high")
    if not arg:
        level = getattr(agent.llm, 'thinking_level', 'off')
        return f"Current thinking level: **{level}**\nUsage: `/think [off|low|medium|high]`"

    level = arg.lower()
    if level not in valid:
        return f"Invalid level '{arg}'. Choose: {', '.join(valid)}"

    agent.llm.thinking_level = level
    return f"Thinking level set to **{level}** for this session."


@_register("/model", "Switch model: /model [name]")
async def cmd_model(agent: "Agent", arg: str) -> str:
    if not arg:
        cfg = get_settings()
        provider = getattr(agent.llm, 'active_provider', 'local')
        if provider == 'local':
            current = cfg.llm.model
        else:
            cloud_provider = cfg.cloud_llm.providers.get(provider)
            current = cloud_provider.model if cloud_provider else '?'
        return f"Current model: **{current}** (provider: {provider})"
    # Set per-session model override — read by process_utterance / process_utterance_stream
    agent._model_override = arg
    return f"Model set to **{arg}** for this session."


@_register("/compact", "Summarize conversation context to save memory")
async def cmd_compact(agent: "Agent", arg: str) -> str:
    if not agent.session:
        return "No active session to compact."
    try:
        summary = await agent.session.compact(agent.llm)
        tokens = agent.session.estimate_tokens()
        return f"Context compacted (~{tokens} tokens remaining).\n\n**Summary:** {summary}"
    except Exception as e:
        return f"Compact failed: {e}"


@_register("/new", "Start a new conversation")
async def cmd_new(agent: "Agent", arg: str) -> str:
    agent.new_session()
    # Reset session usage
    tracker = getattr(agent, 'usage_tracker', None)
    if tracker:
        await tracker.reset_session()
    return "New conversation started."
