"""Adapter that uses the ``claude`` CLI (Claude Code) as an LLM backend.

No API key needed — piggybacks on Claude Code's existing OAuth session.
Invokes ``claude -p --output-format json`` as a subprocess per request.
"""

from __future__ import annotations

import asyncio
import json
import logging
import re
import shutil

from claw.agent_core.claude_adapter import (
    _ChatCompletion,
    _ChatCompletionChunk,
    _Choice,
    _Delta,
    _Function,
    _FunctionDelta,
    _Message,
    _StreamChoice,
    _ToolCall,
    _ToolCallDelta,
    _Usage,
)

log = logging.getLogger(__name__)

_TOOL_CALL_RE = re.compile(r"<tool_call>\s*(\{.*?\})\s*</tool_call>", re.DOTALL)


# ── Prompt building ───────────────────────────────────────────────────────────


def _format_tools_section(tools: list[dict]) -> str:
    """Build the tool instruction block injected into the prompt."""
    lines = [
        "\n## Available Tools",
        "To call a tool, respond with ONLY tool call blocks (no other text):",
        "",
        '<tool_call>{"name": "tool_name", "arguments": {"param": "value"}}</tool_call>',
        "",
        "You may include multiple <tool_call> blocks.",
        "If you do NOT need tools, respond normally with plain text.",
        "",
    ]
    for tool in tools:
        func = tool.get("function", {})
        name = func.get("name", "?")
        desc = func.get("description", "")
        params = func.get("parameters", {})
        props = params.get("properties", {})
        required = set(params.get("required", []))

        param_parts: list[str] = []
        for pname, pschema in props.items():
            req = " [required]" if pname in required else ""
            ptype = pschema.get("type", "any")
            pdesc = pschema.get("description", "")
            param_parts.append(f"    - {pname} ({ptype}{req}): {pdesc}")

        lines.append(f"### {name}")
        if desc:
            lines.append(desc)
        if param_parts:
            lines.append("Parameters:")
            lines.extend(param_parts)
        lines.append("")

    return "\n".join(lines)


def _sanitize(text: str) -> str:
    """Escape tool_call tags in untrusted content to prevent injection."""
    return text.replace("<tool_call>", "&lt;tool_call&gt;").replace(
        "</tool_call>", "&lt;/tool_call&gt;"
    )


def _build_prompt(messages: list[dict], tools: list[dict] | None) -> str:
    """Convert OpenAI-format messages + tools into a single text prompt."""
    parts: list[str] = []

    # System prompt first (trusted — not sanitized)
    for msg in messages:
        if msg.get("role") == "system":
            parts.append(msg.get("content", ""))
            break

    # Tool section early so Claude sees it before the conversation
    if tools:
        parts.append(_format_tools_section(tools))

    # Conversation history
    parts.append("\n## Conversation")
    for msg in messages:
        role = msg.get("role", "")
        content = msg.get("content", "")

        if role == "system":
            continue
        elif role == "user":
            # User input is untrusted — sanitize to prevent tag injection
            parts.append(f"\nUser: {_sanitize(content)}")
        elif role == "assistant":
            tool_calls = msg.get("tool_calls", [])
            if tool_calls:
                calls = []
                for tc in tool_calls:
                    func = tc.get("function", {})
                    calls.append(f'{func.get("name", "?")}({func.get("arguments", "{}")})')
                parts.append(f"\nAssistant [called tools]: {', '.join(calls)}")
                if content:
                    parts.append(f"Assistant: {content}")
            else:
                parts.append(f"\nAssistant: {content}")
        elif role == "tool":
            # Tool results are untrusted — sanitize
            tid = msg.get("tool_call_id", "")
            parts.append(f"\nTool result ({tid}): {_sanitize(content)}")

    parts.append("\nAssistant:")
    return "\n".join(parts)


# ── Response parsing ──────────────────────────────────────────────────────────


def _parse_tool_calls(text: str, valid_names: set[str]) -> list[_ToolCall] | None:
    """Extract ``<tool_call>`` blocks from Claude's response text.

    Only accepts tool calls whose name is in *valid_names* (the tools actually
    offered in this request).  This prevents injection of arbitrary tool calls
    via user input or malicious tool results.
    """
    matches = _TOOL_CALL_RE.findall(text)
    if not matches:
        return None

    tool_calls: list[_ToolCall] = []
    for i, raw in enumerate(matches):
        try:
            data = json.loads(raw)
            name = data.get("name", "")
            if name not in valid_names:
                log.warning("Rejected tool call with unknown name: %s", name)
                continue
            tool_calls.append(
                _ToolCall(
                    id=f"cli_{i}",
                    function=_Function(
                        name=name,
                        arguments=json.dumps(data.get("arguments", {})),
                    ),
                )
            )
        except json.JSONDecodeError:
            log.warning("Bad tool call JSON: %s", raw[:200])

    return tool_calls or None


def _strip_tool_calls(text: str) -> str:
    """Remove ``<tool_call>`` blocks and return remaining text."""
    return _TOOL_CALL_RE.sub("", text).strip()


# ── Fake stream wrapper ──────────────────────────────────────────────────────


class _FakeStream:
    """Yields a completed response as two stream-compatible chunks."""

    def __init__(self, response: _ChatCompletion):
        self._response = response

    def __aiter__(self):
        return self._gen()

    async def _gen(self):
        msg = self._response.choices[0].message
        tc_deltas = None
        if msg.tool_calls:
            tc_deltas = [
                _ToolCallDelta(
                    index=i,
                    id=tc.id,
                    function=_FunctionDelta(
                        name=tc.function.name, arguments=tc.function.arguments
                    ),
                )
                for i, tc in enumerate(msg.tool_calls)
            ]
        # Chunk 1: content + tool calls
        yield _ChatCompletionChunk(
            choices=[_StreamChoice(delta=_Delta(content=msg.content, tool_calls=tc_deltas))]
        )
        # Chunk 2: usage (empty choices — matches OpenAI streaming pattern)
        yield _ChatCompletionChunk(choices=[], usage=self._response.usage)


# ── Namespace objects (match OpenAI client.chat.completions.create) ───────────


class _CLICompletionsNamespace:
    def __init__(self, adapter: ClaudeCLIAdapter):
        self._adapter = adapter

    async def create(self, **kwargs):
        stream = kwargs.pop("stream", False)
        kwargs.pop("stream_options", None)
        kwargs.pop("extra_body", None)
        kwargs.pop("temperature", None)

        messages = kwargs.pop("messages", [])
        tools = kwargs.pop("tools", None)
        model = kwargs.pop("model", None)
        kwargs.pop("max_tokens", None)  # handled by Claude Code internally

        prompt = _build_prompt(messages, tools)

        cmd = ["claude", "-p", "--output-format", "json"]
        if model:
            cmd.extend(["--model", model])

        try:
            proc = await asyncio.create_subprocess_exec(
                *cmd,
                stdin=asyncio.subprocess.PIPE,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
            stdout, stderr = await asyncio.wait_for(
                proc.communicate(input=prompt.encode()),
                timeout=self._adapter.timeout,
            )
        except asyncio.TimeoutError:
            if proc.returncode is None:
                proc.kill()
            raise
        except FileNotFoundError:
            raise RuntimeError("'claude' CLI not found — is Claude Code installed?")

        if proc.returncode != 0:
            err = stderr.decode(errors="replace").strip()
            log.error("claude CLI stderr: %s", err[:500])
            raise RuntimeError(f"claude CLI error (exit {proc.returncode})")

        raw = stdout.decode(errors="replace").strip()
        try:
            data = json.loads(raw)
            text = data.get("result", raw)
        except json.JSONDecodeError:
            text = raw

        # Extract tool calls if tools were offered; validate names against allowlist
        valid_names = {
            t.get("function", {}).get("name") for t in (tools or [])
        } if tools else set()
        tool_calls = _parse_tool_calls(text, valid_names) if tools else None
        if tool_calls:
            content = _strip_tool_calls(text) or None
        else:
            content = text

        # Rough token estimates (chars / 4)
        prompt_toks = len(prompt) // 4
        comp_toks = len(text) // 4

        response = _ChatCompletion(
            choices=[_Choice(message=_Message(content=content, tool_calls=tool_calls))],
            usage=_Usage(prompt_toks, comp_toks, prompt_toks + comp_toks),
        )
        return _FakeStream(response) if stream else response


class _CLIChatNamespace:
    def __init__(self, adapter: ClaudeCLIAdapter):
        self.completions = _CLICompletionsNamespace(adapter)


# ── Main adapter ──────────────────────────────────────────────────────────────


class ClaudeCLIAdapter:
    """Uses the ``claude`` CLI (Claude Code) as an LLM backend.

    No API key needed — piggybacks on Claude Code's existing OAuth session.

    Usage::

        adapter = ClaudeCLIAdapter()
        response = await adapter.chat.completions.create(
            model="sonnet",
            messages=[{"role": "user", "content": "Hello"}],
        )
        print(response.choices[0].message.content)
    """

    def __init__(self, timeout: int = 120):
        if not shutil.which("claude"):
            log.warning("'claude' not found in PATH — claude-cli provider will fail at runtime")
        self.timeout = timeout
        self.chat = _CLIChatNamespace(self)
        self.base_url = "claude-cli"
