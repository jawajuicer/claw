"""Adapter wrapping Anthropic SDK to present an OpenAI-compatible interface.

The rest of the codebase uses the OpenAI SDK format (messages, tools, responses).
This adapter converts between OpenAI and Anthropic formats so the LLMClient can
route to Claude without any changes to the agent or conversation code.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field

log = logging.getLogger(__name__)


# ── Lightweight response objects matching OpenAI's interface ──────────────────


@dataclass
class _Usage:
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0


@dataclass
class _Function:
    name: str = ""
    arguments: str = ""


@dataclass
class _ToolCall:
    id: str = ""
    type: str = "function"
    function: _Function = field(default_factory=_Function)


@dataclass
class _Message:
    role: str = "assistant"
    content: str | None = None
    tool_calls: list[_ToolCall] | None = None

    def model_dump(self) -> dict:
        result: dict = {"role": self.role, "content": self.content}
        if self.tool_calls:
            result["tool_calls"] = [
                {
                    "id": tc.id,
                    "type": tc.type,
                    "function": {
                        "name": tc.function.name,
                        "arguments": tc.function.arguments,
                    },
                }
                for tc in self.tool_calls
            ]
        return result


@dataclass
class _Choice:
    message: _Message = field(default_factory=_Message)
    finish_reason: str | None = None
    index: int = 0


@dataclass
class _ChatCompletion:
    choices: list[_Choice] = field(default_factory=list)
    usage: _Usage = field(default_factory=_Usage)


# ── Streaming response objects ────────────────────────────────────────────────


@dataclass
class _FunctionDelta:
    name: str | None = None
    arguments: str | None = None


@dataclass
class _ToolCallDelta:
    index: int = 0
    id: str | None = None
    type: str = "function"
    function: _FunctionDelta | None = None


@dataclass
class _Delta:
    content: str | None = None
    tool_calls: list[_ToolCallDelta] | None = None


@dataclass
class _StreamChoice:
    delta: _Delta = field(default_factory=_Delta)
    index: int = 0


@dataclass
class _ChatCompletionChunk:
    choices: list[_StreamChoice] = field(default_factory=list)
    usage: _Usage | None = None


# ── Format conversion helpers ─────────────────────────────────────────────────


def _convert_messages(messages: list[dict]) -> tuple[str | None, list[dict]]:
    """Convert OpenAI message format to Anthropic format.

    Returns (system_prompt, anthropic_messages).
    """
    system_prompt = None
    anthropic_messages: list[dict] = []

    for msg in messages:
        role = msg.get("role", "")
        content = msg.get("content", "")

        if role == "system":
            system_prompt = content
            continue

        if role == "user":
            anthropic_messages.append({"role": "user", "content": content})

        elif role == "assistant":
            tool_calls = msg.get("tool_calls", [])
            if tool_calls:
                content_blocks: list[dict] = []
                if content:
                    content_blocks.append({"type": "text", "text": content})
                for tc in tool_calls:
                    func = tc.get("function", {})
                    args_str = func.get("arguments", "{}")
                    try:
                        args = json.loads(args_str) if isinstance(args_str, str) else args_str
                    except json.JSONDecodeError:
                        args = {}
                    content_blocks.append({
                        "type": "tool_use",
                        "id": tc.get("id", ""),
                        "name": func.get("name", ""),
                        "input": args,
                    })
                anthropic_messages.append({"role": "assistant", "content": content_blocks})
            else:
                anthropic_messages.append({"role": "assistant", "content": content or ""})

        elif role == "tool":
            # Anthropic: tool results are user messages with tool_result content blocks.
            # Consecutive tool results must be grouped into a single user message.
            tool_result = {
                "type": "tool_result",
                "tool_use_id": msg.get("tool_call_id", ""),
                "content": content or "",
            }
            if (
                anthropic_messages
                and anthropic_messages[-1]["role"] == "user"
                and isinstance(anthropic_messages[-1]["content"], list)
                and anthropic_messages[-1]["content"]
                and anthropic_messages[-1]["content"][0].get("type") == "tool_result"
            ):
                anthropic_messages[-1]["content"].append(tool_result)
            else:
                anthropic_messages.append({"role": "user", "content": [tool_result]})

    return system_prompt, anthropic_messages


def _convert_tools(tools: list[dict] | None) -> list[dict] | None:
    """Convert OpenAI tool definitions to Anthropic format."""
    if not tools:
        return None
    anthropic_tools = []
    for tool in tools:
        func = tool.get("function", {})
        anthropic_tools.append({
            "name": func.get("name", ""),
            "description": func.get("description", ""),
            "input_schema": func.get("parameters", {"type": "object", "properties": {}}),
        })
    return anthropic_tools


def _convert_response(response) -> _ChatCompletion:
    """Convert Anthropic Message to OpenAI ChatCompletion format."""
    content_text = ""
    tool_calls: list[_ToolCall] = []

    for block in response.content or []:
        if block.type == "thinking":
            continue
        elif block.type == "text":
            content_text += block.text
        elif block.type == "tool_use":
            tool_calls.append(
                _ToolCall(
                    id=block.id,
                    function=_Function(name=block.name, arguments=json.dumps(block.input)),
                )
            )

    usage = _Usage(
        prompt_tokens=response.usage.input_tokens,
        completion_tokens=response.usage.output_tokens,
        total_tokens=response.usage.input_tokens + response.usage.output_tokens,
    )

    return _ChatCompletion(
        choices=[
            _Choice(
                message=_Message(
                    content=content_text or None,
                    tool_calls=tool_calls if tool_calls else None,
                ),
                finish_reason=response.stop_reason,
            )
        ],
        usage=usage,
    )


# ── Stream adapter ────────────────────────────────────────────────────────────


class _StreamAdapter:
    """Wraps an Anthropic raw event stream and yields OpenAI-compatible chunks."""

    def __init__(self, raw_stream):
        self._raw_stream = raw_stream

    def __aiter__(self):
        return self._iterate()

    async def _iterate(self):
        tool_index_map: dict[int, int] = {}  # content_block_index → tool_call_index
        skip_blocks: set[int] = set()
        next_tool_index = 0
        input_tokens = 0
        output_tokens = 0

        async for event in self._raw_stream:
            etype = event.type

            if etype == "message_start":
                if hasattr(event.message, "usage") and event.message.usage:
                    input_tokens = event.message.usage.input_tokens

            elif etype == "content_block_start":
                block = event.content_block
                idx = event.index

                if block.type == "thinking":
                    skip_blocks.add(idx)
                elif block.type == "tool_use":
                    tool_idx = next_tool_index
                    tool_index_map[idx] = tool_idx
                    next_tool_index += 1
                    yield _ChatCompletionChunk(
                        choices=[
                            _StreamChoice(
                                delta=_Delta(
                                    tool_calls=[
                                        _ToolCallDelta(
                                            index=tool_idx,
                                            id=block.id,
                                            function=_FunctionDelta(
                                                name=block.name, arguments=""
                                            ),
                                        )
                                    ]
                                )
                            )
                        ]
                    )
                # text blocks: content arrives via content_block_delta

            elif etype == "content_block_delta":
                idx = event.index
                if idx in skip_blocks:
                    continue

                delta = event.delta
                if delta.type == "text_delta":
                    yield _ChatCompletionChunk(
                        choices=[_StreamChoice(delta=_Delta(content=delta.text))]
                    )
                elif delta.type == "input_json_delta":
                    tool_idx = tool_index_map.get(idx)
                    if tool_idx is not None:
                        yield _ChatCompletionChunk(
                            choices=[
                                _StreamChoice(
                                    delta=_Delta(
                                        tool_calls=[
                                            _ToolCallDelta(
                                                index=tool_idx,
                                                function=_FunctionDelta(
                                                    arguments=delta.partial_json
                                                ),
                                            )
                                        ]
                                    )
                                )
                            ]
                        )

            elif etype == "message_delta":
                if hasattr(event, "usage") and event.usage:
                    output_tokens = event.usage.output_tokens

        # Final chunk with combined usage
        if input_tokens or output_tokens:
            yield _ChatCompletionChunk(
                choices=[],
                usage=_Usage(
                    prompt_tokens=input_tokens,
                    completion_tokens=output_tokens,
                    total_tokens=input_tokens + output_tokens,
                ),
            )


# ── Namespace objects to match OpenAI's client.chat.completions.create() ─────


class _CompletionsNamespace:
    def __init__(self, client):
        self._client = client

    async def create(self, **kwargs):
        stream = kwargs.pop("stream", False)
        kwargs.pop("stream_options", None)  # Anthropic doesn't use this

        messages = kwargs.pop("messages", [])
        tools_openai = kwargs.pop("tools", None)
        model = kwargs.pop("model", "claude-sonnet-4-6")
        max_tokens = kwargs.pop("max_tokens", 4096)
        temperature = kwargs.pop("temperature", 0.7)
        extra_body = kwargs.pop("extra_body", {}) or {}

        system_prompt, anthropic_messages = _convert_messages(messages)
        anthropic_tools = _convert_tools(tools_openai)

        api_kwargs: dict = {
            "model": model,
            "max_tokens": max_tokens,
            "messages": anthropic_messages,
            "temperature": temperature,
        }

        if system_prompt:
            api_kwargs["system"] = system_prompt

        if anthropic_tools:
            api_kwargs["tools"] = anthropic_tools

        # Handle extended thinking from extra_body
        thinking = extra_body.pop("thinking", None)
        if thinking:
            api_kwargs["thinking"] = thinking
            api_kwargs["temperature"] = 1.0  # required when thinking is enabled
            budget = thinking.get("budget_tokens", 0)
            api_kwargs["max_tokens"] = max_tokens + budget

        if stream:
            raw_stream = await self._client.messages.create(**api_kwargs, stream=True)
            return _StreamAdapter(raw_stream)

        response = await self._client.messages.create(**api_kwargs)
        return _convert_response(response)


class _ChatNamespace:
    def __init__(self, client):
        self.completions = _CompletionsNamespace(client)


# ── Main adapter class ────────────────────────────────────────────────────────


class ClaudeAdapter:
    """Wraps AsyncAnthropic to provide an OpenAI-compatible interface.

    Usage::

        adapter = ClaudeAdapter(api_key="sk-ant-...")
        response = await adapter.chat.completions.create(
            model="claude-sonnet-4-6",
            messages=[{"role": "user", "content": "Hello"}],
            max_tokens=1024,
        )
        print(response.choices[0].message.content)
    """

    def __init__(self, api_key: str, timeout: int = 60):
        from anthropic import AsyncAnthropic

        self._client = AsyncAnthropic(api_key=api_key, timeout=timeout)
        self.chat = _ChatNamespace(self._client)
        self.base_url = "https://api.anthropic.com"
