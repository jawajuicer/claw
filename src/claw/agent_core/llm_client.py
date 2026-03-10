"""LLM client via OpenAI-compatible API."""

from __future__ import annotations

import asyncio
import logging

from openai import APITimeoutError, AsyncOpenAI
from openai.types.chat import ChatCompletionMessageParam

from claw.config import get_settings, on_reload

log = logging.getLogger(__name__)


class LLMClient:
    """Async LLM client wrapping an OpenAI-compatible endpoint."""

    def __init__(self) -> None:
        cfg = get_settings().llm
        self._client = AsyncOpenAI(
            base_url=cfg.base_url,
            api_key=cfg.api_key,
            timeout=cfg.timeout,
        )
        self._model = cfg.model
        self._temperature = cfg.temperature
        self._max_tokens = cfg.max_tokens
        self._timeout = cfg.timeout
        self._lock = asyncio.Lock()
        self._simple_lock = asyncio.Lock()  # separate lock for chat_simple (routing etc.)
        on_reload(self._on_config_reload)

    @property
    def busy(self) -> bool:
        """True if an LLM call is currently in progress."""
        return self._lock.locked()

    async def chat(
        self,
        messages: list[ChatCompletionMessageParam],
        tools: list[dict] | None = None,
        model: str | None = None,
    ):
        """Send a chat completion request.

        Args:
            messages: Chat messages.
            tools: OpenAI-format tool definitions.
            model: Override the default model for this request.

        Returns the full ChatCompletion response object.
        """
        effective_model = model or self._model
        kwargs = {
            "model": effective_model,
            "messages": messages,
            "temperature": self._temperature,
            "max_tokens": self._max_tokens,
        }
        if tools:
            kwargs["tools"] = tools

        log.debug("LLM request: model=%s, messages=%d, tools=%d",
                   effective_model, len(messages), len(tools or []))

        async with self._lock:
            try:
                response = await self._client.chat.completions.create(**kwargs)
            except APITimeoutError:
                log.error("LLM call timed out after %ds", self._timeout)
                raise asyncio.TimeoutError(f"LLM call timed out after {self._timeout}s")
        return response

    async def chat_simple(self, prompt: str) -> str:
        """Simple single-turn chat returning just the text content.

        Uses its own lock so it won't block behind a long-running chat() call.
        """
        async with self._simple_lock:
            try:
                response = await self._client.chat.completions.create(
                    model=self._model,
                    messages=[{"role": "user", "content": prompt}],
                    temperature=self._temperature,
                    max_tokens=256,
                )
            except APITimeoutError:
                raise asyncio.TimeoutError("chat_simple timed out")
        return response.choices[0].message.content or ""

    def _on_config_reload(self, settings) -> None:
        cfg = settings.llm
        changed = False
        if cfg.base_url != str(self._client.base_url).rstrip("/"):
            self._client = AsyncOpenAI(
                base_url=cfg.base_url, api_key=cfg.api_key, timeout=cfg.timeout,
            )
            changed = True
        self._model = cfg.model
        self._temperature = cfg.temperature
        self._max_tokens = cfg.max_tokens
        self._timeout = cfg.timeout
        if changed:
            log.info("LLM client reconfigured: %s @ %s", self._model, cfg.base_url)
