"""Ollama LLM client via OpenAI-compatible API."""

from __future__ import annotations

import logging

from openai import AsyncOpenAI
from openai.types.chat import ChatCompletionMessageParam

from claw.config import get_settings, on_reload

log = logging.getLogger(__name__)


class LLMClient:
    """Async LLM client wrapping Ollama's OpenAI-compatible endpoint."""

    def __init__(self) -> None:
        cfg = get_settings().llm
        self._client = AsyncOpenAI(
            base_url=cfg.base_url,
            api_key=cfg.api_key,
        )
        self._model = cfg.model
        self._temperature = cfg.temperature
        self._max_tokens = cfg.max_tokens
        on_reload(self._on_config_reload)

    async def chat(
        self,
        messages: list[ChatCompletionMessageParam],
        tools: list[dict] | None = None,
    ):
        """Send a chat completion request.

        Returns the full ChatCompletion response object.
        """
        kwargs = {
            "model": self._model,
            "messages": messages,
            "temperature": self._temperature,
            "max_tokens": self._max_tokens,
        }
        if tools:
            kwargs["tools"] = tools

        log.debug("LLM request: model=%s, messages=%d, tools=%d",
                   self._model, len(messages), len(tools or []))

        response = await self._client.chat.completions.create(**kwargs)
        return response

    async def chat_simple(self, prompt: str) -> str:
        """Simple single-turn chat returning just the text content."""
        response = await self.chat(
            messages=[{"role": "user", "content": prompt}],
        )
        return response.choices[0].message.content or ""

    def _on_config_reload(self, settings) -> None:
        cfg = settings.llm
        changed = False
        if cfg.base_url != str(self._client.base_url).rstrip("/"):
            self._client = AsyncOpenAI(base_url=cfg.base_url, api_key=cfg.api_key)
            changed = True
        self._model = cfg.model
        self._temperature = cfg.temperature
        self._max_tokens = cfg.max_tokens
        if changed:
            log.info("LLM client reconfigured: %s @ %s", self._model, cfg.base_url)
