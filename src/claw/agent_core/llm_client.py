"""LLM client via OpenAI-compatible API with optional cloud backends."""

from __future__ import annotations

import asyncio
import logging

from collections.abc import AsyncIterator

from openai import APITimeoutError, AsyncOpenAI
from openai.types.chat import ChatCompletionMessageParam
from openai.types.chat.chat_completion_chunk import ChatCompletionChunk

from claw.config import get_settings, on_reload

log = logging.getLogger(__name__)


class LLMClient:
    """Async LLM client wrapping OpenAI-compatible endpoints.

    Supports a local backend (default) and optional cloud providers
    (Claude via Anthropic, Gemini via Google) that use the same
    OpenAI-compatible protocol with different base_url/api_key/model.
    """

    def __init__(self) -> None:
        cfg = get_settings().llm
        self._local_client = AsyncOpenAI(
            base_url=cfg.base_url,
            api_key=cfg.api_key,
            timeout=cfg.timeout,
        )
        self._cloud_clients: dict[str, AsyncOpenAI] = {}
        self._cloud_models: dict[str, str] = {}
        self._cloud_max_tokens: dict[str, int] = {}
        self._cloud_temperature: dict[str, float] = {}
        self._model = cfg.model
        self._temperature = cfg.temperature
        self._max_tokens = cfg.max_tokens
        self._timeout = cfg.timeout
        self._active_provider = "local"
        self._last_serving_provider: str = "local"
        self._thinking_override: str | None = None  # per-session override
        self._lock = asyncio.Lock()
        self._simple_lock = asyncio.Lock()  # separate lock for chat_simple (routing etc.)
        self._init_cloud_clients()
        on_reload(self._on_config_reload)

    def _init_cloud_clients(self) -> None:
        """Initialize cloud provider clients from config + secret store."""
        from claw import secret_store

        cloud_cfg = get_settings().cloud_llm
        self._active_provider = cloud_cfg.active_provider

        # Clear existing cloud clients before re-init
        self._cloud_clients.clear()
        self._cloud_models.clear()
        self._cloud_max_tokens.clear()
        self._cloud_temperature.clear()

        for name, provider in cloud_cfg.providers.items():
            api_key = secret_store.load(provider.api_key_secret) if provider.api_key_secret else None
            if not api_key:
                continue

            extra_headers = {}
            if name == "claude":
                extra_headers["anthropic-version"] = "2023-06-01"

            self._cloud_clients[name] = AsyncOpenAI(
                base_url=provider.base_url,
                api_key=api_key,
                timeout=provider.timeout,
                default_headers=extra_headers if extra_headers else None,
            )
            self._cloud_models[name] = provider.model
            self._cloud_max_tokens[name] = provider.max_tokens
            self._cloud_temperature[name] = provider.temperature
            log.info("Cloud LLM client initialized: %s (%s)", name, provider.model)

        if self._active_provider != "local" and self._active_provider not in self._cloud_clients:
            log.warning("Active provider '%s' not available, falling back to local", self._active_provider)
            self._active_provider = "local"

    @property
    def active_provider(self) -> str:
        return self._active_provider

    @active_provider.setter
    def active_provider(self, value: str) -> None:
        if value != "local" and value not in self._cloud_clients:
            log.warning("Cannot switch to provider '%s' — not configured", value)
            return
        self._active_provider = value
        log.info("LLM provider switched to: %s", value)

    @property
    def busy(self) -> bool:
        """True if an LLM call is currently in progress."""
        return self._lock.locked()

    @property
    def thinking_level(self) -> str:
        if self._thinking_override is not None:
            return self._thinking_override
        return get_settings().llm.thinking

    @thinking_level.setter
    def thinking_level(self, value: str) -> None:
        self._thinking_override = value if value != "off" else None

    @property
    def last_serving_provider(self) -> str:
        return self._last_serving_provider

    def _get_client_and_params(self, model: str | None = None):
        """Get the active client, model, temperature, and max_tokens.

        Returns:
            Tuple of (client, model, temperature, max_tokens).
        """
        provider = self._active_provider
        if provider != "local" and provider in self._cloud_clients:
            return (
                self._cloud_clients[provider],
                model or self._cloud_models[provider],
                self._cloud_temperature.get(provider, self._temperature),
                self._cloud_max_tokens.get(provider, self._max_tokens),
            )
        return (self._local_client, model or self._model, self._temperature, self._max_tokens)

    def _get_failover_chain(self) -> list[str]:
        """Get ordered failover chain from config."""
        chain = get_settings().cloud_llm.failover_chain
        if chain:
            return chain
        # Legacy: failover_to_local
        if get_settings().cloud_llm.failover_to_local and self._active_provider != "local":
            return ["local"]
        return []

    def _should_failover(self, error) -> bool:
        """Check if error is retryable (timeout, 503, 429) vs permanent (400, auth)."""
        if isinstance(error, APITimeoutError):
            return True
        if hasattr(error, "status_code"):
            return error.status_code in (429, 500, 502, 503, 504)
        return True  # default to retryable for unknown errors

    def _get_provider_params(self, provider: str, model_override=None):
        """Get client and params for a specific provider.

        Returns:
            Tuple of (client, model, temperature, max_tokens) or (None, None, None, None).
        """
        if provider == "local":
            return (self._local_client, model_override or self._model, self._temperature, self._max_tokens)
        if provider in self._cloud_clients:
            return (
                self._cloud_clients[provider],
                model_override or self._cloud_models[provider],
                self._cloud_temperature.get(provider, self._temperature),
                self._cloud_max_tokens.get(provider, self._max_tokens),
            )
        return (None, None, None, None)

    def _apply_thinking(self, kwargs: dict, provider: str) -> None:
        """Add thinking/reasoning parameters to kwargs based on provider and level.

        Modifies kwargs in place. Merges into existing extra_body if present.
        """
        level = self.thinking_level

        # Remove any previous thinking keys from extra_body (if present)
        existing = kwargs.get("extra_body", {}) or {}
        for key in ("thinking", "thinking_config", "reasoning_effort"):
            existing.pop(key, None)

        if level == "off":
            # Clean up: remove extra_body if it's now empty
            if not existing:
                kwargs.pop("extra_body", None)
            return  # No thinking params for "off" — local server uses --reasoning-budget 0

        budgets = {"low": 1024, "medium": 4096, "high": 16384}
        budget = budgets.get(level, 0)
        if budget == 0:
            return

        if provider == "claude":
            existing["thinking"] = {"type": "enabled", "budget_tokens": budget}
        elif provider == "gemini":
            existing["thinking_config"] = {"thinking_budget": budget}
        else:
            # Local qwen3 models: use reasoning_effort via extra_body
            existing["reasoning_effort"] = level

        kwargs["extra_body"] = existing

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
        client, effective_model, temperature, max_tokens = self._get_client_and_params(model)
        kwargs = {
            "model": effective_model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
        }
        if tools:
            kwargs["tools"] = tools

        # Add thinking/reasoning parameters
        self._apply_thinking(kwargs, self._active_provider)

        log.debug("LLM request: provider=%s model=%s messages=%d tools=%d",
                   self._active_provider, effective_model, len(messages), len(tools or []))

        async with self._lock:
            # Try primary provider
            try:
                response = await client.chat.completions.create(**kwargs)
                self._last_serving_provider = self._active_provider
                return response
            except (APITimeoutError, Exception) as e:
                if not self._should_failover(e):
                    raise
                log.warning("Provider '%s' failed: %s", self._active_provider, e)

            # Try failover chain
            chain = self._get_failover_chain()
            for provider in chain:
                if provider == self._active_provider:
                    continue  # already tried
                fc, fm, ft, fmax = self._get_provider_params(provider, model)
                if fc is None:
                    continue
                try:
                    failover_kwargs = {**kwargs, "model": fm, "temperature": ft, "max_tokens": fmax}
                    # Rebuild thinking params for new provider
                    self._apply_thinking(failover_kwargs, provider)
                    response = await fc.chat.completions.create(**failover_kwargs)
                    self._last_serving_provider = provider
                    log.info("Failover succeeded: %s -> %s", self._active_provider, provider)
                    return response
                except Exception as e2:
                    log.warning("Failover to '%s' also failed: %s", provider, e2)
                    continue

            raise RuntimeError("All LLM providers failed")

    async def chat_stream(
        self,
        messages: list[ChatCompletionMessageParam],
        tools: list[dict] | None = None,
        model: str | None = None,
    ) -> AsyncIterator[ChatCompletionChunk]:
        """Stream a chat completion, yielding chunks as they arrive.

        Acquires ``self._lock`` internally so that only one LLM request
        (streaming or non-streaming) runs at a time.

        Yields ``ChatCompletionChunk`` objects.  The final chunk carries usage
        stats when ``stream_options={"include_usage": True}``.
        """
        client, effective_model, temperature, max_tokens = self._get_client_and_params(model)
        kwargs: dict = {
            "model": effective_model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
            "stream": True,
            "stream_options": {"include_usage": True},
        }
        if tools:
            kwargs["tools"] = tools

        # Add thinking/reasoning parameters
        self._apply_thinking(kwargs, self._active_provider)

        log.debug(
            "LLM stream request: provider=%s model=%s messages=%d tools=%d",
            self._active_provider, effective_model, len(messages), len(tools or []),
        )

        # Phase 1 (locked): establish stream connection + failover
        async with self._lock:
            stream = None
            try:
                stream = await client.chat.completions.create(**kwargs)
                self._last_serving_provider = self._active_provider
            except (APITimeoutError, Exception) as e:
                if not self._should_failover(e):
                    raise
                log.warning("Stream provider '%s' failed: %s", self._active_provider, e)

            # Failover chain for initial connection
            if stream is None:
                chain = self._get_failover_chain()
                for provider in chain:
                    if provider == self._active_provider:
                        continue
                    fc, fm, ft, fmax = self._get_provider_params(provider, model)
                    if fc is None:
                        continue
                    try:
                        failover_kwargs = {**kwargs, "model": fm, "temperature": ft, "max_tokens": fmax}
                        self._apply_thinking(failover_kwargs, provider)
                        stream = await fc.chat.completions.create(**failover_kwargs)
                        self._last_serving_provider = provider
                        log.info("Stream failover succeeded: %s -> %s", self._active_provider, provider)
                        break
                    except Exception as e2:
                        log.warning("Stream failover to '%s' also failed: %s", provider, e2)
                        continue

            if stream is None:
                raise RuntimeError("All LLM providers failed")

        # Phase 2 (unlocked): stream chunks without holding the lock,
        # allowing chat_simple() and voice-loop chat() to proceed.
        try:
            async for chunk in stream:
                yield chunk
        except APITimeoutError:
            log.error("LLM stream interrupted after %ds", self._timeout)
            raise asyncio.TimeoutError(
                f"LLM stream interrupted after {self._timeout}s"
            )

    async def chat_simple(self, prompt: str) -> str:
        """Simple single-turn chat returning just the text content.

        Uses its own lock so it won't block behind a long-running chat() call.
        Always uses the local LLM — never routed through cloud providers.
        """
        async with self._simple_lock:
            try:
                response = await self._local_client.chat.completions.create(
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
        if cfg.base_url != str(self._local_client.base_url).rstrip("/"):
            self._local_client = AsyncOpenAI(
                base_url=cfg.base_url, api_key=cfg.api_key, timeout=cfg.timeout,
            )
            changed = True
        self._model = cfg.model
        self._temperature = cfg.temperature
        self._max_tokens = cfg.max_tokens
        self._timeout = cfg.timeout

        # Refresh cloud clients
        self._init_cloud_clients()

        if changed:
            log.info("LLM client reconfigured: %s @ %s", self._model, cfg.base_url)
