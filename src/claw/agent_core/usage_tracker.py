"""Cumulative token usage tracking with persistence and cost estimation."""

from __future__ import annotations

import asyncio
import json
import logging
import os
from datetime import date, datetime
from pathlib import Path

from claw.config import PROJECT_ROOT, get_settings

log = logging.getLogger(__name__)

# Pricing per million tokens (input, output) as of 2026
_PRICING = {
    "claude": {"input": 3.0, "output": 15.0},    # Sonnet
    "gemini": {"input": 0.15, "output": 0.60},    # Flash
    "local": {"input": 0.0, "output": 0.0},       # Free
}


class UsageTracker:
    """Tracks cumulative token usage across sessions with JSON persistence.

    Uses asyncio.Lock for non-blocking synchronization in the event loop.
    File I/O (persist) is offloaded to a thread to avoid blocking.
    """

    def __init__(self) -> None:
        self._lock = asyncio.Lock()
        self._session: dict = self._empty_session()
        self._daily: dict[str, dict] = {}
        self._total: dict = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0, "calls": 0, "by_provider": {}}
        self._dirty = False
        self._load_sync()  # sync load at init (before event loop)

    @staticmethod
    def _empty_session() -> dict:
        return {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0, "calls": 0, "by_provider": {}}

    def _persist_file(self) -> Path:
        cfg = get_settings()
        if hasattr(cfg, 'usage') and cfg.usage.persist_file:
            return PROJECT_ROOT / cfg.usage.persist_file
        return PROJECT_ROOT / "data" / "usage.json"

    def _load_sync(self) -> None:
        """Synchronous load for use at __init__ (before event loop)."""
        f = self._persist_file()
        if not f.exists():
            return
        try:
            data = json.loads(f.read_text())
            self._daily = data.get("daily", {})
            self._total = data.get("total", self._total)
        except (json.JSONDecodeError, OSError):
            log.warning("Failed to load usage data")

    async def record(self, prompt_tokens: int, completion_tokens: int, total_tokens: int, provider: str = "local") -> None:
        """Record token usage from a single LLM call."""
        async with self._lock:
            today = date.today().isoformat()

            # Session stats
            self._session["prompt_tokens"] += prompt_tokens
            self._session["completion_tokens"] += completion_tokens
            self._session["total_tokens"] += total_tokens
            self._session["calls"] += 1
            self._session["by_provider"].setdefault(provider, {"prompt": 0, "completion": 0, "calls": 0})
            self._session["by_provider"][provider]["prompt"] += prompt_tokens
            self._session["by_provider"][provider]["completion"] += completion_tokens
            self._session["by_provider"][provider]["calls"] += 1

            # Daily stats
            if today not in self._daily:
                self._daily[today] = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0, "calls": 0, "by_provider": {}}
            day = self._daily[today]
            day["prompt_tokens"] += prompt_tokens
            day["completion_tokens"] += completion_tokens
            day["total_tokens"] += total_tokens
            day["calls"] += 1
            day.setdefault("by_provider", {}).setdefault(provider, {"prompt": 0, "completion": 0, "calls": 0})
            day["by_provider"][provider]["prompt"] += prompt_tokens
            day["by_provider"][provider]["completion"] += completion_tokens
            day["by_provider"][provider]["calls"] += 1

            # Total stats
            self._total["prompt_tokens"] += prompt_tokens
            self._total["completion_tokens"] += completion_tokens
            self._total["total_tokens"] += total_tokens
            self._total["calls"] += 1
            self._total.setdefault("by_provider", {}).setdefault(provider, {"prompt": 0, "completion": 0, "calls": 0})
            self._total["by_provider"][provider]["prompt"] += prompt_tokens
            self._total["by_provider"][provider]["completion"] += completion_tokens
            self._total["by_provider"][provider]["calls"] += 1

            self._dirty = True

    async def reset_session(self) -> None:
        async with self._lock:
            self._session = self._empty_session()

    def get_session_summary(self) -> dict:
        """Read-only snapshot — no lock needed (GIL protects dict reads)."""
        return {**self._session, "cost": self._estimate_cost(self._session)}

    def get_daily_summary(self, day: str | None = None) -> dict:
        d = day or date.today().isoformat()
        stats = self._daily.get(d, self._empty_session())
        return {**stats, "date": d, "cost": self._estimate_cost(stats)}

    def get_total_summary(self) -> dict:
        return {**self._total, "cost": self._estimate_cost(self._total)}

    def get_history(self, days: int = 7) -> list[dict]:
        sorted_days = sorted(self._daily.keys(), reverse=True)[:days]
        return [{"date": d, **self._daily[d], "cost": self._estimate_cost(self._daily[d])} for d in sorted_days]

    @staticmethod
    def _estimate_cost(stats: dict) -> dict:
        """Estimate cloud costs from per-provider token breakdown."""
        total_cost = 0.0
        breakdown = {}
        for provider, counts in stats.get("by_provider", {}).items():
            pricing = _PRICING.get(provider, {"input": 0.0, "output": 0.0})
            input_cost = (counts.get("prompt", 0) / 1_000_000) * pricing["input"]
            output_cost = (counts.get("completion", 0) / 1_000_000) * pricing["output"]
            cost = round(input_cost + output_cost, 6)
            if cost > 0:
                breakdown[provider] = cost
            total_cost += cost
        return {"total": round(total_cost, 4), "breakdown": breakdown}

    async def persist(self) -> None:
        """Save usage data to disk atomically. Offloads I/O to thread pool."""
        async with self._lock:
            if not self._dirty:
                return
            data = {"daily": self._daily, "total": self._total, "last_saved": datetime.now().isoformat()}
            # Don't reset dirty flag yet — write may fail

        # File I/O outside the lock, in a thread to avoid blocking the event loop
        try:
            await asyncio.to_thread(self._write_file, data)
            # Only mark clean after successful write
            async with self._lock:
                self._dirty = False
        except Exception:
            log.exception("Failed to persist usage data")
            # _dirty remains True, so the next persist() call will retry

    def _write_file(self, data: dict) -> None:
        """Synchronous file write (called from thread pool)."""
        f = self._persist_file()
        f.parent.mkdir(parents=True, exist_ok=True)
        tmp = f.with_suffix(".tmp")
        fd = os.open(str(tmp), os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
        with os.fdopen(fd, "w") as fh:
            json.dump(data, fh, indent=2)
        tmp.replace(f)
