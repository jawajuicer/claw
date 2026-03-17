"""In-memory tool usage statistics with ring buffer."""

from __future__ import annotations

import time
from collections import defaultdict
from dataclasses import dataclass, field
from threading import Lock


@dataclass
class ToolCallRecord:
    tool: str
    server: str
    elapsed_s: float
    success: bool
    timestamp: float


class ToolStats:
    """Track per-tool call counts, latency, and error rates.

    Thread-safe ring buffer holding the last ``maxlen`` records,
    plus cumulative per-tool counters that survive buffer rotation.
    """

    def __init__(self, maxlen: int = 1000) -> None:
        self._maxlen = maxlen
        self._records: list[ToolCallRecord] = []
        self._lock = Lock()
        # Cumulative counters (never evicted)
        self._counts: dict[str, int] = defaultdict(int)
        self._errors: dict[str, int] = defaultdict(int)
        self._total_time: dict[str, float] = defaultdict(float)

    def record(
        self,
        tool: str,
        server: str,
        elapsed_s: float,
        success: bool,
    ) -> None:
        with self._lock:
            rec = ToolCallRecord(
                tool=tool,
                server=server,
                elapsed_s=elapsed_s,
                success=success,
                timestamp=time.time(),
            )
            self._records.append(rec)
            if len(self._records) > self._maxlen:
                self._records = self._records[-self._maxlen:]

            self._counts[tool] += 1
            if not success:
                self._errors[tool] += 1
            self._total_time[tool] += elapsed_s

    def summary(self) -> dict[str, dict]:
        """Per-tool summary: count, avg_latency_s, error_rate, last_used."""
        with self._lock:
            # Build last-used from records (most recent first)
            last_used: dict[str, float] = {}
            for rec in reversed(self._records):
                if rec.tool not in last_used:
                    last_used[rec.tool] = rec.timestamp

            result = {}
            for tool in sorted(self._counts):
                count = self._counts[tool]
                errors = self._errors[tool]
                total_time = self._total_time[tool]
                result[tool] = {
                    "count": count,
                    "avg_latency_s": round(total_time / count, 3) if count else 0,
                    "error_rate": round(errors / count, 3) if count else 0,
                    "errors": errors,
                    "last_used": last_used.get(tool),
                }
            return result

    def recent(self, n: int = 20) -> list[dict]:
        """Return the last *n* call records."""
        with self._lock:
            return [
                {
                    "tool": r.tool,
                    "server": r.server,
                    "elapsed_s": r.elapsed_s,
                    "success": r.success,
                    "timestamp": r.timestamp,
                }
                for r in self._records[-n:]
            ]
