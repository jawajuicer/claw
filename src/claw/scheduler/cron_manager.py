"""Cron job manager for recurring scheduled tasks."""

from __future__ import annotations

import json
import logging
import os
from datetime import datetime
from pathlib import Path
from typing import Any
from uuid import uuid4

from claw.config import PROJECT_ROOT, get_settings

log = logging.getLogger(__name__)

# Maximum number of cron jobs to prevent resource exhaustion
MAX_CRON_JOBS = 50


class CronJob:
    """A single cron job definition."""

    def __init__(
        self,
        name: str,
        schedule: str,
        job_type: str = "notification",
        payload: dict[str, Any] | None = None,
        job_id: str | None = None,
    ) -> None:
        self.id = job_id or uuid4().hex[:12]
        self.name = name
        self.schedule = schedule  # cron expression (e.g., "0 9 * * *")
        self.type = job_type  # "notification", "tool", "agent"
        self.payload = payload or {}
        self.last_run: str | None = None
        self.next_run: str | None = None
        self.enabled = True
        self._compute_next_run()

    def _compute_next_run(self) -> None:
        """Compute the next run time from the cron schedule."""
        try:
            from croniter import croniter

            cron = croniter(self.schedule, datetime.now())
            self.next_run = cron.get_next(datetime).isoformat()
        except Exception as e:
            log.warning("Invalid cron expression '%s': %s", self.schedule, e)
            self.next_run = None

    def is_due(self) -> bool:
        """Check if this job is due to run."""
        if not self.enabled or not self.next_run:
            return False
        try:
            next_dt = datetime.fromisoformat(self.next_run)
            return datetime.now() >= next_dt
        except ValueError:
            return False

    def mark_ran(self) -> None:
        """Mark the job as having run and compute next run time."""
        self.last_run = datetime.now().isoformat()
        self._compute_next_run()

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "name": self.name,
            "schedule": self.schedule,
            "type": self.type,
            "payload": self.payload,
            "last_run": self.last_run,
            "next_run": self.next_run,
            "enabled": self.enabled,
        }

    @classmethod
    def from_dict(cls, data: dict) -> CronJob:
        job = cls(
            name=data["name"],
            schedule=data["schedule"],
            job_type=data.get("type", "notification"),
            payload=data.get("payload", {}),
            job_id=data.get("id"),
        )
        job.last_run = data.get("last_run")
        job.next_run = data.get("next_run")
        job.enabled = data.get("enabled", True)
        return job


class CronManager:
    """Manages cron jobs: CRUD, persistence, and execution dispatch."""

    def __init__(self, broadcaster=None, agent=None, router=None, tts=None,
                 chat_lock=None, registry=None) -> None:
        self._broadcaster = broadcaster
        self._agent = agent
        self._router = router
        self._tts = tts
        self._chat_lock = chat_lock
        self._registry = registry
        self._jobs: dict[str, CronJob] = {}
        self._storage_path = PROJECT_ROOT / "data" / "scheduler" / "cron_jobs.json"
        self._load()

    def _load(self) -> None:
        """Load cron jobs from disk."""
        if not self._storage_path.exists():
            return
        try:
            data = json.loads(self._storage_path.read_text())
            if not isinstance(data, list):
                log.warning("Cron jobs file has invalid format (expected list)")
                return
            for job_data in data:
                try:
                    job = CronJob.from_dict(job_data)
                    self._jobs[job.id] = job
                except (KeyError, TypeError, ValueError) as e:
                    log.warning("Skipping invalid cron job entry: %s — %s",
                                job_data.get("id", "?") if isinstance(job_data, dict) else "?", e)
            log.info("Loaded %d cron jobs", len(self._jobs))
        except (json.JSONDecodeError, OSError):
            log.warning("Failed to read cron jobs file")

    def _save(self) -> None:
        """Persist cron jobs to disk atomically."""
        self._storage_path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self._storage_path.with_suffix(".tmp")
        fd = os.open(str(tmp), os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
        with os.fdopen(fd, "w") as f:
            json.dump([j.to_dict() for j in self._jobs.values()], f, indent=2)
        tmp.replace(self._storage_path)

    def create(
        self,
        name: str,
        schedule: str,
        job_type: str = "notification",
        payload: dict | None = None,
    ) -> CronJob:
        """Create a new cron job."""
        if len(self._jobs) >= MAX_CRON_JOBS:
            raise ValueError(f"Maximum number of cron jobs ({MAX_CRON_JOBS}) reached")

        # Validate cron expression
        try:
            from croniter import croniter

            croniter(schedule)
        except (ValueError, KeyError) as e:
            raise ValueError(f"Invalid cron expression: {e}") from e

        if job_type not in ("notification", "tool", "agent"):
            raise ValueError(f"Invalid job type: {job_type}")

        job = CronJob(name=name, schedule=schedule, job_type=job_type, payload=payload)
        self._jobs[job.id] = job
        self._save()
        log.info("Created cron job: %s (%s) [%s]", name, schedule, job_type)
        return job

    def delete(self, job_id: str) -> bool:
        """Delete a cron job by ID."""
        if job_id in self._jobs:
            del self._jobs[job_id]
            self._save()
            log.info("Deleted cron job: %s", job_id)
            return True
        return False

    def list_jobs(self) -> list[dict]:
        """List all cron jobs."""
        return [j.to_dict() for j in self._jobs.values()]

    def get(self, job_id: str) -> CronJob | None:
        return self._jobs.get(job_id)

    async def check_and_dispatch(self) -> None:
        """Check for due jobs and dispatch them.

        Reloads from disk each cycle so that jobs created by the MCP
        tool server (a separate subprocess) are picked up.
        """
        self._load()
        for job in list(self._jobs.values()):
            if not job.is_due():
                continue

            log.info("Dispatching cron job: %s (%s)", job.name, job.type)
            try:
                await self._dispatch(job)
            except Exception:
                log.exception("Cron job '%s' failed", job.name)
            finally:
                job.mark_ran()
                self._save()

    async def _dispatch(self, job: CronJob) -> None:
        """Execute a cron job based on its type."""
        if job.type == "notification":
            await self._dispatch_notification(job)
        elif job.type == "tool":
            await self._dispatch_tool(job)
        elif job.type == "agent":
            await self._dispatch_agent(job)

    async def _dispatch_notification(self, job: CronJob) -> None:
        """Broadcast SSE notification and optionally speak via TTS."""
        message = job.payload.get("message", job.name)
        if self._broadcaster:
            await self._broadcaster.broadcast("cron_notification", {
                "job_id": job.id,
                "job_name": job.name,
                "message": message,
            })

        cfg = get_settings().scheduler
        if cfg.announce_tts and self._tts:
            try:
                await self._tts.speak(message)
            except Exception:
                log.exception("TTS announcement failed for cron job '%s'", job.name)

    async def _dispatch_tool(self, job: CronJob) -> None:
        """Call a specific MCP tool directly."""
        tool_name = job.payload.get("tool_name")
        tool_args = job.payload.get("tool_args", {})
        if not tool_name:
            log.warning("Cron tool job '%s' has no tool_name", job.name)
            return
        if not self._router:
            log.warning("No tool router available for cron job '%s'", job.name)
            return

        result = await self._router.call_tool(tool_name, tool_args)
        log.info("Cron tool job '%s' result: %s", job.name, str(result)[:200])

        # Send result to inbox if available
        if self._broadcaster:
            await self._broadcaster.broadcast("cron_result", {
                "job_id": job.id,
                "job_name": job.name,
                "result": str(result)[:500],
            })

    async def _dispatch_agent(self, job: CronJob) -> None:
        """Process an utterance through the agent with full tool access.

        Uses a temporary session swap to avoid corrupting the main voice
        loop session. Saves/restores the original session under no lock
        because the cron poll runs on the same event loop — cooperative
        scheduling ensures no interleaving within a single dispatch.
        """
        utterance = job.payload.get("utterance", "")
        if not utterance:
            log.warning("Cron agent job '%s' has no utterance", job.name)
            return
        if not self._agent:
            log.warning("No agent available for cron job '%s'", job.name)
            return

        # Get tools so agent jobs can use MCP tools
        tools = None
        if self._registry:
            tools = self._registry.get_openai_tools() or None

        # Use chat_lock if available to prevent concurrent agent access,
        # and swap in a fresh session to avoid corrupting the main session.
        if self._chat_lock:
            async with self._chat_lock:
                original_session = self._agent._session
                self._agent._session = None
                try:
                    result = await self._agent.process_utterance(utterance, tools=tools)
                finally:
                    self._agent._session = original_session
        else:
            original_session = self._agent._session
            self._agent._session = None
            try:
                result = await self._agent.process_utterance(utterance, tools=tools)
            finally:
                self._agent._session = original_session
        log.info("Cron agent job '%s' result: %s", job.name, result[:200])

        # Broadcast result
        if self._broadcaster:
            await self._broadcaster.broadcast("cron_result", {
                "job_id": job.id,
                "job_name": job.name,
                "result": result[:500],
            })
