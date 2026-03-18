"""MCP tool server for cron job management."""

from __future__ import annotations

import json
import os
from datetime import datetime
from pathlib import Path
from uuid import uuid4

from mcp.server.fastmcp import FastMCP

mcp = FastMCP("scheduler")

_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
_CONFIG_YAML = _PROJECT_ROOT / "config.yaml"
_CRON_FILE = _PROJECT_ROOT / "data" / "scheduler" / "cron_jobs.json"


def _load_cron_jobs() -> list[dict]:
    if not _CRON_FILE.exists():
        return []
    try:
        return json.loads(_CRON_FILE.read_text())
    except (json.JSONDecodeError, OSError):
        return []


def _save_cron_jobs(jobs: list[dict]) -> None:
    _CRON_FILE.parent.mkdir(parents=True, exist_ok=True)
    tmp = _CRON_FILE.with_suffix(".tmp")
    fd = os.open(str(tmp), os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
    with os.fdopen(fd, "w") as f:
        json.dump(jobs, f, indent=2)
    tmp.replace(_CRON_FILE)


@mcp.tool()
def create_cron_job(
    name: str,
    schedule: str,
    job_type: str = "notification",
    message: str = "",
    tool_name: str = "",
    tool_args: str = "{}",
    utterance: str = "",
) -> str:
    """Create a recurring cron job.

    Args:
        name: Human-readable name for the job
        schedule: Cron expression (e.g., "0 9 * * *" for 9am daily, "*/30 * * * *" for every 30 min)
        job_type: Type of job - "notification" (TTS + SSE), "tool" (call MCP tool), "agent" (process utterance)
        message: Message text for notification jobs
        tool_name: MCP tool name for tool jobs
        tool_args: JSON string of tool arguments for tool jobs
        utterance: Text to process for agent jobs
    """
    from croniter import croniter

    # Validate cron expression
    try:
        croniter(schedule)
    except (ValueError, KeyError) as e:
        return f"Error: Invalid cron expression '{schedule}': {e}"

    if job_type not in ("notification", "tool", "agent"):
        return f"Error: Invalid job type '{job_type}'. Must be notification, tool, or agent."

    jobs = _load_cron_jobs()
    if len(jobs) >= 50:
        return "Error: Maximum number of cron jobs (50) reached."

    payload: dict = {}
    if job_type == "notification":
        payload["message"] = message or name
    elif job_type == "tool":
        if not tool_name:
            return "Error: tool_name is required for tool jobs."
        payload["tool_name"] = tool_name
        try:
            payload["tool_args"] = json.loads(tool_args) if tool_args else {}
        except json.JSONDecodeError:
            return f"Error: Invalid JSON in tool_args: {tool_args}"
    elif job_type == "agent":
        if not utterance:
            return "Error: utterance is required for agent jobs."
        payload["utterance"] = utterance

    cron = croniter(schedule, datetime.now())
    next_run = cron.get_next(datetime).isoformat()

    job = {
        "id": uuid4().hex[:12],
        "name": name,
        "schedule": schedule,
        "type": job_type,
        "payload": payload,
        "last_run": None,
        "next_run": next_run,
        "enabled": True,
    }
    jobs.append(job)
    _save_cron_jobs(jobs)

    return json.dumps({"status": "created", "job": job}, indent=2)


@mcp.tool()
def list_cron_jobs() -> str:
    """List all scheduled cron jobs with their next run times."""
    jobs = _load_cron_jobs()
    if not jobs:
        return "No cron jobs configured."
    return json.dumps(jobs, indent=2)


@mcp.tool()
def delete_cron_job(job_id: str) -> str:
    """Delete a cron job by its ID.

    Args:
        job_id: The unique ID of the cron job to delete
    """
    jobs = _load_cron_jobs()
    original_len = len(jobs)
    jobs = [j for j in jobs if j.get("id") != job_id]
    if len(jobs) == original_len:
        return f"Error: No cron job found with ID '{job_id}'."
    _save_cron_jobs(jobs)
    return f"Cron job '{job_id}' deleted."


if __name__ == "__main__":
    mcp.run()
