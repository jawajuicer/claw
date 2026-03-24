"""Claude Code CLI relay — passthrough to Claude Code over SSH.

Manages a persistent Claude Code session on a remote dev machine,
relaying messages from any interface (voice, Signal, Discord, CLI, etc.)
via SSH. For voice callers, extracts a spoken summary for TTS.
For text callers, returns the full response as-is.

Uses SSH ControlMaster for connection pooling (one TCP handshake per
activation, all subsequent messages multiplex over it) and retries
transient failures with exponential backoff.

Supports slash commands (/resume, /model, /cost, etc.) that map to
Claude Code CLI flags, giving a terminal-like experience from any interface.

Architecture:
    Claw (any interface) → SSH (ControlMaster) → dev machine → claude --print → response
    The dev machine runs Claude Code with full codebase access.
"""

from __future__ import annotations

import asyncio
import enum
import json
import logging
import re
import shlex
from pathlib import Path

from claw.config import get_settings

log = logging.getLogger(__name__)

# Prepended to relayed messages when voice_mode=True so Claude includes
# a spoken summary alongside its normal response.
_VOICE_INSTRUCTION = (
    "[VOICE RELAY] You are receiving this message from a user speaking through "
    "a voice assistant called Claw. Respond normally to their request. "
    "At the very end of your response, include a <voice_summary> tag with a "
    "brief, natural-language summary (under 30 words) suitable for text-to-speech. "
    "Rules for the voice summary:\n"
    "- No code, file paths, symbols, backticks, or markdown\n"
    "- Speak naturally as if talking to the user\n"
    "- Focus on what you did, what changed, or what you need from them\n"
    "- If you have a question, ask it in the summary\n\n"
    "User says: "
)

_SUMMARY_RE = re.compile(r"<voice_summary>(.*?)</voice_summary>", re.DOTALL)

_FALLBACK_SUMMARY = "Claude responded. Check your screen for details."

# Valid effort levels for /effort command
_EFFORT_LEVELS = {"low", "medium", "high", "max"}

# Valid model aliases (short names accepted by claude CLI)
_MODEL_ALIASES = {"sonnet", "opus", "haiku"}

_DEVNULL = asyncio.subprocess.DEVNULL


class _ErrorKind(enum.Enum):
    """Classification for relay errors to decide retry strategy."""
    TRANSIENT = "transient"   # SSH connection blip — retry with backoff
    TIMEOUT = "timeout"       # Claude took too long — retry with --continue
    FATAL = "fatal"           # auth failure, bad config — do not retry


class ClaudeRelay:
    """Manages a Claude Code CLI session via SSH to a dev machine.

    Supports all Claw interfaces: voice, CLI, Signal, Discord, Telegram.
    Voice callers get a short TTS summary; text callers get the full response.

    Features:
    - SSH ControlMaster for connection pooling (one handshake per activation)
    - Retry with exponential backoff on transient failures
    - Adaptive timeout (longer for first message, shorter for follow-ups)

    Slash commands (/resume, /model, /cost, etc.) are intercepted and
    translated to CLI flags or handled internally.
    """

    def __init__(self) -> None:
        self._session_id: str | None = None
        self.active = False
        self.skip_permissions_override: bool | None = None
        # Per-session overrides (set via slash commands)
        self._model: str | None = None
        self._effort: str | None = None
        self._use_continue: bool = False  # use --continue instead of --resume
        self._cumulative_cost: float = 0.0
        self._turn_count: int = 0
        # SSH ControlMaster state
        self._control_path: str | None = None

    @property
    def available(self) -> bool:
        """Check if relay is configured."""
        cfg = get_settings().claude_relay
        return cfg.enabled and bool(cfg.host) and bool(cfg.user)

    @property
    def session_id(self) -> str | None:
        return self._session_id

    # ── Activation / deactivation ────────────────────────────────────────

    async def activate(self) -> str:
        """Activate relay and warm up the SSH ControlMaster connection.

        Returns a status message for the user.
        """
        self.active = True
        ok = await self._ensure_control_master()
        if ok:
            return "Connected to Claude Code. Go ahead."
        return "Connected to Claude Code (SSH warmup failed, first message may be slow)."

    async def async_reset(self) -> None:
        """End the session, close ControlMaster, and reset all state."""
        await self._close_control_master()
        self._reset_state()

    def reset(self) -> None:
        """Synchronous reset — cleans up socket file, resets state."""
        if self._control_path:
            Path(self._control_path).unlink(missing_ok=True)
            self._control_path = None
        self._reset_state()

    def _reset_state(self) -> None:
        if self._session_id:
            log.info("Claude Code session ended: %s", self._session_id[:12])
        self._session_id = None
        self.active = False
        self.skip_permissions_override = None
        self._model = None
        self._effort = None
        self._use_continue = False
        self._cumulative_cost = 0.0
        self._turn_count = 0

    # ── Slash command handling ──────────────────────────────────────────

    def _handle_command(self, text: str) -> str | None:
        """Handle slash commands. Returns response string, or None to pass through."""
        parts = text.strip().split(None, 1)
        cmd = parts[0].lower()
        arg = parts[1].strip() if len(parts) > 1 else ""

        handler = self._COMMANDS.get(cmd)
        if handler:
            return handler(self, arg)
        return None

    def _cmd_help(self, _arg: str) -> str:
        return (
            "**Claude Code Relay Commands**\n"
            "`/resume [session_id]` — Resume a session (or continue most recent)\n"
            "`/model [name]` — Show or change model (e.g. sonnet, opus, claude-sonnet-4-6)\n"
            "`/effort [level]` — Show or set effort (low, medium, high, max)\n"
            "`/cost` — Show cumulative session cost\n"
            "`/status` — Show session info\n"
            "`/clear` — Start a fresh session\n"
            "`/compact` — Ask Claude to compact/summarize context\n"
            "`/continue` — Continue most recent conversation\n"
            "`/help` — Show this help"
        )

    def _cmd_resume(self, arg: str) -> str:
        if arg:
            self._session_id = arg
            self._use_continue = False
            return f"Session set to `{arg}`. Next message will resume it."
        else:
            self._use_continue = True
            self._session_id = None
            return "Next message will continue the most recent conversation."

    def _cmd_continue(self, _arg: str) -> str:
        self._use_continue = True
        self._session_id = None
        return "Next message will continue the most recent conversation."

    def _cmd_model(self, arg: str) -> str:
        if not arg:
            current = self._model or "default (from config)"
            return f"Current model: `{current}`"
        self._model = arg
        return f"Model set to `{arg}` for this session."

    def _cmd_effort(self, arg: str) -> str:
        if not arg:
            current = self._effort or "default (from config)"
            return f"Current effort: `{current}`"
        if arg.lower() not in _EFFORT_LEVELS:
            return f"Invalid effort level. Choose: {', '.join(sorted(_EFFORT_LEVELS))}"
        self._effort = arg.lower()
        return f"Effort set to `{self._effort}` for this session."

    def _cmd_cost(self, _arg: str) -> str:
        return (
            f"Session cost: ${self._cumulative_cost:.4f}\n"
            f"Turns: {self._turn_count}"
        )

    def _cmd_status(self, _arg: str) -> str:
        lines = [
            f"Session: `{self._session_id or 'none (new)'}`",
            f"Model: `{self._model or 'default'}`",
            f"Effort: `{self._effort or 'default'}`",
            f"Turns: {self._turn_count}",
            f"Cost: ${self._cumulative_cost:.4f}",
            f"Skip permissions: {self._effective_skip_permissions}",
        ]
        return "\n".join(lines)

    def _cmd_clear(self, _arg: str) -> str:
        old_sid = self._session_id
        self._session_id = None
        self._use_continue = False
        self._cumulative_cost = 0.0
        self._turn_count = 0
        if old_sid:
            return f"Session `{old_sid[:12]}...` cleared. Next message starts fresh."
        return "Session cleared. Next message starts fresh."

    def _cmd_compact(self, _arg: str) -> str | None:
        # Can't directly compact in --print mode, but we can ask Claude to do it
        return None  # fall through — send as a real message to Claude

    # Command registry
    _COMMANDS: dict[str, callable] = {
        "/help": _cmd_help,
        "/resume": _cmd_resume,
        "/r": _cmd_resume,
        "/continue": _cmd_continue,
        "/c": _cmd_continue,
        "/model": _cmd_model,
        "/effort": _cmd_effort,
        "/cost": _cmd_cost,
        "/status": _cmd_status,
        "/clear": _cmd_clear,
        "/compact": _cmd_compact,
    }

    # ── SSH ControlMaster ────────────────────────────────────────────────

    async def _ensure_control_master(self) -> bool:
        """Establish or verify the SSH ControlMaster connection.

        Returns True if the master is alive and ready, False on failure
        (caller should fall back to per-message SSH).
        """
        cfg = get_settings().claude_relay
        if cfg.control_persist <= 0:
            return False  # ControlMaster disabled by config

        control_path = f"/tmp/claw-ssh-{cfg.user}@{cfg.host}"

        # Check if existing socket is alive
        if self._control_path and Path(self._control_path).exists():
            try:
                check = await asyncio.create_subprocess_exec(
                    "ssh", "-o", f"ControlPath={self._control_path}",
                    "-O", "check", f"{cfg.user}@{cfg.host}",
                    stdout=_DEVNULL, stderr=_DEVNULL,
                )
                rc = await asyncio.wait_for(check.wait(), timeout=5)
                if rc == 0:
                    return True  # master is alive
            except asyncio.TimeoutError:
                check.kill()
                await check.wait()
            except OSError:
                pass
            log.warning("SSH ControlMaster socket stale, re-establishing")
            Path(self._control_path).unlink(missing_ok=True)
            self._control_path = None

        # Establish new master connection
        # -N = no remote command; with ControlPersist the master forks to
        # background and the foreground -N process exits once ready.
        cmd = [
            "sshpass", "-p", cfg.password, "ssh",
            "-o", "PreferredAuthentications=password",
            "-o", "PubkeyAuthentication=no",
            "-o", "StrictHostKeyChecking=no",
            "-o", "ConnectTimeout=10",
            "-o", "ControlMaster=yes",
            "-o", f"ControlPath={control_path}",
            "-o", f"ControlPersist={cfg.control_persist}",
            "-N",
            f"{cfg.user}@{cfg.host}",
        ]

        try:
            proc = await asyncio.create_subprocess_exec(
                *cmd, stdout=_DEVNULL, stderr=asyncio.subprocess.PIPE,
            )
            _, stderr = await asyncio.wait_for(proc.communicate(), timeout=15)
        except asyncio.TimeoutError:
            proc.kill()
            await proc.wait()
            log.error("SSH ControlMaster establishment timed out")
            return False

        if proc.returncode != 0:
            err = stderr.decode(errors="replace").strip()
            log.error("SSH ControlMaster failed (rc=%d): %s", proc.returncode, err[:200])
            return False

        self._control_path = control_path
        log.info("SSH ControlMaster established: %s", control_path)
        return True

    async def _close_control_master(self) -> None:
        """Gracefully close the ControlMaster connection."""
        if not self._control_path:
            return
        cfg = get_settings().claude_relay
        try:
            proc = await asyncio.create_subprocess_exec(
                "ssh", "-o", f"ControlPath={self._control_path}",
                "-O", "exit", f"{cfg.user}@{cfg.host}",
                stdout=_DEVNULL, stderr=_DEVNULL,
            )
            await asyncio.wait_for(proc.wait(), timeout=5)
        except (asyncio.TimeoutError, OSError):
            pass  # best-effort
        Path(self._control_path).unlink(missing_ok=True)
        self._control_path = None
        log.info("SSH ControlMaster closed")

    # ── SSH + Claude CLI ────────────────────────────────────────────────

    @property
    def _effective_skip_permissions(self) -> bool:
        cfg = get_settings().claude_relay
        if self.skip_permissions_override is not None:
            return self.skip_permissions_override
        return cfg.skip_permissions

    def _build_ssh_command(self, claude_cmd: str) -> list[str]:
        """Build the SSH command to execute claude on the dev machine.

        Uses ControlMaster multiplexing when available (no sshpass needed),
        falls back to standalone sshpass connection otherwise.
        """
        cfg = get_settings().claude_relay

        if self._control_path and Path(self._control_path).exists():
            # Multiplex over existing ControlMaster — no sshpass/password needed
            ssh_parts = [
                "ssh",
                "-o", f"ControlPath={self._control_path}",
                "-o", "StrictHostKeyChecking=no",
                f"{cfg.user}@{cfg.host}",
            ]
        else:
            # Fallback: standalone connection (original behavior)
            ssh_parts = ["sshpass", "-p", cfg.password, "ssh"]
            ssh_parts.extend([
                "-o", "PreferredAuthentications=password",
                "-o", "PubkeyAuthentication=no",
                "-o", "StrictHostKeyChecking=no",
                "-o", "ConnectTimeout=10",
                f"{cfg.user}@{cfg.host}",
            ])

        remote_cmd = (
            f"export PATH=$HOME/.local/bin:$HOME/.npm-global/bin:$PATH && "
            f"cd {shlex.quote(cfg.project_dir)} && "
            f"{claude_cmd}"
        )
        ssh_parts.append(remote_cmd)
        return ssh_parts

    def _build_claude_cmd(
        self, message: str, voice_mode: bool, use_continue: bool,
    ) -> str:
        """Build the claude CLI command string."""
        parts = ["claude", "--print", "--output-format", "json"]

        if self._effective_skip_permissions:
            parts.append("--dangerously-skip-permissions")
        if self._model:
            parts.extend(["--model", shlex.quote(self._model)])
        if self._effort:
            parts.extend(["--effort", shlex.quote(self._effort)])

        # Session continuity
        if use_continue:
            parts.append("--continue")
        elif self._session_id:
            parts.extend(["--resume", self._session_id])

        wrapped = f"{_VOICE_INSTRUCTION}{message}" if voice_mode else message
        parts.append(shlex.quote(wrapped))
        return " ".join(parts)

    # ── Error classification ─────────────────────────────────────────────

    @staticmethod
    def _classify_error(
        returncode: int | None, stderr_text: str, timed_out: bool,
    ) -> _ErrorKind:
        """Classify a relay error to decide retry strategy."""
        if timed_out:
            return _ErrorKind.TIMEOUT
        if returncode == 255:
            # SSH connection-level failure (network, host unreachable, etc.)
            return _ErrorKind.TRANSIENT
        lower = stderr_text.lower()
        if "permission denied" in lower or "authentication failed" in lower:
            return _ErrorKind.FATAL
        if "connection refused" in lower or "connection reset" in lower:
            return _ErrorKind.TRANSIENT
        if "no such file" in lower and "control" in lower:
            return _ErrorKind.TRANSIENT  # stale control socket
        return _ErrorKind.FATAL

    # ── Send (with retry) ────────────────────────────────────────────────

    async def send(
        self,
        message: str,
        timeout: float | None = None,
        voice_mode: bool = False,
    ) -> tuple[str, str | None]:
        """Send a message to Claude Code on the dev machine via SSH.

        Slash commands are intercepted and handled before reaching Claude.
        Transient failures are retried with exponential backoff.

        Args:
            message: The user's message (transcribed speech or typed text).
            timeout: Max seconds to wait (None = auto based on session state).
            voice_mode: If True, request a voice summary for TTS.

        Returns:
            Tuple of (full_response, voice_summary).
        """
        # Handle slash commands
        if message.strip().startswith("/"):
            result = self._handle_command(message)
            if result is not None:
                return (result, result if voice_mode else None)
            # None means fall through (e.g. /compact sends to Claude)

        if not self.available:
            err = "Claude Code relay is not configured. Set claude_relay settings in config.yaml."
            return (err, err if voice_mode else None)

        cfg = get_settings().claude_relay

        # Adaptive timeout: longer for first message, shorter for follow-ups
        if timeout is None:
            timeout = cfg.timeout_initial if not self._session_id else cfg.timeout

        max_attempts = 1 + cfg.max_retries

        for attempt in range(max_attempts):
            # Ensure ControlMaster is alive (no-op if disabled or already alive)
            await self._ensure_control_master()

            # On retry after timeout, use --continue to pick up partial work
            use_continue = self._use_continue or (
                attempt > 0 and self._session_id is not None
            )

            claude_cmd = self._build_claude_cmd(message, voice_mode, use_continue)
            ssh_cmd = self._build_ssh_command(claude_cmd)

            if attempt == 0:
                log.info("Relaying to Claude Code: %s", message[:100])
            else:
                log.info(
                    "Relay retry %d/%d: %s", attempt, cfg.max_retries, message[:80],
                )

            proc = await asyncio.create_subprocess_exec(
                *ssh_cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )

            timed_out = False
            try:
                stdout, stderr = await asyncio.wait_for(
                    proc.communicate(), timeout=timeout,
                )
            except asyncio.TimeoutError:
                proc.kill()
                await proc.wait()
                timed_out = True
                stdout, stderr = b"", b""

            # ── Success ──
            if not timed_out and proc.returncode == 0:
                # Clear _use_continue now that it was consumed
                self._use_continue = False
                return self._parse_response(stdout, voice_mode)

            # ── Failure — classify and maybe retry ──
            stderr_text = stderr.decode(errors="replace").strip()
            kind = self._classify_error(proc.returncode, stderr_text, timed_out)

            is_last_attempt = attempt >= max_attempts - 1

            if kind == _ErrorKind.FATAL or is_last_attempt:
                self._use_continue = False
                if timed_out:
                    err = f"Claude Code timed out after {timeout:.0f}s."
                    if is_last_attempt and cfg.max_retries > 0:
                        err += f" ({cfg.max_retries} retries exhausted)"
                    return (err, err if voice_mode else None)
                err_lines = [
                    ln for ln in stderr_text.splitlines()
                    if not ln.startswith("Warning:") and "known_hosts" not in ln
                ]
                err_clean = "\n".join(err_lines).strip() if err_lines else stderr_text
                log.error("Claude Code relay error (rc=%d): %s", proc.returncode, err_clean[:300])
                short = "There was an error communicating with Claude Code."
                return (f"Claude Code error: {err_clean}", short if voice_mode else None)

            # ── Transient/timeout — retry with backoff ──
            delay = 2 ** attempt  # 1s, 2s
            log.warning(
                "Relay attempt %d/%d failed (%s, rc=%s), retrying in %ds",
                attempt + 1, max_attempts, kind.value, proc.returncode, delay,
            )

            # Invalidate stale ControlMaster on transient SSH failure
            if kind == _ErrorKind.TRANSIENT and self._control_path:
                Path(self._control_path).unlink(missing_ok=True)
                self._control_path = None

            await asyncio.sleep(delay)

        # Should not reach here, but just in case
        self._use_continue = False
        err = "Claude Code relay failed unexpectedly."
        return (err, err if voice_mode else None)

    def _parse_response(
        self, stdout: bytes, voice_mode: bool,
    ) -> tuple[str, str | None]:
        """Parse Claude Code JSON response, track session and cost."""
        raw = stdout.decode(errors="replace").strip()
        result_text = raw
        try:
            data = json.loads(raw)
            result_text = data.get("result") or raw
            # Track session ID for continuity
            new_sid = data.get("session_id")
            if new_sid:
                self._session_id = new_sid
                log.info("Claude Code session: %s", new_sid[:12])
            # Track cost
            cost = data.get("total_cost_usd", 0)
            if cost:
                self._cumulative_cost += cost
                log.info("Claude Code cost: $%.4f (total: $%.4f)", cost, self._cumulative_cost)
        except (json.JSONDecodeError, TypeError):
            log.warning("Could not parse Claude Code JSON output, using raw text")

        self._turn_count += 1

        # Strip voice_summary tag from display response
        display = _SUMMARY_RE.sub("", result_text).strip()

        # Extract voice summary only if voice mode
        summary = None
        if voice_mode:
            summary = self._extract_summary(result_text)

        log.info("Claude Code response: %d chars", len(display))
        return display, summary

    # ── Voice summary extraction ─────────────────────────────────────────

    def _extract_summary(self, text: str) -> str:
        """Extract <voice_summary> from response, with fallback heuristic."""
        match = _SUMMARY_RE.search(text)
        if match:
            return match.group(1).strip()
        return self._heuristic_summary(text)

    @staticmethod
    def _heuristic_summary(text: str) -> str:
        """Generate a brief summary from the first meaningful sentence."""
        clean = re.sub(r"```[\s\S]*?```", "", text)
        clean = re.sub(r"`[^`]+`", "", clean)
        clean = re.sub(r"[#*_~>\[\]]", "", clean)
        clean = re.sub(r"\n+", " ", clean).strip()

        if not clean:
            return _FALLBACK_SUMMARY

        sentences = re.split(r"(?<=[.!?])\s+", clean)
        for s in sentences:
            s = s.strip()
            if len(s) > 15:
                if len(s) > 120:
                    s = s[:120].rsplit(" ", 1)[0] + "."
                return s

        return _FALLBACK_SUMMARY
