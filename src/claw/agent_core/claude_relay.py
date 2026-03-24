"""Claude Code CLI relay — passthrough to Claude Code over SSH.

Manages a persistent Claude Code session on a remote dev machine,
relaying messages from any interface (voice, Signal, Discord, CLI, etc.)
via SSH. For voice callers, extracts a spoken summary for TTS.
For text callers, returns the full response as-is.

Supports slash commands (/resume, /model, /cost, etc.) that map to
Claude Code CLI flags, giving a terminal-like experience from any interface.

Architecture:
    Claw (any interface) → SSH → dev machine → claude --print → response
    The dev machine runs Claude Code with full codebase access.
"""

from __future__ import annotations

import asyncio
import json
import logging
import re
import shlex

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


class ClaudeRelay:
    """Manages a Claude Code CLI session via SSH to a dev machine.

    Supports all Claw interfaces: voice, CLI, Signal, Discord, Telegram.
    Voice callers get a short TTS summary; text callers get the full response.

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

    @property
    def available(self) -> bool:
        """Check if relay is configured."""
        cfg = get_settings().claude_relay
        return cfg.enabled and bool(cfg.host) and bool(cfg.user)

    @property
    def session_id(self) -> str | None:
        return self._session_id

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

    # ── SSH + Claude CLI ────────────────────────────────────────────────

    @property
    def _effective_skip_permissions(self) -> bool:
        cfg = get_settings().claude_relay
        if self.skip_permissions_override is not None:
            return self.skip_permissions_override
        return cfg.skip_permissions

    def _build_ssh_command(self, claude_cmd: str) -> list[str]:
        """Build the SSH command to execute claude on the dev machine."""
        cfg = get_settings().claude_relay
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

    async def send(
        self,
        message: str,
        timeout: float = 180,
        voice_mode: bool = False,
    ) -> tuple[str, str | None]:
        """Send a message to Claude Code on the dev machine via SSH.

        Slash commands are intercepted and handled before reaching Claude.

        Args:
            message: The user's message (transcribed speech or typed text).
            timeout: Max seconds to wait for Claude Code (default 3 min).
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

        # Build the claude command with all current settings
        claude_parts = ["claude", "--print", "--output-format", "json"]

        if self._effective_skip_permissions:
            claude_parts.append("--dangerously-skip-permissions")
        if self._model:
            claude_parts.extend(["--model", self._model])
        if self._effort:
            claude_parts.extend(["--effort", self._effort])

        # Session continuity: --continue (most recent) or --resume <id>
        if self._use_continue:
            claude_parts.append("--continue")
            self._use_continue = False  # only for the next call
        elif self._session_id:
            claude_parts.extend(["--resume", self._session_id])

        # Voice mode: prepend instruction for spoken summary
        if voice_mode:
            wrapped = f"{_VOICE_INSTRUCTION}{message}"
        else:
            wrapped = message

        claude_parts.append(shlex.quote(wrapped))
        claude_cmd = " ".join(claude_parts)
        ssh_cmd = self._build_ssh_command(claude_cmd)

        log.info("Relaying to Claude Code: %s", message[:100])

        proc = await asyncio.create_subprocess_exec(
            *ssh_cmd,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )

        try:
            stdout, stderr = await asyncio.wait_for(
                proc.communicate(), timeout=timeout
            )
        except asyncio.TimeoutError:
            proc.kill()
            await proc.wait()
            log.warning("Claude Code relay timed out after %.0fs", timeout)
            err = "Claude Code timed out. Try a simpler request."
            return (err, err if voice_mode else None)

        if proc.returncode != 0:
            err = stderr.decode(errors="replace").strip()
            err_lines = [
                l for l in err.splitlines()
                if not l.startswith("Warning:") and "known_hosts" not in l
            ]
            err_clean = "\n".join(err_lines).strip() if err_lines else err
            log.error("Claude Code relay error (rc=%d): %s", proc.returncode, err_clean[:300])
            short = "There was an error communicating with Claude Code."
            return (f"Claude Code error: {err_clean}", short if voice_mode else None)

        # Parse JSON response
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

    def reset(self) -> None:
        """End the current session and reset state."""
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
