"""System diagnostics for The Claw."""

from __future__ import annotations

import logging
import shutil
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path

log = logging.getLogger(__name__)


@dataclass
class Check:
    name: str
    status: str  # "pass", "fail", "warn"
    detail: str
    elapsed_ms: float = 0.0


def _timed_check(name: str, fn) -> Check:
    """Run a check function and capture timing."""
    t0 = time.monotonic()
    try:
        status, detail = fn()
    except Exception as e:
        status, detail = "fail", str(e)
    elapsed = (time.monotonic() - t0) * 1000
    return Check(name=name, status=status, detail=detail, elapsed_ms=round(elapsed, 1))


def check_config() -> tuple[str, str]:
    """Validate config.yaml loads without errors."""
    from claw.config import Settings, CONFIG_YAML
    if not CONFIG_YAML.exists():
        return "warn", f"Config file not found at {CONFIG_YAML}"
    try:
        s = Settings.load()
        return "pass", f"Loaded OK ({len(s.model_dump())} sections)"
    except Exception as e:
        return "fail", f"Config error: {e}"


def check_llm() -> tuple[str, str]:
    """Check LLM server connectivity."""
    from claw.config import get_settings
    import urllib.request
    import json

    cfg = get_settings().llm
    url = f"{cfg.base_url}/models"
    try:
        with urllib.request.urlopen(url, timeout=5) as resp:
            data = json.loads(resp.read())
            models = data.get("data", [])
            names = [m.get("id", "?") for m in models[:5]]
            return "pass", f"Connected ({len(models)} models: {', '.join(names)})"
    except Exception as e:
        return "fail", f"Cannot reach {url}: {e}"


def check_audio_devices() -> tuple[str, str]:
    """Check audio device availability."""
    try:
        import sounddevice as sd
        devices = sd.query_devices()
        inputs = [d for d in devices if d.get("max_input_channels", 0) > 0]
        outputs = [d for d in devices if d.get("max_output_channels", 0) > 0]
        return "pass", f"{len(inputs)} input(s), {len(outputs)} output(s)"
    except Exception as e:
        return "warn", f"Audio not available: {e}"


def check_tts() -> tuple[str, str]:
    """Check TTS engine availability."""
    from claw.config import get_settings, PROJECT_ROOT
    cfg = get_settings().tts
    if not cfg.enabled:
        return "warn", "TTS disabled"
    if cfg.engine == "piper":
        model_path = PROJECT_ROOT / cfg.piper_model
        if model_path.exists():
            size_mb = model_path.stat().st_size / 1_000_000
            return "pass", f"Piper model: {model_path.name} ({size_mb:.1f}MB)"
        return "fail", f"Piper model not found: {model_path}"
    elif cfg.engine == "fish_speech":
        import urllib.request
        try:
            urllib.request.urlopen(cfg.fish_speech_url, timeout=3)
            return "pass", f"Fish Speech at {cfg.fish_speech_url}"
        except Exception:
            return "fail", f"Cannot reach Fish Speech at {cfg.fish_speech_url}"
    return "warn", f"Unknown TTS engine: {cfg.engine}"


def check_wake_models() -> tuple[str, str]:
    """Check wake word models exist."""
    from claw.config import get_settings, PROJECT_ROOT
    cfg = get_settings().wake
    found = []
    for model in cfg.model_paths:
        # Check both as direct path and in custom models dir
        custom_dir = PROJECT_ROOT / cfg.custom_models_dir
        if (custom_dir / f"{model}.onnx").exists():
            found.append(model)
        elif (custom_dir / model).exists():
            found.append(model)
        else:
            # openwakeword may have built-in models, so just note it
            found.append(f"{model} (built-in)")
    return "pass", f"{len(found)} model(s): {', '.join(found)}"


def check_memory() -> tuple[str, str]:
    """Check ChromaDB and embedding model."""
    from claw.config import get_settings, PROJECT_ROOT
    cfg = get_settings().memory
    db_path = PROJECT_ROOT / cfg.chroma_path
    if db_path.exists():
        # Count files in ChromaDB
        files = list(db_path.rglob("*"))
        size_mb = sum(f.stat().st_size for f in files if f.is_file()) / 1_000_000
        return "pass", f"ChromaDB at {db_path} ({size_mb:.1f}MB, {len(files)} files)"
    return "warn", f"ChromaDB not initialized yet ({db_path})"


def check_disk_space() -> tuple[str, str]:
    """Check available disk space."""
    from claw.config import PROJECT_ROOT
    usage = shutil.disk_usage(str(PROJECT_ROOT))
    free_gb = usage.free / (1024**3)
    total_gb = usage.total / (1024**3)
    pct_free = (usage.free / usage.total) * 100
    if free_gb < 1:
        return "fail", f"{free_gb:.1f}GB free of {total_gb:.0f}GB ({pct_free:.0f}% free)"
    elif free_gb < 5:
        return "warn", f"{free_gb:.1f}GB free of {total_gb:.0f}GB ({pct_free:.0f}% free)"
    return "pass", f"{free_gb:.1f}GB free of {total_gb:.0f}GB ({pct_free:.0f}% free)"


def check_gpu() -> tuple[str, str]:
    """Detect GPU availability."""
    # NVIDIA
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=name,memory.total", "--format=csv,noheader"],
            capture_output=True, text=True, timeout=5,
        )
        if result.returncode == 0:
            gpus = result.stdout.strip().split("\n")
            return "pass", f"NVIDIA: {'; '.join(g.strip() for g in gpus)}"
    except (FileNotFoundError, subprocess.TimeoutExpired):
        pass
    # AMD ROCm
    if Path("/sys/class/kfd/kfd/topology/nodes").exists():
        return "pass", "AMD ROCm available"
    # Vulkan
    try:
        result = subprocess.run(
            ["vulkaninfo", "--summary"],
            capture_output=True, text=True, timeout=5,
        )
        if result.returncode == 0 and "GPU" in result.stdout:
            return "pass", "Vulkan GPU available"
    except (FileNotFoundError, subprocess.TimeoutExpired):
        pass
    return "warn", "No GPU detected (using CPU)"


def check_permissions() -> tuple[str, str]:
    """Check file permissions on sensitive directories."""
    from claw.config import PROJECT_ROOT
    issues = []
    secrets_dir = PROJECT_ROOT / "data" / "secrets"
    if secrets_dir.exists():
        mode_int = secrets_dir.stat().st_mode & 0o777
        if mode_int != 0o700:
            issues.append(f"data/secrets/ is {oct(mode_int)} (should be 0o700)")
    config_file = PROJECT_ROOT / "config.yaml"
    if config_file.exists():
        mode_int = config_file.stat().st_mode & 0o777
        if mode_int & 0o077:
            issues.append(f"config.yaml is world-readable ({oct(mode_int)})")
    if issues:
        return "warn", "; ".join(issues)
    return "pass", "Permissions OK"


def check_mcp_tools() -> tuple[str, str]:
    """Check MCP tool server scripts exist."""
    from claw.config import get_settings, PROJECT_ROOT
    cfg = get_settings().mcp
    tools_dir = PROJECT_ROOT / cfg.tools_dir
    if not tools_dir.exists():
        return "fail", f"Tools directory not found: {tools_dir}"
    found = []
    missing = []
    for server_name in cfg.enabled_servers:
        server_file = tools_dir / server_name / "server.py"
        if server_file.exists():
            found.append(server_name)
        else:
            missing.append(server_name)
    detail = f"{len(found)} found"
    if missing:
        detail += f", {len(missing)} missing: {', '.join(missing)}"
        return "warn", detail
    return "pass", detail


def check_python() -> tuple[str, str]:
    """Check Python version and key dependencies."""
    v = sys.version_info
    deps = []
    for mod in ["openai", "fastapi", "uvicorn", "chromadb", "yaml"]:
        try:
            __import__(mod)
            deps.append(mod)
        except ImportError:
            pass
    return "pass", f"Python {v.major}.{v.minor}.{v.micro}, {len(deps)} key deps loaded"


def run_doctor() -> int:
    """Run all diagnostics and print report. Returns exit code (0=pass, 1=fail)."""
    print("\n  The Claw \u2014 System Diagnostics\n")

    checks = [
        _timed_check("Python", check_python),
        _timed_check("Config", check_config),
        _timed_check("LLM Server", check_llm),
        _timed_check("Audio Devices", check_audio_devices),
        _timed_check("TTS Engine", check_tts),
        _timed_check("Wake Models", check_wake_models),
        _timed_check("Memory (ChromaDB)", check_memory),
        _timed_check("MCP Tools", check_mcp_tools),
        _timed_check("Disk Space", check_disk_space),
        _timed_check("GPU", check_gpu),
        _timed_check("Permissions", check_permissions),
    ]

    # Symbols and colors
    symbols = {"pass": "\033[32m\u2713\033[0m", "fail": "\033[31m\u2717\033[0m", "warn": "\033[33m!\033[0m"}
    has_fail = False

    for c in checks:
        sym = symbols.get(c.status, "?")
        timing = f"({c.elapsed_ms:.0f}ms)" if c.elapsed_ms > 10 else ""
        print(f"  {sym}  {c.name:.<22s} {c.detail} {timing}")
        if c.status == "fail":
            has_fail = True

    # Summary
    passes = sum(1 for c in checks if c.status == "pass")
    warns = sum(1 for c in checks if c.status == "warn")
    fails = sum(1 for c in checks if c.status == "fail")
    print(f"\n  {passes} passed, {warns} warnings, {fails} failed\n")

    return 1 if has_fail else 0
