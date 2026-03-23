"""Compute backend detection, switching, and llama.cpp rebuild orchestration."""

from __future__ import annotations

import asyncio
import logging
import re
import shutil
import subprocess
from pathlib import Path
from typing import Any, Callable

log = logging.getLogger(__name__)

LLAMA_CPP_DIR = Path("/opt/llama.cpp")

# Backend → cmake flags for llama.cpp build
CMAKE_FLAGS: dict[str, list[str]] = {
    "cpu": [],
    "cuda": ["-DGGML_CUDA=ON"],
    "rocm": ["-DGGML_ROCM=ON"],
    "vulkan": ["-DGGML_VULKAN=ON"],
}

# Backend → apt packages required
APT_PACKAGES: dict[str, list[str]] = {
    "cuda": ["nvidia-cuda-toolkit"],
    "rocm": ["rocm-dev"],
    "vulkan": ["vulkan-tools", "libvulkan-dev"],
}

VALID_BACKENDS = {"cpu", "cuda", "rocm", "vulkan"}

# Module-level build lock to prevent concurrent builds
_build_lock = asyncio.Lock()
_build_running = False
_build_progress: list[dict[str, Any]] = []


def _run(cmd: list[str], **kwargs) -> subprocess.CompletedProcess:
    """Run a subprocess, capturing output."""
    return subprocess.run(cmd, capture_output=True, text=True, timeout=600, **kwargs)


def _dpkg_installed(package: str) -> bool:
    """Check if a dpkg package is installed."""
    try:
        r = _run(["dpkg", "-s", package])
        return r.returncode == 0 and "Status: install ok installed" in r.stdout
    except (subprocess.TimeoutExpired, FileNotFoundError):
        return False


def detect_all() -> dict[str, dict[str, Any]]:
    """Probe for all available compute backends.

    Returns a dict keyed by backend name with:
      - available: bool (hardware present)
      - name: str (GPU name or "CPU")
      - info: str (extra info like VRAM)
      - deps_installed: bool (required packages installed)
    """
    result: dict[str, dict[str, Any]] = {}

    # CPU — always available
    result["cpu"] = {
        "available": True,
        "name": "CPU",
        "info": "Always available",
        "deps_installed": True,
    }

    # NVIDIA CUDA
    try:
        r = _run(["nvidia-smi", "--query-gpu=name,memory.total", "--format=csv,noheader"])
        if r.returncode == 0 and r.stdout.strip():
            parts = r.stdout.strip().split(",", 1)
            gpu_name = parts[0].strip()
            gpu_vram = parts[1].strip() if len(parts) > 1 else ""
            result["cuda"] = {
                "available": True,
                "name": gpu_name,
                "info": gpu_vram,
                "deps_installed": _dpkg_installed("nvidia-cuda-toolkit"),
            }
        else:
            result["cuda"] = {"available": False, "name": "", "info": "", "deps_installed": False}
    except (FileNotFoundError, subprocess.TimeoutExpired):
        result["cuda"] = {"available": False, "name": "", "info": "", "deps_installed": False}

    # AMD ROCm
    rocm_available = False
    rocm_name = ""
    kfd_nodes = Path("/sys/class/kfd/kfd/topology/nodes")
    if kfd_nodes.exists():
        for node in sorted(kfd_nodes.iterdir()):
            props = node / "properties"
            if props.exists():
                content = props.read_text()
                if "gfx_target_version" in content:
                    for line in content.splitlines():
                        if line.startswith("gfx_target_version"):
                            ver = line.split()[-1]
                            if ver != "0":
                                rocm_available = True
                                rocm_name = f"AMD GPU (gfx{ver})"
                                break
            if rocm_available:
                break

    result["rocm"] = {
        "available": rocm_available,
        "name": rocm_name,
        "info": "",
        "deps_installed": _dpkg_installed("rocm-dev") if rocm_available else False,
    }

    # Vulkan (iGPU or discrete)
    vulkan_available = Path("/dev/dri/renderD128").exists()
    vulkan_name = ""
    if vulkan_available:
        try:
            r = _run(["vulkaninfo", "--summary"])
            if r.returncode == 0:
                for line in r.stdout.splitlines():
                    if "deviceName" in line:
                        vulkan_name = line.split("=")[-1].strip()
                        break
        except (FileNotFoundError, subprocess.TimeoutExpired):
            pass

    result["vulkan"] = {
        "available": vulkan_available,
        "name": vulkan_name or ("Vulkan GPU" if vulkan_available else ""),
        "info": "",
        "deps_installed": _dpkg_installed("vulkan-tools") if vulkan_available else False,
    }

    return result


def update_llama_swap_config(backend: str, gpu_layers: int) -> None:
    """Update llama-swap-config.yaml to add/remove --n-gpu-layers from model commands."""
    from claw.config import PROJECT_ROOT

    config_path = PROJECT_ROOT / "llama-swap-config.yaml"

    # Also check home directory (deployment target)
    if not config_path.exists():
        home_path = Path.home() / "claw" / "llama-swap-config.yaml"
        if home_path.exists():
            config_path = home_path

    if not config_path.exists():
        log.warning("llama-swap-config.yaml not found, skipping config update")
        return

    content = config_path.read_text()

    # Remove any existing --n-gpu-layers flags (with their values)
    content = re.sub(r'\s*--n-gpu-layers\s+\d+', '', content)

    if backend != "cpu" and gpu_layers > 0:
        # Add --n-gpu-layers before any existing flags that follow the model path
        # Pattern: lines containing llama-server ... <model-path> → append --n-gpu-layers
        def _add_gpu_layers(match: re.Match) -> str:
            line = match.group(0)
            # Insert before the closing quote or end of cmd value
            return line + f" --n-gpu-layers {gpu_layers}"

        # Match lines containing 'llama-server' command in the YAML
        content = re.sub(
            r'(llama-server\b[^\n]*\.gguf\b[^\n]*)',
            _add_gpu_layers,
            content,
        )

    config_path.write_text(content)
    log.info("Updated llama-swap config: backend=%s, gpu_layers=%s", backend, gpu_layers)


def _find_llama_swap_config() -> Path | None:
    """Locate llama-swap-config.yaml (project root or home deployment)."""
    from claw.config import PROJECT_ROOT

    config_path = PROJECT_ROOT / "llama-swap-config.yaml"
    if config_path.exists():
        return config_path
    home_path = Path.home() / "claw" / "llama-swap-config.yaml"
    if home_path.exists():
        return home_path
    return None


def scan_gguf_models() -> list[dict[str, Any]]:
    """Scan common directories for available GGUF model files."""
    search_dirs: set[Path] = set()
    search_dirs.add(Path.home() / "models")

    # Also check directories referenced in llama-swap config
    config_path = _find_llama_swap_config()
    if config_path and config_path.exists():
        content = config_path.read_text()
        for match in re.finditer(r'--model(?!-)\s+(\S+\.gguf)', content):
            parent = Path(match.group(1)).parent
            if parent.exists():
                search_dirs.add(parent)

    models = []
    seen: set[str] = set()
    for dir_path in search_dirs:
        if not dir_path.exists():
            continue
        for gguf in sorted(dir_path.glob("*.gguf")):
            if gguf.name in seen:
                continue
            seen.add(gguf.name)
            try:
                size = gguf.stat().st_size
            except OSError:
                continue
            models.append({
                "name": gguf.stem,
                "path": str(gguf),
                "size": size,
                "size_gb": round(size / 1e9, 2),
            })

    return sorted(models, key=lambda m: m["size"])


def update_speculative_config(
    enabled: bool,
    draft_model: str = "",
    draft_max: int = 16,
    main_model: str = "",
) -> None:
    """Update llama-swap-config.yaml speculative decoding and model paths.

    When enabled, adds --model-draft, --gpu-layers-draft, and --draft-max.
    If main_model is set, swaps the primary --model GGUF path (first occurrence only).
    When disabled, removes speculative flags.
    """
    config_path = _find_llama_swap_config()
    if not config_path:
        log.warning("llama-swap-config.yaml not found, skipping speculative config update")
        return

    content = config_path.read_text()

    # Remove any existing speculative flags
    content = re.sub(r'\s*--model-draft\s+\S+', '', content)
    content = re.sub(r'\s*--gpu-layers-draft\s+\d+', '', content)
    content = re.sub(r'\s*--draft-max\s+\d+', '', content)

    # Swap the primary model GGUF if specified (first --model only, not --model-draft)
    if main_model:
        content = re.sub(
            r'(--model(?![-\w])\s+)\S+\.gguf',
            rf'\g<1>{main_model}',
            content,
            count=1,
        )

    if enabled and draft_model:
        def _add_spec_flags(match: re.Match) -> str:
            line = match.group(0)
            return (
                line
                + f" --model-draft {draft_model}"
                + f" --gpu-layers-draft 99"
                + f" --draft-max {draft_max}"
            )

        if main_model:
            # Only add speculative flags to the command containing the main model
            escaped = re.escape(Path(main_model).name)
            content = re.sub(
                rf'(llama-server\b[^\n]*{escaped}[^\n]*)',
                _add_spec_flags,
                content,
            )
        else:
            content = re.sub(
                r'(llama-server\b[^\n]*\.gguf\b[^\n]*)',
                _add_spec_flags,
                content,
            )
        log.info("Speculative decoding enabled: main=%s, draft=%s, max=%d", main_model, draft_model, draft_max)
    else:
        log.info("Speculative decoding disabled")

    config_path.write_text(content)


async def switch_backend(
    target: str,
    progress_callback: Callable[[dict[str, Any]], None] | None = None,
) -> dict[str, Any]:
    """Switch the compute backend with 4 phases: deps, build, config, restart.

    Returns {"status": "ok"} on success or {"status": "error", "message": "..."} on failure.
    """
    global _build_running

    if target not in VALID_BACKENDS:
        return {"status": "error", "message": f"Invalid backend: {target}"}

    def emit(phase: str, percent: int, message: str) -> None:
        entry = {"phase": phase, "percent": percent, "message": message}
        _build_progress.append(entry)
        if progress_callback:
            progress_callback(entry)

    try:
        _build_running = True
        _build_progress.clear()

        from claw.config import get_settings
        settings = get_settings()
        gpu_layers = settings.compute.gpu_layers

        # ── Phase 1: Dependencies (0-25%) ──
        emit("deps", 0, f"Checking dependencies for {target}...")

        if target in APT_PACKAGES:
            packages = APT_PACKAGES[target]
            missing = [p for p in packages if not _dpkg_installed(p)]

            if missing:
                emit("deps", 5, f"Installing: {', '.join(missing)}")
                cmd = ["sudo", "apt-get", "install", "-y"] + missing
                try:
                    r = await asyncio.to_thread(
                        subprocess.run, cmd, capture_output=True, text=True, timeout=300
                    )
                    if r.returncode != 0:
                        # sudo may not be available without password
                        manual_cmd = f"sudo apt-get install -y {' '.join(missing)}"
                        emit("deps", 25, f"Package install failed. Run manually: {manual_cmd}")
                        return {
                            "status": "error",
                            "message": f"Package install failed. Run manually:\n{manual_cmd}\n\n{r.stderr[:500]}",
                        }
                    emit("deps", 20, "Dependencies installed")
                except subprocess.TimeoutExpired:
                    return {"status": "error", "message": "Package installation timed out"}
            else:
                emit("deps", 20, "All dependencies already installed")
        else:
            emit("deps", 20, "No additional dependencies needed")

        emit("deps", 25, "Dependencies ready")

        # ── Phase 2: Build llama.cpp (25-80%) ──
        emit("build", 25, "Preparing llama.cpp build...")

        if not LLAMA_CPP_DIR.exists():
            emit("build", 25, "llama.cpp not found at /opt/llama.cpp — skipping build")
        else:
            build_dir = LLAMA_CPP_DIR / "build"
            backup_dir = LLAMA_CPP_DIR / "build.bak"

            # Backup existing build
            if build_dir.exists():
                emit("build", 28, "Backing up current build...")
                if backup_dir.exists():
                    await asyncio.to_thread(shutil.rmtree, backup_dir)
                await asyncio.to_thread(shutil.copytree, build_dir, backup_dir)
                await asyncio.to_thread(shutil.rmtree, build_dir)

            try:
                # cmake configure
                emit("build", 30, "Running cmake configure...")
                cmake_cmd = ["sudo", "cmake", "-B", "build", "-DCMAKE_BUILD_TYPE=Release"]
                cmake_cmd.extend(CMAKE_FLAGS.get(target, []))

                r = await asyncio.to_thread(
                    subprocess.run, cmake_cmd, capture_output=True, text=True,
                    timeout=120, cwd=str(LLAMA_CPP_DIR),
                )
                if r.returncode != 0:
                    raise RuntimeError(f"cmake configure failed:\n{r.stderr[:1000]}")

                emit("build", 45, "cmake configure complete, building...")

                # cmake build
                import os
                nproc = os.cpu_count() or 4
                build_cmd = [
                    "sudo", "cmake", "--build", "build", "--config", "Release",
                    "-j", str(nproc),
                ]
                r = await asyncio.to_thread(
                    subprocess.run, build_cmd, capture_output=True, text=True,
                    timeout=600, cwd=str(LLAMA_CPP_DIR),
                )
                if r.returncode != 0:
                    raise RuntimeError(f"cmake build failed:\n{r.stderr[:1000]}")

                emit("build", 75, "Build complete")

                # Remove backup on success
                if backup_dir.exists():
                    await asyncio.to_thread(shutil.rmtree, backup_dir)
                    emit("build", 78, "Backup cleaned up")

            except Exception as exc:
                # Restore backup on failure
                if backup_dir.exists():
                    if build_dir.exists():
                        await asyncio.to_thread(shutil.rmtree, build_dir)
                    await asyncio.to_thread(shutil.move, str(backup_dir), str(build_dir))
                    emit("build", 80, "Restored previous build from backup")

                emit("build", 80, f"Build failed: {exc}")
                return {"status": "error", "message": str(exc)}

        emit("build", 80, "Build phase complete")

        # ── Phase 3: Update config (80-90%) ──
        emit("config", 82, "Updating llama-swap configuration...")

        try:
            await asyncio.to_thread(update_llama_swap_config, target, gpu_layers)

            # Apply speculative decoding config
            spec = settings.compute.speculative
            spec_model = settings.compute.speculative_model
            spec_max = settings.compute.speculative_draft_max
            spec_main = settings.compute.speculative_main_model
            await asyncio.to_thread(update_speculative_config, spec, spec_model, spec_max, spec_main)

            emit("config", 90, "Configuration updated")
        except Exception as exc:
            emit("config", 90, f"Config update failed: {exc}")
            return {"status": "error", "message": f"Config update failed: {exc}"}

        # ── Phase 4: Restart service (90-100%) ──
        emit("restart", 92, "Restarting llama-swap service...")

        try:
            r = await asyncio.to_thread(
                subprocess.run,
                ["systemctl", "--user", "restart", "llama-swap"],
                capture_output=True, text=True, timeout=30,
            )
            if r.returncode != 0:
                emit("restart", 95, f"Service restart failed (non-critical): {r.stderr[:200]}")
            else:
                emit("restart", 98, "llama-swap restarted")
        except (subprocess.TimeoutExpired, FileNotFoundError) as exc:
            emit("restart", 95, f"Service restart skipped: {exc}")

        emit("restart", 100, "Backend switch complete!")
        return {"status": "ok"}

    except Exception as exc:
        log.exception("switch_backend failed")
        emit("error", 0, f"Unexpected error: {exc}")
        return {"status": "error", "message": str(exc)}
    finally:
        _build_running = False


def is_build_running() -> bool:
    return _build_running


def get_build_progress() -> list[dict[str, Any]]:
    return list(_build_progress)
