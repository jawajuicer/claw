"""Tests for claw.compute — hardware detection, config update, and build orchestration."""

from __future__ import annotations

import subprocess
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest


class TestDetectAll:
    """Test detect_all() with mocked subprocess calls."""

    def test_cpu_always_available(self):
        from claw.compute import detect_all

        with patch("claw.compute._run", side_effect=FileNotFoundError):
            result = detect_all()

        assert result["cpu"]["available"] is True
        assert result["cpu"]["deps_installed"] is True

    def test_nvidia_gpu_detected(self):
        from claw.compute import detect_all

        def mock_run(cmd, **kwargs):
            if "nvidia-smi" in cmd:
                r = MagicMock()
                r.returncode = 0
                r.stdout = "NVIDIA GeForce RTX 4070, 12288 MiB"
                return r
            if "dpkg" in cmd:
                r = MagicMock()
                if "nvidia-cuda-toolkit" in cmd:
                    r.returncode = 0
                    r.stdout = "Status: install ok installed"
                else:
                    r.returncode = 1
                    r.stdout = ""
                return r
            r = MagicMock()
            r.returncode = 1
            r.stdout = ""
            return r

        with patch("claw.compute._run", side_effect=mock_run), \
             patch("claw.compute.Path.exists", return_value=False):
            result = detect_all()

        assert result["cuda"]["available"] is True
        assert result["cuda"]["name"] == "NVIDIA GeForce RTX 4070"
        assert result["cuda"]["info"] == "12288 MiB"
        assert result["cuda"]["deps_installed"] is True

    def test_nvidia_not_present(self):
        from claw.compute import detect_all

        def mock_run(cmd, **kwargs):
            if "nvidia-smi" in cmd:
                raise FileNotFoundError
            r = MagicMock()
            r.returncode = 1
            r.stdout = ""
            return r

        with patch("claw.compute._run", side_effect=mock_run), \
             patch("claw.compute.Path.exists", return_value=False):
            result = detect_all()

        assert result["cuda"]["available"] is False

    def test_vulkan_detected_with_render_node(self, tmp_path):
        from claw.compute import detect_all

        def mock_run(cmd, **kwargs):
            if "nvidia-smi" in cmd:
                raise FileNotFoundError
            if "vulkaninfo" in cmd:
                r = MagicMock()
                r.returncode = 0
                r.stdout = "deviceName = AMD Radeon Graphics"
                return r
            if "dpkg" in cmd:
                r = MagicMock()
                if "vulkan-tools" in cmd:
                    r.returncode = 0
                    r.stdout = "Status: install ok installed"
                else:
                    r.returncode = 1
                    r.stdout = ""
                return r
            r = MagicMock()
            r.returncode = 1
            r.stdout = ""
            return r

        # Mock Path("/dev/dri/renderD128").exists() to return True
        original_exists = Path.exists

        def patched_exists(self):
            if str(self) == "/dev/dri/renderD128":
                return True
            if "kfd" in str(self):
                return False
            return original_exists(self)

        with patch("claw.compute._run", side_effect=mock_run), \
             patch.object(Path, "exists", patched_exists):
            result = detect_all()

        assert result["vulkan"]["available"] is True
        assert "AMD Radeon" in result["vulkan"]["name"]


class TestUpdateLlamaSwapConfig:
    """Test regex-based llama-swap config updates."""

    def test_adds_gpu_layers_to_model_commands(self, tmp_path):
        from claw.compute import update_llama_swap_config

        config = tmp_path / "llama-swap-config.yaml"
        config.write_text(
            'models:\n'
            '  "qwen3.5:0.8b":\n'
            '    cmd: /opt/llama.cpp/build/bin/llama-server -m /models/qwen.gguf --port 5800\n'
            '  "qwen3.5:9b":\n'
            '    cmd: /opt/llama.cpp/build/bin/llama-server -m /models/big.gguf --port 5801\n'
        )

        with patch("claw.config.PROJECT_ROOT", tmp_path):
            update_llama_swap_config("cuda", 99)

        result = config.read_text()
        assert "--n-gpu-layers 99" in result
        assert result.count("--n-gpu-layers") == 2

    def test_removes_gpu_layers_for_cpu(self, tmp_path):
        from claw.compute import update_llama_swap_config

        config = tmp_path / "llama-swap-config.yaml"
        config.write_text(
            'models:\n'
            '  "model":\n'
            '    cmd: /opt/llama.cpp/build/bin/llama-server -m /models/test.gguf --n-gpu-layers 99 --port 5800\n'
        )

        with patch("claw.config.PROJECT_ROOT", tmp_path):
            update_llama_swap_config("cpu", 0)

        result = config.read_text()
        assert "--n-gpu-layers" not in result
        assert "--port 5800" in result

    def test_replaces_existing_gpu_layers(self, tmp_path):
        from claw.compute import update_llama_swap_config

        config = tmp_path / "llama-swap-config.yaml"
        config.write_text(
            'models:\n'
            '  "model":\n'
            '    cmd: /opt/llama.cpp/build/bin/llama-server -m /models/test.gguf --n-gpu-layers 50 --port 5800\n'
        )

        with patch("claw.config.PROJECT_ROOT", tmp_path):
            update_llama_swap_config("cuda", 99)

        result = config.read_text()
        assert "--n-gpu-layers 99" in result
        assert "--n-gpu-layers 50" not in result

    def test_missing_config_file_no_error(self, tmp_path):
        from claw.compute import update_llama_swap_config

        with patch("claw.config.PROJECT_ROOT", tmp_path), \
             patch("pathlib.Path.home", return_value=tmp_path / "nonexistent"):
            # Should not raise
            update_llama_swap_config("cuda", 99)


class TestDpkgInstalled:
    """Test the _dpkg_installed helper."""

    def test_installed_package(self):
        from claw.compute import _dpkg_installed

        r = MagicMock()
        r.returncode = 0
        r.stdout = "Package: foo\nStatus: install ok installed\n"

        with patch("claw.compute._run", return_value=r):
            assert _dpkg_installed("foo") is True

    def test_not_installed(self):
        from claw.compute import _dpkg_installed

        r = MagicMock()
        r.returncode = 1
        r.stdout = ""

        with patch("claw.compute._run", return_value=r):
            assert _dpkg_installed("foo") is False

    def test_dpkg_not_found(self):
        from claw.compute import _dpkg_installed

        with patch("claw.compute._run", side_effect=FileNotFoundError):
            assert _dpkg_installed("foo") is False


class TestValidBackends:
    def test_valid_backends_set(self):
        from claw.compute import VALID_BACKENDS

        assert "cpu" in VALID_BACKENDS
        assert "cuda" in VALID_BACKENDS
        assert "rocm" in VALID_BACKENDS
        assert "vulkan" in VALID_BACKENDS
        assert "metal" not in VALID_BACKENDS

    def test_cmake_flags_mapping(self):
        from claw.compute import CMAKE_FLAGS

        assert CMAKE_FLAGS["cpu"] == []
        assert "-DGGML_CUDA=ON" in CMAKE_FLAGS["cuda"]
        assert "-DGGML_ROCM=ON" in CMAKE_FLAGS["rocm"]
        assert "-DGGML_VULKAN=ON" in CMAKE_FLAGS["vulkan"]
