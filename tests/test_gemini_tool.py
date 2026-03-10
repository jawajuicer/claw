"""Tests for the Gemini MCP tool server."""

from __future__ import annotations

import json
from datetime import date
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest


@pytest.fixture(autouse=True)
def _isolate_gemini(tmp_path):
    """Isolate Gemini server config and state."""
    import mcp_tools.gemini.server as gs

    gs._config = None
    gs._rate_state["date"] = ""
    gs._rate_state["requests"] = 0
    gs._rate_state["grounding_requests"] = 0

    with (
        patch.object(gs, "_PROJECT_ROOT", tmp_path),
        patch.object(gs, "_CONFIG_YAML", tmp_path / "config.yaml"),
    ):
        yield


@pytest.fixture()
def gemini_config(tmp_path):
    """Write a basic gemini config and return the path."""
    import yaml
    import mcp_tools.gemini.server as gs

    config = {
        "gemini": {
            "enabled": True,
            "model": "gemini-2.5-flash",
            "pro_model": "gemini-2.5-pro",
            "temperature": 0.7,
            "max_output_tokens": 2048,
            "web_search": True,
            "document_analysis": True,
            "image_understanding": True,
            "reasoning_fallback": True,
            "daily_request_limit": 5,
            "grounding_daily_limit": 3,
            "log_requests": True,
            "log_dir": "data/gemini/logs",
        }
    }
    config_path = tmp_path / "config.yaml"
    config_path.write_text(yaml.dump(config))
    gs._config = None  # force reload
    return config_path


@pytest.fixture()
def mock_genai():
    """Mock google.generativeai module."""
    mock_response = SimpleNamespace(text="Mock Gemini response")
    mock_model = MagicMock()
    mock_model.generate_content.return_value = mock_response

    mock_module = MagicMock()
    mock_module.GenerativeModel.return_value = mock_model
    return mock_module


class TestConfigLoading:
    def test_load_empty_config(self):
        import mcp_tools.gemini.server as gs
        cfg = gs._load_config()
        assert cfg == {}

    def test_load_gemini_config(self, gemini_config):
        import mcp_tools.gemini.server as gs
        cfg = gs._load_config()
        assert cfg["enabled"] is True
        assert cfg["model"] == "gemini-2.5-flash"

    def test_capability_check_disabled(self):
        import mcp_tools.gemini.server as gs
        gs._config = {"enabled": False, "web_search": True}
        assert gs._is_enabled("web_search") is False

    def test_capability_check_enabled(self):
        import mcp_tools.gemini.server as gs
        gs._config = {"enabled": True, "web_search": True}
        assert gs._is_enabled("web_search") is True

    def test_capability_check_specific_disabled(self):
        import mcp_tools.gemini.server as gs
        gs._config = {"enabled": True, "web_search": False}
        assert gs._is_enabled("web_search") is False


class TestRateLimits:
    def test_under_limit(self, gemini_config):
        import mcp_tools.gemini.server as gs
        assert gs._check_rate_limit() is None

    def test_request_limit_exceeded(self, gemini_config):
        import mcp_tools.gemini.server as gs
        for _ in range(5):
            gs._record_request()
        err = gs._check_rate_limit()
        assert err is not None
        assert "Daily request limit" in err

    def test_grounding_limit_exceeded(self, gemini_config):
        import mcp_tools.gemini.server as gs
        for _ in range(3):
            gs._record_request(grounding=True)
        err = gs._check_rate_limit(grounding=True)
        assert err is not None
        assert "grounding limit" in err

    def test_counters_reset_at_midnight(self, gemini_config):
        import mcp_tools.gemini.server as gs
        gs._rate_state["date"] = "2020-01-01"
        gs._rate_state["requests"] = 999
        # After checking, it should reset because date changed
        gs._record_request()
        assert gs._rate_state["requests"] == 1


class TestWebSearch:
    def test_not_configured(self, gemini_config):
        import mcp_tools.gemini.server as gs
        with (
            patch.object(gs, "_get_api_key", return_value=None),
            patch.object(gs, "_get_genai", return_value=None),
        ):
            result = gs.gemini_web_search("test query")
        assert "not configured" in result.lower()

    def test_disabled_capability(self):
        import mcp_tools.gemini.server as gs
        gs._config = {"enabled": True, "web_search": False}
        result = gs.gemini_web_search("test query")
        assert "disabled" in result.lower()

    def test_success(self, gemini_config, mock_genai):
        import mcp_tools.gemini.server as gs
        with (
            patch.object(gs, "_get_api_key", return_value="AIzaTestKey12345678"),
            patch.object(gs, "_get_genai", return_value=mock_genai),
        ):
            result = gs.gemini_web_search("latest Python news")
        assert result == "Mock Gemini response"
        assert gs._rate_state["grounding_requests"] == 1

    def test_rate_limited(self, gemini_config, mock_genai):
        import mcp_tools.gemini.server as gs
        for _ in range(3):
            gs._record_request(grounding=True)
        with (
            patch.object(gs, "_get_api_key", return_value="AIzaTestKey12345678"),
            patch.object(gs, "_get_genai", return_value=mock_genai),
        ):
            result = gs.gemini_web_search("test")
        assert "limit" in result.lower()


class TestAnalyzeDocument:
    def test_file_not_found(self, gemini_config, mock_genai):
        import mcp_tools.gemini.server as gs
        with (
            patch.object(gs, "_get_api_key", return_value="AIzaTestKey12345678"),
            patch.object(gs, "_get_genai", return_value=mock_genai),
        ):
            result = gs.gemini_analyze_document("/nonexistent/file.txt")
        assert "not found" in result.lower()

    def test_success(self, tmp_path, gemini_config, mock_genai):
        import mcp_tools.gemini.server as gs
        doc = tmp_path / "test.txt"
        doc.write_text("This is a test document with important content.")

        with (
            patch.object(gs, "_get_api_key", return_value="AIzaTestKey12345678"),
            patch.object(gs, "_get_genai", return_value=mock_genai),
        ):
            result = gs.gemini_analyze_document(str(doc), "Summarize this")
        assert result == "Mock Gemini response"


class TestDescribeImage:
    def test_image_not_found(self, gemini_config, mock_genai):
        import mcp_tools.gemini.server as gs
        with (
            patch.object(gs, "_get_api_key", return_value="AIzaTestKey12345678"),
            patch.object(gs, "_get_genai", return_value=mock_genai),
        ):
            result = gs.gemini_describe_image("/nonexistent/image.png")
        assert "not found" in result.lower()

    def test_success(self, tmp_path, gemini_config, mock_genai):
        import mcp_tools.gemini.server as gs
        img = tmp_path / "test.png"
        img.write_bytes(b"\x89PNG\r\n\x1a\n" + b"\x00" * 100)

        with (
            patch.object(gs, "_get_api_key", return_value="AIzaTestKey12345678"),
            patch.object(gs, "_get_genai", return_value=mock_genai),
        ):
            result = gs.gemini_describe_image(str(img), "What is this?")
        assert result == "Mock Gemini response"


class TestAsk:
    def test_gemini_ask_success(self, gemini_config, mock_genai):
        import mcp_tools.gemini.server as gs
        with (
            patch.object(gs, "_get_api_key", return_value="AIzaTestKey12345678"),
            patch.object(gs, "_get_genai", return_value=mock_genai),
        ):
            result = gs.gemini_ask("What is the capital of France?")
        assert result == "Mock Gemini response"
        assert gs._rate_state["requests"] == 1

    def test_gemini_ask_not_configured(self, gemini_config):
        import mcp_tools.gemini.server as gs
        with (
            patch.object(gs, "_get_api_key", return_value=None),
            patch.object(gs, "_get_genai", return_value=None),
        ):
            result = gs.gemini_ask("test question")
        assert "not configured" in result.lower()

    def test_gemini_ask_disabled(self):
        import mcp_tools.gemini.server as gs
        gs._config = {"enabled": False}
        result = gs.gemini_ask("test question")
        assert "disabled" in result.lower()

    def test_gemini_ask_rate_limited(self, gemini_config, mock_genai):
        import mcp_tools.gemini.server as gs
        for _ in range(5):
            gs._record_request()
        result = gs.gemini_ask("test")
        assert "limit" in result.lower()


class TestReason:
    def test_disabled(self):
        import mcp_tools.gemini.server as gs
        gs._config = {"enabled": True, "reasoning_fallback": False}
        result = gs.gemini_reason("Why is the sky blue?")
        assert "disabled" in result.lower()

    def test_success(self, gemini_config, mock_genai):
        import mcp_tools.gemini.server as gs
        with (
            patch.object(gs, "_get_api_key", return_value="AIzaTestKey12345678"),
            patch.object(gs, "_get_genai", return_value=mock_genai),
        ):
            result = gs.gemini_reason("Explain quantum entanglement")
        assert result == "Mock Gemini response"


class TestUsage:
    def test_no_usage(self, gemini_config):
        import mcp_tools.gemini.server as gs
        with patch.object(gs, "_get_api_key", return_value=None):
            result = gs.gemini_usage()
        assert "0/" in result
        assert "Not configured" in result

    def test_with_usage(self, gemini_config):
        import mcp_tools.gemini.server as gs
        gs._record_request()
        gs._record_request(grounding=True)
        with patch.object(gs, "_get_api_key", return_value="AIzaTestKey12345678"):
            result = gs.gemini_usage()
        assert "2/5" in result  # 2 requests out of 5 limit
        assert "AIza" in result


class TestLogging:
    def test_log_written(self, tmp_path, gemini_config, mock_genai):
        import mcp_tools.gemini.server as gs
        with (
            patch.object(gs, "_get_api_key", return_value="AIzaTestKey12345678"),
            patch.object(gs, "_get_genai", return_value=mock_genai),
        ):
            gs.gemini_web_search("test query")

        log_dir = tmp_path / "data" / "gemini" / "logs"
        today = date.today().isoformat()
        log_file = log_dir / f"{today}.jsonl"
        assert log_file.exists()

        entries = [json.loads(line) for line in log_file.read_text().splitlines() if line.strip()]
        assert len(entries) == 1
        assert entries[0]["tool"] == "gemini_web_search"
        assert entries[0]["prompt_preview"] == "test query"

    def test_log_disabled(self, tmp_path, gemini_config, mock_genai):
        import mcp_tools.gemini.server as gs
        cfg = gs._load_config()
        cfg["log_requests"] = False
        with (
            patch.object(gs, "_get_api_key", return_value="AIzaTestKey12345678"),
            patch.object(gs, "_get_genai", return_value=mock_genai),
        ):
            gs.gemini_web_search("test query")

        log_dir = tmp_path / "data" / "gemini" / "logs"
        assert not log_dir.exists() or not list(log_dir.glob("*.jsonl"))

    def test_get_log_entries(self, tmp_path, gemini_config, mock_genai):
        import mcp_tools.gemini.server as gs
        with (
            patch.object(gs, "_get_api_key", return_value="AIzaTestKey12345678"),
            patch.object(gs, "_get_genai", return_value=mock_genai),
        ):
            gs.gemini_web_search("query 1")
            gs.gemini_web_search("query 2")

        entries = gs._get_log_entries()
        assert len(entries) == 2

    def test_wipe_logs(self, tmp_path, gemini_config, mock_genai):
        import mcp_tools.gemini.server as gs
        with (
            patch.object(gs, "_get_api_key", return_value="AIzaTestKey12345678"),
            patch.object(gs, "_get_genai", return_value=mock_genai),
        ):
            gs.gemini_web_search("test")

        count = gs._wipe_logs()
        assert count == 1
        assert gs._get_log_entries() == []
