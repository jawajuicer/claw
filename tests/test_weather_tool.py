"""Tests for mcp_tools/weather/server.py — Weather MCP tool."""

from __future__ import annotations

from unittest.mock import patch

import pytest


# Reset module-level caches between tests
@pytest.fixture(autouse=True)
def _reset_weather_module():
    import mcp_tools.weather.server as ws

    ws._config = None
    ws._IP_LOCATION = None
    yield
    ws._config = None
    ws._IP_LOCATION = None


class TestLoadConfig:
    """Test lazy config loading."""

    def test_no_config_file(self, tmp_path):
        import mcp_tools.weather.server as ws

        ws._CONFIG_YAML = tmp_path / "nonexistent.yaml"
        ws._config = None
        cfg = ws._load_config()
        assert cfg == {}

    def test_reads_weather_section(self, tmp_path):
        import yaml
        import mcp_tools.weather.server as ws

        cfg_file = tmp_path / "config.yaml"
        cfg_file.write_text(yaml.dump({"weather": {"api_key": "test123", "default_location": "Akron, OH"}}))
        ws._CONFIG_YAML = cfg_file
        ws._config = None
        cfg = ws._load_config()
        assert cfg["api_key"] == "test123"
        assert cfg["default_location"] == "Akron, OH"

    def test_caches_config(self, tmp_path):
        import mcp_tools.weather.server as ws

        ws._config = {"api_key": "cached"}
        cfg = ws._load_config()
        assert cfg["api_key"] == "cached"


class TestGetDefaultLocation:
    """Test location resolution: config > IP geolocation."""

    def test_uses_configured_location(self):
        import mcp_tools.weather.server as ws

        ws._config = {"default_location": "New York"}
        loc = ws._get_default_location()
        assert loc == "New York"

    def test_falls_back_to_ip_geolocation(self):
        import mcp_tools.weather.server as ws

        ws._config = {"default_location": ""}
        with patch.object(ws, "_geolocate_by_ip", return_value="Akron, Ohio"):
            loc = ws._get_default_location()
            assert loc == "Akron, Ohio"


class TestGetWeather:
    """Test the get_weather MCP tool."""

    def test_no_location_and_no_default(self):
        import mcp_tools.weather.server as ws

        ws._config = {"default_location": ""}
        ws._IP_LOCATION = ""
        result = ws.get_weather("")
        assert "could not determine" in result.lower()

    def test_with_owm_key_uses_owm(self):
        import mcp_tools.weather.server as ws

        ws._config = {"api_key": "fake_key", "default_location": ""}
        with patch.object(ws, "_owm_current", return_value="Sunny in Akron") as mock_owm:
            result = ws.get_weather("Akron, OH")
            assert result == "Sunny in Akron"
            mock_owm.assert_called_once_with("Akron, OH", "fake_key")

    def test_owm_failure_falls_back_to_open_meteo(self):
        import mcp_tools.weather.server as ws

        ws._config = {"api_key": "fake_key", "default_location": ""}
        with (
            patch.object(ws, "_owm_current", side_effect=Exception("OWM down")),
            patch.object(ws, "_om_current", return_value="Cloudy from OM"),
        ):
            result = ws.get_weather("Akron, OH")
            assert result == "Cloudy from OM"

    def test_no_api_key_uses_open_meteo(self):
        import mcp_tools.weather.server as ws

        ws._config = {"api_key": "", "default_location": ""}
        with patch.object(ws, "_om_current", return_value="Weather data"):
            result = ws.get_weather("London")
            assert result == "Weather data"

    def test_all_apis_fail(self):
        import mcp_tools.weather.server as ws

        ws._config = {"api_key": "", "default_location": ""}
        with patch.object(ws, "_om_current", side_effect=Exception("all down")):
            result = ws.get_weather("Akron, OH")
            assert "failed" in result.lower()


class TestGetForecast:
    """Test the get_forecast MCP tool."""

    def test_days_clamped_to_range(self):
        import mcp_tools.weather.server as ws

        ws._config = {"api_key": "", "default_location": ""}
        with patch.object(ws, "_om_forecast", return_value="forecast") as mock:
            ws.get_forecast("Akron", days=10)
            # days should be clamped to 5
            mock.assert_called_once_with("Akron", 5)

    def test_days_minimum_is_one(self):
        import mcp_tools.weather.server as ws

        ws._config = {"api_key": "", "default_location": ""}
        with patch.object(ws, "_om_forecast", return_value="forecast") as mock:
            ws.get_forecast("Akron", days=0)
            mock.assert_called_once_with("Akron", 1)

    def test_no_location_returns_error(self):
        import mcp_tools.weather.server as ws

        ws._config = {"default_location": ""}
        ws._IP_LOCATION = ""
        result = ws.get_forecast("", days=3)
        assert "could not determine" in result.lower()


class TestWMOCodes:
    """Test WMO weather code mapping."""

    def test_known_codes(self):
        import mcp_tools.weather.server as ws

        assert ws._WMO_CODES[0] == "clear sky"
        assert ws._WMO_CODES[95] == "thunderstorm"
        assert ws._WMO_CODES[71] == "slight snowfall"

    def test_unknown_code_fallback(self):
        import mcp_tools.weather.server as ws

        assert ws._WMO_CODES.get(999, "unknown") == "unknown"
