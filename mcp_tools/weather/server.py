"""Weather MCP server — OpenWeatherMap primary, Open-Meteo fallback."""

from __future__ import annotations

import json
import logging
import os
import urllib.request
import urllib.parse
from datetime import datetime
from pathlib import Path

from mcp.server.fastmcp import FastMCP

log = logging.getLogger(__name__)

mcp = FastMCP("Weather")

# ---------------------------------------------------------------------------
# Config — loaded lazily from config.yaml, cached for process lifetime
# ---------------------------------------------------------------------------
_config: dict | None = None

# Path to project root (server.py is at mcp_tools/weather/server.py)
_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
_CONFIG_YAML = _PROJECT_ROOT / "config.yaml"


def _load_config() -> dict:
    """Read weather config from config.yaml."""
    global _config
    if _config is not None:
        return _config
    if _CONFIG_YAML.exists():
        import yaml
        with open(_CONFIG_YAML) as f:
            data = yaml.safe_load(f) or {}
        _config = data.get("weather", {})
    else:
        _config = {}
    return _config


def _get_owm_key() -> str | None:
    """Resolve OWM API key. Priority: env var > secret store > config.yaml."""
    key = os.environ.get("OWM_API_KEY")
    if key:
        return key
    # Try encrypted secret store
    try:
        import sys
        sys.path.insert(0, str(_PROJECT_ROOT / "src"))
        from claw.secret_store import load as secret_load
        stored = secret_load("owm_api_key")
        if stored:
            return stored
    except Exception:
        pass
    # Fallback to plaintext config
    key = _load_config().get("api_key", "")
    return key or None


def _get_default_location() -> str:
    """Resolve the user's location from config."""
    return _load_config().get("default_location", "")


def _fetch_json(url: str) -> dict:
    """Fetch JSON from a URL with a short timeout."""
    req = urllib.request.Request(url, headers={"User-Agent": "TheClaw/1.0"})
    with urllib.request.urlopen(req, timeout=10) as resp:
        return json.loads(resp.read())


# ---------------------------------------------------------------------------
# OpenWeatherMap (primary)
# ---------------------------------------------------------------------------
_OWM_WEATHER_URL = "https://api.openweathermap.org/data/2.5/weather"
_OWM_FORECAST_URL = "https://api.openweathermap.org/data/2.5/forecast"
_OWM_GEO_URL = "https://api.openweathermap.org/geo/1.0/direct"


def _owm_geocode(location: str, api_key: str) -> tuple[float, float, str]:
    """Geocode using OpenWeatherMap — handles 'City, ST' format natively."""
    params = urllib.parse.urlencode({"q": location, "limit": 1, "appid": api_key})
    data = _fetch_json(f"{_OWM_GEO_URL}?{params}")
    if not data:
        raise ValueError(f"Location '{location}' not found")
    r = data[0]
    parts = [r.get("name", "")]
    if r.get("state"):
        parts.append(r["state"])
    if r.get("country"):
        parts.append(r["country"])
    return r["lat"], r["lon"], ", ".join(parts)


def _owm_current(location: str, api_key: str) -> str:
    """Current weather via OpenWeatherMap."""
    lat, lon, display_name = _owm_geocode(location, api_key)
    params = urllib.parse.urlencode({
        "lat": lat, "lon": lon,
        "appid": api_key, "units": "imperial",
    })
    data = _fetch_json(f"{_OWM_WEATHER_URL}?{params}")

    desc = data["weather"][0]["description"] if data.get("weather") else "unknown"
    main = data.get("main", {})
    wind = data.get("wind", {})

    lines = [f"Weather for {display_name}:"]
    lines.append(f"Condition: {desc}")
    if main.get("temp") is not None:
        lines.append(f"Temperature: {main['temp']:.0f}\u00b0F")
    if main.get("feels_like") is not None:
        lines.append(f"Feels like: {main['feels_like']:.0f}\u00b0F")
    if main.get("humidity") is not None:
        lines.append(f"Humidity: {main['humidity']}%")
    if wind.get("speed") is not None:
        wind_str = f"Wind: {wind['speed']:.0f} mph"
        if wind.get("gust") is not None:
            wind_str += f" (gusts {wind['gust']:.0f} mph)"
        lines.append(wind_str)
    return "\n".join(lines)


def _owm_forecast(location: str, api_key: str, days: int) -> str:
    """Multi-day forecast via OpenWeatherMap (uses 5-day/3-hour endpoint)."""
    lat, lon, display_name = _owm_geocode(location, api_key)
    params = urllib.parse.urlencode({
        "lat": lat, "lon": lon,
        "appid": api_key, "units": "imperial",
    })
    data = _fetch_json(f"{_OWM_FORECAST_URL}?{params}")

    # Group 3-hour forecasts by date, compute daily hi/lo/condition
    daily: dict[str, dict] = {}
    for entry in data.get("list", []):
        date_str = entry["dt_txt"][:10]
        if date_str not in daily:
            daily[date_str] = {"highs": [], "lows": [], "conditions": [], "wind": [], "pop": []}
        m = entry.get("main", {})
        if m.get("temp_max") is not None:
            daily[date_str]["highs"].append(m["temp_max"])
        if m.get("temp_min") is not None:
            daily[date_str]["lows"].append(m["temp_min"])
        if entry.get("weather"):
            daily[date_str]["conditions"].append(entry["weather"][0]["description"])
        if entry.get("wind", {}).get("speed") is not None:
            daily[date_str]["wind"].append(entry["wind"]["speed"])
        if entry.get("pop") is not None:
            daily[date_str]["pop"].append(entry["pop"])

    lines = [f"{days}-day forecast for {display_name}:"]
    for i, (date_str, d) in enumerate(sorted(daily.items())):
        if i >= days:
            break
        dt = datetime.strptime(date_str, "%Y-%m-%d")
        day_name = dt.strftime("%A")
        hi = f"{max(d['highs']):.0f}" if d["highs"] else "?"
        lo = f"{min(d['lows']):.0f}" if d["lows"] else "?"
        # Pick the most common condition
        cond = max(set(d["conditions"]), key=d["conditions"].count) if d["conditions"] else "unknown"
        w = f"{max(d['wind']):.0f}" if d["wind"] else "?"
        pop = f"{max(d['pop']) * 100:.0f}" if d["pop"] else "?"
        lines.append(f"{day_name} ({date_str}): {cond}, {hi}\u00b0/{lo}\u00b0F, {pop}% precip, wind {w} mph")

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Open-Meteo (fallback — no API key needed)
# ---------------------------------------------------------------------------
_OM_WEATHER_URL = "https://api.open-meteo.com/v1/forecast"
_NOMINATIM_URL = "https://nominatim.openstreetmap.org/search"

_WMO_CODES = {
    0: "clear sky", 1: "mainly clear", 2: "partly cloudy", 3: "overcast",
    45: "foggy", 48: "depositing rime fog",
    51: "light drizzle", 53: "moderate drizzle", 55: "dense drizzle",
    56: "light freezing drizzle", 57: "dense freezing drizzle",
    61: "slight rain", 63: "moderate rain", 65: "heavy rain",
    66: "light freezing rain", 67: "heavy freezing rain",
    71: "slight snowfall", 73: "moderate snowfall", 75: "heavy snowfall",
    77: "snow grains",
    80: "slight rain showers", 81: "moderate rain showers", 82: "violent rain showers",
    85: "slight snow showers", 86: "heavy snow showers",
    95: "thunderstorm", 96: "thunderstorm with slight hail", 99: "thunderstorm with heavy hail",
}


def _nominatim_geocode(location: str) -> tuple[float, float, str]:
    """Geocode via Nominatim (OpenStreetMap) — handles 'City, ST' natively."""
    params = urllib.parse.urlencode({"q": location, "format": "json", "limit": 1})
    data = _fetch_json(f"{_NOMINATIM_URL}?{params}")
    if not data:
        raise ValueError(f"Location '{location}' not found")
    r = data[0]
    display = r.get("display_name", location)
    # Shorten "Raleigh, Wake County, North Carolina, United States" → first, third, fourth
    parts = [p.strip() for p in display.split(",")]
    if len(parts) >= 4:
        display = f"{parts[0]}, {parts[-2]}, {parts[-1]}"
    elif len(parts) >= 2:
        display = f"{parts[0]}, {parts[-1]}"
    return float(r["lat"]), float(r["lon"]), display


def _om_current(location: str) -> str:
    """Current weather via Open-Meteo (geocoded by Nominatim)."""
    lat, lon, display_name = _nominatim_geocode(location)
    params = urllib.parse.urlencode({
        "latitude": lat, "longitude": lon,
        "current": "temperature_2m,relative_humidity_2m,apparent_temperature,weather_code,wind_speed_10m,wind_gusts_10m",
        "temperature_unit": "fahrenheit",
        "wind_speed_unit": "mph",
        "timezone": "auto",
    })
    data = _fetch_json(f"{_OM_WEATHER_URL}?{params}")
    current = data.get("current", {})
    code = current.get("weather_code", -1)

    lines = [f"Weather for {display_name}:"]
    lines.append(f"Condition: {_WMO_CODES.get(code, 'unknown')}")
    if current.get("temperature_2m") is not None:
        lines.append(f"Temperature: {current['temperature_2m']}\u00b0F")
    if current.get("apparent_temperature") is not None:
        lines.append(f"Feels like: {current['apparent_temperature']}\u00b0F")
    if current.get("relative_humidity_2m") is not None:
        lines.append(f"Humidity: {current['relative_humidity_2m']}%")
    if current.get("wind_speed_10m") is not None:
        wind_str = f"Wind: {current['wind_speed_10m']} mph"
        if current.get("wind_gusts_10m") is not None:
            wind_str += f" (gusts {current['wind_gusts_10m']} mph)"
        lines.append(wind_str)
    return "\n".join(lines)


def _om_forecast(location: str, days: int) -> str:
    """Multi-day forecast via Open-Meteo (geocoded by Nominatim)."""
    lat, lon, display_name = _nominatim_geocode(location)
    params = urllib.parse.urlencode({
        "latitude": lat, "longitude": lon,
        "daily": "weather_code,temperature_2m_max,temperature_2m_min,precipitation_probability_max,wind_speed_10m_max",
        "temperature_unit": "fahrenheit",
        "wind_speed_unit": "mph",
        "timezone": "auto",
        "forecast_days": days,
    })
    data = _fetch_json(f"{_OM_WEATHER_URL}?{params}")
    daily = data.get("daily", {})
    dates = daily.get("time", [])

    lines = [f"{days}-day forecast for {display_name}:"]
    for i, date in enumerate(dates):
        dt = datetime.strptime(date, "%Y-%m-%d")
        day_name = dt.strftime("%A")
        code = daily.get("weather_code", [])[i] if i < len(daily.get("weather_code", [])) else -1
        hi = daily.get("temperature_2m_max", [None])[i] or "?"
        lo = daily.get("temperature_2m_min", [None])[i] or "?"
        rain = daily.get("precipitation_probability_max", [None])[i] or "?"
        w = daily.get("wind_speed_10m_max", [None])[i] or "?"
        lines.append(f"{day_name} ({date}): {_WMO_CODES.get(code, 'unknown')}, {hi}\u00b0/{lo}\u00b0F, {rain}% precip, wind {w} mph")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# MCP tool endpoints
# ---------------------------------------------------------------------------

@mcp.tool()
def get_weather(location: str = "") -> str:
    """Get the current weather for a location. Auto-detects location if not specified.

    Args:
        location: City name, e.g. "Akron, OH", "New York", "London". Leave empty to auto-detect.
    """
    if not location:
        location = _get_default_location()
    if not location:
        return "Could not determine your location. Please say something like 'weather in Akron, Ohio'."
    api_key = _get_owm_key()
    if api_key:
        try:
            return _owm_current(location, api_key)
        except Exception as e:
            log.warning("OpenWeatherMap failed, falling back to Open-Meteo: %s", e)
    # Fallback to Open-Meteo
    try:
        return _om_current(location)
    except ValueError as e:
        return str(e)
    except Exception:
        return f"Failed to fetch weather for '{location}'"


@mcp.tool()
def get_forecast(location: str = "", days: int = 3) -> str:
    """Get the weather forecast for the next few days. Auto-detects location if not specified.

    Args:
        location: City name, e.g. "Akron, OH", "New York", "London". Leave empty to auto-detect.
        days: Number of days to forecast (1-5). Defaults to 3.
    """
    if not location:
        location = _get_default_location()
    if not location:
        return "Could not determine your location. Please specify a city."
    days = max(1, min(5, days))
    api_key = _get_owm_key()
    if api_key:
        try:
            return _owm_forecast(location, api_key, days)
        except Exception as e:
            log.warning("OpenWeatherMap forecast failed, falling back to Open-Meteo: %s", e)
    # Fallback to Open-Meteo
    try:
        return _om_forecast(location, days)
    except ValueError as e:
        return str(e)
    except Exception:
        return f"Failed to fetch forecast for '{location}'"


if __name__ == "__main__":
    mcp.run()
