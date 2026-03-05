"""Shared fixtures for The Claw test suite."""

from __future__ import annotations

# ---------------------------------------------------------------------------
# Pre-import mocks for C-level dependencies that may not be available
# in CI or headless environments (PortAudio, ONNX, etc.).
# These MUST happen before any claw module is imported.
# ---------------------------------------------------------------------------
import sys
from unittest.mock import MagicMock

# sounddevice requires PortAudio — mock it if unavailable
if "sounddevice" not in sys.modules:
    _mock_sd = MagicMock()
    _mock_sd.InputStream = MagicMock()
    _mock_sd.play = MagicMock()
    _mock_sd.wait = MagicMock()
    _mock_sd.default = MagicMock()
    _mock_sd.default.device = (0, 0)
    _mock_sd.query_devices = MagicMock(return_value=[])
    _mock_sd.CallbackFlags = type("CallbackFlags", (), {"__bool__": lambda self: False})
    sys.modules["sounddevice"] = _mock_sd

# openwakeword + its sub-modules
for _mod in ("openwakeword", "openwakeword.model", "openwakeword.utils"):
    if _mod not in sys.modules:
        sys.modules[_mod] = MagicMock()

# faster-whisper
if "faster_whisper" not in sys.modules:
    sys.modules["faster_whisper"] = MagicMock()

# ---------------------------------------------------------------------------

import json  # noqa: E402
from types import SimpleNamespace  # noqa: E402
from unittest.mock import AsyncMock, MagicMock, patch  # noqa: E402

import pytest  # noqa: E402


# ---------------------------------------------------------------------------
# Settings / config fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(autouse=True)
def _isolate_settings():
    """Reset the settings singleton between tests so no test leaks state."""
    import claw.config as cfg_mod

    original = cfg_mod._settings
    cfg_mod._settings = None
    original_callbacks = cfg_mod._reload_callbacks[:]
    cfg_mod._reload_callbacks.clear()
    yield
    cfg_mod._settings = original
    cfg_mod._reload_callbacks[:] = original_callbacks


@pytest.fixture()
def tmp_config(tmp_path):
    """Provide a temporary config.yaml and patch PROJECT_ROOT / CONFIG_YAML."""
    import claw.config as cfg_mod

    config_yaml = tmp_path / "config.yaml"
    config_yaml.write_text("")

    with (
        patch.object(cfg_mod, "PROJECT_ROOT", tmp_path),
        patch.object(cfg_mod, "CONFIG_YAML", config_yaml),
    ):
        yield config_yaml


@pytest.fixture()
def settings(tmp_config):
    """Return a Settings instance backed by a temporary (empty) config.yaml."""
    from claw.config import Settings

    return Settings.load()


# ---------------------------------------------------------------------------
# Mock LLM client
# ---------------------------------------------------------------------------

def _make_chat_response(content: str = "Hello!", tool_calls=None):
    """Build a fake OpenAI ChatCompletion response."""
    message = SimpleNamespace(
        content=content,
        tool_calls=tool_calls,
        model_dump=lambda: {
            "role": "assistant",
            "content": content,
            "tool_calls": (
                [
                    {
                        "id": tc.id,
                        "type": "function",
                        "function": {
                            "name": tc.function.name,
                            "arguments": tc.function.arguments,
                        },
                    }
                    for tc in tool_calls
                ]
                if tool_calls
                else None
            ),
        },
    )
    choice = SimpleNamespace(message=message)
    usage = SimpleNamespace(prompt_tokens=10, completion_tokens=5, total_tokens=15)
    return SimpleNamespace(choices=[choice], usage=usage)


def _make_tool_call(call_id: str, name: str, arguments: dict | None = None):
    """Build a fake tool_call object matching the OpenAI SDK shape."""
    return SimpleNamespace(
        id=call_id,
        type="function",
        function=SimpleNamespace(
            name=name,
            arguments=json.dumps(arguments or {}),
        ),
    )


@pytest.fixture()
def mock_llm():
    """AsyncMock of LLMClient with sensible defaults."""
    llm = AsyncMock()
    llm.chat = AsyncMock(return_value=_make_chat_response("Test response"))
    llm.chat_simple = AsyncMock(return_value="NONE")
    llm.busy = False
    return llm


# ---------------------------------------------------------------------------
# Mock memory store / retriever
# ---------------------------------------------------------------------------

@pytest.fixture()
def mock_memory_store():
    store = MagicMock()
    store.conversations = MagicMock()
    store.facts = MagicMock()
    store.categories = MagicMock()
    store.query_facts.return_value = []
    store.query_conversations.return_value = []
    store.query_categories.return_value = []
    store.stats.return_value = {"conversations": 0, "facts": 0, "categories": 0}
    return store


@pytest.fixture()
def mock_retriever(mock_memory_store):
    from claw.memory_engine.retriever import MemoryRetriever

    retriever = MemoryRetriever(mock_memory_store)
    return retriever


# ---------------------------------------------------------------------------
# Mock tool router
# ---------------------------------------------------------------------------

@pytest.fixture()
def mock_tool_router():
    router = AsyncMock()
    router.call_tool = AsyncMock(return_value="tool result")
    return router


# ---------------------------------------------------------------------------
# Helpers exposed to test modules
# ---------------------------------------------------------------------------

@pytest.fixture()
def make_chat_response():
    """Expose the helper so tests can build custom responses."""
    return _make_chat_response


@pytest.fixture()
def make_tool_call():
    """Expose the tool-call builder."""
    return _make_tool_call
