"""Tests for claw.audio_pipeline.transcriber — Transcriber (faster-whisper)."""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import numpy as np
import pytest


@pytest.fixture()
def transcriber(settings):
    with patch("claw.audio_pipeline.transcriber.WhisperModel"):
        from claw.audio_pipeline.transcriber import Transcriber

        t = Transcriber()
        yield t


class TestInit:
    """Test Transcriber initialization."""

    def test_default_model_size(self, transcriber):
        assert transcriber._model_size == "base"

    def test_default_compute_type(self, transcriber):
        assert transcriber._compute_type == "int8"

    def test_model_not_loaded_initially(self, transcriber):
        assert transcriber._model is None


class TestLoad:
    """Test model loading."""

    def test_load_creates_model(self, settings):
        with patch("claw.audio_pipeline.transcriber.WhisperModel") as MockModel:
            from claw.audio_pipeline.transcriber import Transcriber

            t = Transcriber()
            t.load()
            MockModel.assert_called_once_with("base", device="cpu", compute_type="int8")
            assert t._model is not None


class TestTranscribeSync:
    """Test synchronous transcription."""

    def test_transcribe_sync_raises_if_not_loaded(self, transcriber):
        audio = np.zeros(16000, dtype=np.float32)
        with pytest.raises(RuntimeError, match="not loaded"):
            transcriber._transcribe_sync(audio)

    def test_transcribe_sync_returns_text(self, transcriber):
        mock_model = MagicMock()
        segment1 = SimpleNamespace(text="  Hello world  ")
        segment2 = SimpleNamespace(text="  how are you  ")
        info = SimpleNamespace(language="en", language_probability=0.99)
        mock_model.transcribe.return_value = ([segment1, segment2], info)
        transcriber._model = mock_model

        audio = np.zeros(16000, dtype=np.float32)
        result = transcriber._transcribe_sync(audio)
        assert result == "Hello world how are you"

    def test_transcribe_sync_uses_vad_filter(self, transcriber):
        mock_model = MagicMock()
        mock_model.transcribe.return_value = ([], SimpleNamespace(language="en", language_probability=0.9))
        transcriber._model = mock_model

        transcriber._transcribe_sync(np.zeros(16000, dtype=np.float32))
        call_kwargs = mock_model.transcribe.call_args
        assert call_kwargs.kwargs.get("vad_filter") is True or call_kwargs[1].get("vad_filter") is True


class TestTranscribeAsync:
    """Test async transcription wrapper."""

    async def test_transcribe_delegates_to_sync(self, transcriber):
        mock_model = MagicMock()
        segment = SimpleNamespace(text="async test")
        info = SimpleNamespace(language="en", language_probability=0.95)
        mock_model.transcribe.return_value = ([segment], info)
        transcriber._model = mock_model

        audio = np.zeros(16000, dtype=np.float32)
        result = await transcriber.transcribe(audio)
        assert result == "async test"

    async def test_transcribe_empty_segments(self, transcriber):
        mock_model = MagicMock()
        info = SimpleNamespace(language="en", language_probability=0.5)
        mock_model.transcribe.return_value = ([], info)
        transcriber._model = mock_model

        result = await transcriber.transcribe(np.zeros(16000, dtype=np.float32))
        assert result == ""


class TestConfigReload:
    """Test config reload handler."""

    def test_reload_updates_language(self, transcriber, settings):
        settings.whisper.language = "es"
        settings.whisper.beam_size = 3
        settings.whisper.model_size = "base"
        settings.whisper.compute_type = "int8"

        transcriber._on_config_reload(settings)
        assert transcriber._language == "es"
        assert transcriber._beam_size == 3

    def test_reload_triggers_model_reload_on_size_change(self, transcriber, settings):
        settings.whisper.model_size = "large-v2"
        settings.whisper.compute_type = "int8"
        settings.whisper.language = "en"
        settings.whisper.beam_size = 5

        with patch.object(transcriber, "load") as mock_load:
            transcriber._on_config_reload(settings)
            mock_load.assert_called_once()

    def test_reload_no_model_reload_when_unchanged(self, transcriber, settings):
        settings.whisper.model_size = "base"
        settings.whisper.compute_type = "int8"
        settings.whisper.language = "fr"
        settings.whisper.beam_size = 5

        with patch.object(transcriber, "load") as mock_load:
            transcriber._on_config_reload(settings)
            mock_load.assert_not_called()
