"""Tests for claw.audio_pipeline.tts.manager — TTSManager."""

from __future__ import annotations

import io
import wave
from unittest.mock import MagicMock, patch

import numpy as np
import pytest


def _make_tts_audio(pcm_data: bytes, sample_rate: int = 16000, channels: int = 1, sample_width: int = 2):
    """Build a TTSAudio instance without triggering import side effects."""
    from claw.audio_pipeline.tts.engine import TTSAudio

    return TTSAudio(pcm_data=pcm_data, sample_rate=sample_rate, channels=channels, sample_width=sample_width)


@pytest.fixture()
def tts_manager(settings):
    from claw.audio_pipeline.tts.manager import TTSManager

    mgr = TTSManager()
    yield mgr


class TestInit:
    """Test TTSManager initialization."""

    def test_not_speaking_initially(self, tts_manager):
        assert tts_manager.is_speaking is False

    def test_no_engine_initially(self, tts_manager):
        assert tts_manager._engine is None


class TestInitialize:
    """Test engine initialization."""

    def test_disabled_tts_skips_init(self, settings, tts_manager):
        settings.tts.enabled = False
        with patch("claw.audio_pipeline.tts.manager.get_settings", return_value=settings):
            tts_manager.initialize()
        assert tts_manager._engine is None

    def test_engine_load_failure_disables_tts(self, settings, tts_manager):
        settings.tts.enabled = True

        mock_engine = MagicMock()
        mock_engine.load.side_effect = RuntimeError("model not found")

        with (
            patch("claw.audio_pipeline.tts.manager.get_settings", return_value=settings),
            patch.object(type(tts_manager), "_create_engine", staticmethod(lambda cfg: mock_engine)),
        ):
            tts_manager.initialize()
        assert tts_manager._engine is None

    def test_successful_init_sets_engine(self, settings, tts_manager):
        settings.tts.enabled = True

        mock_engine = MagicMock()

        with (
            patch("claw.audio_pipeline.tts.manager.get_settings", return_value=settings),
            patch.object(type(tts_manager), "_create_engine", staticmethod(lambda cfg: mock_engine)),
        ):
            tts_manager.initialize()
        assert tts_manager._engine is mock_engine
        mock_engine.load.assert_called_once()


class TestSpeak:
    """Test text-to-speech speak method."""

    async def test_speak_does_nothing_without_engine(self, tts_manager):
        # Should not raise
        await tts_manager.speak("Hello")

    async def test_speak_sets_and_clears_speaking_flag(self, tts_manager):
        audio = _make_tts_audio(pcm_data=b"\x00" * 3200, sample_rate=16000)
        mock_engine = MagicMock()
        mock_engine.synthesize.return_value = audio
        tts_manager._engine = mock_engine

        await tts_manager.speak("Hello")
        assert tts_manager.is_speaking is False


class TestSynthesizeWav:
    """Test WAV synthesis for admin API."""

    async def test_synthesize_wav_returns_empty_without_engine(self, tts_manager):
        result = await tts_manager.synthesize_wav("Hello")
        assert result == b""

    async def test_synthesize_wav_returns_valid_wav(self, tts_manager):
        pcm_data = np.zeros(16000, dtype=np.int16).tobytes()
        audio = _make_tts_audio(pcm_data=pcm_data, sample_rate=16000)
        mock_engine = MagicMock()
        mock_engine.synthesize.return_value = audio
        tts_manager._engine = mock_engine

        result = await tts_manager.synthesize_wav("test")
        assert len(result) > 0

        # Verify it is a valid WAV file
        buf = io.BytesIO(result)
        with wave.open(buf, "rb") as wf:
            assert wf.getnchannels() == 1
            assert wf.getsampwidth() == 2
            assert wf.getframerate() == 16000


class TestPcmToWav:
    """Test static WAV encoding helper."""

    def test_pcm_to_wav_structure(self, tts_manager):
        from claw.audio_pipeline.tts.manager import TTSManager

        pcm = np.zeros(8000, dtype=np.int16).tobytes()
        audio = _make_tts_audio(pcm_data=pcm, sample_rate=22050, channels=1, sample_width=2)
        wav_bytes = TTSManager._pcm_to_wav(audio)

        buf = io.BytesIO(wav_bytes)
        with wave.open(buf, "rb") as wf:
            assert wf.getframerate() == 22050
            assert wf.getnchannels() == 1


class TestShutdown:
    """Test engine shutdown."""

    def test_shutdown_with_engine(self, tts_manager):
        mock_engine = MagicMock()
        tts_manager._engine = mock_engine
        tts_manager.shutdown()
        mock_engine.shutdown.assert_called_once()
        assert tts_manager._engine is None

    def test_shutdown_without_engine(self, tts_manager):
        # Should not raise
        tts_manager.shutdown()


class TestConfigReload:
    """Test engine reconfiguration on config reload."""

    def test_disables_engine_on_reload(self, tts_manager, settings):
        mock_engine = MagicMock()
        tts_manager._engine = mock_engine
        settings.tts.enabled = False

        tts_manager._on_config_reload(settings)
        mock_engine.shutdown.assert_called()
        assert tts_manager._engine is None

    def test_reinitializes_on_engine_change(self, tts_manager, settings):
        mock_engine = MagicMock()
        mock_engine.engine_name = "piper"
        tts_manager._engine = mock_engine
        settings.tts.enabled = True
        settings.tts.engine = "fish_speech"

        with patch.object(tts_manager, "initialize"):
            tts_manager._on_config_reload(settings)
            mock_engine.shutdown.assert_called()
