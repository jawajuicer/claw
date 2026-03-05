"""Tests for claw.audio_pipeline.wake_word — WakeWordDetector."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import numpy as np
import pytest


@pytest.fixture()
def detector(settings):
    with (
        patch("claw.audio_pipeline.wake_word.openwakeword"),
        patch("claw.audio_pipeline.wake_word.OWWModel"),
    ):
        from claw.audio_pipeline.wake_word import WakeWordDetector

        det = WakeWordDetector()
        yield det


class TestInit:
    """Test detector initialization."""

    def test_initial_state(self, detector):
        assert detector._model is None
        assert detector._paused is False

    def test_default_threshold(self, detector):
        assert detector._default_threshold == 0.5


class TestProcessChunk:
    """Test audio chunk processing for wake word detection."""

    def test_returns_none_when_no_model(self, detector):
        chunk = np.zeros(1280, dtype=np.float32)
        assert detector.process_chunk(chunk) is None

    def test_returns_none_when_paused(self, detector):
        detector._model = MagicMock()
        detector._paused = True
        chunk = np.zeros(1280, dtype=np.float32)
        assert detector.process_chunk(chunk) is None

    def test_detects_wake_word_above_threshold(self, detector):
        mock_model = MagicMock()
        mock_model.predict.return_value = {"hey_jarvis_v0.1": 0.9}
        detector._model = mock_model
        detector._default_threshold = 0.5

        chunk = np.zeros(1280, dtype=np.float32)
        result = detector.process_chunk(chunk)
        assert result == "hey_jarvis_v0.1"
        mock_model.reset.assert_called_once()

    def test_below_threshold_returns_none(self, detector):
        mock_model = MagicMock()
        mock_model.predict.return_value = {"hey_jarvis_v0.1": 0.2}
        detector._model = mock_model
        detector._default_threshold = 0.5

        chunk = np.zeros(1280, dtype=np.float32)
        result = detector.process_chunk(chunk)
        assert result is None

    def test_uses_per_model_threshold(self, detector):
        mock_model = MagicMock()
        mock_model.predict.return_value = {"custom_model": 0.4}
        detector._model = mock_model
        detector._thresholds = {"custom_model": 0.3}

        chunk = np.zeros(1280, dtype=np.float32)
        result = detector.process_chunk(chunk)
        assert result == "custom_model"

    def test_converts_float32_to_int16(self, detector):
        mock_model = MagicMock()
        mock_model.predict.return_value = {}
        detector._model = mock_model

        chunk = np.random.randn(1280).astype(np.float32) * 0.5
        detector.process_chunk(chunk)
        # Verify the model was called with int16 data
        call_args = mock_model.predict.call_args[0][0]
        assert call_args.dtype == np.int16


class TestPauseResume:
    """Test pause/resume for TTS avoidance."""

    def test_pause_sets_flag(self, detector):
        detector._model = MagicMock()
        detector.pause()
        assert detector._paused is True
        detector._model.reset.assert_called_once()

    def test_resume_clears_flag(self, detector):
        detector._model = MagicMock()
        detector._paused = True
        detector.resume()
        assert detector._paused is False

    def test_pause_without_model(self, detector):
        # Should not raise
        detector.pause()
        assert detector._paused is True


class TestReset:
    """Test model state reset."""

    def test_reset_with_model(self, detector):
        detector._model = MagicMock()
        detector.reset()
        detector._model.reset.assert_called_once()

    def test_reset_without_model(self, detector):
        detector._model = None
        # Should not raise
        detector.reset()


class TestConfigReload:
    """Test that config reload updates thresholds and optionally reloads model."""

    def test_reload_updates_thresholds(self, detector, settings):
        settings.wake.default_threshold = 0.7
        settings.wake.thresholds = {"custom": 0.3}
        settings.wake.model_paths = detector._model_paths  # same as current

        detector._on_config_reload(settings)
        assert detector._default_threshold == 0.7
        assert detector._thresholds == {"custom": 0.3}

    def test_reload_triggers_model_reload_on_path_change(self, detector, settings):
        settings.wake.model_paths = ["new_model"]
        settings.wake.default_threshold = 0.5
        settings.wake.thresholds = {}

        with patch.object(detector, "load") as mock_load:
            detector._on_config_reload(settings)
            mock_load.assert_called_once()
        assert detector._model_paths == ["new_model"]
