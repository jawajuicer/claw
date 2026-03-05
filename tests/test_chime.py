"""Tests for claw.audio_pipeline.chime — play_listening_chime."""

from __future__ import annotations

from unittest.mock import patch

import numpy as np


class TestPlayListeningChime:
    """Test the chime playback function."""

    def test_chime_plays_when_enabled(self, settings):
        settings.audio.chime_enabled = True
        settings.audio.chime_volume = 0.5
        settings.audio.chime_frequency = 880
        settings.audio.chime_duration_ms = 200

        with (
            patch("claw.audio_pipeline.chime.get_settings", return_value=settings),
            patch("claw.audio_pipeline.chime.sd") as mock_sd,
        ):
            from claw.audio_pipeline.chime import play_listening_chime

            play_listening_chime()
            mock_sd.play.assert_called_once()
            mock_sd.wait.assert_called_once()

            # Check audio is float32
            played_audio = mock_sd.play.call_args[0][0]
            assert played_audio.dtype == np.float32

    def test_chime_skipped_when_disabled(self, settings):
        settings.audio.chime_enabled = False

        with (
            patch("claw.audio_pipeline.chime.get_settings", return_value=settings),
            patch("claw.audio_pipeline.chime.sd") as mock_sd,
        ):
            from claw.audio_pipeline.chime import play_listening_chime

            play_listening_chime()
            mock_sd.play.assert_not_called()

    def test_chime_volume_applied(self, settings):
        settings.audio.chime_enabled = True
        settings.audio.chime_volume = 0.1
        settings.audio.chime_frequency = 880
        settings.audio.chime_duration_ms = 200

        with (
            patch("claw.audio_pipeline.chime.get_settings", return_value=settings),
            patch("claw.audio_pipeline.chime.sd") as mock_sd,
        ):
            from claw.audio_pipeline.chime import play_listening_chime

            play_listening_chime()
            played = mock_sd.play.call_args[0][0]
            # With volume 0.1, max amplitude should be approximately 0.1
            assert np.max(np.abs(played)) <= 0.15

    def test_chime_handles_playback_error(self, settings):
        settings.audio.chime_enabled = True
        settings.audio.chime_frequency = 880
        settings.audio.chime_duration_ms = 200

        with (
            patch("claw.audio_pipeline.chime.get_settings", return_value=settings),
            patch("claw.audio_pipeline.chime.sd") as mock_sd,
        ):
            mock_sd.play.side_effect = RuntimeError("audio device error")
            from claw.audio_pipeline.chime import play_listening_chime

            # Should not raise — error is caught internally
            play_listening_chime()

    def test_chime_uses_correct_sample_rate(self, settings):
        settings.audio.chime_enabled = True
        settings.audio.chime_frequency = 880
        settings.audio.chime_duration_ms = 200

        with (
            patch("claw.audio_pipeline.chime.get_settings", return_value=settings),
            patch("claw.audio_pipeline.chime.sd") as mock_sd,
        ):
            from claw.audio_pipeline.chime import play_listening_chime

            play_listening_chime()
            call_kwargs = mock_sd.play.call_args
            assert call_kwargs.kwargs.get("samplerate") == 44100 or call_kwargs[1].get("samplerate") == 44100

    def test_chime_audio_has_correct_length(self, settings):
        settings.audio.chime_enabled = True
        settings.audio.chime_frequency = 880
        settings.audio.chime_duration_ms = 200
        settings.audio.chime_volume = 1.0

        with (
            patch("claw.audio_pipeline.chime.get_settings", return_value=settings),
            patch("claw.audio_pipeline.chime.sd") as mock_sd,
        ):
            from claw.audio_pipeline.chime import play_listening_chime

            play_listening_chime()
            played = mock_sd.play.call_args[0][0]
            # 200ms at 44100 Hz = 8820 samples (two halves of 4410 each)
            expected_samples = int(44100 * 200 / 1000)
            assert len(played) == expected_samples
