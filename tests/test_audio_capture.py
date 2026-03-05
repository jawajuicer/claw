"""Tests for claw.audio_pipeline.capture — AudioCapture."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import numpy as np
import pytest


@pytest.fixture()
def capture(settings):
    with patch("claw.audio_pipeline.capture.sd"):
        from claw.audio_pipeline.capture import AudioCapture

        cap = AudioCapture()
        yield cap


class TestInit:
    """Test AudioCapture initialization."""

    def test_default_sample_rate(self, capture):
        assert capture.sample_rate == 16000

    def test_default_channels(self, capture):
        assert capture.channels == 1

    def test_buffer_is_empty(self, capture):
        assert len(capture._buffer) == 0


class TestStartStop:
    """Test stream lifecycle."""

    def test_start_creates_stream(self, settings):
        with patch("claw.audio_pipeline.capture.sd") as mock_sd:
            from claw.audio_pipeline.capture import AudioCapture

            cap = AudioCapture()
            cap.start()
            mock_sd.InputStream.assert_called_once()
            mock_sd.InputStream.return_value.start.assert_called_once()

    def test_stop_closes_stream(self, settings):
        with patch("claw.audio_pipeline.capture.sd"):
            from claw.audio_pipeline.capture import AudioCapture

            cap = AudioCapture()
            cap._stream = MagicMock()
            cap.stop()
            assert cap._stream is None

    def test_stop_when_no_stream(self, capture):
        # Should not raise
        capture.stop()


class TestReadChunk:
    """Test buffer reading."""

    def test_read_chunk_returns_none_on_empty(self, capture):
        assert capture.read_chunk() is None

    def test_read_chunk_returns_data(self, capture):
        chunk = np.zeros(1280, dtype=np.float32)
        capture._buffer.append(chunk)
        result = capture.read_chunk()
        assert result is not None
        assert len(result) == 1280

    def test_read_chunk_pops_from_buffer(self, capture):
        capture._buffer.append(np.zeros(1280, dtype=np.float32))
        capture._buffer.append(np.ones(1280, dtype=np.float32))
        first = capture.read_chunk()
        assert np.all(first == 0)
        second = capture.read_chunk()
        assert np.all(second == 1)


class TestDrainAndFlush:
    """Test drain_buffer and flush."""

    def test_drain_buffer_returns_all_chunks(self, capture):
        for i in range(5):
            capture._buffer.append(np.full(1280, i, dtype=np.float32))
        chunks = capture.drain_buffer()
        assert len(chunks) == 5
        assert len(capture._buffer) == 0

    def test_drain_empty_buffer(self, capture):
        chunks = capture.drain_buffer()
        assert chunks == []

    def test_flush_clears_buffer(self, capture):
        capture._buffer.append(np.zeros(1280, dtype=np.float32))
        capture.flush()
        assert len(capture._buffer) == 0


class TestCallback:
    """Test the sounddevice callback."""

    def test_callback_appends_mono_data(self, capture):
        # Simulate stereo-like input (N, 1) as sounddevice provides
        indata = np.random.randn(1280, 1).astype(np.float32)
        capture._callback(indata, 1280, None, MagicMock(return_value=False))
        assert len(capture._buffer) == 1
        assert capture._buffer[0].shape == (1280,)

    def test_callback_copies_data(self, capture):
        indata = np.zeros((1280, 1), dtype=np.float32)
        capture._callback(indata, 1280, None, MagicMock(return_value=False))
        # Modify the original — stored copy should be unaffected
        indata[0, 0] = 999.0
        assert capture._buffer[0][0] == 0.0


class TestRecordUntilSilence:
    """Test the async recording loop.

    record_until_silence calls drain_buffer() first (consuming buffered data),
    then polls read_chunk() in a loop. We mock read_chunk to feed data
    sequentially so the loop terminates properly.
    """

    async def test_stops_on_silence_after_speech(self, capture, settings):
        """Simulate speech followed by silence to trigger stop."""
        speech_chunk = np.full(1280, 0.1, dtype=np.float32)
        silent_chunk = np.full(1280, 0.001, dtype=np.float32)

        silence_count = int(settings.audio.silence_duration * settings.audio.sample_rate / settings.audio.block_size) + 5

        # Build a sequence: speech chunks then silence chunks
        chunks = [speech_chunk.copy() for _ in range(5)]
        chunks += [silent_chunk.copy() for _ in range(silence_count)]

        call_count = 0

        def mock_read_chunk():
            nonlocal call_count
            if call_count < len(chunks):
                c = chunks[call_count]
                call_count += 1
                return c
            return None

        # drain_buffer will return empty (no pre-buffered data before we mock)
        capture.drain_buffer = lambda: list(chunks[:5])
        # read_chunk feeds the remaining silence chunks
        remaining = chunks[5:]
        idx = 0

        def read_remaining():
            nonlocal idx
            if idx < len(remaining):
                c = remaining[idx]
                idx += 1
                return c
            return None

        capture.read_chunk = read_remaining

        audio = await capture.record_until_silence()
        assert len(audio) > 0

    async def test_stops_at_max_duration(self, capture, settings):
        """Feed enough speech chunks to hit the max duration limit."""
        max_chunks = int(settings.audio.max_record_seconds * settings.audio.sample_rate / settings.audio.block_size)
        speech_chunk = np.full(1280, 0.1, dtype=np.float32)

        all_chunks = [speech_chunk.copy() for _ in range(max_chunks + 10)]

        # drain_buffer returns first batch
        capture.drain_buffer = lambda: all_chunks[:10]

        remaining = all_chunks[10:]
        idx = 0

        def read_remaining():
            nonlocal idx
            if idx < len(remaining):
                c = remaining[idx]
                idx += 1
                return c
            return None

        capture.read_chunk = read_remaining

        audio = await capture.record_until_silence()
        assert len(audio) > 0
