"""Audio capture from USB microphone via sounddevice."""

from __future__ import annotations

import asyncio
import logging
from collections import deque

import numpy as np
import sounddevice as sd

from claw.config import get_settings

log = logging.getLogger(__name__)


class AudioCapture:
    """Captures audio from a microphone using sounddevice.

    The sounddevice callback runs on a C-level audio thread and writes frames
    into a thread-safe deque. The async methods read from this deque on the
    asyncio event loop.
    """

    def __init__(self) -> None:
        cfg = get_settings().audio
        self.sample_rate = cfg.sample_rate
        self.channels = cfg.channels
        self.block_size = cfg.block_size
        self.device_index = cfg.device_index
        self.silence_threshold = cfg.silence_threshold
        self.silence_duration = cfg.silence_duration
        self.max_record_seconds = cfg.max_record_seconds

        self._buffer: deque[np.ndarray] = deque(maxlen=self.sample_rate * self.max_record_seconds // self.block_size)
        self._stream: sd.InputStream | None = None

    def _callback(self, indata: np.ndarray, frames: int, time_info: object, status: sd.CallbackFlags) -> None:
        if status:
            log.warning("Audio callback status: %s", status)
        self._buffer.append(indata[:, 0].copy())

    def start(self) -> None:
        """Open the audio input stream."""
        self._stream = sd.InputStream(
            samplerate=self.sample_rate,
            channels=self.channels,
            blocksize=self.block_size,
            dtype="float32",
            device=self.device_index,
            callback=self._callback,
        )
        self._stream.start()
        log.info(
            "Audio capture started (device=%s, rate=%d, block=%d)",
            self.device_index or "default",
            self.sample_rate,
            self.block_size,
        )

    def stop(self) -> None:
        """Close the audio input stream."""
        if self._stream is not None:
            self._stream.stop()
            self._stream.close()
            self._stream = None
            log.info("Audio capture stopped")

    def read_chunk(self) -> np.ndarray | None:
        """Pop one chunk from the buffer (non-blocking). Returns None if empty."""
        try:
            return self._buffer.popleft()
        except IndexError:
            return None

    async def record_until_silence(self) -> np.ndarray:
        """Record audio until silence is detected.

        Drains any buffered audio first (preserves speech from continuous
        utterances like "hey jarvis tell me a joke"), then waits for speech
        to begin before starting silence detection.

        Returns the full recorded audio as a float32 numpy array.
        """
        cfg = get_settings().audio

        # Drain any buffered audio (speech from before/during chime)
        frames: list[np.ndarray] = self.drain_buffer()
        silent_chunks = 0
        silence_chunks_needed = int(cfg.silence_duration * cfg.sample_rate / cfg.block_size)
        max_chunks = int(cfg.max_record_seconds * cfg.sample_rate / cfg.block_size)
        heard_speech = any(
            float(np.sqrt(np.mean(c ** 2))) >= cfg.silence_threshold for c in frames
        )

        log.info("Recording... (silence_threshold=%.4f, timeout=%ds, buffered=%d, speech=%s)",
                 cfg.silence_threshold, cfg.max_record_seconds, len(frames), heard_speech)

        while len(frames) < max_chunks:
            chunk = self.read_chunk()
            if chunk is None:
                await asyncio.sleep(0.01)
                continue

            frames.append(chunk)
            rms = float(np.sqrt(np.mean(chunk ** 2)))

            if rms >= cfg.silence_threshold:
                heard_speech = True
                silent_chunks = 0
            elif heard_speech:
                # Only count silence after we've heard speech
                silent_chunks += 1
                if silent_chunks >= silence_chunks_needed:
                    log.info("Silence detected after %d chunks", len(frames))
                    break

        if len(frames) >= max_chunks:
            log.warning("Recording hit max duration (%ds)", cfg.max_record_seconds)

        audio = np.concatenate(frames) if frames else np.array([], dtype=np.float32)
        log.info("Recorded %.2f seconds of audio", len(audio) / cfg.sample_rate)
        return audio

    def drain_buffer(self) -> list[np.ndarray]:
        """Pop all chunks from the buffer and return them."""
        chunks = list(self._buffer)
        self._buffer.clear()
        return chunks

    def flush(self) -> None:
        """Discard all buffered audio."""
        self._buffer.clear()
