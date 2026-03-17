"""Audio capture from USB microphone via sounddevice."""

from __future__ import annotations

import asyncio
import logging
from collections import deque
from typing import TYPE_CHECKING

import numpy as np
import sounddevice as sd

from claw.config import get_settings

if TYPE_CHECKING:
    from claw.audio_pipeline.vad import StreamingVAD

log = logging.getLogger(__name__)


class AudioCapture:
    """Captures audio from a microphone using sounddevice.

    The sounddevice callback runs on a C-level audio thread and writes frames
    into a thread-safe deque. The async methods read from this deque on the
    asyncio event loop.
    """

    # AGC constants
    _AGC_TARGET_RMS = 0.08
    _AGC_ATTACK = 0.3
    _AGC_RELEASE = 0.05
    _AGC_MIN_GAIN = 0.02
    _AGC_MAX_GAIN = 15.0

    def __init__(self, vad: StreamingVAD | None = None) -> None:
        cfg = get_settings().audio
        self.sample_rate = cfg.sample_rate
        self.channels = cfg.channels
        self.block_size = cfg.block_size
        self.device_index = cfg.device_index
        self.silence_duration = cfg.silence_duration
        self.max_record_seconds = cfg.max_record_seconds
        self._agc_silence_threshold = cfg.agc_silence_threshold
        self._agc_gain_ceiling = cfg.agc_gain_ceiling
        self._vad = vad

        self._buffer: deque[np.ndarray] = deque(maxlen=self.sample_rate * self.max_record_seconds // self.block_size)
        self._stream: sd.InputStream | None = None

        # AGC state
        self._agc_gain = 1.0

        # Audio metrics (updated every chunk by _condition_audio)
        self.last_rms: float = 0.0
        self.last_agc_gain: float = 1.0
        self.noise_floor: float = 0.0
        self._noise_floor_alpha = 0.01  # slow EMA for noise floor

        # High-pass filter state (initialized in start())
        self._hp_sos: np.ndarray | None = None
        self._hp_zi: np.ndarray | None = None

    def _condition_audio(self, chunk: np.ndarray) -> np.ndarray:
        """Apply audio conditioning: high-pass filter, AGC, soft clipping."""
        # 1. High-pass filter (80Hz) — removes rumble, HVAC hum, DC offset
        if self._hp_sos is not None:
            from scipy.signal import sosfilt

            chunk, self._hp_zi = sosfilt(self._hp_sos, chunk, zi=self._hp_zi)
            chunk = chunk.astype(np.float32)
        else:
            # Fallback: simple DC removal
            chunk = chunk - np.mean(chunk)

        # 2. AGC — normalize loudness to target RMS
        rms = float(np.sqrt(np.mean(chunk**2)))
        if rms > 1e-6:
            desired_gain = self._AGC_TARGET_RMS / rms
            desired_gain = np.clip(desired_gain, self._AGC_MIN_GAIN, self._AGC_MAX_GAIN)
            # Asymmetric smoothing: fast attack (reduce gain quickly), slow release
            if desired_gain < self._agc_gain:
                alpha = self._AGC_ATTACK
            else:
                alpha = self._AGC_RELEASE
            self._agc_gain += alpha * (desired_gain - self._agc_gain)
        chunk = chunk * self._agc_gain

        # Update exposed metrics
        self.last_rms = rms
        self.last_agc_gain = self._agc_gain
        # Noise floor: slow EMA of RMS (tracks ambient noise level)
        self.noise_floor += self._noise_floor_alpha * (rms - self.noise_floor)

        # 3. Soft clipping — compress peaks that exceed ±1.0
        chunk = np.tanh(chunk)

        return chunk.astype(np.float32)

    def _callback(self, indata: np.ndarray, frames: int, time_info: object, status: sd.CallbackFlags) -> None:
        if status:
            log.warning("Audio callback status: %s", status)
        chunk = indata[:, 0].copy()
        chunk = self._condition_audio(chunk)
        self._buffer.append(chunk)

    def start(self) -> None:
        """Open the audio input stream.

        If the configured device fails (e.g. stale index after hardware change),
        automatically falls back to the system default device.
        """
        device = self.device_index
        try:
            self._stream = self._open_stream(device)
        except sd.PortAudioError:
            if device is not None:
                log.warning(
                    "Audio device %s failed, falling back to system default",
                    device,
                )
                device = None
                self._stream = self._open_stream(device)
            else:
                raise
        self._stream.start()

        # Initialize high-pass filter coefficients
        try:
            from scipy.signal import butter, sosfilt_zi

            self._hp_sos = butter(2, 80, btype="high", fs=self.sample_rate, output="sos")
            self._hp_zi = sosfilt_zi(self._hp_sos)
        except ImportError:
            log.info("scipy not available, using DC removal instead of high-pass filter")
            self._hp_sos = None

        log.info(
            "Audio capture started (device=%s, rate=%d, block=%d)",
            device if device is not None else "default",
            self.sample_rate,
            self.block_size,
        )

    def _open_stream(self, device: int | None) -> sd.InputStream:
        """Create an InputStream for the given device."""
        return sd.InputStream(
            samplerate=self.sample_rate,
            channels=self.channels,
            blocksize=self.block_size,
            dtype="float32",
            device=device,
            callback=self._callback,
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
        """Record audio until speech ends (VAD) or silence is detected (RMS fallback).

        Uses Silero VAD when available to detect actual speech boundaries.
        Falls back to amplitude-based detection if VAD is not loaded.

        Drains any buffered audio first (preserves speech from continuous
        utterances like "hey claw tell me a joke"), then waits for speech
        to begin before starting end-of-speech detection.

        Returns the full recorded audio as a float32 numpy array.
        """
        if self._vad is not None:
            return await self._record_with_vad()
        return await self._record_with_rms()

    async def _record_with_vad(self) -> np.ndarray:
        """Record using Silero VAD for end-of-speech detection."""
        cfg = get_settings().audio
        vad = self._vad
        vad.reset()

        frames: list[np.ndarray] = self.drain_buffer()
        max_chunks = int(cfg.max_record_seconds * cfg.sample_rate / cfg.block_size)
        # How many consecutive non-speech chunks before we stop
        silence_chunks_needed = int(cfg.silence_duration * cfg.sample_rate / cfg.block_size)
        silent_chunks = 0

        # Check buffered audio for speech
        heard_speech = False
        for chunk in frames:
            if vad.is_speech(chunk):
                heard_speech = True

        log.info("Recording (VAD mode, threshold=%.2f, timeout=%ds, buffered=%d, speech=%s)",
                 vad.threshold, cfg.max_record_seconds, len(frames), heard_speech)

        while len(frames) < max_chunks:
            chunk = self.read_chunk()
            if chunk is None:
                await asyncio.sleep(0.01)
                continue

            frames.append(chunk)
            speech_detected = vad.is_speech(chunk)

            if speech_detected:
                heard_speech = True
                silent_chunks = 0
            elif heard_speech:
                silent_chunks += 1
                if silent_chunks >= silence_chunks_needed:
                    log.info("VAD: speech ended after %d chunks (%.1fs)",
                             len(frames), len(frames) * cfg.block_size / cfg.sample_rate)
                    break

        if len(frames) >= max_chunks:
            log.warning("Recording hit max duration (%ds)", cfg.max_record_seconds)

        audio = np.concatenate(frames) if frames else np.array([], dtype=np.float32)
        log.info("Recorded %.2f seconds of audio", len(audio) / cfg.sample_rate)
        return audio

    async def _record_with_rms(self) -> np.ndarray:
        """Record using RMS amplitude for silence detection (fallback)."""
        cfg = get_settings().audio

        frames: list[np.ndarray] = self.drain_buffer()
        silent_chunks = 0
        silence_chunks_needed = int(cfg.silence_duration * cfg.sample_rate / cfg.block_size)
        max_chunks = int(cfg.max_record_seconds * cfg.sample_rate / cfg.block_size)
        threshold = self._agc_silence_threshold
        heard_speech = any(
            float(np.sqrt(np.mean(c ** 2))) >= threshold for c in frames
        )

        log.info("Recording (RMS mode, threshold=%.4f, gain_ceiling=%.1f, timeout=%ds, buffered=%d, speech=%s)",
                 threshold, self._agc_gain_ceiling, cfg.max_record_seconds, len(frames), heard_speech)

        while len(frames) < max_chunks:
            chunk = self.read_chunk()
            if chunk is None:
                await asyncio.sleep(0.01)
                continue

            frames.append(chunk)
            rms = float(np.sqrt(np.mean(chunk ** 2)))

            if rms >= threshold and self._agc_gain < self._agc_gain_ceiling:
                heard_speech = True
                silent_chunks = 0
            elif heard_speech:
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

    def get_metrics(self) -> dict:
        """Return current audio metrics for diagnostics."""
        return {
            "rms": round(self.last_rms, 6),
            "agc_gain": round(self.last_agc_gain, 3),
            "noise_floor": round(self.noise_floor, 6),
            "sample_rate": self.sample_rate,
            "block_size": self.block_size,
            "device_index": self.device_index,
        }

    def flush(self) -> None:
        """Discard all buffered audio."""
        self._buffer.clear()
