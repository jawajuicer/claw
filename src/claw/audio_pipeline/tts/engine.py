"""Abstract TTS engine interface and audio data container."""

from __future__ import annotations

import abc
from dataclasses import dataclass
from typing import Iterator


@dataclass
class TTSAudio:
    """Raw PCM audio produced by a TTS engine."""

    pcm_data: bytes  # Raw PCM int16 mono
    sample_rate: int
    channels: int = 1
    sample_width: int = 2  # bytes per sample (int16)


class TTSEngine(abc.ABC):
    """Base class for text-to-speech engines."""

    @abc.abstractmethod
    def load(self) -> None:
        """Load the model / connect to the server."""

    @abc.abstractmethod
    def synthesize(self, text: str) -> TTSAudio:
        """Synthesize text into a single audio buffer."""

    @abc.abstractmethod
    def synthesize_stream(self, text: str) -> Iterator[bytes]:
        """Yield raw PCM int16 mono chunks as they're produced."""

    @abc.abstractmethod
    def get_sample_rate(self) -> int:
        """Return the output sample rate in Hz."""

    def shutdown(self) -> None:
        """Release resources. Override if needed."""

    @property
    @abc.abstractmethod
    def engine_name(self) -> str:
        """Human-readable engine name."""
