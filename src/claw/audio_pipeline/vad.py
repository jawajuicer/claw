"""Streaming Voice Activity Detection using Silero VAD (ONNX).

Uses the same Silero VAD v6 model bundled with faster-whisper.
Processes 512-sample windows with maintained hidden state for real-time
speech/silence detection during recording.
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np

log = logging.getLogger(__name__)

# Silero VAD v6 constants
_WINDOW_SIZE = 512  # samples per VAD window (32ms at 16kHz)
_CONTEXT_SIZE = 64  # context samples prepended to each window


def _find_vad_model() -> str:
    """Locate the Silero VAD ONNX model bundled with faster-whisper."""
    try:
        from faster_whisper.utils import get_assets_path

        path = Path(get_assets_path()) / "silero_vad_v6.onnx"
        if path.exists():
            return str(path)
    except ImportError:
        pass

    raise FileNotFoundError(
        "Silero VAD model not found. Install faster-whisper: pip install faster-whisper"
    )


class StreamingVAD:
    """Streaming voice activity detector using Silero VAD ONNX model.

    Processes audio chunk-by-chunk, maintaining internal state (h, c)
    across calls. Returns per-window speech probabilities.
    """

    def __init__(self, threshold: float = 0.5) -> None:
        self.threshold = threshold
        self.last_speech_prob: float = 0.0  # exposed for diagnostics
        self._session = None
        self._h: np.ndarray | None = None
        self._c: np.ndarray | None = None
        self._context = np.zeros(_CONTEXT_SIZE, dtype=np.float32)

    def load(self) -> None:
        """Load the ONNX model."""
        import onnxruntime

        model_path = _find_vad_model()
        opts = onnxruntime.SessionOptions()
        opts.inter_op_num_threads = 1
        opts.intra_op_num_threads = 1
        opts.enable_cpu_mem_arena = False
        opts.log_severity_level = 4

        self._session = onnxruntime.InferenceSession(
            model_path,
            providers=["CPUExecutionProvider"],
            sess_options=opts,
        )
        self.reset()
        log.info("Streaming VAD loaded (threshold=%.2f)", self.threshold)

    def reset(self) -> None:
        """Reset hidden state for a new recording session."""
        self._h = np.zeros((1, 1, 128), dtype=np.float32)
        self._c = np.zeros((1, 1, 128), dtype=np.float32)
        self._context = np.zeros(_CONTEXT_SIZE, dtype=np.float32)

    def process_chunk(self, audio: np.ndarray) -> list[float]:
        """Process an audio chunk and return speech probabilities.

        Args:
            audio: 1D float32 array of any length (will be split into
                   512-sample windows internally).

        Returns:
            List of speech probabilities, one per 512-sample window.
            Partial windows at the end are zero-padded.
        """
        if self._session is None:
            raise RuntimeError("VAD not loaded — call load() first")

        probabilities = []

        # Split chunk into 512-sample windows
        offset = 0
        while offset < len(audio):
            window = audio[offset : offset + _WINDOW_SIZE]

            # Zero-pad if the last window is short
            if len(window) < _WINDOW_SIZE:
                window = np.pad(window, (0, _WINDOW_SIZE - len(window)))

            # Prepend context from previous window
            inp = np.concatenate([self._context, window]).reshape(1, -1)
            output, self._h, self._c = self._session.run(
                None, {"input": inp, "h": self._h, "c": self._c}
            )
            prob = float(output.flat[0])
            probabilities.append(prob)

            # Update context for next window
            self._context = window[-_CONTEXT_SIZE:]
            offset += _WINDOW_SIZE

        return probabilities

    def is_speech(self, audio: np.ndarray) -> bool:
        """Check if any window in the chunk contains speech.

        Returns True if any 512-sample window exceeds the threshold.
        """
        probs = self.process_chunk(audio)
        if probs:
            self.last_speech_prob = max(probs)
        return any(p >= self.threshold for p in probs)

    def max_speech_prob(self, audio: np.ndarray) -> float:
        """Return the maximum speech probability across all windows in the chunk."""
        probs = self.process_chunk(audio)
        return max(probs) if probs else 0.0
