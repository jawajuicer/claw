"""Text-to-speech subsystem for The Claw."""

from claw.audio_pipeline.tts.engine import TTSAudio, TTSEngine
from claw.audio_pipeline.tts.manager import TTSManager

__all__ = ["TTSAudio", "TTSEngine", "TTSManager"]
