
"""Local English speech transcription for BibleCast using faster-whisper."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from threading import Lock


@dataclass(frozen=True)
class TranscriptionResult:
    """Result of local speech transcription."""

    success: bool
    text: str = ""
    language: str = ""
    error: str = ""


class WhisperTranscriber:
    """Transcribe audio locally, forcing English for BibleCast."""

    def __init__(
        self,
        model_size: str = "small",
        device: str = "cpu",
        compute_type: str = "int8",
    ) -> None:
        self.model_size = model_size
        self.device = device
        self.compute_type = compute_type
        self._model = None
        self._lock = Lock()

    def _get_model(self):
        """Load the model only when it is first needed."""
        if self._model is None:
            from faster_whisper import WhisperModel

            self._model = WhisperModel(
                self.model_size,
                device=self.device,
                compute_type=self.compute_type,
            )

        return self._model

    def transcribe(self, audio_path: str | Path) -> TranscriptionResult:
        """Return an English transcript without interpreting Bible references."""
        audio = Path(audio_path).expanduser()

        if not audio.is_file():
            return TranscriptionResult(
                success=False,
                error=f"Audio file not found: {audio}",
            )

        try:
            with self._lock:
                model = self._get_model()
                segments, info = model.transcribe(
                    str(audio),
                    language="en",
                    beam_size=5,
                    vad_filter=False,
                )
                text = " ".join(
                    segment.text.strip() for segment in segments
                ).strip()

            if not text:
                return TranscriptionResult(
                    success=False,
                    language=info.language or "en",
                    error="No speech was recognized in the audio.",
                )

            return TranscriptionResult(
                success=True,
                text=text,
                language=info.language or "en",
            )

        except Exception as exc:
            return TranscriptionResult(
                success=False,
                error=f"Speech transcription failed: "
                f"{type(exc).__name__}: {exc}",
            )
