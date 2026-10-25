
"""Local English speech transcription for BibleCast."""

from __future__ import annotations

import os

from dataclasses import dataclass
from pathlib import Path
from threading import Lock

import numpy as np


@dataclass(frozen=True)
class TranscriptionResult:
    """Local transcription result with diagnostic quality indicators."""

    success: bool
    text: str = ""
    language: str = ""
    error: str = ""
    avg_logprob: float | None = None
    no_speech_probability: float | None = None
    quality: str = "unknown"


class WhisperTranscriber:
    """Transcribe English audio locally from files or memory."""

    SAMPLE_RATE = 16_000

    def __init__(
        self,
        model_size: str | None = None,
        device: str = "cpu",
        compute_type: str = "int8",
    ) -> None:
        # BIBLECAST_WHISPER_MODEL lets a slower computer pick a faster model
        # (for example "base.en" or "tiny.en") without changing code.
        self.model_size = (
            model_size
            or os.environ.get("BIBLECAST_WHISPER_MODEL")
            or "small"
        )
        self.device = device
        self.compute_type = compute_type
        self._model = None
        self._lock = Lock()

    def _get_model(self):
        """Load the speech model once and reuse it."""
        if self._model is None:
            from faster_whisper import WhisperModel

            self._model = WhisperModel(
                self.model_size,
                device=self.device,
                compute_type=self.compute_type,
            )

        return self._model

    def warm_up(self) -> None:
        """Load the speech model now so the first live segment is fast."""
        with self._lock:
            self._get_model()

    def transcribe(self, audio_path: str | Path) -> TranscriptionResult:
        """Transcribe an existing audio file."""
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
                    temperature=0.0,
                    condition_on_previous_text=False,
                    vad_filter=False,
                )
                return self._collect_result(segments, info)

        except Exception as exc:
            return self._failure(exc)

    def transcribe_audio(
        self,
        audio: np.ndarray,
        sample_rate: int = SAMPLE_RATE,
    ) -> TranscriptionResult:
        """Transcribe mono floating-point audio without writing a file.

        Input must be a one-dimensional NumPy array containing audio
        normalized approximately to the range [-1.0, 1.0].
        """
        if sample_rate != self.SAMPLE_RATE:
            return TranscriptionResult(
                success=False,
                error=(
                    f"Expected {self.SAMPLE_RATE} Hz audio, "
                    f"received {sample_rate} Hz."
                ),
            )

        if not isinstance(audio, np.ndarray):
            return TranscriptionResult(
                success=False,
                error="Audio must be provided as a NumPy array.",
            )

        if audio.ndim != 1:
            return TranscriptionResult(
                success=False,
                error="Audio must be mono, one-dimensional audio.",
            )

        if audio.size < sample_rate:
            return TranscriptionResult(
                success=False,
                error="Audio segment is too short to transcribe reliably.",
            )

        if not np.isfinite(audio).all():
            return TranscriptionResult(
                success=False,
                error="Audio contains invalid numeric samples.",
            )

        samples = np.asarray(audio, dtype=np.float32)

        if float(np.max(np.abs(samples))) < 0.001:
            return TranscriptionResult(
                success=False,
                error="Audio is silent or too quiet.",
            )

        try:
            with self._lock:
                model = self._get_model()
                # Live audio favours speed: greedy decoding, and a voice
                # activity filter so pauses are skipped instead of decoded.
                segments, info = model.transcribe(
                    samples,
                    language="en",
                    beam_size=1,
                    temperature=0.0,
                    condition_on_previous_text=False,
                    without_timestamps=True,
                    vad_filter=True,
                    vad_parameters={"min_silence_duration_ms": 300},
                )
                return self._collect_result(segments, info)

        except Exception as exc:
            return self._failure(exc)

    @staticmethod
    def _collect_result(segments, info) -> TranscriptionResult:
        """Collect transcription text and model diagnostics."""
        collected = list(segments)
        text = " ".join(
            segment.text.strip()
            for segment in collected
            if segment.text.strip()
        ).strip()

        if not text:
            return TranscriptionResult(
                success=False,
                language=getattr(info, "language", None) or "en",
                error="No speech was recognized in the audio.",
                quality="low",
            )

        logprobs = [
            float(segment.avg_logprob)
            for segment in collected
            if getattr(segment, "avg_logprob", None) is not None
        ]
        silence_scores = [
            float(segment.no_speech_prob)
            for segment in collected
            if getattr(segment, "no_speech_prob", None) is not None
        ]

        average_logprob = (
            sum(logprobs) / len(logprobs) if logprobs else None
        )
        max_no_speech = max(silence_scores) if silence_scores else None

        # These are review indicators, not calibrated probabilities
        # that the words or Bible reference are correct.
        if (
            average_logprob is not None
            and average_logprob < -1.0
        ):
            quality = "low"
        elif (
            average_logprob is not None
            and average_logprob < -0.5
        ):
            quality = "review"
        else:
            quality = "unverified"

        if max_no_speech is not None and max_no_speech >= 0.8:
            quality = "review"

        return TranscriptionResult(
            success=True,
            text=text,
            language=getattr(info, "language", None) or "en",
            avg_logprob=average_logprob,
            no_speech_probability=max_no_speech,
            quality=quality,
        )

    @staticmethod
    def _failure(exc: Exception) -> TranscriptionResult:
        return TranscriptionResult(
            success=False,
            error=(
                f"Speech transcription failed: "
                f"{type(exc).__name__}: {exc}"
            ),
            quality="low",
        )
