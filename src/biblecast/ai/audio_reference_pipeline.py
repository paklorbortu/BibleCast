
"""Coordinate audio transcription and local Bible passage matching."""

from dataclasses import dataclass
from pathlib import Path

from biblecast.ai.passage_candidates import (
    PassageCandidate,
    find_passage_candidates,
)
from biblecast.ai.whisper_transcriber import (
    TranscriptionResult,
    WhisperTranscriber,
)


@dataclass(frozen=True)
class AudioReferenceResult:
    """Internal result of processing audio into passage candidates."""

    success: bool
    transcription: TranscriptionResult | None = None
    candidates: tuple[PassageCandidate, ...] = ()
    error: str = ""


class AudioReferencePipeline:
    """Transcribe audio and find local Bible passage candidates.

    This pipeline suggests passages only. It never previews or projects
    a passage and never approves a candidate on the operator's behalf.
    """

    def __init__(
        self,
        transcriber: WhisperTranscriber | None = None,
    ) -> None:
        self.transcriber = transcriber or WhisperTranscriber()

    def process(
        self,
        audio_path: str | Path,
        translation_code: str,
        limit: int = 5,
    ) -> AudioReferenceResult:
        """Transcribe audio and return ranked passage candidates."""

        transcription = self.transcriber.transcribe(audio_path)

        if not transcription.success:
            return AudioReferenceResult(
                success=False,
                transcription=transcription,
                error=transcription.error or "Audio transcription failed.",
            )

        transcript = transcription.text.strip()
        if not transcript:
            return AudioReferenceResult(
                success=False,
                transcription=transcription,
                error="No speech was recognized.",
            )

        candidates = find_passage_candidates(
            translation_code,
            transcript,
            limit=limit,
        )

        if not candidates:
            return AudioReferenceResult(
                success=False,
                transcription=transcription,
                error="No matching Bible passages were found.",
            )

        return AudioReferenceResult(
            success=True,
            transcription=transcription,
            candidates=tuple(candidates),
        )
