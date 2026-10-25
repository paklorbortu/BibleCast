
"""Coordinate local audio transcription and Bible passage matching."""

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import logging

from biblecast.ai.passage_candidates import (
    PassageCandidate,
    find_passage_candidates,
)
from biblecast.ai.whisper_transcriber import (
    TranscriptionResult,
    WhisperTranscriber,
)


logger = logging.getLogger(__name__)

@dataclass(frozen=True)
class AudioReferenceResult:
    """Internal transcription and passage candidate result."""

    success: bool
    transcription: TranscriptionResult | None = None
    candidates: tuple[PassageCandidate, ...] = ()
    error: str = ""


class AudioReferencePipeline:
    """Transcribe locally and suggest passages for operator review."""

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
        """Process an existing audio file."""
        transcription = self.transcriber.transcribe(audio_path)
        return self._match_transcription(
            transcription,
            translation_code,
            limit,
        )

    def process_audio(
        self,
        audio: np.ndarray,
        translation_code: str,
        sample_rate: int = 16_000,
        limit: int = 5,
    ) -> AudioReferenceResult:
        """Process microphone samples directly from memory."""
        transcription = self.transcriber.transcribe_audio(
            audio,
            sample_rate=sample_rate,
        )
        return self._match_transcription(
            transcription,
            translation_code,
            limit,
        )

    @staticmethod
    def _match_transcription(
        transcription: TranscriptionResult,
        translation_code: str,
        limit: int,
    ) -> AudioReferenceResult:
        """Turn a successful transcription into candidate passages."""
        if not transcription.success:
            return AudioReferenceResult(
                success=False,
                transcription=transcription,
                error=transcription.error or "Audio transcription failed.",
            )

        transcript = transcription.text.strip()
        logger.info(
            "BibleCast audio diagnostic: transcript=%r, "
            "quality=%s, avg_logprob=%s, no_speech_probability=%s",
            transcript,
            transcription.quality,
            transcription.avg_logprob,
            transcription.no_speech_probability,
        )
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
