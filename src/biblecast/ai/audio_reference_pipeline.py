
"""Coordinate local audio transcription and Bible passage matching."""

from dataclasses import dataclass
from pathlib import Path

import logging
import re

import numpy as np

from biblecast.ai.passage_candidates import (
    PassageCandidate,
    find_passage_candidates,
)
from biblecast.ai.whisper_transcriber import (
    TranscriptionResult,
    WhisperTranscriber,
)


logger = logging.getLogger(__name__)

# Live audio arrives in short overlapping segments, so a quotation can be
# split across two of them. Keep a little recent speech to join them again.
_CONTEXT_WORDS = 14
_MAX_OVERLAP_WORDS = 6
_CONFIDENCE_RANK = {"high": 3, "medium": 2, "low": 1}


def _word_key(word: str) -> str:
    return re.sub(r"[^a-z0-9]", "", word.casefold())


def merge_overlapping_speech(previous: str, current: str) -> str:
    """Join two transcripts, removing words repeated by segment overlap."""
    previous_words = previous.split()[-_CONTEXT_WORDS:]
    current_words = current.split()

    longest = min(
        _MAX_OVERLAP_WORDS, len(previous_words), len(current_words)
    )
    for size in range(longest, 0, -1):
        if [
            _word_key(w) for w in previous_words[-size:]
        ] == [_word_key(w) for w in current_words[:size]]:
            return " ".join(previous_words + current_words[size:])

    return " ".join(previous_words + current_words)


def _prefer(first: PassageCandidate, second: PassageCandidate):
    """Return the stronger of two candidates for the same passage."""
    def strength(item):
        return (_CONFIDENCE_RANK.get(item.confidence, 0), item.score)

    return first if strength(first) >= strength(second) else second


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
        self._recent_speech = ""

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
        return self._match_live_transcription(
            transcription,
            translation_code,
            limit,
        )

    def reset_context(self) -> None:
        """Forget recent speech, e.g. when a new listening session starts."""
        self._recent_speech = ""

    def _match_live_transcription(
        self,
        transcription: TranscriptionResult,
        translation_code: str,
        limit: int,
    ) -> AudioReferenceResult:
        """Match this segment, and this segment joined to the previous one."""
        current = (transcription.text or "").strip()
        result = self._match_transcription(
            transcription, translation_code, limit
        )

        if not transcription.success or not current:
            return result

        previous = self._recent_speech
        self._recent_speech = current

        if not previous:
            return result

        merged_text = merge_overlapping_speech(previous, current)
        merged = find_passage_candidates(
            translation_code, merged_text, limit=limit
        )
        if not merged:
            return result

        combined: dict[str, PassageCandidate] = {
            item.reference: item for item in result.candidates
        }
        for item in merged:
            existing = combined.get(item.reference)
            combined[item.reference] = (
                item if existing is None else _prefer(existing, item)
            )

        ranked = sorted(
            combined.values(),
            key=lambda item: (
                -_CONFIDENCE_RANK.get(item.confidence, 0),
                -item.score,
                item.reference,
            ),
        )[:limit]

        logger.info(
            "BibleCast joined speech=%r, candidates=%s",
            merged_text,
            [(c.reference, c.confidence) for c in ranked],
        )

        return AudioReferenceResult(
            success=True,
            transcription=transcription,
            candidates=tuple(ranked),
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

        logger.info(
            "BibleCast match result: %s",
            [(c.reference, c.match_type, c.confidence, round(c.score, 2))
             for c in candidates] or "no confident match",
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
