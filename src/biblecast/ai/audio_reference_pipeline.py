
"""Coordinate transcription, Bible reference interpretation, and validation."""

from dataclasses import dataclass
from pathlib import Path

from biblecast.ai.gemma_text import (
    BibleReferenceResult,
    GemmaTextAdapter,
)
from biblecast.ai.reference_validator import (
    ValidatedReference,
    validate_reference,
)
from biblecast.ai.whisper_transcriber import (
    TranscriptionResult,
    WhisperTranscriber,
)


@dataclass(frozen=True)
class AudioReferenceResult:
    """Result of processing one audio recording."""

    success: bool
    transcription: TranscriptionResult | None = None
    reference: BibleReferenceResult | None = None
    validated_reference: ValidatedReference | None = None
    error: str = ""


class AudioReferencePipeline:
    """Run the offline audio-to-validated-reference workflow."""

    def __init__(
        self,
        transcriber: WhisperTranscriber | None = None,
        interpreter: GemmaTextAdapter | None = None,
    ) -> None:
        self.transcriber = transcriber or WhisperTranscriber()
        self.interpreter = interpreter or GemmaTextAdapter()

    def process(
        self,
        audio_path: str | Path,
        translation_code: str,
    ) -> AudioReferenceResult:
        """Process audio and validate the interpreted reference."""

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

        reference = self.interpreter.interpret(transcript)

        if not reference.success:
            return AudioReferenceResult(
                success=False,
                transcription=transcription,
                reference=reference,
                error=reference.error or "Bible reference interpretation failed.",
            )

        validated = validate_reference(translation_code, reference)

        if not validated.success:
            return AudioReferenceResult(
                success=False,
                transcription=transcription,
                reference=reference,
                validated_reference=validated,
                error=validated.error or "Bible reference validation failed.",
            )

        return AudioReferenceResult(
            success=True,
            transcription=transcription,
            reference=reference,
            validated_reference=validated,
        )
