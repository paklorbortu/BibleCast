
from pathlib import Path
from unittest.mock import patch

from biblecast.ai.audio_reference_pipeline import AudioReferencePipeline
from biblecast.ai.gemma_text import BibleReferenceResult
from biblecast.ai.reference_validator import ValidatedReference
from biblecast.ai.whisper_transcriber import TranscriptionResult


class FakeTranscriber:
    def __init__(self, result):
        self.result = result
        self.audio_path = None

    def transcribe(self, audio_path):
        self.audio_path = audio_path
        return self.result


class FakeInterpreter:
    def __init__(self, result):
        self.result = result
        self.transcript = None

    def interpret(self, transcript):
        self.transcript = transcript
        return self.result


def make_pipeline(transcription, interpretation):
    transcriber = FakeTranscriber(transcription)
    interpreter = FakeInterpreter(interpretation)

    pipeline = AudioReferencePipeline(
        transcriber=transcriber,
        interpreter=interpreter,
    )

    return pipeline, transcriber, interpreter


def test_successfully_processes_and_validates_reference():
    transcription = TranscriptionResult(
        success=True,
        text="Please open John three sixteen.",
        language="en",
    )
    interpretation = BibleReferenceResult(
        success=True,
        book="John",
        chapter="3",
        verse="16",
        confidence="high",
    )
    validated = ValidatedReference(
        success=True,
        book="John",
        chapter=3,
        start_verse=16,
        end_verse=16,
        verses=(
            {
                "book_name": "John",
                "chapter": 3,
                "verse": 16,
                "text": "For God so loved the world...",
            },
        ),
    )

    pipeline, transcriber, interpreter = make_pipeline(
        transcription,
        interpretation,
    )

    with patch(
        "biblecast.ai.audio_reference_pipeline.validate_reference",
        return_value=validated,
    ) as validate:
        result = pipeline.process("sermon.wav", "WEB")

    assert result.success is True
    assert result.transcription == transcription
    assert result.reference == interpretation
    assert result.validated_reference == validated
    assert result.error == ""

    assert transcriber.audio_path == "sermon.wav"
    assert interpreter.transcript == "Please open John three sixteen."
    validate.assert_called_once_with("WEB", interpretation)


def test_returns_error_when_transcription_fails():
    transcription = TranscriptionResult(
        success=False,
        error="Audio file could not be decoded.",
    )
    pipeline, _, interpreter = make_pipeline(
        transcription,
        BibleReferenceResult(success=False),
    )

    with patch(
        "biblecast.ai.audio_reference_pipeline.validate_reference"
    ) as validate:
        result = pipeline.process(Path("sermon.wav"), "WEB")

    assert result.success is False
    assert result.error == "Audio file could not be decoded."
    assert result.transcription == transcription
    assert result.reference is None
    assert result.validated_reference is None
    interpreter.transcript is None
    validate.assert_not_called()


def test_rejects_empty_transcription():
    transcription = TranscriptionResult(
        success=True,
        text="   ",
        language="en",
    )
    pipeline, _, interpreter = make_pipeline(
        transcription,
        BibleReferenceResult(success=False),
    )

    with patch(
        "biblecast.ai.audio_reference_pipeline.validate_reference"
    ) as validate:
        result = pipeline.process("sermon.wav", "WEB")

    assert result.success is False
    assert result.error == "No speech was recognized."
    assert result.reference is None
    assert interpreter.transcript is None
    validate.assert_not_called()


def test_returns_error_when_interpretation_fails():
    transcription = TranscriptionResult(
        success=True,
        text="Some unclear speech.",
        language="en",
    )
    interpretation = BibleReferenceResult(
        success=False,
        error="Gemma inference timed out.",
    )
    pipeline, _, interpreter = make_pipeline(
        transcription,
        interpretation,
    )

    with patch(
        "biblecast.ai.audio_reference_pipeline.validate_reference"
    ) as validate:
        result = pipeline.process("sermon.wav", "WEB")

    assert result.success is False
    assert result.reference == interpretation
    assert result.validated_reference is None
    assert result.error == "Gemma inference timed out."
    assert interpreter.transcript == "Some unclear speech."
    validate.assert_not_called()


def test_returns_error_when_reference_validation_fails():
    transcription = TranscriptionResult(
        success=True,
        text="Open John three sixteen.",
        language="en",
    )
    interpretation = BibleReferenceResult(
        success=True,
        book="John",
        chapter="3",
        verse="16",
        confidence="high",
    )
    invalid_reference = ValidatedReference(
        success=False,
        error="The reference does not exist.",
    )
    pipeline, _, _ = make_pipeline(transcription, interpretation)

    with patch(
        "biblecast.ai.audio_reference_pipeline.validate_reference",
        return_value=invalid_reference,
    ):
        result = pipeline.process("sermon.wav", "WEB")

    assert result.success is False
    assert result.reference == interpretation
    assert result.validated_reference == invalid_reference
    assert result.error == "The reference does not exist."


def test_uses_fallback_error_messages_when_errors_are_empty():
    transcription = TranscriptionResult(
        success=False,
        error="",
    )
    pipeline, _, _ = make_pipeline(
        transcription,
        BibleReferenceResult(success=False),
    )

    result = pipeline.process("sermon.wav", "WEB")

    assert result.success is False
    assert result.error == "Audio transcription failed."
