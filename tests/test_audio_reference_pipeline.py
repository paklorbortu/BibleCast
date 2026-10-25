
from pathlib import Path
from unittest.mock import patch

import numpy as np

from biblecast.ai.audio_reference_pipeline import AudioReferencePipeline
from biblecast.ai.passage_candidates import PassageCandidate
from biblecast.ai.whisper_transcriber import TranscriptionResult


class FakeTranscriber:
    def __init__(self, result):
        self.result = result
        self.audio_path = None

    def transcribe(self, audio_path):
        self.audio_path = audio_path
        return self.result

    def transcribe_audio(self, audio, sample_rate=16_000):
        return self.result


def make_pipeline(transcription):
    transcriber = FakeTranscriber(transcription)
    return AudioReferencePipeline(transcriber=transcriber), transcriber


def sample_candidates():
    return [
        PassageCandidate(
            reference="Joshua 1:9",
            text="Be strong and courageous.",
            score=1.0,
            match_type="quote",
        ),
        PassageCandidate(
            reference="Deuteronomy 31:6",
            text="Be strong and courageous. Do not fear.",
            score=1.0,
            match_type="quote",
        ),
    ]


def test_returns_ranked_candidates_from_transcription():
    transcription = TranscriptionResult(
        success=True,
        text="Be strong and courageous.",
        language="en",
    )
    pipeline, transcriber = make_pipeline(transcription)
    candidates = sample_candidates()

    with patch(
        "biblecast.ai.audio_reference_pipeline.find_passage_candidates",
        return_value=candidates,
    ) as find_candidates:
        result = pipeline.process("sermon.wav", "WEB", limit=2)

    assert result.success is True
    assert result.transcription == transcription
    assert result.candidates == tuple(candidates)
    assert result.error == ""
    assert transcriber.audio_path == "sermon.wav"
    find_candidates.assert_called_once_with(
        "WEB",
        "Be strong and courageous.",
        limit=2,
    )


def test_returns_error_when_transcription_fails():
    transcription = TranscriptionResult(
        success=False,
        error="Audio file could not be decoded.",
    )
    pipeline, _ = make_pipeline(transcription)

    with patch(
        "biblecast.ai.audio_reference_pipeline.find_passage_candidates"
    ) as find_candidates:
        result = pipeline.process(Path("sermon.wav"), "WEB")

    assert result.success is False
    assert result.error == "Audio file could not be decoded."
    assert result.candidates == ()
    find_candidates.assert_not_called()


def test_rejects_empty_transcription():
    transcription = TranscriptionResult(
        success=True,
        text="   ",
        language="en",
    )
    pipeline, _ = make_pipeline(transcription)

    with patch(
        "biblecast.ai.audio_reference_pipeline.find_passage_candidates"
    ) as find_candidates:
        result = pipeline.process("sermon.wav", "WEB")

    assert result.success is False
    assert result.error == "No speech was recognized."
    assert result.candidates == ()
    find_candidates.assert_not_called()


def test_returns_error_when_no_candidates_are_found():
    transcription = TranscriptionResult(
        success=True,
        text="Unrelated speech.",
        language="en",
    )
    pipeline, _ = make_pipeline(transcription)

    with patch(
        "biblecast.ai.audio_reference_pipeline.find_passage_candidates",
        return_value=[],
    ):
        result = pipeline.process("sermon.wav", "WEB")

    assert result.success is False
    assert result.error == "No matching Bible passages were found."
    assert result.candidates == ()


def test_uses_fallback_error_when_transcription_error_is_empty():
    transcription = TranscriptionResult(
        success=False,
        error="",
    )
    pipeline, _ = make_pipeline(transcription)

    result = pipeline.process("sermon.wav", "WEB")

    assert result.success is False
    assert result.error == "Audio transcription failed."


def test_pipeline_only_suggests_candidates():
    transcription = TranscriptionResult(
        success=True,
        text="Open John three sixteen.",
        language="en",
    )
    pipeline, _ = make_pipeline(transcription)

    with patch(
        "biblecast.ai.audio_reference_pipeline.find_passage_candidates",
        return_value=[
            PassageCandidate(
                reference="John 3:16",
                text="For God so loved the world...",
                score=1.0,
                match_type="explicit",
            )
        ],
    ):
        result = pipeline.process("sermon.wav", "WEB")

    assert result.success is True
    assert result.candidates[0].reference == "John 3:16"
    assert not hasattr(result, "go_live")
    assert not hasattr(result, "approved")


def test_merge_overlapping_speech_removes_repeated_words():
    from biblecast.ai.audio_reference_pipeline import (
        merge_overlapping_speech,
    )

    assert merge_overlapping_speech(
        "the Lord is my", "is my shepherd I shall lack nothing"
    ) == "the Lord is my shepherd I shall lack nothing"
    assert merge_overlapping_speech(
        "hello there", "completely different words"
    ) == "hello there completely different words"


def test_live_quote_split_across_two_segments_is_found():
    first = TranscriptionResult(
        success=True, text="the Lord is my", language="en"
    )
    second = TranscriptionResult(
        success=True, text="is my shepherd I shall lack nothing",
        language="en",
    )
    transcriber = FakeTranscriber(first)
    pipeline = AudioReferencePipeline(transcriber=transcriber)
    psalm = PassageCandidate(
        reference="Psalms 23:1",
        text="The LORD is my shepherd; I shall lack nothing.",
        score=1.0,
        match_type="quote",
        confidence="high",
    )

    def fake_find(translation_code, text, limit=5):
        return [psalm] if "shepherd" in text and "the lord" in text.lower() else []

    with patch(
        "biblecast.ai.audio_reference_pipeline.find_passage_candidates",
        side_effect=fake_find,
    ):
        audio = np.zeros(16_000, dtype=np.float32)
        assert not pipeline.process_audio(audio, "WEB").success

        transcriber.result = second
        result = pipeline.process_audio(audio, "WEB")

    assert result.success
    assert result.candidates[0].reference == "Psalms 23:1"

