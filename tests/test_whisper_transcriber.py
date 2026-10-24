
from pathlib import Path
from unittest.mock import MagicMock

from biblecast.ai.whisper_transcriber import WhisperTranscriber


def test_missing_audio_returns_error(tmp_path: Path):
    transcriber = WhisperTranscriber()

    result = transcriber.transcribe(tmp_path / "missing.wav")

    assert not result.success
    assert "Audio file not found" in result.error


def test_transcription_forces_english(monkeypatch, tmp_path: Path):
    audio = tmp_path / "sample.wav"
    audio.write_bytes(b"test")

    segment = MagicMock()
    segment.text = " Please open Psalm 23 verse 1. "

    info = MagicMock()
    info.language = "en"

    model = MagicMock()
    model.transcribe.return_value = ([segment], info)

    transcriber = WhisperTranscriber()
    monkeypatch.setattr(transcriber, "_get_model", lambda: model)

    result = transcriber.transcribe(audio)

    assert result.success
    assert result.text == "Please open Psalm 23 verse 1."
    assert result.language == "en"

    model.transcribe.assert_called_once_with(
        str(audio),
        language="en",
        beam_size=5,
        vad_filter=False,
    )


def test_empty_transcription_returns_error(monkeypatch, tmp_path: Path):
    audio = tmp_path / "silence.wav"
    audio.write_bytes(b"test")

    info = MagicMock()
    info.language = "en"

    model = MagicMock()
    model.transcribe.return_value = ([], info)

    transcriber = WhisperTranscriber()
    monkeypatch.setattr(transcriber, "_get_model", lambda: model)

    result = transcriber.transcribe(audio)

    assert not result.success
    assert "No speech was recognized" in result.error
