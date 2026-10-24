
import subprocess
from pathlib import Path
from unittest.mock import MagicMock

from biblecast.ai.gemma_text import GemmaTextAdapter


def make_adapter(tmp_path: Path) -> GemmaTextAdapter:
    model = tmp_path / "gemma.gguf"
    model.write_bytes(b"test model")

    adapter = GemmaTextAdapter(model_path=model)
    return adapter


def mock_successful_process(monkeypatch, output: str):
    completed = MagicMock()
    completed.returncode = 0
    completed.stdout = output
    completed.stderr = ""

    monkeypatch.setattr(
        "biblecast.ai.gemma_text.shutil.which",
        lambda executable: "llama-cli.exe",
    )
    monkeypatch.setattr(
        "biblecast.ai.gemma_text.subprocess.run",
        lambda *args, **kwargs: completed,
    )


def test_extracts_explicit_reference(tmp_path, monkeypatch):
    adapter = make_adapter(tmp_path)
    mock_successful_process(
        monkeypatch,
        "BOOK=John;CHAPTER=3;VERSE=16;CONFIDENCE=high",
    )

    result = adapter.interpret("What is John three sixteen?")

    assert result.success
    assert result.book == "John"
    assert result.chapter == "3"
    assert result.verse == "16"
    assert result.confidence == "high"


def test_extracts_quoted_passage(tmp_path, monkeypatch):
    adapter = make_adapter(tmp_path)
    mock_successful_process(
        monkeypatch,
        "BOOK=Psalm;CHAPTER=23;VERSE=1;CONFIDENCE=high",
    )

    result = adapter.interpret("The Lord is my shepherd.")

    assert result.success
    assert result.book == "Psalm"
    assert result.chapter == "23"
    assert result.verse == "1"


def test_unknown_reference_is_low_confidence(tmp_path, monkeypatch):
    adapter = make_adapter(tmp_path)
    mock_successful_process(
        monkeypatch,
        "BOOK=UNKNOWN;CHAPTER=UNKNOWN;"
        "VERSE=UNKNOWN;CONFIDENCE=low",
    )

    result = adapter.interpret("A sentence with no identifiable passage.")

    assert result.success
    assert result.book == "UNKNOWN"
    assert result.chapter == "UNKNOWN"
    assert result.verse == "UNKNOWN"
    assert result.confidence == "low"


def test_empty_transcript_is_rejected(tmp_path):
    adapter = make_adapter(tmp_path)

    result = adapter.interpret("   ")

    assert not result.success
    assert "empty" in result.error.lower()


def test_missing_model_returns_error(tmp_path, monkeypatch):
    adapter = GemmaTextAdapter(model_path=tmp_path / "missing.gguf")

    monkeypatch.setattr(
        "biblecast.ai.gemma_text.shutil.which",
        lambda executable: "llama-cli.exe",
    )

    result = adapter.interpret("Open John 3:16.")

    assert not result.success
    assert "model file not found" in result.error.lower()


def test_missing_executable_returns_error(tmp_path, monkeypatch):
    adapter = make_adapter(tmp_path)

    monkeypatch.setattr(
        "biblecast.ai.gemma_text.shutil.which",
        lambda executable: None,
    )

    result = adapter.interpret("Open John 3:16.")

    assert not result.success
    assert "executable" in result.error.lower()


def test_timeout_returns_error(tmp_path, monkeypatch):
    adapter = make_adapter(tmp_path)

    monkeypatch.setattr(
        "biblecast.ai.gemma_text.shutil.which",
        lambda executable: "llama-cli.exe",
    )

    def raise_timeout(*args, **kwargs):
        raise subprocess.TimeoutExpired("llama-cli", 90)

    monkeypatch.setattr(
        "biblecast.ai.gemma_text.subprocess.run",
        raise_timeout,
    )

    result = adapter.interpret("Open John 3:16.")

    assert not result.success
    assert "timed out" in result.error.lower()


def test_malformed_output_is_rejected(tmp_path, monkeypatch):
    adapter = make_adapter(tmp_path)
    mock_successful_process(monkeypatch, "I think it is John chapter three.")

    result = adapter.interpret("Open John 3:16.")

    assert not result.success
    assert "unrecognized" in result.error.lower()


def test_missing_reference_fields_cannot_remain_high_confidence(
    tmp_path, monkeypatch
):
    adapter = make_adapter(tmp_path)
    mock_successful_process(
        monkeypatch,
        "BOOK=John;CHAPTER=3;VERSE=UNKNOWN;CONFIDENCE=high",
    )

    result = adapter.interpret("John chapter three.")

    assert result.success
    assert result.verse == "UNKNOWN"
    assert result.confidence == "low"
