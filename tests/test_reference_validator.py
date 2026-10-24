
from unittest.mock import patch

from biblecast.ai.gemma_text import BibleReferenceResult
from biblecast.ai.reference_validator import validate_reference


def make_result(
    book="John",
    chapter="3",
    verse="16",
    confidence="high",
    success=True,
):
    return BibleReferenceResult(
        success=success,
        book=book,
        chapter=chapter,
        verse=verse,
        confidence=confidence,
    )


def test_valid_single_verse_returns_scripture():
    scripture = {
        "book_name": "John",
        "chapter": 3,
        "verse": 16,
        "text": "For God so loved the world.",
    }

    with (
        patch(
            "biblecast.ai.reference_validator.list_books",
            return_value=["Genesis", "John", "Revelation"],
        ),
        patch(
            "biblecast.ai.reference_validator.get_verse",
            return_value=scripture,
        ) as get_verse,
    ):
        result = validate_reference("WEB", make_result())

    assert result.success
    assert result.book == "John"
    assert result.verses == (scripture,)
    get_verse.assert_called_once_with("WEB", "John", 3, 16)


def test_rejects_low_confidence():
    with patch(
        "biblecast.ai.reference_validator.list_books"
    ) as list_books:
        result = validate_reference(
            "WEB",
            make_result(confidence="low"),
        )

    assert not result.success
    assert "confidence" in result.error.lower()
    list_books.assert_not_called()


def test_rejects_unknown_book():
    with patch(
        "biblecast.ai.reference_validator.list_books",
        return_value=["Genesis", "John"],
    ):
        result = validate_reference(
            "WEB",
            make_result(book="NotABook"),
        )

    assert not result.success
    assert "not available" in result.error.lower()


def test_rejects_invalid_chapter():
    result = validate_reference(
        "WEB",
        make_result(chapter="UNKNOWN"),
    )

    assert not result.success
    assert "chapter" in result.error.lower()


def test_rejects_invalid_verse_format():
    result = validate_reference(
        "WEB",
        make_result(verse="UNKNOWN"),
    )

    assert not result.success
    assert "verse" in result.error.lower()


def test_rejects_nonexistent_single_verse():
    with (
        patch(
            "biblecast.ai.reference_validator.list_books",
            return_value=["John"],
        ),
        patch(
            "biblecast.ai.reference_validator.get_verse",
            return_value=None,
        ),
    ):
        result = validate_reference("WEB", make_result())

    assert not result.success


def test_valid_range_returns_all_verses():
    scripture = [
        {"book_name": "John", "chapter": 3, "verse": 16, "text": "Verse 16"},
        {"book_name": "John", "chapter": 3, "verse": 17, "text": "Verse 17"},
    ]

    with (
        patch(
            "biblecast.ai.reference_validator.list_books",
            return_value=["John"],
        ),
        patch(
            "biblecast.ai.reference_validator.get_verse_range",
            return_value=scripture,
        ) as get_range,
    ):
        result = validate_reference(
            "WEB",
            make_result(verse="16-17"),
        )

    assert result.success
    assert result.start_verse == 16
    assert result.end_verse == 17
    assert len(result.verses) == 2
    get_range.assert_called_once_with("WEB", "John", 3, 16, 17)


def test_rejects_incomplete_range():
    with (
        patch(
            "biblecast.ai.reference_validator.list_books",
            return_value=["John"],
        ),
        patch(
            "biblecast.ai.reference_validator.get_verse_range",
            return_value=[
                {"book_name": "John", "chapter": 3, "verse": 16, "text": "Verse 16"}
            ],
        ),
    ):
        result = validate_reference(
            "WEB",
            make_result(verse="16-17"),
        )

    assert not result.success
    assert "incomplete" in result.error.lower()


def test_rejects_failed_gemma_result():
    result = validate_reference(
        "WEB",
        make_result(success=False),
    )

    assert not result.success
