
from biblecast.bible.reference_resolver import parse_reference


def test_parse_single_verse():
    assert parse_reference("John 3:16") == {
        "book": "John",
        "chapter": 3,
        "verse": 16,
        "end_verse": None,
    }


def test_parse_verse_range():
    assert parse_reference("John 3:16 to verse 20") == {
        "book": "John",
        "chapter": 3,
        "verse": 16,
        "end_verse": 20,
    }


def test_parse_chapter_and_verse_words():
    assert parse_reference("John chapter 3 verse 16") == {
        "book": "John",
        "chapter": 3,
        "verse": 16,
        "end_verse": None,
    }


def test_reject_invalid_reference_format():
    assert parse_reference("John") is None
    assert parse_reference("John 3:20-16") is None
