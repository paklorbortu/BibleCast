
from biblecast.bible.reference_resolver import (
    canonical_book_name,
    parse_reference,
)


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


def test_spoken_book_names_map_to_stored_names():
    assert canonical_book_name("Psalm") == "Psalms"
    assert canonical_book_name("first John") == "1 John"
    assert canonical_book_name("Second Corinthians") == "2 Corinthians"
    assert canonical_book_name("3rd John") == "3 John"
    assert canonical_book_name("Revelations") == "Revelation"
    assert canonical_book_name("John") == "John"


def test_parse_reference_uses_book_aliases():
    assert parse_reference("Psalm 23:1")["book"] == "Psalms"
    assert parse_reference("first John 4:8")["book"] == "1 John"

