
from biblecast.bible.navigation import find_verse_index


def make_verse(book_name, chapter, verse):
    return {
        "book_name": book_name,
        "chapter": chapter,
        "verse": verse,
        "text": f"Text of {book_name} {chapter}:{verse}",
    }


def test_find_verse_index_returns_correct_position():
    verses = [
        make_verse("Genesis", 1, 1),
        make_verse("Genesis", 1, 2),
        make_verse("Genesis", 2, 1),
        make_verse("Exodus", 1, 1),
    ]

    assert find_verse_index(verses, "Genesis", 1, 1) == 0
    assert find_verse_index(verses, "Genesis", 2, 1) == 2
    assert find_verse_index(verses, "Exodus", 1, 1) == 3


def test_find_verse_index_ignores_book_name_case():
    verses = [make_verse("John", 3, 16)]

    assert find_verse_index(verses, "john", 3, 16) == 0


def test_find_verse_index_returns_none_when_missing():
    verses = [
        make_verse("John", 3, 16),
        make_verse("John", 3, 17),
    ]

    assert find_verse_index(verses, "John", 3, 18) is None
    assert find_verse_index(verses, "Mark", 1, 1) is None
