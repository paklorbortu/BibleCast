
import re

from biblecast.bible.database import get_verse


_ORDINALS = {
    "first": "1", "1st": "1",
    "second": "2", "2nd": "2",
    "third": "3", "3rd": "3",
}

# Spoken or common alternative names mapped to the stored book names.
_BOOK_ALIASES = {
    "psalm": "Psalms",
    "psalms": "Psalms",
    "song of songs": "Song of Solomon",
    "songs of solomon": "Song of Solomon",
    "canticles": "Song of Solomon",
    "revelations": "Revelation",
    "revelation of john": "Revelation",
    "acts of the apostles": "Acts",
    "philippian": "Philippians",
    "ephesian": "Ephesians",
    "colossian": "Colossians",
    "galatian": "Galatians",
}


# Names used to accept a chapter-only reference ("Job chapter 38"). Requiring
# a real book name keeps ordinary speech such as "point 3" from matching.
_BOOK_NAMES = (
    "Genesis", "Exodus", "Leviticus", "Numbers", "Deuteronomy", "Joshua",
    "Judges", "Ruth", "1 Samuel", "2 Samuel", "1 Kings", "2 Kings",
    "1 Chronicles", "2 Chronicles", "Ezra", "Nehemiah", "Esther", "Job",
    "Psalms", "Proverbs", "Ecclesiastes", "Song of Solomon", "Isaiah",
    "Jeremiah", "Lamentations", "Ezekiel", "Daniel", "Hosea", "Joel",
    "Amos", "Obadiah", "Jonah", "Micah", "Nahum", "Habakkuk", "Zephaniah",
    "Haggai", "Zechariah", "Malachi", "Matthew", "Mark", "Luke", "John",
    "Acts", "Romans", "1 Corinthians", "2 Corinthians", "Galatians",
    "Ephesians", "Philippians", "Colossians", "1 Thessalonians",
    "2 Thessalonians", "1 Timothy", "2 Timothy", "Titus", "Philemon",
    "Hebrews", "James", "1 Peter", "2 Peter", "1 John", "2 John",
    "3 John", "Jude", "Revelation",
)
KNOWN_BOOKS = frozenset(name.casefold() for name in _BOOK_NAMES)


def canonical_book_name(name):
    """Map a spoken or alternative book name to its stored form.

    Examples: "Psalm" -> "Psalms", "first John" -> "1 John".
    Unknown names are returned unchanged so exact names still work.
    """
    words = name.split()
    if not words:
        return name

    prefix = ""
    first = words[0].casefold().rstrip(".")
    if len(words) > 1 and first in _ORDINALS:
        prefix = _ORDINALS[first] + " "
        words = words[1:]
    elif len(words) > 1 and words[0] in {"1", "2", "3"}:
        prefix = words[0] + " "
        words = words[1:]

    base = " ".join(words)
    base = _BOOK_ALIASES.get(base.casefold(), base)
    return (prefix + base).strip()


REFERENCE_PATTERN = re.compile(
    r"""
    ^\s*
    (?P<book>
        (?:[1-3]\s*)?
        [A-Za-z]+
        (?:\s+[A-Za-z]+)*
    )
    \s+
    (?:(?:chapter)\s+)?
    (?P<chapter>\d+)
    \s*
    (?:
        :\s*|\s+(?:verse|verses)\s+
    )
    (?P<verse>\d+)
    (?:
        \s*(?:-|to)\s*
        (?:(?:verse|verses)\s+)?
        (?P<end_verse>\d+)
    )?
    \s*$
    """,
    re.IGNORECASE | re.VERBOSE,
)



def parse_reference(reference):
    """Parse a reference such as John 3:16 or John 3:16-20."""
    match = REFERENCE_PATTERN.match(reference)

    if not match:
        return None

    book = " ".join(match.group("book").split())
    book = re.sub(r"\s+chapter\s*$", "", book, flags=re.IGNORECASE)
    book = canonical_book_name(book)

    chapter = int(match.group("chapter"))
    verse = int(match.group("verse"))
    end_verse = match.group("end_verse")

    if chapter < 1 or verse < 1:
        return None

    if end_verse is not None:
        end_verse = int(end_verse)
        if end_verse < verse:
            return None

    return {
        "book": book,
        "chapter": chapter,
        "verse": verse,
        "end_verse": end_verse,
    }



CHAPTER_PATTERN = re.compile(
    r"""
    ^\s*
    (?P<book>
        (?:[1-3]\s*)?
        [A-Za-z]+
        (?:\s+[A-Za-z]+)*?
    )
    \s+
    (?:chapter\s+)?
    (?P<chapter>\d+)
    \s*$
    """,
    re.IGNORECASE | re.VERBOSE,
)


def parse_chapter_reference(reference):
    """Parse a book and chapter with no verse, such as "Job chapter 38".

    The result points at verse 1 so the operator sees the start of the
    chapter. Only real Bible book names are accepted.
    """
    match = CHAPTER_PATTERN.match(reference)
    if not match:
        return None

    book = canonical_book_name(" ".join(match.group("book").split()))
    chapter = int(match.group("chapter"))

    if chapter < 1 or book.casefold() not in KNOWN_BOOKS:
        return None

    return {
        "book": book,
        "chapter": chapter,
        "verse": 1,
        "end_verse": None,
        "chapter_only": True,
    }


def resolve_reference(
    translation_code,
    reference,
    allow_chapter_only=False,
):
    """Resolve a reference to its first verse and optional range.

    With allow_chapter_only, "Job 38" resolves to the first verse of the
    chapter and the result is marked with chapter_only=True.
    """
    parsed = parse_reference(reference)

    if parsed is None and allow_chapter_only:
        parsed = parse_chapter_reference(reference)

    if parsed is None:
        return None

    verse = get_verse(
        translation_code,
        parsed["book"],
        parsed["chapter"],
        parsed["verse"],
    )

    if verse is None:
        return None

    return {
        **parsed,
        "first_verse": verse,
        "is_range": parsed["end_verse"] is not None,
    }
