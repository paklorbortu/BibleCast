
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



def resolve_reference(translation_code, reference):
    """Resolve a reference to its first verse and optional range."""
    parsed = parse_reference(reference)

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
