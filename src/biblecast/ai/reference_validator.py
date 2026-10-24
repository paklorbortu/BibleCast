
"""Validate Gemma's Bible reference against an installed translation."""

from __future__ import annotations

import re
from dataclasses import dataclass

from biblecast.ai.gemma_text import BibleReferenceResult
from biblecast.bible.database import (
    get_verse,
    get_verse_range,
    list_books,
)


_VERSE_PATTERN = re.compile(
    r"^\s*(\d+)(?:\s*(?:-|to)\s*(?:verse\s*)?(\d+))?\s*$",
    re.IGNORECASE,
)


@dataclass(frozen=True)
class ValidatedReference:
    """A verified reference and its scripture text."""

    success: bool
    book: str = ""
    chapter: int | None = None
    start_verse: int | None = None
    end_verse: int | None = None
    verses: tuple[dict, ...] = ()
    error: str = ""


def validate_reference(
    translation_code: str,
    result: BibleReferenceResult,
) -> ValidatedReference:
    """Validate a Gemma result and retrieve the complete passage."""

    if not result.success:
        return ValidatedReference(
            success=False,
            error=result.error or "Gemma did not return a usable result.",
        )

    if result.confidence not in {"medium", "high"}:
        return ValidatedReference(
            success=False,
            error="Reference confidence is too low for preview.",
        )

    book = result.book.strip()
    if not book or book.upper() == "UNKNOWN":
        return ValidatedReference(
            success=False,
            error="The Bible book could not be identified.",
        )

    if (
        not result.chapter.strip().isdigit()
        or int(result.chapter) < 1
    ):
        return ValidatedReference(
            success=False,
            error="The chapter is missing or invalid.",
        )

    match = _VERSE_PATTERN.fullmatch(result.verse.strip())
    if not match:
        return ValidatedReference(
            success=False,
            error="The verse number or range is missing or invalid.",
        )

    chapter = int(result.chapter)
    start_verse = int(match.group(1))
    end_verse = int(match.group(2) or start_verse)

    if start_verse < 1 or end_verse < start_verse:
        return ValidatedReference(
            success=False,
            error="The verse range is invalid.",
        )

    installed_books = list_books(translation_code)
    canonical_book = next(
        (
            installed_book
            for installed_book in installed_books
            if installed_book.casefold() == book.casefold()
        ),
        None,
    )

    if canonical_book is None:
        return ValidatedReference(
            success=False,
            error=f"Book '{book}' is not available in {translation_code}.",
        )

    if start_verse == end_verse:
        verse = get_verse(
            translation_code,
            canonical_book,
            chapter,
            start_verse,
        )
        verses = [verse] if verse is not None else []
    else:
        verses = get_verse_range(
            translation_code,
            canonical_book,
            chapter,
            start_verse,
            end_verse,
        )

    if not verses:
        return ValidatedReference(
            success=False,
            error="The reference does not exist or the passage is incomplete.",
        )

    expected_count = end_verse - start_verse + 1
    if len(verses) != expected_count:
        return ValidatedReference(
            success=False,
            error="The passage is incomplete in the selected translation.",
        )

    return ValidatedReference(
        success=True,
        book=canonical_book,
        chapter=chapter,
        start_verse=start_verse,
        end_verse=end_verse,
        verses=tuple(verses),
    )

