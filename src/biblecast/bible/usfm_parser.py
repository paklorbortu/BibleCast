
import re


_MARKER_PATTERN = re.compile(r"\\(id|h|c|v)\s+([^\s]+)")
_BOOK_ID_PATTERN = re.compile(r"(?m)^\\id\s+(\S+)")
_BOOK_NAME_PATTERN = re.compile(r"(?m)^\\h\s+(.+?)\s*$")



def _clean_verse_text(text: str) -> str:
    """Remove USFM formatting and footnotes while preserving verse text."""

    # Remove footnotes and cross-references, including their contents.
    text = re.sub(r"\\f\b.*?\\f\*", " ", text, flags=re.DOTALL)
    text = re.sub(r"\\x\b.*?\\x\*", " ", text, flags=re.DOTALL)

    # Remove caller markers that sometimes appear before footnotes.
    text = re.sub(r"\\[+]?fr\b", " ", text)
    text = re.sub(r"\\[+]?ft\b", " ", text)

    # Convert word markup to its displayed text.
    # Example: \w For|strong="G1063"\w* becomes For
    text = re.sub(
        r"\\\+?w\s+(.*?)\|[^\\]*?\\\+?w\*",
        r"\1",
        text,
        flags=re.DOTALL,
    )

    # Remove any remaining Strong's-style attributes that escaped word markup.
    text = re.sub(r'\|strong="[^"]*"', "", text)
    text = re.sub(r'\|lemma="[^"]*"', "", text)
    text = re.sub(r'\|x-morph="[^"]*"', "", text)

    # Remove remaining USFM markers while preserving their surrounding text.
    text = re.sub(r"\\[+a-zA-Z][a-zA-Z0-9+_-]*\*", "", text)
    text = re.sub(r"\\[+a-zA-Z][a-zA-Z0-9+_-]*\b", "", text)

    # Normalize whitespace introduced by markup and line breaks.
    text = re.sub(r"\s+", " ", text).strip()
    return text



def parse_usfm(content: str) -> dict:
    """Parse one USFM book into book metadata and numbered verses."""

    book_id_match = _BOOK_ID_PATTERN.search(content)
    book_name_match = _BOOK_NAME_PATTERN.search(content)

    if not book_id_match:
        raise ValueError("USFM content is missing a book ID.")

    if not book_name_match:
        raise ValueError("USFM content is missing a book heading.")

    book_id = book_id_match.group(1)
    book_name = book_name_match.group(1).strip()

    markers = list(_MARKER_PATTERN.finditer(content))
    chapters = {}
    current_chapter = None

    for index, marker in enumerate(markers):
        marker_type = marker.group(1)
        marker_value = marker.group(2)

        if marker_type == "c":
            try:
                current_chapter = int(marker_value)
            except ValueError as exc:
                raise ValueError(
                    f"Invalid chapter number: {marker_value}"
                ) from exc

            if current_chapter < 1:
                raise ValueError("Chapter numbers must be positive.")

            chapters.setdefault(current_chapter, [])

        elif marker_type == "v":
            if current_chapter is None:
                raise ValueError("Found a verse before a chapter marker.")

            try:
                verse_number = int(marker_value)
            except ValueError as exc:
                raise ValueError(
                    f"Invalid verse number: {marker_value}"
                ) from exc

            if verse_number < 1:
                raise ValueError("Verse numbers must be positive.")

            end = (
                markers[index + 1].start()
                if index + 1 < len(markers)
                else len(content)
            )

            
            raw_verse_text = content[marker.end():end]
            verse_text = _clean_verse_text(raw_verse_text)

            # Some USFM editions include verse markers whose only content
            # is a footnote describing an omitted or alternate-text verse.
            # Do not store footnote commentary as scripture.
            if not verse_text:
                if re.search(r"\\f\b.*?\\f\*", raw_verse_text, re.DOTALL):
                    continue

                raise ValueError(
                    f"Verse {current_chapter}:{verse_number} is empty."
                )

            if any(
                item["number"] == verse_number
                for item in chapters[current_chapter]
            ):
                raise ValueError(
                    f"Duplicate verse {current_chapter}:{verse_number}."
                )

            chapters[current_chapter].append(
                {"number": verse_number, "text": verse_text}
            )

    if not chapters or not any(chapters.values()):
        raise ValueError("No chapter verses were found in the USFM content.")

    return {
        "book_id": book_id,
        "name": book_name,
        "chapters": [
            {
                "number": chapter_number,
                "verses": sorted(verses, key=lambda verse: verse["number"]),
            }
            for chapter_number, verses in sorted(chapters.items())
        ],
    }
