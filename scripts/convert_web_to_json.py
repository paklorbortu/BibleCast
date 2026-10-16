
import json
import re
from pathlib import Path

from biblecast.bible.usfm_parser import parse_usfm


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SOURCE_DIR = PROJECT_ROOT / "data" / "translations" / "WEB" / "usfm_source"
OUTPUT_PATH = PROJECT_ROOT / "data" / "translations" / "WEB" / "web_translation.json"

# Canonical Protestant Bible order, used to assign stable book numbers.
BIBLE_BOOKS = [
    "Genesis", "Exodus", "Leviticus", "Numbers", "Deuteronomy",
    "Joshua", "Judges", "Ruth", "1 Samuel", "2 Samuel",
    "1 Kings", "2 Kings", "1 Chronicles", "2 Chronicles", "Ezra",
    "Nehemiah", "Esther", "Job", "Psalms", "Proverbs",
    "Ecclesiastes", "Song of Solomon", "Isaiah", "Jeremiah",
    "Lamentations", "Ezekiel", "Daniel", "Hosea", "Joel", "Amos",
    "Obadiah", "Jonah", "Micah", "Nahum", "Habakkuk", "Zephaniah",
    "Haggai", "Zechariah", "Malachi",
    "Matthew", "Mark", "Luke", "John", "Acts", "Romans",
    "1 Corinthians", "2 Corinthians", "Galatians", "Ephesians",
    "Philippians", "Colossians", "1 Thessalonians",
    "2 Thessalonians", "1 Timothy", "2 Timothy", "Titus",
    "Philemon", "Hebrews", "James", "1 Peter", "2 Peter",
    "1 John", "2 John", "3 John", "Jude", "Revelation",
]


def main():
    files = sorted(SOURCE_DIR.glob("*.usfm"))
    books_by_name = {}
    errors = []

    for path in files:
        content = path.read_text(encoding="utf-8")

        # Ignore front matter and glossary files without chapter/verse markers.
        if r"\c " not in content or r"\v " not in content:
            continue

        try:
            parsed = parse_usfm(content)
            name = parsed["name"].strip()

            if name in books_by_name:
                raise ValueError(f"Duplicate book name: {name}")

            books_by_name[name] = parsed

        except Exception as exc:
            errors.append(f"{path.name}: {exc}")

    if errors:
        raise RuntimeError(
            "Conversion stopped because some books failed to parse:\n"
            + "\n".join(errors)
        )

    missing = [name for name in BIBLE_BOOKS if name not in books_by_name]
    unexpected = [name for name in books_by_name if name not in BIBLE_BOOKS]

    if missing or unexpected or len(books_by_name) != 66:
        raise RuntimeError(
            f"Book verification failed. Missing: {missing}; "
            f"unexpected: {unexpected}; parsed: {len(books_by_name)}"
        )

    books = []

    for number, name in enumerate(BIBLE_BOOKS, start=1):
        parsed = books_by_name[name]

        books.append({
            "number": number,
            "name": name,
            "testament": "OT" if number <= 39 else "NT",
            "chapters": parsed["chapters"],
        })

    translation = {
        "code": "WEB",
        "name": "World English Bible",
        "language": "en",
        "source": "https://ebible.org/engwebp/",
        "license_name": "Public domain",
        "license_url": "https://ebible.org/copyright.htm",
        "books": books,
    }

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_PATH.write_text(
        json.dumps(translation, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    chapter_count = sum(len(book["chapters"]) for book in books)
    verse_count = sum(
        len(chapter["verses"])
        for book in books
        for chapter in book["chapters"]
    )

    print("WEB conversion completed successfully.")
    print(f"Books: {len(books)}")
    print(f"Chapters: {chapter_count}")
    print(f"Verses: {verse_count}")
    print(f"Output: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
