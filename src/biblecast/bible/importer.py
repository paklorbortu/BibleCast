
import json
import sqlite3
from pathlib import Path

from biblecast.bible.database import get_connection



def import_translation(
    json_path: str | Path,
    *,
    replace_existing: bool = False,
) -> int:
    """Import a Bible translation from a UTF-8 JSON file."""

    path = Path(json_path)

    with path.open("r", encoding="utf-8") as file:
        data = json.load(file)

    required = ("code", "name", "books")
    if any(not data.get(key) for key in required):
        raise ValueError("Translation must include code, name and books.")

    code = str(data["code"]).strip().upper()
    name = str(data["name"]).strip()

    if not code or not name:
        raise ValueError("Translation code and name cannot be empty.")

    books = data["books"]
    if not isinstance(books, list) or not books:
        raise ValueError("Translation must contain a non-empty books list.")

    seen_books = set()
    seen_book_numbers = set()
    total_verses = 0

    for book in books:
        book_name = str(book.get("name", "")).strip()
        book_number = book.get("number")
        testament = book.get("testament")
        chapters = book.get("chapters")

        if not book_name or not isinstance(book_number, int):
            raise ValueError("Each book needs a name and integer number.")

        if book_number < 1 or book_name.casefold() in seen_books:
            raise ValueError(f"Invalid or duplicate book: {book_name}")

        if book_number in seen_book_numbers:
            raise ValueError(f"Duplicate book number: {book_number}")

        if testament not in ("OT", "NT"):
            raise ValueError(f"Invalid testament for {book_name}.")

        if not isinstance(chapters, list) or not chapters:
            raise ValueError(f"{book_name} must contain chapters.")

        seen_books.add(book_name.casefold())
        seen_book_numbers.add(book_number)
        seen_chapters = set()

        for chapter in chapters:
            chapter_number = chapter.get("number")
            verses = chapter.get("verses")

            if not isinstance(chapter_number, int) or chapter_number < 1:
                raise ValueError(f"Invalid chapter in {book_name}.")

            if chapter_number in seen_chapters:
                raise ValueError(f"Duplicate chapter in {book_name}.")

            if not isinstance(verses, list) or not verses:
                raise ValueError(
                    f"{book_name} chapter {chapter_number} has no verses."
                )

            seen_chapters.add(chapter_number)
            seen_verses = set()

            for verse in verses:
                verse_number = verse.get("number")
                text = verse.get("text")

                if not isinstance(verse_number, int) or verse_number < 1:
                    raise ValueError(
                        f"Invalid verse number in {book_name}."
                    )

                if verse_number in seen_verses:
                    raise ValueError(
                        f"Duplicate verse in {book_name} "
                        f"{chapter_number}."
                    )

                if not isinstance(text, str) or not text.strip():
                    raise ValueError(
                        f"Empty verse text in {book_name} "
                        f"{chapter_number}:{verse_number}."
                    )

                seen_verses.add(verse_number)
                total_verses += 1

    # Validate everything before modifying the database.
    with get_connection() as connection:
        existing = connection.execute(
            "SELECT id FROM translations WHERE code = ?",
            (code,),
        ).fetchone()

        if existing and not replace_existing:
            raise ValueError(
                f"Translation {code} already exists. "
                "Set replace_existing=True to replace it explicitly."
            )

        if existing:
            connection.execute(
                "DELETE FROM translations WHERE id = ?",
                (existing["id"],),
            )

        cursor = connection.execute(
            """
            INSERT INTO translations
                (code, name, language, source, license_name,
                 license_url, is_installed)
            VALUES (?, ?, ?, ?, ?, ?, 0)
            """,
            (
                code,
                name,
                data.get("language", "en"),
                data.get("source"),
                data.get("license_name"),
                data.get("license_url"),
            ),
        )
        translation_id = cursor.lastrowid

        for book in books:
            cursor = connection.execute(
                """
                INSERT INTO books
                    (translation_id, book_number, name, testament, chapters)
                VALUES (?, ?, ?, ?, ?)
                """,
                (
                    translation_id,
                    book["number"],
                    book["name"].strip(),
                    book["testament"],
                    len(book["chapters"]),
                ),
            )
            book_id = cursor.lastrowid

            for chapter in book["chapters"]:
                for verse in chapter["verses"]:
                    connection.execute(
                        """
                        INSERT INTO verses
                            (translation_id, book_id, chapter, verse, text)
                        VALUES (?, ?, ?, ?, ?)
                        """,
                        (
                            translation_id,
                            book_id,
                            chapter["number"],
                            verse["number"],
                            verse["text"].strip(),
                        ),
                    )

        connection.execute(
            """
            UPDATE translations
            SET is_installed = 1
            WHERE id = ?
            """,
            (translation_id,),
        )

    return total_verses
