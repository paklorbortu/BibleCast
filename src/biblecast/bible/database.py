
import os
import re
import sqlite3
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[3]
DATA_DIR = PROJECT_ROOT / "data"
DEFAULT_DATABASE_PATH = DATA_DIR / "biblecast.db"


def get_database_path():
    """Allow tests to use a separate temporary database."""
    override = os.environ.get("BIBLECAST_DATABASE_PATH")
    if override:
        return Path(override)
    return DEFAULT_DATABASE_PATH


def get_connection():
    database_path = get_database_path()
    database_path.parent.mkdir(parents=True, exist_ok=True)

    connection = sqlite3.connect(database_path)
    connection.row_factory = sqlite3.Row
    connection.execute("PRAGMA foreign_keys = ON")
    return connection


def initialize_database():
    with get_connection() as connection:
        connection.executescript(
            """
            CREATE TABLE IF NOT EXISTS translations (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                code TEXT NOT NULL UNIQUE,
                name TEXT NOT NULL,
                language TEXT NOT NULL DEFAULT 'en',
                source TEXT,
                license_name TEXT,
                license_url TEXT,
                is_installed INTEGER NOT NULL DEFAULT 0
                    CHECK (is_installed IN (0, 1))
            );

            CREATE TABLE IF NOT EXISTS books (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                translation_id INTEGER NOT NULL,
                book_number INTEGER NOT NULL,
                name TEXT NOT NULL,
                testament TEXT NOT NULL
                    CHECK (testament IN ('OT', 'NT')),
                chapters INTEGER NOT NULL CHECK (chapters > 0),
                UNIQUE (translation_id, book_number),
                UNIQUE (translation_id, name),
                FOREIGN KEY (translation_id)
                    REFERENCES translations(id) ON DELETE CASCADE
            );

            CREATE TABLE IF NOT EXISTS verses (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                translation_id INTEGER NOT NULL,
                book_id INTEGER NOT NULL,
                chapter INTEGER NOT NULL CHECK (chapter > 0),
                verse INTEGER NOT NULL CHECK (verse > 0),
                text TEXT NOT NULL,
                UNIQUE (translation_id, book_id, chapter, verse),
                FOREIGN KEY (translation_id)
                    REFERENCES translations(id) ON DELETE CASCADE,
                FOREIGN KEY (book_id)
                    REFERENCES books(id) ON DELETE CASCADE
            );

            CREATE INDEX IF NOT EXISTS idx_verses_reference
            ON verses (translation_id, book_id, chapter, verse);

            CREATE INDEX IF NOT EXISTS idx_books_name
            ON books (translation_id, name);
            """
        )


def list_translations(installed_only=True):
    with get_connection() as connection:
        if installed_only:
            rows = connection.execute(
                """
                SELECT code, name, language
                FROM translations
                WHERE is_installed = 1
                ORDER BY name
                """
            ).fetchall()
        else:
            rows = connection.execute(
                """
                SELECT code, name, language, is_installed
                FROM translations
                ORDER BY name
                """
            ).fetchall()

        return [dict(row) for row in rows]


def get_verse(translation_code, book_name, chapter, verse):
    with get_connection() as connection:
        row = connection.execute(
            """
            SELECT
                t.code AS translation_code,
                t.name AS translation_name,
                b.name AS book_name,
                v.chapter,
                v.verse,
                v.text
            FROM verses v
            JOIN translations t
                ON t.id = v.translation_id
            JOIN books b
                ON b.id = v.book_id
            WHERE t.code = ?
              AND LOWER(b.name) = LOWER(?)
              AND v.chapter = ?
              AND v.verse = ?
              AND t.is_installed = 1
            """,
            (translation_code, book_name, chapter, verse),
        ).fetchone()

        return dict(row) if row else None



def get_book_verses(translation_code, book_name, chapter):
    """Return all verses in a chapter in canonical verse order."""
    with get_connection() as connection:
        rows = connection.execute(
            """
            SELECT
                t.code AS translation_code,
                t.name AS translation_name,
                b.name AS book_name,
                v.chapter,
                v.verse,
                v.text
            FROM verses v
            JOIN translations t ON t.id = v.translation_id
            JOIN books b ON b.id = v.book_id
            WHERE t.code = ?
              AND LOWER(b.name) = LOWER(?)
              AND v.chapter = ?
              AND t.is_installed = 1
            ORDER BY v.verse
            """,
            (translation_code, book_name, chapter),
        ).fetchall()

        return [dict(row) for row in rows]



def search_verses(translation_code, query, limit=50):
    """Find verses by a phrase, ignoring punctuation and case."""
    query = re.sub(r"[^\w\s]", " ", query.casefold())
    query = " ".join(query.split())

    if not query:
        return []

    with get_connection() as connection:
        rows = connection.execute(
            """
            SELECT
                t.code AS translation_code,
                t.name AS translation_name,
                b.name AS book_name,
                b.book_number,
                v.chapter,
                v.verse,
                v.text
            FROM verses v
            JOIN translations t ON t.id = v.translation_id
            JOIN books b ON b.id = v.book_id
            WHERE t.code = ?
              AND t.is_installed = 1
            ORDER BY b.book_number, v.chapter, v.verse
            """,
            (translation_code,),
        ).fetchall()

    results = []

    for row in rows:
        verse = dict(row)
        normalized_text = re.sub(
            r"[^\w\s]", " ", verse["text"].casefold()
        )
        normalized_text = " ".join(normalized_text.split())

        if query in normalized_text:
            verse.pop("book_number", None)
            results.append(verse)

            if len(results) >= limit:
                break

    return results


def get_verse_range(translation_code, book_name, chapter, start_verse, end_verse):
    """Return a validated range of verses from a single chapter."""
    if start_verse < 1 or end_verse < start_verse:
        return []

    with get_connection() as connection:
        rows = connection.execute(
            """
            SELECT
                t.code AS translation_code,
                t.name AS translation_name,
                b.name AS book_name,
                v.chapter,
                v.verse,
                v.text
            FROM verses v
            JOIN translations t ON t.id = v.translation_id
            JOIN books b ON b.id = v.book_id
            WHERE t.code = ?
              AND LOWER(b.name) = LOWER(?)
              AND v.chapter = ?
              AND v.verse BETWEEN ? AND ?
              AND t.is_installed = 1
            ORDER BY v.verse
            """,
            (
                translation_code,
                book_name,
                chapter,
                start_verse,
                end_verse,
            ),
        ).fetchall()

    verses = [dict(row) for row in rows]

    # Reject incomplete ranges instead of silently casting only part.
    expected_count = end_verse - start_verse + 1
    if len(verses) != expected_count:
        return []

    return verses



def list_books(translation_code):
    """Return the installed Bible books in canonical order."""
    with get_connection() as connection:
        rows = connection.execute(
            """
            SELECT b.name
            FROM books b
            JOIN translations t ON t.id = b.translation_id
            WHERE t.code = ?
              AND t.is_installed = 1
            ORDER BY b.book_number
            """,
            (translation_code,),
        ).fetchall()

    return [row["name"] for row in rows]



def list_chapters(translation_code, book_name):
    """Return the available chapter numbers for a Bible book."""
    with get_connection() as connection:
        rows = connection.execute(
            """
            SELECT DISTINCT b.name, v.chapter
            FROM verses v
            JOIN books b ON b.id = v.book_id
            JOIN translations t ON t.id = v.translation_id
            WHERE t.code = ?
              AND LOWER(b.name) = LOWER(?)
              AND t.is_installed = 1
            ORDER BY v.chapter
            """,
            (translation_code, book_name),
        ).fetchall()

    return [row["chapter"] for row in rows]


def list_verse_numbers(translation_code, book_name, chapter):
    """Return the available verse numbers for a chapter."""
    with get_connection() as connection:
        rows = connection.execute(
            """
            SELECT v.verse
            FROM verses v
            JOIN books b ON b.id = v.book_id
            JOIN translations t ON t.id = v.translation_id
            WHERE t.code = ?
              AND LOWER(b.name) = LOWER(?)
              AND v.chapter = ?
              AND t.is_installed = 1
            ORDER BY v.verse
            """,
            (translation_code, book_name, chapter),
        ).fetchall()

    return [row["verse"] for row in rows]
