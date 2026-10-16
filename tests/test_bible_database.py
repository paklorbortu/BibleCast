
import sqlite3
import os
from biblecast.bible.database import (
    get_verse,
    search_verses,
    initialize_database,
)
from biblecast.bible.importer import import_translation
from pathlib import Path


def test_database_schema_can_be_created(tmp_path):
    database_path = tmp_path / "test_biblecast.db"

    with sqlite3.connect(database_path) as connection:
        connection.executescript(
            """
            CREATE TABLE translations (
                id INTEGER PRIMARY KEY,
                code TEXT NOT NULL UNIQUE,
                name TEXT NOT NULL,
                is_installed INTEGER NOT NULL DEFAULT 0
            );

            CREATE TABLE books (
                id INTEGER PRIMARY KEY,
                translation_id INTEGER NOT NULL,
                book_number INTEGER NOT NULL,
                name TEXT NOT NULL,
                testament TEXT NOT NULL,
                chapters INTEGER NOT NULL,
                UNIQUE (translation_id, book_number),
                FOREIGN KEY (translation_id)
                    REFERENCES translations(id)
            );

            CREATE TABLE verses (
                id INTEGER PRIMARY KEY,
                translation_id INTEGER NOT NULL,
                book_id INTEGER NOT NULL,
                chapter INTEGER NOT NULL,
                verse INTEGER NOT NULL,
                text TEXT NOT NULL,
                UNIQUE (translation_id, book_id, chapter, verse),
                FOREIGN KEY (translation_id)
                    REFERENCES translations(id),
                FOREIGN KEY (book_id)
                    REFERENCES books(id)
            );
            """
        )

        tables = connection.execute(
            """
            SELECT name FROM sqlite_master
            WHERE type = 'table'
            """
        ).fetchall()

    assert {row[0] for row in tables} >= {
        "translations",
        "books",
        "verses",
    }


def test_search_verses_ignores_punctuation(tmp_path, monkeypatch):
    database_path = tmp_path / "test_biblecast.db"
    monkeypatch.setenv("BIBLECAST_DATABASE_PATH", str(database_path))

    initialize_database()

    # Use the real WEB translation file to test actual scripture text.
    translation_path = (
        Path(__file__).resolve().parents[1]
        / "data"
        / "translations"
        / "WEB"
        / "web_translation.json"
    )
    import_translation(translation_path)

    results = search_verses("WEB", "In the beginning God created")

    assert results
    assert any(
        verse["book_name"] == "Genesis"
        and verse["chapter"] == 1
        and verse["verse"] == 1
        for verse in results
    )

    verse = get_verse("WEB", "Genesis", 1, 1)
    assert verse is not None
    assert "In the beginning" in verse["text"]
