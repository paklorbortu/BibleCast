
import json
import sqlite3

import pytest

from biblecast.bible.database import (
    get_verse,
    initialize_database,
    list_translations,
)
from biblecast.bible.importer import import_translation


@pytest.fixture
def isolated_database(tmp_path, monkeypatch):
    database_path = tmp_path / "test_biblecast.db"
    monkeypatch.setenv(
        "BIBLECAST_DATABASE_PATH",
        str(database_path),
    )
    initialize_database()
    return database_path


def make_translation(tmp_path, code="TEST"):
    data = {
        "code": code,
        "name": "Test Bible",
        "language": "en",
        "source": "Automated test fixture",
        "license_name": "Test-only content",
        "license_url": "",
        "books": [
            {
                "number": 43,
                "name": "John",
                "testament": "NT",
                "chapters": [
                    {
                        "number": 3,
                        "verses": [
                            {
                                "number": 16,
                                "text": "For God so loved the world.",
                            },
                            {
                                "number": 17,
                                "text": "For God sent not his Son.",
                            },
                        ],
                    }
                ],
            }
        ],
    }

    path = tmp_path / f"{code.lower()}.json"
    path.write_text(
        json.dumps(data),
        encoding="utf-8",
    )
    return path, data


def test_import_and_retrieve_verse(
    tmp_path,
    isolated_database,
):
    path, _ = make_translation(tmp_path)

    count = import_translation(path)

    assert count == 2

    verse = get_verse("TEST", "John", 3, 16)

    assert verse is not None
    assert verse["text"] == "For God so loved the world."
    assert verse["translation_code"] == "TEST"
    assert verse["book_name"] == "John"

    translations = list_translations()
    assert len(translations) == 1
    assert translations[0]["code"] == "TEST"


def test_duplicate_translation_is_rejected(
    tmp_path,
    isolated_database,
):
    path, _ = make_translation(tmp_path)
    import_translation(path)

    with pytest.raises(ValueError, match="already exists"):
        import_translation(path)


def test_invalid_translation_is_rejected(
    tmp_path,
    isolated_database,
):
    path, data = make_translation(tmp_path)
    data["books"][0]["chapters"][0]["verses"][0]["text"] = ""

    path.write_text(
        json.dumps(data),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="Empty verse text"):
        import_translation(path)

    assert list_translations() == []

    with sqlite3.connect(isolated_database) as connection:
        translation_count = connection.execute(
            "SELECT COUNT(*) FROM translations"
        ).fetchone()[0]

    assert translation_count == 0


def test_missing_verse_returns_none(
    tmp_path,
    isolated_database,
):
    path, _ = make_translation(tmp_path)
    import_translation(path)

    assert get_verse("TEST", "John", 3, 99) is None
