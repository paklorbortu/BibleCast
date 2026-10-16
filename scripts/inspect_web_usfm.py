
from pathlib import Path

from biblecast.bible.usfm_parser import parse_usfm


source_dir = Path("data/translations/WEB/usfm_source")
genesis_file = source_dir / "02-GENengwebp.usfm"

content = genesis_file.read_text(encoding="utf-8")
book = parse_usfm(content)

print(f"Book: {book['name']}")
print(f"Book ID: {book['book_id']}")
print(f"Chapters found: {len(book['chapters'])}")

total_verses = sum(
    len(chapter["verses"])
    for chapter in book["chapters"]
)
print(f"Verses found: {total_verses}")

print("\nFirst five verses:")
for verse in book["chapters"][0]["verses"][:5]:
    print(f"Genesis 1:{verse['number']} {verse['text']}")

print("\nLast verse in chapter 1:")
last_verse = book["chapters"][0]["verses"][-1]
print(f"Genesis 1:{last_verse['number']} {last_verse['text']}")
