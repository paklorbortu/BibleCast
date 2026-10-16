
from pathlib import Path

from biblecast.bible.usfm_parser import parse_usfm


SOURCE_DIR = Path("data/translations/WEB/usfm_source")

files = sorted(SOURCE_DIR.glob("*.usfm"))

passed = []
failed = []
skipped = []
total_chapters = 0
total_verses = 0

for path in files:
    content = path.read_text(encoding="utf-8")

    # Front matter and glossary files are not ordinary Bible books.
    if r"\c " not in content or r"\v " not in content:
        skipped.append(path.name)
        continue

    try:
        book = parse_usfm(content)

        chapter_count = len(book["chapters"])
        verse_count = sum(
            len(chapter["verses"])
            for chapter in book["chapters"]
        )

        passed.append(
            (path.name, book["name"], chapter_count, verse_count)
        )
        total_chapters += chapter_count
        total_verses += verse_count

    except Exception as exc:
        failed.append((path.name, str(exc)))


print(f"USFM files found: {len(files)}")
print(f"Books parsed successfully: {len(passed)}")
print(f"Files skipped (no chapter/verse markers): {len(skipped)}")
print(f"Books with errors: {len(failed)}")
print(f"Total chapters parsed: {total_chapters}")
print(f"Total verses parsed: {total_verses}")

if failed:
    print("\nFAILED FILES:")
    for filename, error in failed:
        print(f"- {filename}: {error}")

if skipped:
    print("\nSKIPPED FILES:")
    for filename in skipped:
        print(f"- {filename}")

print("\nPARSED BOOKS:")
for filename, name, chapters, verses in passed:
    print(f"- {name}: {chapters} chapters, {verses} verses ({filename})")
