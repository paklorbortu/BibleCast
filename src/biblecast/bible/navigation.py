def find_verse_index(verses, book_name, chapter, verse_number):
    """Find a verse's position in the canonical Bible sequence."""
    for index, verse in enumerate(verses):
        if (
            verse["book_name"].casefold() == book_name.casefold()
            and verse["chapter"] == chapter
            and verse["verse"] == verse_number
        ):
            return index

    return None