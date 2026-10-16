import re
import sys

from PySide6.QtCore import Qt, QStringListModel
from PySide6.QtWidgets import (
    QApplication,
    QCompleter,
    QComboBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMainWindow,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

from biblecast.bible.database import (
    get_book_verses,
    get_verse_range,
    list_books,
    list_chapters,
    list_translations,
    list_verse_numbers,
)
from biblecast.bible.reference_resolver import resolve_reference


class ProjectorWindow(QMainWindow):
    """Window for displaying one Bible verse at a time."""

    def __init__(self, title="BibleCast Projector"):
        super().__init__()
        self.setWindowTitle(title)
        self.setMinimumSize(900, 500)

        self.reference = QLabel("BibleCast")
        self.reference.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.reference.setTextFormat(Qt.TextFormat.RichText)
        self.reference.setStyleSheet(
            """
            color: white;
            font-size: 25px;
            font-weight: 600;
            padding-bottom: 8px;
            """
        )

        self.verse = QLabel(
            "Your selected Bible verse will appear here."
        )
        self.verse.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.verse.setWordWrap(True)
        self.verse.setStyleSheet(
            """
            color: white;
            font-size: 38px;
            font-weight: 500;
            """
        )

        layout = QVBoxLayout()
        layout.setContentsMargins(80, 55, 80, 55)
        layout.setSpacing(28)
        layout.addStretch()
        layout.addWidget(self.reference)
        layout.addWidget(self.verse)
        layout.addStretch()

        container = QWidget()
        container.setLayout(layout)
        container.setStyleSheet("background-color: #111827;")
        self.setCentralWidget(container)

    def show_verse(self, reference, text, translation_code):
        """Update the display with a reference, verse text and translation."""
        header = f"{reference} ({translation_code})"
        self.reference.setText(
            f'<div style="text-align:center">'
            f'<u>{header}</u>'
            f"</div>"
        )
        self.verse.setText(text)


class BibleCastWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("BibleCast")
        self.setMinimumSize(780, 620)

        # Preview and live output remain independent.
        self.preview_projector = ProjectorWindow(
            "BibleCast Preview"
        )
        self.projector = ProjectorWindow(
            "BibleCast Live Projector"
        )

        self.selected_reference = ""
        self.selected_text = ""
        self.passage_verses = []
        self.passage_index = 0

        # Track the autocomplete stage and selected book.
        self.suggestion_stage = "book"
        self.suggestion_book = None

        title = QLabel("BibleCast")
        title.setStyleSheet(
            "font-size: 28px; font-weight: bold;"
        )

        subtitle = QLabel(
            "Search, select and project Bible passages."
        )

        self.translation = QComboBox()
        self.load_translations()
        self.translation.currentIndexChanged.connect(
            self.on_translation_changed
        )

        self.reference = QLineEdit()
        self.reference.setPlaceholderText(
            "Enter a reference, e.g. John 3:16 or John 3:16-20"
        )
        self.reference.returnPressed.connect(self.find_verse)

        # A single completer is reused for book, chapter and verse options.
        self.suggestion_model = QStringListModel(self)

        self.reference_completer = QCompleter(
            self.suggestion_model,
            self,
        )
        self.reference_completer.setCaseSensitivity(
            Qt.CaseSensitivity.CaseInsensitive
        )
        self.reference_completer.setCompletionMode(
            QCompleter.CompletionMode.PopupCompletion
        )
        self.reference_completer.setFilterMode(
            Qt.MatchFlag.MatchStartsWith
        )

        self.reference.setCompleter(self.reference_completer)
        self.reference.textEdited.connect(
            self.update_reference_suggestions
        )
        self.reference_completer.activated.connect(
            self.complete_reference_part
        )

        self.refresh_book_completer()

        self.verse_text = QLabel(
            "Your selected scripture will appear here."
        )
        self.verse_text.setWordWrap(True)
        self.verse_text.setMinimumHeight(100)
        self.verse_text.setAlignment(
            Qt.AlignmentFlag.AlignTop | Qt.AlignmentFlag.AlignLeft
        )
        self.verse_text.setStyleSheet(
            """
            font-size: 18px;
            padding: 12px;
            border: 1px solid #D1D5DB;
            border-radius: 6px;
            """
        )

        self.status = QLabel("Ready.")
        self.status.setWordWrap(True)

        search_button = QPushButton("Find Verse")
        preview_button = QPushButton("Preview")
        live_button = QPushButton("Go Live")
        clear_button = QPushButton("Clear Projector")

        search_button.clicked.connect(self.find_verse)
        preview_button.clicked.connect(self.preview_verse)
        live_button.clicked.connect(self.go_live)
        clear_button.clicked.connect(self.clear_projector)

        layout = QVBoxLayout()
        layout.setContentsMargins(28, 28, 28, 28)
        layout.setSpacing(14)

        layout.addWidget(title)
        layout.addWidget(subtitle)
        layout.addSpacing(12)

        layout.addWidget(QLabel("Bible translation"))
        layout.addWidget(self.translation)

        layout.addWidget(QLabel("Scripture reference"))
        layout.addWidget(self.reference)
        layout.addWidget(search_button)

        layout.addWidget(QLabel("Verse preview"))
        layout.addWidget(self.verse_text)

        buttons = QHBoxLayout()
        buttons.addWidget(preview_button)
        buttons.addWidget(live_button)
        buttons.addWidget(clear_button)
        layout.addLayout(buttons)

        layout.addWidget(self.status)
        layout.addStretch()

        container = QWidget()
        container.setLayout(layout)
        self.setCentralWidget(container)

    def load_translations(self):
        """Populate the dropdown with installed Bible translations."""
        translations = list_translations(installed_only=True)

        for translation in translations:
            self.translation.addItem(
                translation["name"],
                translation["code"],
            )

        if self.translation.count() == 0:
            self.translation.addItem(
                "No translations installed",
                None,
            )

    def on_translation_changed(self):
        self.refresh_book_completer()

        # The selected verse may not exist in the newly selected translation.
        self.clear_selection()
        self.status.setText(
            "Translation changed. Enter or select a reference."
        )

    def refresh_book_completer(self):
        """Load the available book names for the current translation."""
        translation_code = self.translation.currentData()

        if not translation_code:
            self.suggestion_model.setStringList([])
            return

        self.suggestion_model.setStringList(
            list_books(translation_code)
        )
        self.suggestion_stage = "book"
        self.suggestion_book = None

    def update_reference_suggestions(self, text):
        """Suggest complete references to avoid losing the book or chapter."""
        translation_code = self.translation.currentData()

        if not translation_code:
            self.hide_suggestions()
            return

        text = text.strip()

        if not text:
            self.suggestion_stage = "book"
            self.suggestion_book = None
            self.set_suggestions(list_books(translation_code), "")
            return

        books = list_books(translation_code)

        exact_book = next(
            (
                book
                for book in sorted(books, key=len, reverse=True)
                if text.casefold() == book.casefold()
                or text.casefold().startswith(book.casefold() + " ")
            ),
            None,
        )

        # Stage 1: Suggest book names.
        if exact_book is None:
            self.suggestion_stage = "book"
            self.suggestion_book = None

            matches = [
                book for book in books
                if book.casefold().startswith(text.casefold())
            ]

            self.set_suggestions(matches, text)
            return

        self.suggestion_book = exact_book
        remainder = text[len(exact_book):].strip()

        # Stage 2: Suggest complete book and chapter references.
        if ":" not in remainder:
            self.suggestion_stage = "chapter"

            chapters = list_chapters(
                translation_code,
                exact_book,
            )

            chapter_prefix = remainder

            matches = [
                f"{exact_book} {chapter}:"
                for chapter in chapters
                if str(chapter).startswith(chapter_prefix)
            ]

            self.set_suggestions(matches, text)
            return

        # Stage 3: Suggest complete book, chapter and verse references.
        chapter_text, verse_text = remainder.split(":", 1)

        if not chapter_text.isdigit():
            self.hide_suggestions()
            return

        if "-" in verse_text or "to" in verse_text.casefold():
            self.hide_suggestions()
            return

        chapter = int(chapter_text)

        verses = list_verse_numbers(
            translation_code,
            exact_book,
            chapter,
        )

        if not verses:
            self.hide_suggestions()
            return

        self.suggestion_stage = "verse"

        matches = [
            f"{exact_book} {chapter}:{verse}"
            for verse in verses
            if str(verse).startswith(verse_text)
        ]

        self.set_suggestions(matches, text)

    def set_suggestions(self, suggestions, prefix):
        """Display suggestions matching the complete reference prefix."""
        self.suggestion_model.setStringList(suggestions)
        self.reference_completer.setCompletionPrefix(prefix)

        if suggestions and self.reference.hasFocus():
            self.reference_completer.complete()
        else:
            self.hide_suggestions()

    def hide_suggestions(self):
        self.reference_completer.popup().hide()

    
    def complete_reference_part(self, selection):
        """Accept a book or complete reference without reconstructing it."""
        translation_code = self.translation.currentData()

        if not translation_code:
            return

        books = list_books(translation_code)

        # Selecting a book starts chapter selection.
        selected_book = next(
            (
                book for book in books
                if selection.casefold() == book.casefold()
            ),
            None,
        )

        if selected_book:
            self.suggestion_book = selected_book
            self.suggestion_stage = "chapter"

            self.reference.setText(f"{selected_book} ")
            self.reference.setCursorPosition(len(self.reference.text()))

            self.update_reference_suggestions(
                self.reference.text()
            )
            return

        # The selection is already a complete reference, such as
        # "Genesis 1:" or "Genesis 1:3".
        self.reference.setText(selection)
        self.reference.setCursorPosition(len(selection))

        if selection.endswith(":"):
            self.suggestion_stage = "verse"
            self.update_reference_suggestions(selection)
        else:
            self.suggestion_stage = "complete"
            self.hide_suggestions()


    def find_verse(self):
        """Retrieve a single verse, a verse range or a whole chapter."""
        reference = self.reference.text().strip()
        translation_code = self.translation.currentData()

        if not reference:
            self.status.setText("Enter a Bible reference first.")
            return

        if not translation_code:
            self.status.setText(
                "Install a Bible translation before searching."
            )
            return

        # Handle chapter-only references such as John 1.
        chapter_match = re.fullmatch(
            r"(.+?)\s+(\d+)",
            reference,
        )

        if chapter_match:
            book_name = chapter_match.group(1).strip()
            chapter = int(chapter_match.group(2))

            canonical_book = next(
                (
                    book
                    for book in list_books(translation_code)
                    if book.casefold() == book_name.casefold()
                ),
                None,
            )

            if canonical_book is not None:
                verses = get_book_verses(
                    translation_code,
                    canonical_book,
                    chapter,
                )

                if verses:
                    self.passage_verses = verses
                    self.passage_index = 0
                    self.update_selected_verse()

                    self.status.setText(
                        f"Found {canonical_book} {chapter}, "
                        f"{len(verses)} verses. "
                        f"Selected {self.selected_reference}."
                    )
                    return

                self.clear_selection()
                self.status.setText(
                    "That chapter was not found in the selected translation."
                )
                return

        # Resolve individual verses and ranges.
        result = resolve_reference(
            translation_code,
            reference,
        )

        if result is None:
            self.clear_selection()
            self.status.setText(
                "No matching verse was found. "
                "Check the book, chapter and verse."
            )
            return

        first_verse = result["first_verse"]
        end_verse = result["end_verse"]

        if end_verse is not None:
            verses = get_verse_range(
                translation_code,
                first_verse["book_name"],
                first_verse["chapter"],
                first_verse["verse"],
                end_verse,
            )

            if not verses:
                self.clear_selection()
                self.status.setText(
                    "The requested range is invalid or incomplete."
                )
                return
        else:
            verses = [first_verse]

        self.passage_verses = verses
        self.passage_index = 0
        self.update_selected_verse()

        self.status.setText(
            f"Found {len(verses)} verse(s). "
            f"Selected {self.selected_reference}, ready to preview or cast."
        )

    def clear_selection(self):
        self.selected_reference = ""
        self.selected_text = ""
        self.passage_verses = []
        self.passage_index = 0
        self.verse_text.setText("No verse selected.")

    def update_selected_verse(self):
        """Refresh the operator's verse preview."""
        if not self.passage_verses:
            self.clear_selection()
            return

        verse = self.passage_verses[self.passage_index]

        self.selected_reference = (
            f"{verse['book_name']} "
            f"{verse['chapter']}:{verse['verse']}"
        )
        self.selected_text = verse["text"]
        self.verse_text.setText(self.selected_text)

    def preview_verse(self):
        """Display the selected verse in the separate preview window."""
        if not self.selected_text:
            self.status.setText(
                "Find a valid Bible passage before previewing."
            )
            return

        self.preview_projector.show_verse(
            self.selected_reference,
            self.selected_text,
            self.translation.currentData(),
        )
        self.preview_projector.show()
        self.preview_projector.raise_()
        self.preview_projector.activateWindow()

        self.status.setText(
            f"Previewing {self.selected_reference}. "
            "The live projector has not been changed."
        )

    def go_live(self):
        """Display the selected verse on the live projector."""
        if not self.selected_text:
            self.status.setText(
                "Find a valid Bible passage before going live."
            )
            return

        self.projector.show_verse(
            self.selected_reference,
            self.selected_text,
            self.translation.currentData(),
        )
        self.projector.showFullScreen()
        self.projector.raise_()
        self.projector.activateWindow()

        self.status.setText(
            f"Now displaying {self.selected_reference} live."
        )

    def clear_projector(self):
        """Clear live output without removing the selected verse."""
        self.projector.show_verse("BibleCast", "", "")
        self.status.setText("Live projector cleared.")


def main():
    app = QApplication(sys.argv)
    app.setApplicationName("BibleCast")

    window = BibleCastWindow()
    window.show()

    sys.exit(app.exec())


if __name__ == "__main__":
    main()