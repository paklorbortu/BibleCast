import re
import sys

from PySide6.QtGui import QShortcut, QKeySequence
from biblecast.ai.audio_reference_pipeline import AudioReferencePipeline
from biblecast.bible.navigation import find_verse_index
from PySide6.QtCore import (
    Qt, 
    QStringListModel,
    QThread,
    Signal,
)
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
    QFileDialog,
    QListWidget,
    QListWidgetItem,
)
from biblecast.bible.database import (
    get_all_verses,
    get_book_verses,
    get_verse_range,
    list_books,
    list_chapters,
    list_translations,
    list_verse_numbers,
)
from biblecast.bible.reference_resolver import resolve_reference


class CandidateSearchThread(QThread):
    """Find Bible passage candidates without blocking the UI."""

    result_ready = Signal(object, str)

    def __init__(self, audio_path, translation_code, parent=None):
        super().__init__(parent)
        self.audio_path = audio_path
        self.translation_code = translation_code

    def run(self):
        try:
            pipeline = AudioReferencePipeline()
            result = pipeline.process(
                self.audio_path,
                self.translation_code,
            )
            self.result_ready.emit(result, "")
        except Exception as exc:
            self.result_ready.emit(None, str(exc))

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
        self.is_live = False
        self.candidate_thread = None
        self.audio_candidates = []

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

        self.mode_indicator = QLabel("● READY")
        self.mode_indicator.setAlignment(
            Qt.AlignmentFlag.AlignLeft
        )
        self.mode_indicator.setStyleSheet(
            """
            color: #64748B;
            background-color: #F1F5F9;
            font-size: 12px;
            font-weight: 700;
            padding: 8px 12px;
            border-radius: 6px;
            """
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
        previous_button = QPushButton("◀ Previous Verse")
        next_button = QPushButton("Next Verse ▶")
        preview_button = QPushButton("Preview")
        live_button = QPushButton("Go Live")
        clear_button = QPushButton("Clear Projector")

        audio_button = QPushButton("Find Passages from Audio")
        self.candidate_list = QListWidget()
        self.candidate_list.setMinimumHeight(140)
        self.candidate_list.setStyleSheet(
            """
            QListWidget {
                background-color: #111827;
                color: #F9FAFB;
                border: 1px solid #374151;
                border-radius: 6px;
                padding: 4px;
            }
            QListWidget::item {
                padding: 8px;
                border-bottom: 1px solid #374151;
            }
            QListWidget::item:selected {
                background-color: #374151;
                color: white;
            }
            """
        )
        self.candidate_list.setToolTip(
            "Suggested Bible passages. Select one to review it."
        )

        use_candidate_button = QPushButton("Use Selected Candidate")
        use_candidate_button.setEnabled(False)

        audio_button.clicked.connect(self.find_candidates_from_audio)
        self.candidate_list.currentRowChanged.connect(
            lambda row: use_candidate_button.setEnabled(row >= 0)
        )
        use_candidate_button.clicked.connect(self.use_selected_candidate)

        self.use_candidate_button = use_candidate_button

        search_button.clicked.connect(self.find_verse)
        previous_button.clicked.connect(self.previous_verse)
        next_button.clicked.connect(self.next_verse)
        preview_button.clicked.connect(self.preview_verse)
        live_button.clicked.connect(self.go_live)
        clear_button.clicked.connect(self.clear_projector)

        # Keyboard shortcuts for live presentation control.
        self.next_shortcut = QShortcut(
            QKeySequence(Qt.Key.Key_Right),
            self,
        )
        self.next_shortcut.activated.connect(self.next_verse)

        self.previous_shortcut = QShortcut(
            QKeySequence(Qt.Key.Key_Left),
            self,
        )
        self.previous_shortcut.activated.connect(self.previous_verse)

        self.go_live_shortcut = QShortcut(
            QKeySequence("Ctrl+Return"),
            self,
        )
        self.go_live_shortcut.activated.connect(self.go_live)

        self.clear_projector_shortcut = QShortcut(
            QKeySequence(Qt.Key.Key_Escape),
            self,
        )
        self.clear_projector_shortcut.activated.connect(
            self.clear_projector
        )

        layout = QVBoxLayout()
        layout.setContentsMargins(28, 28, 28, 28)
        layout.setSpacing(14)

        layout.addWidget(title)
        layout.addWidget(subtitle)
        layout.addWidget(self.mode_indicator)
        layout.addSpacing(12)

        layout.addWidget(QLabel("Bible translation"))
        layout.addWidget(self.translation)

        layout.addWidget(QLabel("Scripture reference"))
        layout.addWidget(self.reference)
        layout.addWidget(search_button)
        layout.addWidget(audio_button)
        layout.addWidget(QLabel("Suggested passages"))
        layout.addWidget(self.candidate_list)
        layout.addWidget(use_candidate_button)

        navigation_buttons = QHBoxLayout()
        navigation_buttons.addWidget(previous_button)
        navigation_buttons.addWidget(next_button)
        layout.addLayout(navigation_buttons)

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



    def find_candidates_from_audio(self):
        """Choose an audio file and search for local Bible passages."""
        if (
            self.candidate_thread is not None
            and self.candidate_thread.isRunning()
        ):
            self.status.setText("Audio analysis is already running.")
            return

        translation_code = self.translation.currentData()
        if not translation_code:
            self.status.setText(
                "Install a Bible translation before searching."
            )
            return

        audio_path, _ = QFileDialog.getOpenFileName(
            self,
            "Choose sermon audio",
            "",
            "Audio files (*.wav *.mp3 *.m4a *.flac *.ogg);;All files (*)",
        )
        if not audio_path:
            return

        self.candidate_list.clear()
        self.audio_candidates = []
        self.use_candidate_button.setEnabled(False)
        self.status.setText(
            "Analyzing audio and searching the installed Bible..."
        )

        thread = CandidateSearchThread(
            audio_path,
            translation_code,
            self,
        )
        self.candidate_thread = thread
        thread.result_ready.connect(self.on_candidate_search_result)
        thread.finished.connect(self.on_candidate_search_finished)
        thread.start()

    def on_candidate_search_result(self, result, error):
        """Display candidate references and verse text, never the transcript."""
        if error:
            self.status.setText(f"Audio analysis failed: {error}")
            return

        if result is None:
            self.status.setText("Audio analysis failed.")
            return

        if not result.success:
            self.status.setText(result.error or "No candidates found.")
            return

        self.audio_candidates = list(result.candidates)
        self.candidate_list.clear()

        for candidate in self.audio_candidates:
            label = (
                f"{candidate.reference}  |  "
                f"{candidate.match_type} match  |  "
                f"{candidate.score:.0%}\n"
                f"{candidate.text}"
            )
            item = QListWidgetItem(label)
            self.candidate_list.addItem(item)

        self.status.setText(
            f"Found {len(self.audio_candidates)} candidate passage(s). "
            "Select one to review it."
        )

    def on_candidate_search_finished(self):
        """Allow another search after the worker has stopped."""
        self.candidate_thread = None

    def use_selected_candidate(self):
        """Load a chosen candidate into the normal preview selection."""
        row = self.candidate_list.currentRow()

        if row < 0 or row >= len(self.audio_candidates):
            self.status.setText("Select a candidate first.")
            return

        translation_code = self.translation.currentData()
        candidate = self.audio_candidates[row]

        resolved = resolve_reference(
            translation_code,
            candidate.reference,
        )
        if resolved is None:
            self.status.setText(
                "That candidate could not be resolved in the selected "
                "translation. Try searching again."
            )
            return

        all_verses = get_all_verses(translation_code)
        first_verse = resolved["first_verse"]

        index = find_verse_index(
            all_verses,
            first_verse["book_name"],
            first_verse["chapter"],
            first_verse["verse"],
        )
        if index is None:
            self.status.setText(
                "Could not locate the selected candidate in the Bible."
            )
            return

        self.reference.setText(candidate.reference)
        self.passage_verses = all_verses
        self.passage_index = index
        self.update_selected_verse()

        self.status.setText(
            f"Selected {self.selected_reference} for review. "
            "It is not live. Use Preview or Go Live when ready."
        )


    
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
                    all_verses = get_all_verses(translation_code)

                    self.passage_index = find_verse_index(
                        all_verses,
                        verses[0]["book_name"],
                        verses[0]["chapter"],
                        verses[0]["verse"],
                    )

                    if self.passage_index is None:
                        self.clear_selection()
                        self.status.setText(
                            "Could not locate the starting verse in the Bible."
                        )
                        return

                    self.passage_verses = all_verses
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

        # Validate ranges before enabling full-Bible navigation.
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

        all_verses = get_all_verses(translation_code)

        self.passage_index = find_verse_index(
            all_verses,
            first_verse["book_name"],
            first_verse["chapter"],
            first_verse["verse"],
        )

        if self.passage_index is None:
            self.clear_selection()
            self.status.setText(
                "Could not locate the starting verse in the Bible."
            )
            return

        self.passage_verses = all_verses
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


    def previous_verse(self):
        """Move to the previous verse within the selected passage."""
        if not self.passage_verses:
            self.status.setText("Find a passage before navigating.")
            return

        if self.passage_index == 0:
            self.status.setText("Already at the first verse of the Bible.")
            return

        self.passage_index -= 1
        self.update_selected_verse()
        self.update_navigation_status()

    def next_verse(self):
        """Move to the next verse within the selected passage."""
        if not self.passage_verses:
            self.status.setText("Find a passage before navigating.")
            return

        if self.passage_index >= len(self.passage_verses) - 1:
            self.status.setText("Already at the last verse of the Bible.")
            return

        self.passage_index += 1
        self.update_selected_verse()
        self.update_navigation_status()

    def update_navigation_status(self):
        """Update the status and refresh live output when appropriate."""
        position = (
            f"{self.passage_index + 1} of {len(self.passage_verses)}"
        )

        if self.is_live:
            self.projector.show_verse(
                self.selected_reference,
                self.selected_text,
                self.translation.currentData(),
            )
            self.status.setText(
                f"Now displaying {self.selected_reference} live "
                f"({position})."
            )
        else:
            self.status.setText(
                f"Selected {self.selected_reference} ({position}). "
                "Preview or go live when ready."
            )

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
        self.is_live = True
        self.mode_indicator.setText("● LIVE")
        self.mode_indicator.setStyleSheet(
            """
            color: #B91C1C;
            background-color: #FEE2E2;
            font-size: 12px;
            font-weight: 700;
            padding: 8px 12px;
            border-radius: 6px;
            """
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
        self.is_live = False
        self.mode_indicator.setText("● READY")
        self.mode_indicator.setStyleSheet(
            """
            color: #64748B;
            background-color: #F1F5F9;
            font-size: 12px;
            font-weight: 700;
            padding: 8px 12px;
            border-radius: 6px;
            """
        )
        self.status.setText("Live projector cleared.")


def main():
    app = QApplication(sys.argv)
    app.setApplicationName("BibleCast")

    window = BibleCastWindow()
    window.show()

    sys.exit(app.exec())


if __name__ == "__main__":
    main()