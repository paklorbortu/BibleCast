
from unittest.mock import patch

import pytest

from biblecast.ai.passage_candidates import (
    clear_quote_cache,
    find_passage_candidates,
)


@pytest.fixture(autouse=True)
def _fresh_quote_cache():
    """Each test patches its own verses, so never reuse a cached index."""
    clear_quote_cache()
    yield
    clear_quote_cache()


def _fake_resolver(known):
    """Build a resolve_reference stand-in that needs no database."""

    def resolve(translation_code, reference):
        return known.get(reference)

    return resolve


def _resolved(book, chapter, verse, text):
    return {
        "book": book,
        "chapter": chapter,
        "verse": verse,
        "end_verse": None,
        "is_range": False,
        "first_verse": {"text": text},
    }


JOHN_3_16 = _resolved("John", 3, 16, "For God so loved the world.")


def test_resolves_explicit_spoken_reference_without_gemma():
    with patch(
        "biblecast.ai.passage_candidates.get_all_verses"
    ) as get_all_verses, patch(
        "biblecast.ai.passage_candidates.resolve_reference",
        _fake_resolver({"John 3:16": JOHN_3_16}),
    ):
        candidates = find_passage_candidates(
            "WEB",
            "Please open John three sixteen.",
        )

    assert len(candidates) == 1
    assert candidates[0].reference == "John 3:16"
    assert candidates[0].match_type == "explicit"
    assert candidates[0].confidence == "high"
    assert candidates[0].score == 1.0
    get_all_verses.assert_not_called()


def test_resolves_numeric_explicit_reference():
    with patch(
        "biblecast.ai.passage_candidates.resolve_reference",
        _fake_resolver({"John 3:16": JOHN_3_16}),
    ):
        candidates = find_passage_candidates("WEB", "Open John 3:16")

    assert candidates
    assert candidates[0].reference == "John 3:16"
    assert candidates[0].match_type == "explicit"


def test_explicit_reference_ignores_leading_filler_words():
    with patch(
        "biblecast.ai.passage_candidates.resolve_reference",
        _fake_resolver({"Romans 8:28": _resolved(
            "Romans", 8, 28, "All things work together for good."
        )}),
    ):
        candidates = find_passage_candidates(
            "WEB", "let's read Romans 8 28 together"
        )

    assert candidates[0].reference == "Romans 8:28"


def test_ranks_quoted_passages_by_ordered_match():
    verses = [
        {
            "book_name": "Genesis",
            "chapter": 1,
            "verse": 1,
            "text": "In the beginning God created the heavens and the earth.",
        },
        {
            "book_name": "Joshua",
            "chapter": 1,
            "verse": 9,
            "text": "Be strong and courageous. Do not be afraid.",
        },
        {
            "book_name": "Psalm",
            "chapter": 23,
            "verse": 1,
            "text": "Yahweh is my shepherd: I shall lack nothing.",
        },
    ]

    with patch(
        "biblecast.ai.passage_candidates.get_all_verses",
        return_value=verses,
    ):
        candidates = find_passage_candidates(
            "WEB",
            "Be strong and courageous",
        )

    assert candidates
    assert candidates[0].reference == "Joshua 1:9"
    assert candidates[0].match_type == "quote"
    assert candidates[0].score > 0


def test_returns_multiple_ranked_candidates_for_ambiguous_quote():
    verses = [
        {
            "book_name": "Joshua",
            "chapter": 1,
            "verse": 9,
            "text": "Be strong and courageous.",
        },
        {
            "book_name": "Deuteronomy",
            "chapter": 31,
            "verse": 6,
            "text": "Be strong and courageous. Do not fear.",
        },
        {
            "book_name": "Genesis",
            "chapter": 1,
            "verse": 1,
            "text": "In the beginning God created the heavens and the earth.",
        },
    ]

    with patch(
        "biblecast.ai.passage_candidates.get_all_verses",
        return_value=verses,
    ):
        candidates = find_passage_candidates(
            "WEB",
            "Be strong and courageous",
            limit=2,
        )

    assert len(candidates) == 2
    assert candidates[0].score >= candidates[1].score
    assert candidates[0].reference == "Joshua 1:9"
    assert candidates[1].reference == "Deuteronomy 31:6"


def test_returns_no_candidates_for_empty_or_unrelated_input():
    with patch(
        "biblecast.ai.passage_candidates.get_all_verses",
        return_value=[],
    ):
        assert find_passage_candidates("WEB", "   ") == []

    with patch(
        "biblecast.ai.passage_candidates.get_all_verses",
        return_value=[],
    ):
        assert find_passage_candidates(
            "WEB",
            "   ",
        ) == []


def test_rejects_single_common_word_as_quote():
    with patch(
        "biblecast.ai.passage_candidates.get_all_verses"
    ) as get_all_verses:
        candidates = find_passage_candidates(
            "WEB",
            "one",
        )

    assert candidates == []
    get_all_verses.assert_not_called()


def test_rejects_short_ambiguous_fragment():
    verses = [
        {
            "book_name": "John",
            "chapter": 10,
            "verse": 30,
            "text": "I and the Father are one.",
        },
    ]

    with patch(
        "biblecast.ai.passage_candidates.get_all_verses",
        return_value=verses,
    ):
        candidates = find_passage_candidates(
            "WEB",
            "Father one",
        )

    assert candidates == []


def test_matches_ordered_direct_quote():
    verses = [
        {
            "book_name": "Psalm",
            "chapter": 23,
            "verse": 1,
            "text": "Yahweh is my shepherd I shall lack nothing.",
        },
        {
            "book_name": "John",
            "chapter": 3,
            "verse": 16,
            "text": (
                "For God so loved the world that he gave "
                "his one and only Son."
            ),
        },
    ]

    with patch(
        "biblecast.ai.passage_candidates.get_all_verses",
        return_value=verses,
    ):
        candidates = find_passage_candidates(
            "WEB",
            "Yahweh is my shepherd",
        )

    assert candidates
    assert candidates[0].reference == "Psalm 23:1"
    assert candidates[0].match_type == "quote"
    assert candidates[0].evidence == "yahweh is my shepherd"


PSALM = {
    "book_name": "Psalms", "chapter": 23, "verse": 1,
    "text": "Yahweh is my shepherd: I shall lack nothing.",
}


def _patched(verses):
    return patch(
        "biblecast.ai.passage_candidates.get_all_verses",
        return_value=verses,
    )


def test_exact_quote_is_high_confidence_and_reported_once():
    verses = [
        PSALM,
        {"book_name": "Psalms", "chapter": 23, "verse": 2,
         "text": "He makes me lie down in green pastures."},
        {"book_name": "Psalms", "chapter": 23, "verse": 3,
         "text": "He restores my soul."},
    ]

    with _patched(verses):
        candidates = find_passage_candidates(
            "WEB", "Yahweh is my shepherd I shall lack nothing"
        )

    assert [c.reference for c in candidates] == ["Psalms 23:1"]
    assert candidates[0].confidence == "high"
    assert candidates[0].text == PSALM["text"]


def test_quote_crossing_a_verse_boundary_reports_the_range():
    verses = [
        {"book_name": "John", "chapter": 1, "verse": 1,
         "text": "In the beginning was the Word."},
        {"book_name": "John", "chapter": 1, "verse": 2,
         "text": "The Word was with God and the Word was God."},
    ]

    with _patched(verses):
        candidates = find_passage_candidates(
            "WEB", "the beginning was the Word the Word was with God"
        )

    assert candidates[0].reference == "John 1:1-2"
    assert "In the beginning" in candidates[0].text


def test_partial_quote_is_labelled_lower_confidence():
    verses = [{
        "book_name": "John", "chapter": 10, "verse": 30,
        "text": "I and the Father are one.",
    }]

    with _patched(verses):
        candidates = find_passage_candidates(
            "WEB",
            "I and the Father are one and we should all be united",
        )

    assert candidates
    assert candidates[0].confidence == "low"


def test_low_confidence_matches_are_dropped_beside_a_high_match():
    verses = [
        {"book_name": "Genesis", "chapter": 1, "verse": 1,
         "text": "In the beginning God created the heavens and the earth."},
        {"book_name": "Jeremiah", "chapter": 10, "verse": 11,
         "text": "They will perish from the heavens and the earth."},
    ]

    with _patched(verses):
        candidates = find_passage_candidates(
            "WEB",
            "In the beginning God created the heavens and the earth",
        )

    assert [c.reference for c in candidates] == ["Genesis 1:1"]


def test_ordinary_speech_returns_nothing():
    with _patched([PSALM]):
        assert find_passage_candidates(
            "WEB", "let us pray and thank the Lord for the food"
        ) == []
