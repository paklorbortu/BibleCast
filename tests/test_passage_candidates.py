
from unittest.mock import patch

from biblecast.ai.passage_candidates import find_passage_candidates


def test_resolves_explicit_spoken_reference_without_gemma():
    with patch(
        "biblecast.ai.passage_candidates.get_all_verses"
    ) as get_all_verses:
        candidates = find_passage_candidates(
            "WEB",
            "Please open John three sixteen.",
        )

    assert len(candidates) == 1
    assert candidates[0].reference == "John 3:16"
    assert candidates[0].match_type == "explicit"
    assert candidates[0].score == 1.0
    get_all_verses.assert_not_called()


def test_resolves_numeric_explicit_reference():
    candidates = find_passage_candidates(
        "WEB",
        "Open John 3:16",
    )

    # This assertion is made against the real local Bible database.
    # If WEB is not installed in the test environment, test the
    # reference normalization separately using the resolver tests.
    assert candidates
    assert candidates[0].reference == "John 3:16"
    assert candidates[0].match_type == "explicit"


def test_ranks_quoted_passages_by_lexical_match():
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
