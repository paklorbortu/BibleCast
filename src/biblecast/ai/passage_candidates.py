
"""Find explicit Bible references and rank local passage candidates."""

from __future__ import annotations

import re
from collections import Counter
from dataclasses import dataclass

from biblecast.bible.database import get_all_verses
from biblecast.bible.reference_resolver import resolve_reference


@dataclass(frozen=True)
class PassageCandidate:
    """A passage suggested for operator review."""

    reference: str
    text: str
    score: float
    match_type: str


_NUMBER_WORDS = {
    "one": 1, "two": 2, "three": 3, "four": 4, "five": 5,
    "six": 6, "seven": 7, "eight": 8, "nine": 9, "ten": 10,
    "eleven": 11, "twelve": 12, "thirteen": 13, "fourteen": 14,
    "fifteen": 15, "sixteen": 16, "seventeen": 17,
    "eighteen": 18, "nineteen": 19, "twenty": 20,
    "thirty": 30, "forty": 40, "fifty": 50, "sixty": 60,
    "seventy": 70, "eighty": 80, "ninety": 90,
}

_STOP_WORDS = {
    "a", "an", "and", "are", "as", "at", "be", "been", "being",
    "but", "by", "do", "for", "from", "he", "her", "him", "his",
    "i", "in", "is", "it", "its", "me", "my", "of", "on", "or",
    "our", "she", "that", "the", "their", "them", "there", "they",
    "this", "to", "us", "was", "we", "were", "will", "with", "you",
    "your",
}


def _normalize_spoken_numbers(text: str) -> str:
    """Convert common spoken numbers to digits."""

    words = text.split()
    normalized = []
    index = 0

    while index < len(words):
        word = re.sub(r"[^\w]", "", words[index].casefold())

        # Handle numbers such as "twenty three".
        if (
            word in _NUMBER_WORDS
            and _NUMBER_WORDS[word] >= 20
            and _NUMBER_WORDS[word] % 10 == 0
            and index + 1 < len(words)
        ):
            next_word = re.sub(
                r"[^\w]", "", words[index + 1].casefold()
            )
            if (
                next_word in _NUMBER_WORDS
                and 1 <= _NUMBER_WORDS[next_word] <= 9
            ):
                normalized.append(
                    str(_NUMBER_WORDS[word] + _NUMBER_WORDS[next_word])
                )
                index += 2
                continue

        if word in _NUMBER_WORDS:
            normalized.append(str(_NUMBER_WORDS[word]))
        else:
            normalized.append(words[index])

        index += 1

    return " ".join(normalized)


def _extract_explicit_reference(
    translation_code: str,
    transcript: str,
) -> dict | None:
    """Find and resolve an explicit reference in a transcript."""

    normalized = _normalize_spoken_numbers(transcript)

    normalized = re.sub(
        r"\b(?:please\s+)?(?:open|show|display|read)\s+",
        "",
        normalized,
        flags=re.IGNORECASE,
    )
    normalized = re.sub(
        r"\bturn\s+to\s+",
        "",
        normalized,
        flags=re.IGNORECASE,
    )
    normalized = re.sub(
        r"\bchapter\s+(\d+)\s+verse\s+(\d+)\b",
        r"\1:\2",
        normalized,
        flags=re.IGNORECASE,
    )
    normalized = re.sub(
        r"\b(\d+)\s+verse\s+(\d+)\b",
        r"\1:\2",
        normalized,
        flags=re.IGNORECASE,
    )

    patterns = [
        re.compile(
            r"\b(?:[1-3]\s+)?[A-Za-z]+"
            r"(?:\s+[A-Za-z]+){0,3}\s+"
            r"\d+\s*:\s*\d+"
            r"(?:\s*(?:-|to)\s*(?:verse\s*)?\d+)?",
            re.IGNORECASE,
        ),
        re.compile(
            r"\b((?:[1-3]\s+)?[A-Za-z]+"
            r"(?:\s+[A-Za-z]+){0,3})\s+"
            r"(\d+)\s+(\d+)\b",
            re.IGNORECASE,
        ),
    ]

    for pattern_index, pattern in enumerate(patterns):
        for match in pattern.finditer(normalized):
            if pattern_index == 0:
                candidate_text = match.group(0).strip()
            else:
                candidate_text = (
                    f"{match.group(1)} "
                    f"{match.group(2)}:{match.group(3)}"
                )

            resolved = resolve_reference(
                translation_code,
                candidate_text,
            )
            if resolved is not None:
                return resolved

    return None


def _tokens(text: str) -> list[str]:
    """Normalize text into meaningful lexical tokens."""

    tokens = re.findall(r"[a-z]+", text.casefold())
    return [
        token
        for token in tokens
        if token not in _STOP_WORDS and len(token) > 1
    ]


def _score_passage(query_tokens: list[str], verse_text: str) -> float:
    """Score the proportion of query words found in a verse."""

    verse_tokens = set(_tokens(verse_text))
    query_counts = Counter(query_tokens)

    if not query_counts:
        return 0.0

    matched = sum(
        count
        for token, count in query_counts.items()
        if token in verse_tokens
    )
    return matched / sum(query_counts.values())


def find_passage_candidates(
    translation_code: str,
    transcript: str,
    limit: int = 5,
) -> list[PassageCandidate]:
    """Return an explicit passage or ranked local Bible verse candidates.

    Results are suggestions only. This function never controls projection.
    """

    transcript = transcript.strip()
    if not transcript or limit < 1:
        return []

    # Prefer an explicit Bible reference when one can be resolved.
    resolved = _extract_explicit_reference(
        translation_code,
        transcript,
    )

    if resolved is not None:
        verse = resolved["first_verse"]
        reference = (
            f'{resolved["book"]} {resolved["chapter"]}:'
            f'{resolved["verse"]}'
        )

        return [
            PassageCandidate(
                reference=reference,
                text=verse["text"],
                score=1.0,
                match_type="explicit",
            )
        ]

    # Otherwise search the installed translation for lexical matches.
    query_tokens = _tokens(transcript)
    if not query_tokens:
        return []

    candidates = []

    for verse in get_all_verses(translation_code):
        score = _score_passage(query_tokens, verse["text"])

        if score <= 0:
            continue

        candidates.append(
            PassageCandidate(
                reference=(
                    f'{verse["book_name"]} {verse["chapter"]}:'
                    f'{verse["verse"]}'
                ),
                text=verse["text"],
                score=score,
                match_type="quote",
            )
        )

    candidates.sort(
        key=lambda candidate: (
            -candidate.score,
            len(_tokens(candidate.text)),
            candidate.reference,
        )
    )

    return candidates[:limit]
