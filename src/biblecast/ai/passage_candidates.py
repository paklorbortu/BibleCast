
"""Find explicit Bible references and rank local passage candidates."""
from __future__ import annotations

import re
from collections import Counter
from dataclasses import dataclass
from difflib import SequenceMatcher
from functools import lru_cache
from biblecast.bible.database import get_all_verses
from biblecast.bible.reference_resolver import resolve_reference


@dataclass(frozen=True)
class PassageCandidate:
    """A passage suggested for operator review."""

    reference: str
    text: str
    score: float
    match_type: str
    confidence: str = "low"
    evidence: str = ""

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


# INSERT THE NEW HELPERS HERE

_QUOTE_STOP_WORDS = _STOP_WORDS 


def _quote_tokens(text: str) -> list[str]:
    """Tokenize a quotation while preserving word order."""
    return re.findall(r"[a-z]+", text.casefold())


@lru_cache(maxsize=4)
def _build_quote_index(translation_code: str) -> tuple[dict, ...]:
    """Cache verse text and ordered tokens for each translation."""
    indexed = []

    for verse in get_all_verses(translation_code):
        indexed.append({
            "reference": (
                f'{verse["book_name"]} '
                f'{verse["chapter"]}:{verse["verse"]}'
            ),
            "book_name": verse["book_name"],
            "chapter": verse["chapter"],
            "verse": verse["verse"],
            "text": verse["text"],
            "tokens": tuple(_quote_tokens(verse["text"])),
        })

    return tuple(indexed)


def _build_quote_windows(
    verses: tuple[dict, ...],
) -> list[dict]:
    """Build searchable windows of one to three consecutive verses."""
    windows = []

    for start in range(len(verses)):
        for size in (1, 2, 3):
            end = start + size

            if end > len(verses):
                break

            group = verses[start:end]

            if any(
                group[i]["book_name"] != group[0]["book_name"]
                or group[i]["chapter"] != group[0]["chapter"]
                or group[i]["verse"] != group[i - 1]["verse"] + 1
                for i in range(1, len(group))
            ):
                break

            windows.append({
                "reference": group[0]["reference"],
                "text": " ".join(item["text"] for item in group),
                "tokens": tuple(
                    token
                    for item in group
                    for token in item["tokens"]
                ),
            })

    return windows


def _match_ordered_quote(
    transcript: str,
    translation_code: str,
    limit: int,
) -> list[PassageCandidate]:
    """Find quotations supported by an ordered run of matching words."""
    query = _quote_tokens(transcript)

    informative = [
        token for token in query
        if token not in _QUOTE_STOP_WORDS and len(token) > 2
    ]

    if len(query) < 4 or len(informative) < 2:
        return []

    verses = _build_quote_index(translation_code)
    windows = _build_quote_windows(verses)
    ranked = []

    for window in windows:
        target = list(window["tokens"])
        matcher = SequenceMatcher(
            None, query, target, autojunk=False
        )
        block = matcher.find_longest_match(
            0, len(query), 0, len(target)
        )

        matched_tokens = query[
            block.a:block.a + block.size
        ]
        matched_informative = [
            token for token in matched_tokens
            if token not in _QUOTE_STOP_WORDS and len(token) > 2
        ]

        if block.size < 4 or len(matched_informative) < 2:
            continue

        coverage = block.size / len(query)
        if coverage < 0.5:
            continue

        score = coverage * (
            len(matched_informative) / len(informative)
        )

        ranked.append(
            PassageCandidate(
                reference=window["reference"],
                text=window["text"],
                score=score,
                match_type="quote",
                confidence="medium",
                evidence=" ".join(matched_tokens),
            )
        )
    ranked.sort(
        key=lambda item: (
            -item.score,
            len(_quote_tokens(item.text)),
            item.reference,
        )        
    )
    return ranked[:limit]



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
    # Search for a sufficiently strong ordered quotation.
    # Paraphrase matching will be added as a separate stage.
    return _match_ordered_quote(
        transcript,
        translation_code,
        limit,
    )