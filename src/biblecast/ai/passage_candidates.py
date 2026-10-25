
"""Find explicit Bible references and rank local passage candidates."""

from __future__ import annotations

import re
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

            # Leading filler such as "let's read" may be captured as part
            # of the book name, so retry with fewer leading words.
            words = candidate_text.split()
            for start in range(len(words) - 1):
                resolved = resolve_reference(
                    translation_code,
                    " ".join(words[start:]),
                )
                if resolved is not None:
                    return resolved

    return None


_MIN_QUERY_WORDS = 4
_MIN_INFORMATIVE_WORDS = 2
_MIN_COVERAGE = 0.5
_HIGH_COVERAGE = 0.85
_MEDIUM_COVERAGE = 0.65
_MAX_WINDOW_VERSES = 3
_RARE_TOKENS_USED = 3


@dataclass(frozen=True)
class _QuoteIndex:
    """Cached verse data and a word to verse lookup for one translation."""

    verses: tuple[dict, ...]
    postings: dict[str, tuple[int, ...]]


def _quote_tokens(text: str) -> list[str]:
    """Tokenize a quotation while preserving word order."""
    return re.findall(r"[a-z]+", text.casefold())


def _is_informative(token: str) -> bool:
    return token not in _STOP_WORDS and len(token) > 2


@lru_cache(maxsize=4)
def _build_quote_index(translation_code: str) -> _QuoteIndex:
    """Cache verse tokens and a word lookup for each translation."""
    verses = []
    postings: dict[str, list[int]] = {}

    for position, verse in enumerate(get_all_verses(translation_code)):
        tokens = tuple(_quote_tokens(verse["text"]))
        verses.append({
            "book_name": verse["book_name"],
            "chapter": verse["chapter"],
            "verse": verse["verse"],
            "text": verse["text"],
            "tokens": tokens,
        })
        for token in set(tokens):
            postings.setdefault(token, []).append(position)

    return _QuoteIndex(
        verses=tuple(verses),
        postings={
            token: tuple(positions)
            for token, positions in postings.items()
        },
    )


def clear_quote_cache() -> None:
    """Drop cached verse data, e.g. after a translation is imported."""
    _build_quote_index.cache_clear()


def _consecutive(group: tuple[dict, ...]) -> bool:
    """True when the verses are adjacent in the same book and chapter."""
    return all(
        group[i]["book_name"] == group[0]["book_name"]
        and group[i]["chapter"] == group[0]["chapter"]
        and group[i]["verse"] == group[i - 1]["verse"] + 1
        for i in range(1, len(group))
    )


def _reference_label(group: list[dict]) -> str:
    first, last = group[0], group[-1]
    label = f'{first["book_name"]} {first["chapter"]}:{first["verse"]}'
    if last is not first:
        label += f'-{last["verse"]}'
    return label


def _confidence_for(coverage: float) -> str:
    if coverage >= _HIGH_COVERAGE:
        return "high"
    if coverage >= _MEDIUM_COVERAGE:
        return "medium"
    return "low"


def _match_ordered_quote(
    transcript: str,
    translation_code: str,
    limit: int,
) -> list[PassageCandidate]:
    """Find quotations supported by an ordered run of matching words.

    Only verses sharing the rarest query words are examined, each
    alongside up to two following verses, so a quotation that crosses a
    verse boundary is still found. Each verse is reported once.
    """
    query = _quote_tokens(transcript)
    informative = [token for token in query if _is_informative(token)]

    if len(query) < _MIN_QUERY_WORDS or (
        len(informative) < _MIN_INFORMATIVE_WORDS
    ):
        return []

    index = _build_quote_index(translation_code)
    verses = index.verses

    # Candidate verses come from the rarest informative query words, so one
    # misheard word does not hide the real verse.
    distinct = sorted(
        set(informative),
        key=lambda token: len(index.postings.get(token, ())),
    )
    rare = [t for t in distinct if t in index.postings][:_RARE_TOKENS_USED]
    candidate_verses = {
        position
        for token in rare
        for position in index.postings[token]
    }

    seen_windows = set()
    best: dict[str, PassageCandidate] = {}

    for position in candidate_verses:
        for start in range(
            max(0, position - _MAX_WINDOW_VERSES + 1), position + 1
        ):
            for size in range(1, _MAX_WINDOW_VERSES + 1):
                end = start + size
                if end > len(verses) or (start, size) in seen_windows:
                    continue
                group = verses[start:end]
                if not _consecutive(group):
                    break
                seen_windows.add((start, size))

                target: list[str] = []
                owners: list[int] = []
                for offset, item in enumerate(group):
                    target.extend(item["tokens"])
                    owners.extend([offset] * len(item["tokens"]))

                if len(set(informative) & set(target)) < _MIN_INFORMATIVE_WORDS:
                    continue

                block = SequenceMatcher(
                    None, query, target, autojunk=False
                ).find_longest_match(0, len(query), 0, len(target))

                matched = query[block.a:block.a + block.size]
                matched_informative = [
                    t for t in matched if _is_informative(t)
                ]
                if block.size < _MIN_QUERY_WORDS or (
                    len(matched_informative) < _MIN_INFORMATIVE_WORDS
                ):
                    continue

                coverage = block.size / len(query)
                if coverage < _MIN_COVERAGE:
                    continue

                score = coverage * (
                    len(matched_informative) / len(informative)
                )

                # Report only the verses the matched words actually touch.
                first = owners[block.b]
                last = owners[block.b + block.size - 1]
                touched = list(group[first:last + 1])
                reference = _reference_label(touched)

                candidate = PassageCandidate(
                    reference=reference,
                    text=" ".join(item["text"] for item in touched),
                    score=score,
                    match_type="quote",
                    confidence=_confidence_for(coverage),
                    evidence=" ".join(matched),
                )

                current = best.get(reference)
                if current is None or candidate.score > current.score:
                    best[reference] = candidate

    results = list(best.values())

    # Weak partial matches add noise next to a confident match.
    if any(item.confidence == "high" for item in results):
        results = [item for item in results if item.confidence != "low"]

    ranked = sorted(
        results,
        key=lambda item: (
            -item.score,
            len(_quote_tokens(item.text)),
            item.reference,
        ),
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
                confidence="high",
            )
        ]

    # Otherwise look for an ordered quotation. Paraphrase matching will be
    # added as a separate stage.
    return _match_ordered_quote(
        transcript,
        translation_code,
        limit,
    )
