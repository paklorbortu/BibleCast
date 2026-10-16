
import pytest

from biblecast.bible.usfm_parser import parse_usfm


SAMPLE_USFM = r"""
\id GEN World English Bible (WEB)
\ide UTF-8
\h Genesis
\c 1
\p
\v 1 \w In|strong="H8064"\w* \w the|strong="H1254"\w* beginning, God created the earth.\f + \fr 1:1 \ft A sample footnote.\f*
\v 2 The earth was empty.
\c 2
\p
\v 1 God made another beginning.
"""


def test_parse_book_metadata_and_chapters():
    result = parse_usfm(SAMPLE_USFM)

    assert result["book_id"] == "GEN"
    assert result["name"] == "Genesis"
    assert len(result["chapters"]) == 2
    assert result["chapters"][0]["number"] == 1
    assert result["chapters"][1]["number"] == 2


def test_parse_verses_and_remove_markup():
    result = parse_usfm(SAMPLE_USFM)

    verses = result["chapters"][0]["verses"]

    assert verses[0] == {
        "number": 1,
        "text": "In the beginning, God created the earth.",
    }
    assert verses[1]["number"] == 2
    assert "footnote" not in verses[0]["text"]
    assert "strong=" not in verses[0]["text"]


def test_reject_missing_book_id():
    with pytest.raises(ValueError, match="book ID"):
        parse_usfm(r"\h Genesis \c 1 \v 1 Text")



def test_reject_verse_before_chapter():
    content = (
        r"\id GEN World English Bible" + "\n"
        + r"\h Genesis" + "\n"
        + r"\v 1 Text"
    )

    with pytest.raises(ValueError, match="before a chapter"):
        parse_usfm(content)



def test_reject_empty_verse():
    content = r"\id GEN World English Bible (WEB)" + "\n"
    content += r"\h Genesis" + "\n"
    content += r"\c 1" + "\n"
    content += r"\v 1 \f + \ft Footnote only.\f*" + "\n"

    with pytest.raises(ValueError, match="No chapter verses were found"):
        parse_usfm(content)



def test_skip_footnote_only_verse():
    content = (
        r"\id LUK World English Bible (WEB)" + "\n"
        + r"\h Luke" + "\n"
        + r"\c 17" + "\n"
        + r"\v 35 Two will be together." + "\n"
        + r"\v 36 \f + \fr 17:36 \ft Some manuscripts add a sentence.\f*" + "\n"
        + r"\v 37 They will be taken." + "\n"
    )

    result = parse_usfm(content)
    verses = result["chapters"][0]["verses"]

    assert [verse["number"] for verse in verses] == [35, 37]
    assert all("Some manuscripts" not in verse["text"] for verse in verses)
