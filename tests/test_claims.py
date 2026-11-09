"""Tests for the rule-based claim splitter."""

from groundcheck.claims import ClaimSplitter


def test_split_simple_sentences():
    splitter = ClaimSplitter(split_clauses=False)
    text = "The earth orbits the sun. Water boils at 100 degrees Celsius."
    claims = splitter.split(text)
    assert len(claims) == 2
    assert claims[0].text == "The earth orbits the sun."
    assert claims[1].text == "Water boils at 100 degrees Celsius."
    assert text[claims[0].start_char : claims[0].end_char] == claims[0].text
    assert text[claims[1].start_char : claims[1].end_char] == claims[1].text


def test_split_clauses_semicolon():
    splitter = ClaimSplitter(split_clauses=True)
    text = "Python was created in 1991; Guido van Rossum designed it."
    claims = splitter.split(text)
    assert len(claims) == 2
    assert claims[0].text == "Python was created in 1991"
    assert claims[1].text == "Guido van Rossum designed it."
    assert text[claims[0].start_char : claims[0].end_char] == claims[0].text
    assert text[claims[1].start_char : claims[1].end_char] == claims[1].text


def test_split_with_abbreviations_and_decimals():
    splitter = ClaimSplitter(split_clauses=False)
    text = "Dr. Smith arrived at 3.14 PM from the U.S. He gave a lecture on physics."
    claims = splitter.split(text)
    assert len(claims) == 2
    assert "Dr. Smith arrived" in claims[0].text
    assert "He gave a lecture" in claims[1].text


def test_empty_and_whitespace_input():
    splitter = ClaimSplitter()
    assert splitter.split("") == []
    assert splitter.split("   \n\t  ") == []


def test_short_claim_filtering():
    splitter = ClaimSplitter(min_claim_length=10)
    claims = splitter.split("Hi. This is a longer meaningful sentence.")
    assert len(claims) == 1
    assert "This is a longer" in claims[0].text
