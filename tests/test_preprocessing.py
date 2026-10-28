"""
Tests for ml/preprocessing.py
Covers empty text, whitespace normalization, unicode, boundary limits, and sentence splitting.
"""

import pytest
from ml.preprocessing import (
    normalize_whitespace,
    clean_text_for_ml,
    validate_input_text,
    extract_sentences,
    MAX_TEXT_CHARACTERS,
)


def test_empty_text():
    res = validate_input_text("")
    assert not res["is_valid"]
    assert "empty" in res["error"].lower()


def test_none_input():
    res = validate_input_text(None)
    assert not res["is_valid"]
    assert res["error"] is not None


def test_very_short_text():
    res = validate_input_text("Hello there friend.")
    assert not res["is_valid"]
    assert "too brief" in res["error"].lower()


def test_short_text_warning():
    # 10 words - valid but below recommended 25 words
    text = "The quick brown fox jumps over the lazy dog today."
    res = validate_input_text(text)
    assert res["is_valid"]
    assert res["warning"] is not None
    assert not res["is_optimal"]


def test_optimal_text():
    # > 25 words
    text = (
        "Machine learning models rely on statistical representations of language to predict "
        "the probability of words and sentences. Higher dimensional n-gram representations capture "
        "subtle characteristics of human writing style compared to synthetic generation."
    )
    res = validate_input_text(text)
    assert res["is_valid"]
    assert res["is_optimal"]
    assert res["warning"] is None


def test_whitespace_normalization():
    dirty = "   Multiple    spaces\t\tand\n\n\n\nnewlines   "
    clean = normalize_whitespace(dirty)
    assert "    " not in clean
    assert "\t" not in clean
    assert "\n\n\n" not in clean


def test_unicode_and_special_chars():
    unicode_text = "“Smart quotes”, non-breaking\u00A0space, and em-dash—should be handled properly."
    res = validate_input_text(unicode_text)
    assert res["is_valid"]
    assert res["char_count"] > 0


def test_sentence_extraction():
    text = "First sentence here! Second sentence is here? Third sentence concludes."
    sentences = extract_sentences(text)
    assert len(sentences) == 3
    assert sentences[0] == "First sentence here!"
    assert sentences[1] == "Second sentence is here?"
    assert sentences[2] == "Third sentence concludes."


def test_excessive_length():
    huge_text = "word " * (MAX_TEXT_CHARACTERS // 3)
    res = validate_input_text(huge_text)
    assert not res["is_valid"]
    assert "exceeds" in res["error"].lower()
