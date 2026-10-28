"""
TextGuard AI - Text Preprocessing Module
Provides robust, reusable text validation, cleaning, and normalization functions.
"""

import re
import unicodedata
from typing import Dict, Any, Tuple


# Minimum word threshold for statistically meaningful linguistic & ML analysis
MIN_RECOMMENDED_WORDS = 25
MIN_REQUIRED_WORDS = 5
MAX_TEXT_CHARACTERS = 100_000


def normalize_whitespace(text: str) -> str:
    """
    Standardize spaces, tabs, and newlines without collapsing paragraph breaks.

    Args:
        text (str): Input text string.

    Returns:
        str: Text with normalized horizontal spaces and clean newlines.
    """
    if not isinstance(text, str):
        return ""
    # Normalize unicode forms (NFKC handles ligatures, compatibility characters)
    text = unicodedata.normalize("NFKC", text)
    # Convert carriage returns to standard line breaks
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    # Replace non-breaking spaces and tabs with regular spaces
    text = re.sub(r"[ \t\u00A0\u2000-\u200B\u202F\u205F\u3000]+", " ", text)
    # Keep up to two consecutive newlines (paragraph separation)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def clean_text_for_ml(text: str) -> str:
    """
    Clean text for machine learning vectorization while preserving vocabulary
    and stylistic cues.

    Args:
        text (str): Raw input text.

    Returns:
        str: Cleaned text suitable for TF-IDF vectorization.
    """
    if not isinstance(text, str):
        return ""
    # Normalize whitespace
    text = normalize_whitespace(text)
    # Remove control characters except standard whitespace
    text = re.sub(r"[\x00-\x08\x0B\x0C\x0E-\x1F\x7F]", "", text)
    return text


def validate_input_text(text: Any) -> Dict[str, Any]:
    """
    Validate user input text to ensure it meets requirements for analysis.

    Args:
        text (Any): Input object from GUI or API.

    Returns:
        Dict[str, Any]: Dictionary containing validation status, counts, and messages.
    """
    result: Dict[str, Any] = {
        "is_valid": False,
        "is_optimal": False,
        "error": None,
        "warning": None,
        "char_count": 0,
        "word_count": 0,
        "cleaned_text": "",
    }

    if text is None or not isinstance(text, str):
        result["error"] = "Input text is required and must be a string."
        return result

    cleaned = normalize_whitespace(text)
    char_count = len(cleaned)
    words = re.findall(r"\b[\w'-]+\b", cleaned)
    word_count = len(words)

    result["char_count"] = char_count
    result["word_count"] = word_count
    result["cleaned_text"] = cleaned

    if char_count == 0 or word_count == 0:
        result["error"] = "Text editor is empty. Please enter or paste some text to analyze."
        return result

    if char_count > MAX_TEXT_CHARACTERS:
        result["error"] = (
            f"Text length exceeds maximum allowed limit ({MAX_TEXT_CHARACTERS:,} characters). "
            f"Your text contains {char_count:,} characters."
        )
        return result

    if word_count < MIN_REQUIRED_WORDS:
        result["error"] = (
            f"Text is too brief for analysis ({word_count} words). "
            f"Please provide at least {MIN_REQUIRED_WORDS} words."
        )
        return result

    result["is_valid"] = True

    if word_count < MIN_RECOMMENDED_WORDS:
        result["warning"] = (
            f"Notice: The text contains only {word_count} words. "
            f"Machine learning and linguistic confidence are significantly higher on texts "
            f"with at least {MIN_RECOMMENDED_WORDS} words."
        )
    else:
        result["is_optimal"] = True

    return result


def extract_sentences(text: str) -> list[str]:
    """
    Split text into sentences using regex boundary detection.
    Lightweight, CPU-friendly alternative to heavy spaCy models.

    Args:
        text (str): Input text.

    Returns:
        list[str]: Non-empty list of detected sentence strings.
    """
    if not text:
        return []
    # Standard sentence terminal regex pattern
    sentence_end = re.compile(r"(?<=[.!?])\s+(?=[A-Z0-9\"'‘“])")
    raw_sentences = sentence_end.split(text.strip())
    sentences = [s.strip() for s in raw_sentences if s.strip()]
    return sentences if sentences else [text.strip()]
