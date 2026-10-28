"""
Tests for analysis/text_statistics.py
Covers token extraction, TTR diversity, sentence std dev, transitions, and readability.
"""

import pytest
from analysis.text_statistics import LinguisticAnalyzer


def test_basic_counts():
    text = "The quick brown fox. It jumps over the dog."
    analyzer = LinguisticAnalyzer(text)
    assert analyzer.word_count == 9
    assert analyzer.sentence_count == 2
    assert analyzer.char_count_total > 0
    assert analyzer.paragraph_count == 1


def test_lexical_diversity():
    # Repetitive words vs all unique words
    repetitive = "the the the the the the the the"
    diverse = "apple banana cherry date elderberry fig grape"

    ana_rep = LinguisticAnalyzer(repetitive)
    ana_div = LinguisticAnalyzer(diverse)

    assert ana_rep.ttr < 0.2
    assert ana_div.ttr == 1.0


def test_sentence_length_std_burstiness():
    # Uniform sentence lengths: 5 words each
    uniform = "One two three four five. Six seven eight nine ten. Eleven twelve thirteen fourteen fifteen."
    # Highly varied sentence lengths: 3 words, 10 words, 2 words
    varied = "Short first part. Here is a much longer and more complex descriptive second sentence. End now."

    ana_uni = LinguisticAnalyzer(uniform)
    ana_var = LinguisticAnalyzer(varied)

    assert ana_uni.sentence_length_std < 0.5
    assert ana_var.sentence_length_std > 2.0


def test_transition_words_detection():
    text = "Furthermore, the data suggests this conclusion. Moreover, we must observe results. Therefore, it is clear."
    analyzer = LinguisticAnalyzer(text)
    assert analyzer.transition_word_count >= 3
    found_words = [t[0] for t in analyzer.detected_transitions]
    assert "furthermore" in found_words
    assert "moreover" in found_words
    assert "therefore" in found_words


def test_flesch_readability():
    simple_text = "The cat sat on the mat. The dog ran in the park. It was a nice day."
    analyzer = LinguisticAnalyzer(simple_text)
    assert analyzer.flesch_reading_ease > 70.0  # Very easy to read
    assert analyzer.flesch_kincaid_grade < 5.0   # Elementary grade level
