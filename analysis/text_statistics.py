"""
TextGuard AI - Linguistic & Stylometric Statistics Module
Computes detailed stylistic, lexical, and structural features from input text.
"""

import math
import re
import string
from collections import Counter
from typing import Dict, Any, List


# Set of common discourse markers and transition words frequently found in formal and synthetic writing
COMMON_TRANSITION_WORDS = {
    "furthermore", "moreover", "consequently", "therefore", "nevertheless",
    "nonetheless", "additionally", "subsequently", "accordingly", "similarly",
    "conversely", "meanwhile", "in conclusion", "in summary", "overall",
    "on the other hand", "for instance", "for example", "in particular",
    "as a result", "notably", "essentially", "crucially", "ultimately",
    "in addition", "to begin with", "in contrast", "hence", "thus"
}


class LinguisticAnalyzer:
    """
    Extracts descriptive linguistic, structural, and vocabulary metrics from text.
    These statistics provide educational insight into sentence rhythm, word complexity,
    and lexical diversity.
    """

    def __init__(self, text: str):
        self.raw_text = text or ""
        self.normalized_text = text.strip() if text else ""
        self._analyze()

    def _count_syllables_word(self, word: str) -> int:
        """
        Estimate syllable count of an English word using vowel grouping heuristics.
        """
        word = word.lower().strip()
        if not word:
            return 0
        if len(word) <= 3:
            return 1

        # Remove trailing 'e', 'es', 'ed' that don't constitute syllables
        word = re.sub(r"(?:[^laeiouy]|ed|es|e)$", "", word)
        vowel_groups = re.findall(r"[aeiouy]{1,2}", word)
        return max(1, len(vowel_groups))

    def _analyze(self):
        """
        Perform all extraction calculations on the input text.
        """
        # Paragraph extraction
        raw_paragraphs = [p.strip() for p in self.normalized_text.split("\n\n") if p.strip()]
        self.paragraph_count = max(1, len(raw_paragraphs)) if self.normalized_text else 0

        # Sentence extraction
        sentence_end = re.compile(r"(?<=[.!?])\s+(?=[A-Z0-9\"'‘“])")
        raw_sentences = sentence_end.split(self.normalized_text)
        self.sentences: List[str] = [s.strip() for s in raw_sentences if s.strip()]
        self.sentence_count = len(self.sentences)

        # Word tokens
        self.words: List[str] = re.findall(r"\b[\w'-]+\b", self.normalized_text)
        self.word_count = len(self.words)
        self.words_lower: List[str] = [w.lower() for w in self.words]

        # Character counts
        self.char_count_total = len(self.normalized_text)
        self.char_count_no_spaces = len(re.sub(r"\s+", "", self.normalized_text))

        # Sentence lengths (word counts per sentence)
        self.sentence_lengths: List[int] = []
        for s in self.sentences:
            s_words = re.findall(r"\b[\w'-]+\b", s)
            self.sentence_lengths.append(len(s_words))

        # Average and standard deviation of sentence lengths (Burstiness)
        if self.sentence_lengths:
            self.avg_sentence_length = sum(self.sentence_lengths) / len(self.sentence_lengths)
            variance = sum((x - self.avg_sentence_length) ** 2 for x in self.sentence_lengths) / len(self.sentence_lengths)
            self.sentence_length_std = math.sqrt(variance)
        else:
            self.avg_sentence_length = 0.0
            self.sentence_length_std = 0.0

        # Average word length
        if self.words:
            total_word_chars = sum(len(w) for w in self.words)
            self.avg_word_length = total_word_chars / self.word_count
        else:
            self.avg_word_length = 0.0

        # Vocabulary diversity (Type-Token Ratio TTR & Root TTR)
        unique_words = set(self.words_lower)
        self.unique_word_count = len(unique_words)
        if self.word_count > 0:
            self.ttr = self.unique_word_count / self.word_count
            self.root_ttr = self.unique_word_count / math.sqrt(self.word_count)
        else:
            self.ttr = 0.0
            self.root_ttr = 0.0

        # Punctuation counts and density
        punct_chars = [c for c in self.normalized_text if c in string.punctuation or c in "“”‘’—…"]
        self.punctuation_count = len(punct_chars)
        self.punctuation_density = (
            (self.punctuation_count / self.char_count_total * 100)
            if self.char_count_total > 0
            else 0.0
        )

        # Transition words usage
        text_lower = self.normalized_text.lower()
        transition_matches = 0
        detected_transitions = []
        for phrase in COMMON_TRANSITION_WORDS:
            count = len(re.findall(r"\b" + re.escape(phrase) + r"\b", text_lower))
            if count > 0:
                transition_matches += count
                detected_transitions.append((phrase, count))
        self.transition_word_count = transition_matches
        self.detected_transitions = sorted(detected_transitions, key=lambda x: x[1], reverse=True)

        # Syllables & Readability (Flesch Reading Ease & Grade Level)
        total_syllables = sum(self._count_syllables_word(w) for w in self.words)
        self.avg_syllables_per_word = (
            (total_syllables / self.word_count) if self.word_count > 0 else 0.0
        )

        if self.word_count > 0 and self.sentence_count > 0:
            # Flesch Reading Ease formula
            asl = self.avg_sentence_length
            asw = self.avg_syllables_per_word
            self.flesch_reading_ease = max(0.0, min(100.0, 206.835 - (1.015 * asl) - (84.6 * asw)))
            # Flesch-Kincaid Grade Level
            self.flesch_kincaid_grade = max(0.0, (0.39 * asl) + (11.8 * asw) - 15.59)
        else:
            self.flesch_reading_ease = 0.0
            self.flesch_kincaid_grade = 0.0

        # Top repeated words (excluding short stopwords)
        stopword_simple = {
            "the", "and", "a", "an", "of", "to", "in", "is", "that", "for",
            "it", "as", "was", "with", "on", "are", "by", "this", "be", "at",
            "from", "or", "have", "not", "but", "what", "all", "were", "when"
        }
        content_words = [w for w in self.words_lower if len(w) > 2 and w not in stopword_simple]
        word_frequencies = Counter(content_words)
        self.top_repeated_words = word_frequencies.most_common(5)

    def to_dict(self) -> Dict[str, Any]:
        """
        Export all calculated statistics as a clean serializable dictionary.
        """
        return {
            "word_count": self.word_count,
            "char_count_total": self.char_count_total,
            "char_count_no_spaces": self.char_count_no_spaces,
            "sentence_count": self.sentence_count,
            "paragraph_count": self.paragraph_count,
            "avg_word_length": round(self.avg_word_length, 2),
            "avg_sentence_length": round(self.avg_sentence_length, 2),
            "sentence_length_std": round(self.sentence_length_std, 2),
            "sentence_lengths": self.sentence_lengths,
            "unique_word_count": self.unique_word_count,
            "lexical_diversity_ttr": round(self.ttr, 3),
            "root_ttr": round(self.root_ttr, 2),
            "punctuation_count": self.punctuation_count,
            "punctuation_density_pct": round(self.punctuation_density, 2),
            "transition_word_count": self.transition_word_count,
            "detected_transitions": self.detected_transitions[:6],
            "flesch_reading_ease": round(self.flesch_reading_ease, 1),
            "flesch_kincaid_grade": round(self.flesch_kincaid_grade, 1),
            "top_repeated_words": self.top_repeated_words,
        }
