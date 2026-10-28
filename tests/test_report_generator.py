"""
Tests for analysis/report_generator.py and data/history_manager.py
Validates report exports (TXT, JSON, PDF) and persistent history operations.
"""

import os
import json
import pytest
from analysis.report_generator import ReportGenerator
from data.history_manager import HistoryManager


@pytest.fixture
def sample_report_data():
    return {
        "title": "TextGuard AI - Test Report",
        "generated_at": "2026-10-27 21:00:00",
        "application": "TextGuard AI v1.0.0",
        "institution": "American International University-Bangladesh (AIUB)",
        "course": "CSC 4162 - Programming in Python",
        "model_metadata": {
            "model_name": "TF-IDF Logistic Classifier",
            "model_version": "1.0.0",
            "is_calibrated": True,
        },
        "prediction": {
            "verdict": "More Likely Human-Written",
            "verdict_code": "HUMAN_WRITTEN",
            "ai_probability_pct": 12.5,
            "human_probability_pct": 87.5,
            "confidence_level": "High",
            "confidence_score": 75.0,
            "is_borderline": False,
            "explanation": "Natural sentence variance and informal vocabulary patterns.",
        },
        "linguistic_metrics": {
            "word_count": 45,
            "character_count_total": 280,
            "character_count_no_spaces": 235,
            "sentence_count": 3,
            "paragraph_count": 1,
            "avg_word_length": 5.2,
            "avg_sentence_length_words": 15.0,
            "burstiness_sentence_std": 4.5,
            "lexical_diversity_ttr": 0.85,
            "root_ttr": 5.7,
            "punctuation_count": 4,
            "punctuation_density_pct": 1.4,
            "transition_word_count": 1,
            "flesch_reading_ease": 65.0,
            "flesch_kincaid_grade": 7.5,
            "top_repeated_words": [("programming", 3), ("python", 2)],
        },
        "input_text": "Sample text for testing report exports in various formats.",
        "disclaimer": "Educational classification estimate only.",
    }


def test_txt_export(sample_report_data, tmp_path):
    out_file = str(tmp_path / "report.txt")
    result = ReportGenerator.export_txt(sample_report_data, out_file)
    assert os.path.exists(result)
    with open(result, "r", encoding="utf-8") as f:
        content = f.read()
    assert "TEXTGUARD AI" in content
    assert "More Likely Human-Written" in content
    assert "87.50%" in content


def test_json_export(sample_report_data, tmp_path):
    out_file = str(tmp_path / "report.json")
    result = ReportGenerator.export_json(sample_report_data, out_file)
    assert os.path.exists(result)
    with open(result, "r", encoding="utf-8") as f:
        data = json.load(f)
    assert data["prediction"]["verdict_code"] == "HUMAN_WRITTEN"
    assert data["linguistic_metrics"]["word_count"] == 45


def test_pdf_export(sample_report_data, tmp_path):
    out_file = str(tmp_path / "report.pdf")
    result = ReportGenerator.export_pdf(sample_report_data, out_file)
    assert os.path.exists(result)
    assert os.path.getsize(result) > 1000  # Generated PDF has binary content


def test_history_manager_crud(tmp_path):
    test_db = str(tmp_path / "history_test.json")
    hm = HistoryManager(filepath=test_db)

    # 1. Add record
    pred = {"verdict": "More Likely Human-Written", "ai_probability": 15.0, "human_probability": 85.0}
    stats = {"word_count": 50, "char_count_total": 300, "sentence_count": 3}
    rid = hm.add_record("Sample input text for history testing.", pred, stats)
    assert rid is not None
    assert len(hm.get_all_records()) == 1

    # 2. Search record
    matches = hm.search_records("history testing")
    assert len(matches) == 1
    assert matches[0]["id"] == rid

    # 3. Privacy controls: disable save
    hm.update_privacy_settings(save_enabled=False, full_text_enabled=True)
    rid2 = hm.add_record("Another text", pred, stats)
    assert rid2 is None
    assert len(hm.get_all_records()) == 1

    # 4. Delete single record
    deleted = hm.delete_record(rid)
    assert deleted
    assert len(hm.get_all_records()) == 0
