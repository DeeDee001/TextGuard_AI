"""
Tests for ml/predictor.py
Validates predictor loading, inference accuracy on distinct samples,
uncertainty handling, and edge case resilience.
"""

import pytest
from ml.predictor import TextGuardPredictor


@pytest.fixture(scope="module")
def predictor():
    return TextGuardPredictor()


def test_predictor_initialization(predictor):
    assert predictor.is_ready
    assert predictor.pipeline is not None
    assert "TextGuard" in predictor.model_name


def test_human_sample_prediction(predictor):
    text = (
        "When I first started programming in college, I honestly had no clue what I was doing. "
        "I spent hours trying to fix silly syntax errors and forgot semicolons all the time. "
        "Taking a walk around the lake helped clear my head before I finally got it working."
    )
    result = predictor.predict(text)
    assert result["success"]
    assert "Human" in result["verdict"]
    assert result["human_probability"] > 50.0
    assert result["confidence_level"] in ["Moderate", "High"]


def test_ai_sample_prediction(predictor):
    text = (
        "Artificial intelligence represents a critical paradigm in contemporary computational discourse. "
        "With increasing integration across diverse industrial sectors, organizational stakeholders "
        "are systematically reevaluating traditional operational frameworks. Data indicates that efficiency is subject to continuous enhancement."
    )
    result = predictor.predict(text)
    assert result["success"]
    assert "AI" in result["verdict"]
    assert result["ai_probability"] > 50.0


def test_empty_input_handling(predictor):
    result = predictor.predict("")
    assert not result["success"]
    assert "Invalid" in result["verdict"]


def test_sub_minimum_input_rejection(predictor):
    # Less than 5 words triggers validation error
    result = predictor.predict("Hi there.")
    assert not result["success"]
    assert "too brief" in result["error"].lower()


def test_short_input_warning(predictor):
    # 8 words: passes validation but includes warning banner
    text = "This is a brief passage of eight words."
    result = predictor.predict(text)
    assert result["success"]
    assert result["warning"] is not None


def test_missing_model_handling():
    with pytest.raises(FileNotFoundError):
        TextGuardPredictor(model_path="non_existent_model_file.joblib")
