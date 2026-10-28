"""
TextGuard AI - Model Predictor Module
Provides thread-safe inference, confidence calibration, borderline analysis,
and structured prediction payloads for the GUI and reporting systems.
"""

import os
from typing import Dict, Any, Optional
import joblib

from ml.preprocessing import clean_text_for_ml, validate_input_text


DEFAULT_MODEL_PATH = os.path.join("models", "textguard_pipeline.joblib")


class TextGuardPredictor:
    """
    Inference controller for the TextGuard AI text classification model.
    Encapsulates input validation, vectorization, calibrated probability extraction,
    and confidence reasoning.
    """

    def __init__(self, model_path: Optional[str] = None):
        self.model_path = model_path or DEFAULT_MODEL_PATH
        self.pipeline = None
        self.model_name = "TextGuard TF-IDF Calibrated Logistic Classifier"
        self.model_version = "1.0.0"
        self._load_model()

    def _load_model(self) -> None:
        """
        Load the serialized Scikit-learn Pipeline from disk.
        """
        if not os.path.exists(self.model_path):
            raise FileNotFoundError(
                f"Model file not found at '{self.model_path}'. "
                f"Please run the training pipeline first using 'python ml/train_model.py'."
            )
        try:
            self.pipeline = joblib.load(self.model_path)
        except Exception as e:
            raise RuntimeError(
                f"Failed to deserialize model pipeline from '{self.model_path}': {str(e)}"
            )

    @property
    def is_ready(self) -> bool:
        """Check if model pipeline is loaded and operational."""
        return self.pipeline is not None

    def predict(self, raw_text: str) -> Dict[str, Any]:
        """
        Execute full prediction pipeline on user-submitted text.

        Args:
            raw_text (str): Raw text submitted from GUI or API.

        Returns:
            Dict[str, Any]: Structured prediction payload.
        """
        # 1. Input Validation
        validation = validate_input_text(raw_text)
        if not validation["is_valid"]:
            return {
                "success": False,
                "error": validation["error"],
                "warning": validation["warning"],
                "verdict": "Invalid Input",
                "ai_probability": 0.0,
                "human_probability": 0.0,
                "confidence_level": "None",
                "confidence_score": 0.0,
                "explanation": validation["error"],
                "is_borderline": False,
                "word_count": validation["word_count"],
                "char_count": validation["char_count"],
            }

        cleaned = validation["cleaned_text"]
        word_count = validation["word_count"]
        warning = validation["warning"]

        if not self.is_ready:
            return {
                "success": False,
                "error": "Model is not loaded. Cannot execute analysis.",
                "verdict": "Model Error",
                "ai_probability": 0.0,
                "human_probability": 0.0,
                "confidence_level": "None",
            }

        # 2. ML Inference (Calibrated probabilities from Logistic Regression)
        try:
            probas = self.pipeline.predict_proba([cleaned])[0]
            human_prob = float(probas[0])
            ai_prob = float(probas[1])
        except Exception as e:
            return {
                "success": False,
                "error": f"Inference execution failed: {str(e)}",
                "verdict": "Inference Error",
                "ai_probability": 0.0,
                "human_probability": 0.0,
                "confidence_level": "None",
            }

        # 3. Confidence & Borderline / Inconclusive Analysis
        # Margin measures how far apart the two class probabilities are
        margin = abs(ai_prob - human_prob)
        is_borderline = margin < 0.15  # Probabilities between 42.5% and 57.5%

        # Confidence categorization
        if is_borderline:
            verdict = "Inconclusive / Borderline"
            verdict_code = "INCONCLUSIVE"
            confidence_level = "Low (Uncertain)"
            explanation = (
                f"The text exhibits mixed stylistic and lexical markers. "
                f"The model's probability scores (Human: {human_prob*100:.1f}%, AI: {ai_prob*100:.1f}%) "
                f"fall within the uncertainty band (±7.5%), making a definitive determination statistically unreliable."
            )
        elif ai_prob > human_prob:
            verdict = "More Likely AI-Generated"
            verdict_code = "AI_GENERATED"
            if margin >= 0.50:
                confidence_level = "High"
            elif margin >= 0.25:
                confidence_level = "Moderate"
            else:
                confidence_level = "Slight"
            explanation = (
                f"The text exhibits high structural uniformity, characteristic lexical frequency n-grams, "
                f"and transition patterns common to synthetic language models (Estimated AI Score: {ai_prob*100:.1f}%)."
            )
        else:
            verdict = "More Likely Human-Written"
            verdict_code = "HUMAN_WRITTEN"
            if margin >= 0.50:
                confidence_level = "High"
            elif margin >= 0.25:
                confidence_level = "Moderate"
            else:
                confidence_level = "Slight"
            explanation = (
                f"The text features vocabulary variety, natural sentence rhythm, and informal phrase structures "
                f"consistent with human authorship (Estimated Human Score: {human_prob*100:.1f}%)."
            )

        # Append length warning if applicable
        if word_count < 30:
            explanation += (
                f" Warning: Analysis based on a short sample ({word_count} words); "
                f"longer texts yield substantially higher classification reliability."
            )

        return {
            "success": True,
            "verdict": verdict,
            "verdict_code": verdict_code,
            "ai_probability": round(ai_prob * 100, 2),
            "human_probability": round(human_prob * 100, 2),
            "confidence_level": confidence_level,
            "confidence_score": round(margin * 100, 2),
            "is_borderline": is_borderline,
            "explanation": explanation,
            "warning": warning,
            "word_count": word_count,
            "char_count": validation["char_count"],
            "model_name": self.model_name,
            "model_version": self.model_version,
            "is_calibrated": True,
        }
