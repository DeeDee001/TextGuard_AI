"""
TextGuard AI - Model Evaluation & Diagnostic Script
Performs deep performance diagnostics, error analysis, and cross-validation verification.
"""

import json
import os
import joblib
import pandas as pd
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score


def run_evaluation(csv_path="AI_vs_Human_Text_Classification_Benchmark.csv", model_path="models/textguard_pipeline.joblib"):
    """
    Load trained model and evaluate on benchmark data with diagnostics.
    """
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Trained model not found at: {model_path}. Please run ml/train_model.py first.")

    pipeline = joblib.load(model_path)
    print(f"[*] Loaded model pipeline from: {model_path}")
    print(f"    - Vectorizer: {pipeline.named_steps['tfidf']}")
    print(f"    - Classifier: {pipeline.named_steps['clf']}")

    df = pd.read_csv(csv_path, usecols=["Raw_Text", "Label", "Source_Model"]).dropna()
    # Test on a diverse slice of 5000 random samples across all source models
    sample_df = df.sample(n=min(5000, len(df)), random_state=123)

    X = sample_df["Raw_Text"].values
    y = sample_df["Label"].values
    models = sample_df["Source_Model"].values

    preds = pipeline.predict(X)
    probas = pipeline.predict_proba(X)[:, 1]

    print("\n" + "=" * 60)
    print("DIAGNOSTIC EVALUATION ACROSS GENERATION SOURCES")
    print("=" * 60)

    # Breakdown by Source Model
    unique_sources = set(models)
    for src in sorted(unique_sources):
        mask = (models == src)
        sub_y = y[mask]
        sub_preds = preds[mask]
        acc = (sub_y == sub_preds).mean() * 100
        print(f"Source: {src:<20} | Samples: {mask.sum():<5} | Detection Accuracy: {acc:.2f}%")

    print("=" * 60)
    print("\nOverall Classification Report:")
    print(classification_report(y, preds, target_names=["Human", "AI"]))


if __name__ == "__main__":
    run_evaluation()
