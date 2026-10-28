"""
TextGuard AI - Machine Learning Training Pipeline
Trains, validates, and serializes a TF-IDF + Classifier pipeline using the
AI vs Human Text Classification Benchmark dataset.
"""

import json
import os
import sys
import time
from typing import Dict, Any

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline


DATASET_FILENAME = "AI_vs_Human_Text_Classification_Benchmark.csv"
MODEL_DIR = "models"
MODEL_PATH = os.path.join(MODEL_DIR, "textguard_pipeline.joblib")
EVAL_RESULTS_PATH = os.path.join(MODEL_DIR, "evaluation_results.json")
CONFUSION_MATRIX_PATH = os.path.join(MODEL_DIR, "confusion_matrix.png")
METADATA_PATH = os.path.join(MODEL_DIR, "model_metadata.json")


def load_and_preprocess_dataset(csv_path: str, sample_size: int = None) -> tuple:
    """
    Load dataset, handle missing records, and return train/test splits.
    """
    print(f"[*] Loading dataset from: {csv_path}...")
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"Dataset file not found at: {csv_path}")

    # Load only necessary columns to optimize RAM and speed
    df = pd.read_csv(csv_path, usecols=["Raw_Text", "Label", "Source_Model"])

    # Clean missing values
    df = df.dropna(subset=["Raw_Text", "Label"])
    df["Raw_Text"] = df["Raw_Text"].astype(str)
    df["Label"] = df["Label"].astype(int)

    total_records = len(df)
    print(f"[*] Loaded {total_records:,} valid records.")
    print(f"    - Human-written (Label 0): {(df['Label'] == 0).sum():,} samples")
    print(f"    - AI-generated (Label 1):  {(df['Label'] == 1).sum():,} samples")

    if sample_size and sample_size < total_records:
        print(f"[*] Subsampling to {sample_size:,} balanced records for rapid training...")
        df = df.groupby("Label", group_keys=False).apply(
            lambda x: x.sample(n=sample_size // 2, random_state=42)
        )

    X = df["Raw_Text"].values
    y = df["Label"].values

    # Stratified 80/20 train/test split to guarantee identical label proportions
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42, stratify=y
    )

    print(f"[*] Dataset split completed:")
    print(f"    - Training set:   {len(X_train):,} samples")
    print(f"    - Evaluation set: {len(X_test):,} samples")

    return X_train, X_test, y_train, y_test


def build_pipeline(max_features: int = 20000) -> Pipeline:
    """
    Construct a unified Scikit-learn Pipeline combining TF-IDF vectorization
    and Logistic Regression classifier.
    """
    vectorizer = TfidfVectorizer(
        ngram_range=(1, 2),
        sublinear_tf=True,
        max_features=max_features,
        strip_accents="unicode",
        min_df=3,
        max_df=0.90,
    )

    classifier = LogisticRegression(
        C=1.5,
        max_iter=1000,
        solver="lbfgs",
        class_weight="balanced",
        random_state=42,
    )

    return Pipeline([
        ("tfidf", vectorizer),
        ("clf", classifier),
    ])


def train_and_evaluate(
    csv_path: str = DATASET_FILENAME,
    max_features: int = 20000,
    sample_size: int = None,
) -> Dict[str, Any]:
    """
    Execute end-to-end model training, evaluation, and artifact generation.
    """
    os.makedirs(MODEL_DIR, exist_ok=True)
    start_time = time.time()

    X_train, X_test, y_train, y_test = load_and_preprocess_dataset(csv_path, sample_size)

    # Build Pipeline
    print(f"[*] Building TF-IDF (1-2 ngrams, max_features={max_features:,}) + Logistic Regression Pipeline...")
    pipeline = build_pipeline(max_features=max_features)

    # Train model
    print("[*] Training model on training dataset...")
    train_start = time.time()
    pipeline.fit(X_train, y_train)
    train_duration = time.time() - train_start
    print(f"[+] Training completed in {train_duration:.2f} seconds.")

    # Evaluate on held-out test split
    print("[*] Evaluating model on independent test split...")
    y_pred = pipeline.predict(X_test)
    y_proba = pipeline.predict_proba(X_test)[:, 1]

    # Calculate metrics
    accuracy = float(accuracy_score(y_test, y_pred))
    precision_human = float(precision_score(y_test, y_pred, pos_label=0))
    recall_human = float(recall_score(y_test, y_pred, pos_label=0))
    f1_human = float(f1_score(y_test, y_pred, pos_label=0))

    precision_ai = float(precision_score(y_test, y_pred, pos_label=1))
    recall_ai = float(recall_score(y_test, y_pred, pos_label=1))
    f1_ai = float(f1_score(y_test, y_pred, pos_label=1))

    f1_macro = float(f1_score(y_test, y_pred, average="macro"))
    f1_weighted = float(f1_score(y_test, y_pred, average="weighted"))
    roc_auc = float(roc_auc_score(y_test, y_proba))
    cm = confusion_matrix(y_test, y_pred).tolist()

    report_str = classification_report(
        y_test, y_pred, target_names=["Human-written (0)", "AI-generated (1)"]
    )
    print("\n" + "=" * 50)
    print("CLASSIFICATION REPORT ON HELD-OUT TEST DATA")
    print("=" * 50)
    print(report_str)
    print(f"Test Accuracy: {accuracy * 100:.2f}%")
    print(f"ROC-AUC Score: {roc_auc:.4f}")
    print(f"Confusion Matrix:\n  [[TN={cm[0][0]}, FP={cm[0][1]}],\n   [FN={cm[1][0]}, TP={cm[1][1]}]]")
    print("=" * 50 + "\n")

    # Save model pipeline
    print(f"[*] Serializing trained pipeline to: {MODEL_PATH}...")
    joblib.dump(pipeline, MODEL_PATH, compress=3)
    model_size_mb = os.path.getsize(MODEL_PATH) / (1024 * 1024)
    print(f"[+] Model saved successfully ({model_size_mb:.2f} MB).")

    # Generate and save Confusion Matrix plot
    print(f"[*] Generating confusion matrix visualization: {CONFUSION_MATRIX_PATH}...")
    fig, ax = plt.subplots(figsize=(6, 5), dpi=150)
    cm_arr = np.array(cm)
    cax = ax.matshow(cm_arr, cmap="Blues", alpha=0.85)
    plt.colorbar(cax)

    classes = ["Human-written", "AI-generated"]
    ax.set_xticks([0, 1])
    ax.set_yticks([0, 1])
    ax.set_xticklabels(classes, fontsize=10)
    ax.set_yticklabels(classes, fontsize=10)
    ax.set_xlabel("Predicted Label", fontsize=11, fontweight="bold", labelpad=10)
    ax.set_ylabel("True Label", fontsize=11, fontweight="bold")
    ax.set_title("TextGuard AI - Confusion Matrix (Test Set)", fontsize=12, fontweight="bold", pad=15)

    for i in range(2):
        for j in range(2):
            count_val = cm_arr[i, j]
            pct_val = count_val / len(y_test) * 100
            color = "white" if count_val > len(y_test) * 0.25 else "black"
            ax.text(
                j, i, f"{count_val:,}\n({pct_val:.1f}%)",
                ha="center", va="center", color=color, fontsize=11, fontweight="bold"
            )

    plt.tight_layout()
    plt.savefig(CONFUSION_MATRIX_PATH, dpi=150)
    plt.close()
    print(f"[+] Confusion matrix plot saved.")

    # Save evaluation results dictionary
    eval_data = {
        "dataset_name": "AI vs Human Text Classification Benchmark",
        "dataset_total_samples": len(X_train) + len(X_test),
        "train_samples": len(X_train),
        "test_samples": len(X_test),
        "train_duration_seconds": round(train_duration, 2),
        "total_duration_seconds": round(time.time() - start_time, 2),
        "model_type": "TF-IDF Vectorizer + Calibrated Logistic Regression",
        "vectorizer_max_features": max_features,
        "metrics": {
            "accuracy": round(accuracy, 4),
            "accuracy_pct": round(accuracy * 100, 2),
            "roc_auc": round(roc_auc, 4),
            "f1_macro": round(f1_macro, 4),
            "f1_weighted": round(f1_weighted, 4),
            "human_class": {
                "precision": round(precision_human, 4),
                "recall": round(recall_human, 4),
                "f1_score": round(f1_human, 4),
                "support": int((y_test == 0).sum()),
            },
            "ai_class": {
                "precision": round(precision_ai, 4),
                "recall": round(recall_ai, 4),
                "f1_score": round(f1_ai, 4),
                "support": int((y_test == 1).sum()),
            },
            "confusion_matrix": {
                "true_negative": int(cm[0][0]),
                "false_positive": int(cm[0][1]),
                "false_negative": int(cm[1][0]),
                "true_positive": int(cm[1][1]),
            },
        },
    }

    with open(EVAL_RESULTS_PATH, "w", encoding="utf-8") as f:
        json.dump(eval_data, f, indent=4)
    print(f"[+] Evaluation results saved to: {EVAL_RESULTS_PATH}")

    # Save model metadata
    metadata = {
        "model_name": "TextGuard TF-IDF Logistic Classifier",
        "model_version": "1.0.0",
        "created_date": time.strftime("%Y-%m-%d %H:%M:%S"),
        "author": "AIUB Python Project Team",
        "course": "CSC 4162 - Programming in Python",
        "accuracy": f"{accuracy * 100:.2f}%",
        "f1_macro": round(f1_macro, 4),
        "roc_auc": round(roc_auc, 4),
        "input_features": f"TF-IDF n-grams (1,2) with {max_features:,} vocabulary size",
    }
    with open(METADATA_PATH, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=4)
    print(f"[+] Model metadata saved to: {METADATA_PATH}")

    return eval_data


if __name__ == "__main__":
    # Allow passing sample size via command line for testing if needed
    sample = int(sys.argv[1]) if len(sys.argv) > 1 else None
    train_and_evaluate(sample_size=sample)
