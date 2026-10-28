"""
TextGuard AI — AI Text Detection System Using Python & Machine Learning
Main Application Entry Point.

American International University-Bangladesh (AIUB)
Course: CSC 4162 - Programming in Python
"""

import os
import sys
import argparse


def main():
    """Main CLI and GUI launcher."""
    parser = argparse.ArgumentParser(
        description="TextGuard AI — Intelligent Text Authorship Analysis System"
    )
    parser.add_argument(
        "--train", action="store_true", help="Train machine learning model on benchmark dataset"
    )
    parser.add_argument(
        "--evaluate", action="store_true", help="Run model diagnostics and evaluation on test split"
    )
    parser.add_argument(
        "--cli", type=str, help="Run prediction on a string directly from terminal"
    )

    args = parser.parse_args()

    # CLI Model Training Option
    if args.train:
        from ml.train_model import train_and_evaluate
        print("[*] Launching model training pipeline...")
        train_and_evaluate()
        return

    # CLI Evaluation Option
    if args.evaluate:
        from ml.evaluate_model import run_evaluation
        print("[*] Launching model evaluation suite...")
        run_evaluation()
        return

    # CLI Direct Inference Option
    if args.cli:
        from ml.predictor import TextGuardPredictor
        from analysis.text_statistics import LinguisticAnalyzer
        predictor = TextGuardPredictor()
        res = predictor.predict(args.cli)
        stats = LinguisticAnalyzer(args.cli).to_dict()
        print("\n" + "=" * 50)
        print("TextGuard AI — CLI Result")
        print("=" * 50)
        print(f"Verdict        : {res['verdict']}")
        print(f"AI Likelihood  : {res['ai_probability']:.2f}%")
        print(f"Human Likelihood: {res['human_probability']:.2f}%")
        print(f"Confidence     : {res['confidence_level']}")
        print(f"Word Count     : {stats['word_count']}")
        print(f"Diversity (TTR): {stats['lexical_diversity_ttr']}")
        print("=" * 50)
        return

    # Default: Launch Desktop Graphical User Interface
    print("[*] Launching TextGuard AI Desktop Interface...")
    from gui.app import launch_app
    launch_app()


if __name__ == "__main__":
    main()
