"""
Main script for Premier League project.

This is the file that the graders will run:

    python main.py

It loads the data, trains several models, evaluates them,
and prints a summary.
"""

import numpy as np
from sklearn.metrics import accuracy_score, confusion_matrix

from src.data_loader import load_and_split
from src.models import (
    train_random_forest,
    train_knn,
    train_logistic_regression,
)
from src.evaluation import evaluate_model


def main() -> None:
    """Load data, train several models, and compare their performance."""
    print("=" * 70)
    print("Premier League Match Outcome Prediction")
    print("=" * 70)

    # --------------------------------------------------------------
    # 1. Load and preprocess data
    # --------------------------------------------------------------
    print("\n1. Loading and preprocessing data...")
    X_train, X_test, y_train, y_test = load_and_split()
    print(f"   Train size: {X_train.shape}")
    print(f"   Test size:  {X_test.shape}")

    # --------------------------------------------------------------
    # 2. Baseline model: always predict 'home win'
    # --------------------------------------------------------------
    print("\n2. Baseline model (always predict 'home win')...")
    # In data_loader.py we encoded: 2 = home win
    baseline_pred = np.full_like(y_test, fill_value=2)
    baseline_acc = accuracy_score(y_test, baseline_pred)
    print(f"Baseline accuracy: {baseline_acc:.3f}")
    print("Baseline confusion matrix:")
    print(confusion_matrix(y_test, baseline_pred))

    # --------------------------------------------------------------
    # 3. Train machine-learning models
    # --------------------------------------------------------------
    print("\n3. Training ML models...")
    rf_model = train_random_forest(X_train, y_train)
    knn_model = train_knn(X_train, y_train)
    lr_model = train_logistic_regression(X_train, y_train)
    print("   ✓ All models trained.")

    # --------------------------------------------------------------
    # 4. Evaluate models on the test set
    # --------------------------------------------------------------
    print("\n4. Evaluating ML models on test set...")
    rf_acc = evaluate_model(rf_model, X_test, y_test, "Random Forest")
    knn_acc = evaluate_model(knn_model, X_test, y_test, "KNN")
    lr_acc = evaluate_model(lr_model, X_test, y_test, "Logistic Regression")

    # --------------------------------------------------------------
    # 5. Summary and winner
    # --------------------------------------------------------------
    results = {
        "Baseline (Home Win)": baseline_acc,
        "Random Forest": rf_acc,
        "KNN": knn_acc,
        "Logistic Regression": lr_acc,
    }
    winner = max(results, key=results.get)

    print("\n" + "=" * 70)
    print("Summary of test accuracies:")
    for name, acc in results.items():
        print(f" - {name}: {acc:.3f}")
    print("-" * 70)
    print(f"Best model: {winner} ({results[winner]:.3f} accuracy)")
    print("=" * 70)


if __name__ == "__main__":
    # This line makes sure main() runs when we type `python main.py`
    main()
