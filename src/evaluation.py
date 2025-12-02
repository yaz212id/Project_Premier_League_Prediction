"""Evaluation helpers for Premier League models."""

import numpy as np
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix


def evaluate_model(
    model,
    X_test: np.ndarray,
    y_test: np.ndarray,
    model_name: str,
    print_report: bool = True,
) -> float:
    """
    Evaluate a model on the test set and optionally print metrics.

    Returns
    -------
    accuracy : float
        Overall classification accuracy on the test set.
    """
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)

    if print_report:
        print(f"\n{model_name} Results")
        print("-" * (len(model_name) + 8))
        print(f"Accuracy: {acc:.3f}")
        print("\nClassification report:")
        # zero_division=0 avoids ugly warnings if a class is never predicted
        print(classification_report(y_test, y_pred, zero_division=0))
        print("Confusion matrix:")
        print(confusion_matrix(y_test, y_pred))

    return acc
