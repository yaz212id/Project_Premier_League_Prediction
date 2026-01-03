"""
Main script for Premier League project.

Run:
  python main.py

Outputs saved in ./results/ :
- comparison_2024_2025.csv
- model_accuracy_summary_2024_2025.png
- logistic_regression_coefficients.png
- table_actual_2024_2025.csv
- table_predicted_2024_2025.csv
- comparison_2025_2026_first14.csv
- table_actual_2025_2026_first14.csv
- table_predicted_2025_2026_first14.csv
"""

from __future__ import annotations

import os
import numpy as np
import pandas as pd

from src.data_loader import FEATURE_COLUMNS, load_future_season_features, load_train_test_with_metadata
from src.evaluation import (
    accuracy_ci_wilson,
    build_league_table,
    mcnemar_pvalue_vs_baseline,
    plot_accuracy_summary,
    plot_logistic_regression_coefficients,
    simple_accuracy,
)
from src.models import (
    train_gradient_boosting,
    train_knn,
    train_logistic_regression,
    train_random_forest,
)


def predict_codes(model, X: np.ndarray) -> np.ndarray:
    """
    Predict class codes. If the model has:
      - predict_proba
      - attribute draw_pred_boost > 1
    then we boost P(draw) before taking argmax.
    """
    boost = float(getattr(model, "draw_pred_boost", 1.0))
    if hasattr(model, "predict_proba") and boost != 1.0:
        probs = model.predict_proba(X)
        probs[:, 1] *= boost
        probs = probs / probs.sum(axis=1, keepdims=True)
        return probs.argmax(axis=1)
    return model.predict(X)


def main() -> None:
    print("\nPremier League Match Outcome Prediction\n" + "=" * 55)

    results_dir = os.path.join(os.path.dirname(__file__), "results")
    os.makedirs(results_dir, exist_ok=True)

    # -----------------------------------------------------------------
    # 1) Load train/test (test season = 2024/25)
    # -----------------------------------------------------------------
    print("\n1) Loading data...")
    X_train, X_test, y_train, y_test, train_df, test_df = load_train_test_with_metadata()

    # Baseline: always predict home win (2)
    y_pred_base = np.full_like(y_test, 2)

    # -----------------------------------------------------------------
    # 2) Train models
    # -----------------------------------------------------------------
    print("\n2) Training models...")
    models = {
        "Baseline (Home Win)": None,
        "Random Forest": train_random_forest(X_train, y_train, random_state=42, draw_class_weight=1.0, draw_pred_boost=1.0),
        "KNN": train_knn(X_train, y_train, n_neighbors=25),
        "Logistic Regression": train_logistic_regression(X_train, y_train, random_state=42, draw_class_weight=1.0, draw_pred_boost=1.0),
        "Gradient Boosting": train_gradient_boosting(X_train, y_train, random_state=42),
    }

    # -----------------------------------------------------------------
    # 3) Evaluate on 2024/25 test season
    # -----------------------------------------------------------------
    print("\n3) Evaluating on 2024/25 test season...")
    rows = []

    # baseline
    base_acc = simple_accuracy(y_test, y_pred_base)
    k = int((y_pred_base == y_test).sum())
    lo, hi = accuracy_ci_wilson(k, len(y_test))
    rows.append(
        {"model": "Baseline (Home Win)", "accuracy": base_acc, "ci_low": lo, "ci_high": hi, "p_value": np.nan}
    )

    best_name = "Baseline (Home Win)"
    best_model = None
    best_acc = base_acc
    best_pred_test = y_pred_base

    for name, model in models.items():
        if model is None:
            continue

        y_pred = predict_codes(model, X_test)
        acc = simple_accuracy(y_test, y_pred)
        k = int((y_pred == y_test).sum())
        lo, hi = accuracy_ci_wilson(k, len(y_test))
        pv = mcnemar_pvalue_vs_baseline(y_test, y_pred, y_pred_base)

        rows.append({"model": name, "accuracy": acc, "ci_low": lo, "ci_high": hi, "p_value": pv})

        if acc > best_acc:
            best_acc = acc
            best_name = name
            best_model = model
            best_pred_test = y_pred

    summary_df = pd.DataFrame(rows).sort_values("accuracy", ascending=False).reset_index(drop=True)
    summary_csv = os.path.join(results_dir, "comparison_2024_2025.csv")
    summary_df.to_csv(summary_csv, index=False)

    fig_path = os.path.join(results_dir, "model_accuracy_summary_2024_2025.png")
    plot_accuracy_summary(
        summary_df.sort_values("model"),
        save_path=fig_path,
        title="Model comparison on 2024/25 (accuracy + 95% CI, p-value vs baseline)",
    )

    print(f"\nSaved model comparison CSV: {summary_csv}")
    print(f"Saved accuracy figure:      {fig_path}")
    print(f"\nBest model on 2024/25: {best_name} (accuracy={best_acc:.3f})")

    # logistic regression coefficients plot (interpretability)
    if "Logistic Regression" in models and models["Logistic Regression"] is not None:
        coef_path = os.path.join(results_dir, "logistic_regression_coefficients.png")
        plot_logistic_regression_coefficients(models["Logistic Regression"], FEATURE_COLUMNS, coef_path)
        print(f"Saved logistic regression coefficient plot: {coef_path}")

    # -----------------------------------------------------------------
    # 4) League table for 2024/25: actual vs predicted (best model)
    # -----------------------------------------------------------------
    print("\n4) Building 2024/25 league tables (actual vs predicted)...")
    test_df = test_df.copy()
    test_df["pred_code"] = best_pred_test

    actual_2425 = build_league_table(test_df, "result_code")
    pred_2425 = build_league_table(test_df, "pred_code")

    actual_csv = os.path.join(results_dir, "table_actual_2024_2025.csv")
    pred_csv = os.path.join(results_dir, "table_predicted_2024_2025.csv")
    actual_2425.to_csv(actual_csv)
    pred_2425.to_csv(pred_csv)

    print(f"Saved: {actual_csv}")
    print(f"Saved: {pred_csv}")
    print("\nTop 8 (predicted 2024/25):")
    print(pred_2425.head(8))

    # -----------------------------------------------------------------
    # 5) Predict 2025/26 first 14 matchweeks
    # -----------------------------------------------------------------
    print("\n5) Predicting 2025/26 first 14 matchweeks...")
    future_df, X_future = load_future_season_features()
    if best_model is None:
        # if baseline was best (unlikely), just use it
        future_pred = np.full(shape=(len(future_df),), fill_value=2, dtype=int)
    else:
        future_pred = predict_codes(best_model, X_future)

    future_df = future_df.copy()
    future_df["pred_code"] = future_pred

    # comparison file (fixtures)
    comp_cols = ["Date", "round", "home", "away", "result", "result_code", "pred_code"]
    comp = future_df[comp_cols].copy()
    comp_csv = os.path.join(results_dir, "comparison_2025_2026_first14.csv")
    comp.to_csv(comp_csv, index=False)

    # league tables
    actual_future = build_league_table(future_df, "result_code")   # skips NaN safely
    pred_future = build_league_table(future_df, "pred_code")       # always available

    actual_future_csv = os.path.join(results_dir, "table_actual_2025_2026_first14.csv")
    pred_future_csv = os.path.join(results_dir, "table_predicted_2025_2026_first14.csv")
    actual_future.to_csv(actual_future_csv)
    pred_future.to_csv(pred_future_csv)

    print(f"Saved: {comp_csv}")
    print(f"Saved: {actual_future_csv}")
    print(f"Saved: {pred_future_csv}")

    print("\nTop 8 (predicted 2025/26 after 14 matchweeks):")
    print(pred_future.head(8))
    print("\nTop 8 (actual 2025/26 after 14 matchweeks - only matches with results present):")
    print(actual_future.head(8))


if __name__ == "__main__":
    main()
