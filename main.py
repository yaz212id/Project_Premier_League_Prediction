"""
Main script for Premier League project.

This is the file that the graders will run:

    python main.py
"""

import os
import numpy as np
from sklearn.metrics import accuracy_score, confusion_matrix

from src.data_loader import (
    load_train_test_with_metadata,
    load_future_season_features,
)
from src.models import (
    train_random_forest,
    train_knn,
    train_logistic_regression,
    train_gradient_boosting,
)
from src.evaluation import (
    evaluate_model,
    build_league_table,
    plot_model_accuracies,
)

# Dossier où l'on sauvegarde les résultats (images, CSV)
# -> Project_Premier_League_Prediction/results
RESULTS_DIR = os.path.join(os.path.dirname(__file__), "results")


def main() -> None:
    """Load data, train models, compare them and build league tables."""
    # Crée le dossier results/ s'il n'existe pas
    os.makedirs(RESULTS_DIR, exist_ok=True)

    print("=" * 70)
    print("Premier League Match Outcome Prediction")
    print("=" * 70)

    # --------------------------------------------------------------
    # 1. Load and preprocess data
    # --------------------------------------------------------------
    print("\n1. Loading and preprocessing data...")
    (
        X_train,
        X_test,
        y_train,
        y_test,
        train_df,
        test_df,
    ) = load_train_test_with_metadata()
    print(f"   Train size: {X_train.shape}")  # (n_train_matches, 4)
    print(f"   Test size:  {X_test.shape}")   # (380, 4) pour 2024/25

    # --------------------------------------------------------------
    # 2. Baseline model: always predict 'home win'
    # --------------------------------------------------------------
    print("\n2. Baseline model (always predict 'home win')...")
    # 2 = home win (voir result_map dans data_loader._add_basic_features)
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
    gb_model = train_gradient_boosting(X_train, y_train)
    print("   ✓ All models trained.")

    # Dictionnaire pour retrouver facilement un modèle par son nom
    trained_models = {
        "Random Forest": rf_model,
        "KNN": knn_model,
        "Logistic Regression": lr_model,
        "Gradient Boosting": gb_model,
    }

    # --------------------------------------------------------------
    # 4. Evaluate models on the test set
    # --------------------------------------------------------------
    print("\n4. Evaluating ML models on test set...")
    rf_acc = evaluate_model(rf_model, X_test, y_test, "Random Forest")
    knn_acc = evaluate_model(knn_model, X_test, y_test, "KNN")
    lr_acc = evaluate_model(lr_model, X_test, y_test, "Logistic Regression")
    gb_acc = evaluate_model(gb_model, X_test, y_test, "Gradient Boosting")

    # --------------------------------------------------------------
    # 5. Summary, best model, and accuracy plot
    # --------------------------------------------------------------
    results = {
        "Baseline (Home Win)": baseline_acc,
        "Random Forest": rf_acc,
        "KNN": knn_acc,
        "Logistic Regression": lr_acc,
        "Gradient Boosting": gb_acc,
    }

    # Meilleur modèle (en termes d'accuracy)
    winner_name = max(results, key=results.get)
    best_model = trained_models.get(winner_name, rf_model)  # fallback RF

    print("\n" + "=" * 70)
    print("Summary of test accuracies:")
    for name, acc in results.items():
        print(f" - {name}: {acc:.3f}")
    print("-" * 70)
    print(f"Best model: {winner_name} ({results[winner_name]:.3f} accuracy)")
    print("=" * 70)

    # Graphique des accuracies
    acc_plot_path = os.path.join(RESULTS_DIR, "model_accuracies_2024_2025.png")
    plot_model_accuracies(results, save_path=acc_plot_path)
    print(f"\nSaved accuracy plot to: {acc_plot_path}")

    # --------------------------------------------------------------
    # 6. League tables for 2024/2025 (test season)
    # --------------------------------------------------------------
    print("\n6. Building league tables for 2024/2025 test season...")

    # Table réelle (à partir de result_code)
    actual_table_2425 = build_league_table(test_df, result_col="result_code")

    # Table prédite par le meilleur modèle
    test_df = test_df.copy()
    test_df["pred_code"] = best_model.predict(X_test)
    predicted_table_2425 = build_league_table(test_df, result_col="pred_code")

    # Sauvegarde
    actual_csv_2425 = os.path.join(RESULTS_DIR, "table_actual_2024_2025.csv")
    pred_csv_2425 = os.path.join(RESULTS_DIR, "table_predicted_2024_2025.csv")
    actual_table_2425.to_csv(actual_csv_2425)
    predicted_table_2425.to_csv(pred_csv_2425)

    print(f"Saved actual 2024/2025 table to:   {actual_csv_2425}")
    print(f"Saved predicted 2024/2025 table to: {pred_csv_2425}")

    print("\nTop 8 – actual 2024/2025 table:")
    print(actual_table_2425.head(8))
    print("\nTop 8 – predicted 2024/2025 table (best model):")
    print(predicted_table_2425.head(8))

    # --------------------------------------------------------------
    # 7. League tables for first 14 games of 2025/2026
    # --------------------------------------------------------------
    print("\n7. Predicting first 14 games of 2025/2026...")

    future_df, X_future = load_future_season_features()
    # future_df contient déjà result_code (réel) grâce à _add_basic_features

    # Prédictions du meilleur modèle sur ces 14 matches
    future_df = future_df.copy()
    future_df["pred_code"] = best_model.predict(X_future)

    # Classements réel et prédit (sur ces 14 matches uniquement)
    actual_table_future = build_league_table(future_df, result_col="result_code")
    predicted_table_future = build_league_table(future_df, result_col="pred_code")

    actual_csv_future = os.path.join(
        RESULTS_DIR, "table_actual_2025_2026_first14.csv"
    )
    pred_csv_future = os.path.join(
        RESULTS_DIR, "table_predicted_2025_2026_first14.csv"
    )
    actual_table_future.to_csv(actual_csv_future)
    predicted_table_future.to_csv(pred_csv_future)

    print(f"\nSaved actual 2025/26 (first 14 games) table to:   {actual_csv_future}")
    print(f"Saved predicted 2025/26 (first 14 games) table to: {pred_csv_future}")

    print("\nActual table 2025/26 – first 14 games:")
    print(actual_table_future.head(8))
    print("\nPredicted table 2025/26 – first 14 games (best model):")
    print(predicted_table_future.head(8))


if __name__ == "__main__":
    # This line makes sure main() runs when we type `python main.py`
    main()
