"""
Main script for Premier League project.

Ce script sera exécuté par les correcteurs :

    python main.py

Pipeline :
- charge les données historiques,
- construit les features,
- entraîne plusieurs modèles (baseline, RF, KNN, LogReg),
- évalue les modèles sur la saison 2024/2025 (test),
- choisit le meilleur modèle,
- prédit les 14 premiers matches de 2025/2026,
- construit le classement réel vs ML pour cette saison partielle,
- sauvegarde le tableau dans results/.
"""

import os

import numpy as np
from sklearn.metrics import accuracy_score, confusion_matrix

from src.data_loader import (
    load_and_split,
    load_future_season_for_prediction,
)
from src.models import (
    train_random_forest,
    train_knn,
    train_logistic_regression,
)
from src.evaluation import evaluate_model, compute_points_table


def main() -> None:
    """Load data, train several models, and compare their performance."""
    print("=" * 70)
    print("Premier League Match Outcome Prediction")
    print("=" * 70)

    # ------------------------------------------------------------------
    # 1. Chargement et pré-processing des données historiques
    # ------------------------------------------------------------------
    print("\n1. Loading and preprocessing historical data...")
    # X_train, X_test : features train/test (forme, goal diff, etc.)
    # y_train, y_test : labels (0/1/2) pour chaque match
    X_train, X_test, y_train, y_test = load_and_split()
    print(f"   Train size: {X_train.shape}")
    print(f"   Test size:  {X_test.shape}")

    # ------------------------------------------------------------------
    # 2. Baseline très naïve : toujours prédire "home win"
    # ------------------------------------------------------------------
    print("\n2. Baseline model (always predict 'home win')...")
    # baseline_pred : array de la même taille que y_test rempli avec 2
    # (2 = home win dans notre encodage)
    baseline_pred = np.full_like(y_test, fill_value=2)
    baseline_acc = accuracy_score(y_test, baseline_pred)
    print(f"Baseline accuracy: {baseline_acc:.3f}")
    print("Baseline confusion matrix:")
    print(confusion_matrix(y_test, baseline_pred))

    # ------------------------------------------------------------------
    # 3. Entraînement des modèles ML
    # ------------------------------------------------------------------
    print("\n3. Training ML models...")
    # rf_model : RandomForest entraîné sur X_train / y_train
    rf_model = train_random_forest(X_train, y_train)
    # knn_model : KNN entraîné
    knn_model = train_knn(X_train, y_train)
    # lr_model : Logistic Regression entraînée
    lr_model = train_logistic_regression(X_train, y_train)
    print("   ✓ All models trained.")

    # Dictionnaire pour garder les objets modèles (utile pour récupérer le winner)
    trained_models = {
        "Random Forest": rf_model,
        "KNN": knn_model,
        "Logistic Regression": lr_model,
    }

    # ------------------------------------------------------------------
    # 4. Évaluation sur le test set (saison 2024/2025)
    # ------------------------------------------------------------------
    print("\n4. Evaluating ML models on 2024/2025 test season...")
    # rf_acc, knn_acc, lr_acc : accuracies respectives
    rf_acc = evaluate_model(rf_model, X_test, y_test, "Random Forest")
    knn_acc = evaluate_model(knn_model, X_test, y_test, "KNN")
    lr_acc = evaluate_model(lr_model, X_test, y_test, "Logistic Regression")

    # ------------------------------------------------------------------
    # 5. Récap et meilleur modèle
    # ------------------------------------------------------------------
    results = {
        "Baseline (Home Win)": baseline_acc,
        "Random Forest": rf_acc,
        "KNN": knn_acc,
        "Logistic Regression": lr_acc,
    }
    # winner_name : nom du modèle avec la meilleure accuracy (hors baseline si tu veux)
    winner_name = max(results, key=results.get)
    # best_model : instance du modèle gagnant (RandomForest / KNN / LogReg)
    best_model = trained_models.get(winner_name, rf_model)  # fallback RF

    print("\n" + "=" * 70)
    print("Summary of test accuracies:")
    for name, acc in results.items():
        print(f" - {name}: {acc:.3f}")
    print("-" * 70)
    print(f"Best model on test season: {winner_name} ({results[winner_name]:.3f} accuracy)")
    print("=" * 70)

    # ------------------------------------------------------------------
    # 6. Prédiction des 14 premiers matches de 2025/2026
    # ------------------------------------------------------------------
    print("\n6. Predicting first 14 matchdays of 2025/2026...")

    # X_future : features pour les matches 2025/2026
    # y_future : résultats réels (0/1/2) de ces matches
    # meta_future : DataFrame (Date, home, away)
    X_future, y_future, meta_future = load_future_season_for_prediction()

    # y_future_pred : prédictions du best_model sur ces matches
    y_future_pred = best_model.predict(X_future)

    # future_acc : accuracy du modèle sur ces 14 matches
    future_acc = accuracy_score(y_future, y_future_pred)
    print(f"\nAccuracy of {winner_name} on 2025/2026 partial season: {future_acc:.3f}")

    # ------------------------------------------------------------------
    # 7. Construction des classements (réel vs ML) sur 2025/2026 (14 journées)
    # ------------------------------------------------------------------
    from src.evaluation import compute_points_table  # déjà importé en haut, juste pour lisibilité

    # table_actual : classement réel (points_actual)
    table_actual = compute_points_table(
        matches=meta_future,
        result_codes=y_future,
        points_column="points_actual",
    )

    # table_pred : classement prédit par le modèle (points_pred)
    table_pred = compute_points_table(
        matches=meta_future,
        result_codes=y_future_pred,
        points_column="points_pred",
    )

    # On merge les deux tableaux pour comparer par équipe
    league_comparison = table_actual.merge(
        table_pred, on="team", how="outer"
    ).fillna(0)

    # On ajoute une colonne "delta" = points_pred - points_actual
    league_comparison["delta_points"] = (
        league_comparison["points_pred"] - league_comparison["points_actual"]
    )

    # Tri par points réels décroissants (classement officiel)
    league_comparison = league_comparison.sort_values(
        "points_actual", ascending=False
    ).reset_index(drop=True)

    print("\nLeague table comparison on 2025/2026 partial season (first 14 matchdays):")
    print(league_comparison)

    # ------------------------------------------------------------------
    # 8. Sauvegarde dans le dossier results/
    # ------------------------------------------------------------------
    results_dir = os.path.join(os.path.dirname(__file__), "results")
    os.makedirs(results_dir, exist_ok=True)

    output_path = os.path.join(results_dir, "league_table_2025_2026_partial.csv")
    league_comparison.to_csv(output_path, index=False)

    print(f"\nSaved league comparison table to: {output_path}")


if __name__ == "__main__":
    # Cette ligne garantit que main() est exécuté quand on fait `python main.py`
    main()

