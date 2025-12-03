"""Evaluation helpers for Premier League models."""

from typing import Dict

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix


def evaluate_model(
    model,
    X_test: np.ndarray,
    y_test: np.ndarray,
    model_name: str,
    print_report: bool = True,
) -> float:
    """
    Évalue un modèle sur le test set et affiche quelques métriques.

    Parameters
    ----------
    model : objet sklearn
        Modèle déjà entraîné (RandomForest, KNN, LogisticRegression, etc.).
    X_test : np.ndarray
        Features du test set.
    y_test : np.ndarray
        Labels réels du test set.
    model_name : str
        Nom du modèle (pour l'affichage).
    print_report : bool
        Si True, affiche accuracy, classification_report, confusion_matrix.

    Returns
    -------
    accuracy : float
        Accuracy globale du modèle sur le test set.
    """
    # y_pred : prédictions de classe 0/1/2
    y_pred = model.predict(X_test)

    # acc : accuracy globale
    acc = accuracy_score(y_test, y_pred)

    if print_report:
        print(f"\n{model_name} Results")
        print("-" * (len(model_name) + 8))
        print(f"Accuracy: {acc:.3f}")
        print("\nClassification report:")
        print(classification_report(y_test, y_pred, zero_division=0))
        print("Confusion matrix:")
        print(confusion_matrix(y_test, y_pred))

    return acc


def compute_points_table(
    matches: pd.DataFrame,
    result_codes: np.ndarray,
    points_column: str,
) -> pd.DataFrame:
    """
    Construit un classement (table de points) à partir de prédictions ou de résultats.

    Parameters
    ----------
    matches : pd.DataFrame
        DataFrame avec au moins les colonnes 'home' et 'away'.
        Une ligne = un match (home vs away).
    result_codes : np.ndarray
        Résultats 0/1/2 pour chaque match :
            0 = away win
            1 = draw
            2 = home win
    points_column : str
        Nom de la colonne de points dans le tableau retourné
        (ex: 'points_actual', 'points_pred').

    Returns
    -------
    table : pd.DataFrame
        Colonnes : ['team', points_column], classées par points décroissants.
    """
    # table_points : dict[str, int] -> points cumulés pour chaque équipe
    table_points: Dict[str, int] = {}

    # On parcourt chaque match avec son résultat
    for (_, row), res in zip(matches.iterrows(), result_codes):
        home_team = row["home"]
        away_team = row["away"]

        # Initialisation à 0 si l'équipe n'est pas encore dans le dict
        table_points.setdefault(home_team, 0)
        table_points.setdefault(away_team, 0)

        # Attribution des points selon le code 0/1/2
        if res == 2:  # home win
            table_points[home_team] += 3
        elif res == 1:  # draw
            table_points[home_team] += 1
            table_points[away_team] += 1
        elif res == 0:  # away win
            table_points[away_team] += 3

    # On convertit le dict en DataFrame
    table = pd.DataFrame(
        {
            "team": list(table_points.keys()),
            points_column: list(table_points.values()),
        }
    )

    # Tri par points décroissants
    table = table.sort_values(points_column, ascending=False).reset_index(drop=True)
    return table
