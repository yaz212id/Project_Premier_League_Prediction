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

import matplotlib.pyplot as plt
from collections import defaultdict
import pandas as pd


def build_league_table(matches_df: pd.DataFrame, result_col: str) -> pd.DataFrame:
    """
    Construit un classement à partir d'un DataFrame de matches.

    Paramètres
    ----------
    matches_df : DataFrame avec au moins les colonnes :
        - 'home' (équipe domicile)
        - 'away' (équipe extérieur)
        - result_col (0 = away win, 1 = draw, 2 = home win)
    result_col : str
        Nom de la colonne contenant le résultat (ex: 'result_code' ou 'pred_code').

    Retour
    ------
    table : DataFrame indexé par équipe avec colonnes :
        - P  (matches joués)
        - W  (victoires)
        - D  (nuls)
        - L  (défaites)
        - Pts (points)
    """
    table = defaultdict(lambda: {"P": 0, "W": 0, "D": 0, "L": 0, "Pts": 0})

    for _, row in matches_df.iterrows():
        home = row["home"]
        away = row["away"]
        res = row[result_col]  # 0 / 1 / 2

        # Tous les matches comptent pour 1 joué
        table[home]["P"] += 1
        table[away]["P"] += 1

        if res == 2:  # home win
            table[home]["W"] += 1
            table[home]["Pts"] += 3
            table[away]["L"] += 1
        elif res == 0:  # away win
            table[away]["W"] += 1
            table[away]["Pts"] += 3
            table[home]["L"] += 1
        elif res == 1:  # draw
            table[home]["D"] += 1
            table[away]["D"] += 1
            table[home]["Pts"] += 1
            table[away]["Pts"] += 1

    table_df = pd.DataFrame.from_dict(table, orient="index")
    table_df.index.name = "Team"

    # Tri par points décroissants puis par victoires (juste pour stabiliser)
    table_df = table_df.sort_values(by=["Pts", "W"], ascending=False)

    # Ajout d'une colonne "Position"
    table_df.insert(0, "Pos", range(1, len(table_df) + 1))

    return table_df


def plot_model_accuracies(results: dict, save_path: str | None = None) -> None:
    """
    Dessine un barplot des scores de chaque modèle.

    results : dict {nom_du_modèle: accuracy}
    save_path : si non None, enregistre le graphique dans ce fichier.
    """
    names = list(results.keys())
    accuracies = list(results.values())

    plt.figure(figsize=(8, 4))
    plt.bar(names, accuracies)
    plt.ylabel("Accuracy")
    plt.ylim(0, 1)
    plt.title("Model accuracies on 2024/2025 test season")
    plt.xticks(rotation=20, ha="right")

    plt.tight_layout()

    if save_path is not None:
        plt.savefig(save_path, bbox_inches="tight")
    else:
        plt.show()
