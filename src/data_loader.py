"""
Data loading and feature creation for Premier League project.

- Charge les saisons 2018–2025 pour l'entraînement / test.
- Prépare les features (forme récente, goal diff) pour le modèle.
- Prépare aussi une saison future (2025/2026, 14 premières journées)
  pour faire des prédictions et comparer au classement réel.
"""

import os
from typing import Tuple, List

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------
# 0. Constantes globales
# ---------------------------------------------------------------------

# Colonnes de features utilisées par tous les modèles.
# Ici :
#   home_form      : moyenne des points des 5 derniers matches à domicile
#   away_form      : moyenne des points des 5 derniers matches à l'extérieur
#   home_gd_form   : moyenne des goal differences des 5 derniers matches à domicile
#   away_gd_form   : idem à l'extérieur
FEATURE_COLUMNS: List[str] = [
    "home_form",
    "away_form",
    "home_gd_form",
    "away_gd_form",
]

# Dossier où se trouvent les CSV bruts
# -> data/raw/PL_2018_2019_data.csv, etc.
DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "raw")

# Fichiers des saisons utilisées pour TRAIN + TEST
SEASONS = [
    "PL_2018_2019_data.csv",
    "PL_2019_2020_data.csv",
    "PL_2020_2021_data.csv",
    "PL_2021_2022_data.csv",
    "PL_2022_2023_data.csv",
    "PL_2023_2024_data.csv",
    "PL_2024_2025_data.csv",
]

# Fichier de la saison FUTURE 2025/2026 (14 premières journées)
FUTURE_SEASON_FILE = "epl-2025-GMTStandardTime.csv"


# ---------------------------------------------------------------------
# 1. Fonctions internes pour charger et nettoyer les CSV
# ---------------------------------------------------------------------
def _load_season(path: str) -> pd.DataFrame:
    """
    Charge une saison et garde uniquement les colonnes utiles.

    Parameters
    ----------
    path : str
        Chemin complet vers un fichier CSV de Premier League.

    Returns
    -------
    df : pd.DataFrame
        Dataframe nettoyé avec colonnes :
        Date, home, away, home_goals, away_goals, result
    """
    # Football-data a une première ligne de description -> on la saute
    df = pd.read_csv(path, skiprows=1)

    # Map des noms bruts vers des noms plus simples
    columns_map = {
        "HomeTeam": "home",
        "AwayTeam": "away",
        "FTHG": "home_goals",
        "FTAG": "away_goals",
        "FTR": "result",  # H / D / A
    }
    df = df.rename(columns=columns_map)

    keep_cols = ["Date", "home", "away", "home_goals", "away_goals", "result"]
    df = df[keep_cols].copy()
    return df


def _load_all_seasons() -> pd.DataFrame:
    """
    Charge et concatène toutes les saisons d'entraînement / test.
    """
    dfs: List[pd.DataFrame] = []

    for fname in SEASONS:
        full_path = os.path.join(DATA_DIR, fname)
        if not os.path.exists(full_path):
            raise FileNotFoundError(f"Missing CSV file: {full_path}")
        dfs.append(_load_season(full_path))

    df_all = pd.concat(dfs, ignore_index=True)
    return df_all


# ---------------------------------------------------------------------
# 2. Création des features (forme, goal diff, etc.)
# ---------------------------------------------------------------------
def _add_basic_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Ajoute le label (result_code) et les features de "forme" pour chaque match.

    - result_code : 0 = away win, 1 = draw, 2 = home win
    - home_form / away_form : moyenne des points sur les 5 derniers matches
    - home_gd_form / away_gd_form : moyenne du goal diff sur 5 matches
    """
    # --------------------------------------------------------------
    # a) Date + label numérique
    # --------------------------------------------------------------
    df["Date"] = pd.to_datetime(df["Date"], dayfirst=True, errors="coerce")
    df = df.sort_values("Date").reset_index(drop=True)

    result_map = {"A": 0, "D": 1, "H": 2}
    df["result_code"] = df["result"].map(result_map)

    # --------------------------------------------------------------
    # b) Points par match pour chaque équipe
    # --------------------------------------------------------------
    def result_to_points(res: str) -> tuple[int, int]:
        """Retourne (points_home, points_away) pour un résultat donné."""
        if res == "H":
            return 3, 0
        if res == "A":
            return 0, 3
        if res == "D":
            return 1, 1
        return 0, 0

    pts = df["result"].apply(result_to_points)
    df["home_points"] = pts.apply(lambda x: x[0])
    df["away_points"] = pts.apply(lambda x: x[1])

    # Goal difference pour chaque équipe
    df["home_gd"] = df["home_goals"] - df["away_goals"]
    df["away_gd"] = -df["home_gd"]

    # --------------------------------------------------------------
    # c) On passe en "long format" pour calculer la forme par équipe
    # --------------------------------------------------------------
    home_long = df[["Date", "home", "home_points", "home_gd"]].rename(
        columns={"home": "team", "home_points": "points", "home_gd": "gd"}
    )
    away_long = df[["Date", "away", "away_points", "away_gd"]].rename(
        columns={"away": "team", "away_points": "points", "away_gd": "gd"}
    )

    long_df = pd.concat([home_long, away_long], ignore_index=True)
    long_df = long_df.sort_values(["team", "Date"])

    window = 5  # nombre de matches pour la moyenne mobile

    # Moyenne mobile des points : on shift(1) pour n'utiliser que le passé
    long_df["form_pts"] = (
        long_df.groupby("team")["points"]
        .rolling(window=window, min_periods=1)
        .mean()
        .shift(1)
        .reset_index(level=0, drop=True)
    )
    # Moyenne mobile du goal diff
    long_df["form_gd"] = (
        long_df.groupby("team")["gd"]
        .rolling(window=window, min_periods=1)
        .mean()
        .shift(1)
        .reset_index(level=0, drop=True)
    )

    # Les premiers matches n'ont pas d'historique -> NaN, on met 0
    long_df[["form_pts", "form_gd"]] = long_df[["form_pts", "form_gd"]].fillna(0.0)

    # --------------------------------------------------------------
    # d) Merge des features de forme sur le dataframe original
    # --------------------------------------------------------------
    home_form = long_df[["Date", "team", "form_pts", "form_gd"]].rename(
        columns={
            "team": "home",
            "form_pts": "home_form",
            "form_gd": "home_gd_form",
        }
    )
    away_form = long_df[["Date", "team", "form_pts", "form_gd"]].rename(
        columns={
            "team": "away",
            "form_pts": "away_form",
            "form_gd": "away_gd_form",
        }
    )

    df = df.merge(home_form, on=["Date", "home"], how="left")
    df = df.merge(away_form, on=["Date", "away"], how="left")

    df[["home_form", "away_form", "home_gd_form", "away_gd_form"]] = df[
        ["home_form", "away_form", "home_gd_form", "away_gd_form"]
    ].fillna(0.0)

    return df


# ---------------------------------------------------------------------
# 3. Fonction principale : train / test split
# ---------------------------------------------------------------------
def load_and_split(
    test_start: str = "2024-08-01",
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Charge les saisons historiques, construit les features et fait le split.

    Parameters
    ----------
    test_start : str
        Toutes les matches à partir de cette date vont dans le test set.

    Returns
    -------
    X_train, X_test, y_train, y_test : np.ndarray
    """
    df = _load_all_seasons()
    df = _add_basic_features(df)

    cutoff_date = pd.Timestamp(test_start)
    train_df = df[df["Date"] < cutoff_date].copy()
    test_df = df[df["Date"] >= cutoff_date].copy()

    # On utilise les features définies en haut du fichier
    X_train = train_df[FEATURE_COLUMNS].values
    y_train = train_df["result_code"].values

    X_test = test_df[FEATURE_COLUMNS].values
    y_test = test_df["result_code"].values

    return X_train, X_test, y_train, y_test


# ---------------------------------------------------------------------
# 4. Chargement de la saison FUTURE 2025/2026 pour la prédiction
# ---------------------------------------------------------------------
def load_future_season_for_prediction() -> Tuple[np.ndarray, np.ndarray, pd.DataFrame]:
    """
    Prépare les matches de la saison 2025/2026 (14 premières journées).

    Returns
    -------
    X_future : np.ndarray
        Features pour les matches futurs.
    y_future : np.ndarray
        Labels réels (0/1/2) pour ces matches (utile pour comparer).
    meta_future : pd.DataFrame
        Infos sur les matches (Date, home, away) pour construire le classement.
    """
    future_path = os.path.join(DATA_DIR, FUTURE_SEASON_FILE)
    if not os.path.exists(future_path):
        raise FileNotFoundError(f"Missing future season CSV: {future_path}")

    df_future = _load_season(future_path)
    df_future = _add_basic_features(df_future)

    X_future = df_future[FEATURE_COLUMNS].values
    y_future = df_future["result_code"].values

    # On garde Date, home, away pour reconstruire les classements
    meta_future = df_future[["Date", "home", "away"]].copy()

    return X_future, y_future, meta_future
