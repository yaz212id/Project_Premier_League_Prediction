"""
Data loading and feature creation for Premier League project.
"""

import os
from typing import Tuple, List

import numpy as np
import pandas as pd

# ------------------------------------------------------------
# 1. Colonnes de features utilisées par tous les modèles
# ------------------------------------------------------------
# X[...] sera un tableau numpy avec exactement ces colonnes :
FEATURE_COLUMNS: List[str] = [
    "home_form",      # forme (points moyens) de l'équipe à domicile
    "away_form",      # forme (points moyens) de l'équipe à l'extérieur
    "home_gd_form",   # forme en différence de buts à domicile
    "away_gd_form",   # forme en différence de buts à l'extérieur
]

# Dossier où se trouvent les CSV bruts
# => Project_Premier_League_Prediction/data/raw/...
DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "raw")

# Saisons historiques utilisées pour l'entraînement / test
SEASONS = [
    "PL_2018_2019_data.csv",
    "PL_2019_2020_data.csv",
    "PL_2020_2021_data.csv",
    "PL_2021_2022_data.csv",
    "PL_2022_2023_data.csv",
    "PL_2023_2024_data.csv",
    "PL_2024_2025_data.csv",   # saison de test (380 matches)
]

# Fichier qui contient la saison 2025/2026 complète (1–38),
# mais on ne gardera que les 14 premières journées
FUTURE_FILENAME = "epl-2025-GMTStandardTime.csv"


# ------------------------------------------------------------
# 2. Chargement d'une saison historique football-data
# ------------------------------------------------------------
def _load_season(path: str) -> pd.DataFrame:
    """
    Charge un CSV football-data.co.uk (saisons 2018–2025) et renvoie
    un DataFrame propre avec les colonnes :
        Date, home, away, home_goals, away_goals, result (H/D/A)
    """
    # football-data a une ligne de description en première ligne
    df = pd.read_csv(path, skiprows=1)

    columns_map = {
        "HomeTeam": "home",
        "AwayTeam": "away",
        "FTHG": "home_goals",
        "FTAG": "away_goals",
        "FTR": "result",   # H / D / A
        "Date": "Date",
    }

    # On vérifie que les colonnes attendues existent
    missing = [c for c in columns_map.keys() if c not in df.columns]
    if missing:
        raise ValueError(
            f"Le fichier {path} ne contient pas les colonnes attendues.\n"
            f"Colonnes manquantes : {missing}\n"
            f"Colonnes trouvées : {list(df.columns)}"
        )

    df = df.rename(columns=columns_map)

    keep_cols = ["Date", "home", "away", "home_goals", "away_goals", "result"]
    df = df[keep_cols].copy()
    return df


def _load_all_seasons() -> pd.DataFrame:
    """
    Charge et concatène toutes les saisons historiques dans un seul DataFrame.
    """
    dfs: List[pd.DataFrame] = []
    for fname in SEASONS:
        full_path = os.path.join(DATA_DIR, fname)
        if not os.path.exists(full_path):
            raise FileNotFoundError(f"Missing CSV file: {full_path}")
        dfs.append(_load_season(full_path))

    df_all = pd.concat(dfs, ignore_index=True)
    return df_all


# ------------------------------------------------------------
# 3. Construction des features (forme, label, etc.)
# ------------------------------------------------------------
def _add_basic_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Ajoute :
      - result_code (0 = away win, 1 = draw, 2 = home win)
      - home_form / away_form : points moyens sur les 5 derniers matches
      - home_gd_form / away_gd_form : diff. de buts moyenne sur 5 matches
    """
    # Date en datetime + tri
    df["Date"] = pd.to_datetime(df["Date"], dayfirst=True, errors="coerce")
    df = df.sort_values("Date").reset_index(drop=True)

    # Encodage du résultat en entier
    result_map = {"A": 0, "D": 1, "H": 2}
    df["result_code"] = df["result"].map(result_map)

    # Points par match pour chaque équipe
    def result_to_points(res: str) -> tuple[int, int]:
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

    # Différence de buts
    df["home_gd"] = df["home_goals"] - df["away_goals"]
    df["away_gd"] = -df["home_gd"]

    # Passage en format "long" (une ligne par équipe et par match)
    home_long = df[["Date", "home", "home_points", "home_gd"]].rename(
        columns={"home": "team", "home_points": "points", "home_gd": "gd"}
    )
    away_long = df[["Date", "away", "away_points", "away_gd"]].rename(
        columns={"away": "team", "away_points": "points", "away_gd": "gd"}
    )

    long_df = pd.concat([home_long, away_long], ignore_index=True)
    long_df = long_df.sort_values(["team", "Date"])

    # Rolling moyenne sur les 5 derniers matches (shift(1) = seulement le passé)
    window = 5
    long_df["form_pts"] = (
        long_df.groupby("team")["points"]
        .rolling(window=window, min_periods=1)
        .mean()
        .shift(1)
        .reset_index(level=0, drop=True)
    )
    long_df["form_gd"] = (
        long_df.groupby("team")["gd"]
        .rolling(window=window, min_periods=1)
        .mean()
        .shift(1)
        .reset_index(level=0, drop=True)
    )

    # Pas d’historique au tout début → NaN → 0
    long_df[["form_pts", "form_gd"]] = long_df[["form_pts", "form_gd"]].fillna(0.0)

    # Retour en format "large" avec colonnes home_* et away_*
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


# ------------------------------------------------------------
# 4. Split train / test pour les modèles
# ------------------------------------------------------------
def load_and_split(
    test_start: str = "2024-08-01",
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Version simple : ne renvoie que les tableaux numpy (comme au début du projet).
    """
    df = _load_all_seasons()
    df = _add_basic_features(df)

    cutoff_date = pd.Timestamp(test_start)
    train_df = df[df["Date"] < cutoff_date].copy()
    test_df = df[df["Date"] >= cutoff_date].copy()

    X_train = train_df[FEATURE_COLUMNS].values
    y_train = train_df["result_code"].values
    X_test = test_df[FEATURE_COLUMNS].values
    y_test = test_df["result_code"].values

    return X_train, X_test, y_train, y_test


def load_train_test_with_metadata(
    test_start: str = "2024-08-01",
):
    """
    Version enrichie : renvoie aussi les DataFrames train_df / test_df
    (avec les noms d'équipes, les dates, etc.) pour construire les classements.
    """
    df = _load_all_seasons()
    df = _add_basic_features(df)

    cutoff_date = pd.Timestamp(test_start)
    train_df = df[df["Date"] < cutoff_date].copy()
    test_df = df[df["Date"] >= cutoff_date].copy()

    X_train = train_df[FEATURE_COLUMNS].values
    y_train = train_df["result_code"].values
    X_test = test_df[FEATURE_COLUMNS].values
    y_test = test_df["result_code"].values

    return X_train, X_test, y_train, y_test, train_df, test_df


# ------------------------------------------------------------
# 5. Chargement de la saison future (14 premières journées 2025/26)
# ------------------------------------------------------------
def _load_future_matches(path: str) -> pd.DataFrame:
    """
    Charge le fichier epl-2025-GMTStandardTime.csv et renvoie
    uniquement les matches des 14 premières journées avec colonnes :
        Date, home, away, home_goals, away_goals, result, round
    """
    df = pd.read_csv(path)

    # On renomme les colonnes importantes
    df = df.rename(
        columns={
            "Home Team": "home",
            "Away Team": "away",
            "Round Number": "round",
        }
    )

    # On parse la date (format du style "11/08/2025 20:00")
    df["Date"] = pd.to_datetime(df["Date"], dayfirst=True, errors="coerce")

    # On ne garde que les 14 premières journées
    df = df[df["round"] <= 14].copy()

    # On parse le score "4 - 2" ou "4–2" -> home_goals, away_goals
    def parse_score(s):
        if isinstance(s, str):
            s_clean = s.replace(" ", "")
            for sep in ["-", "–", "—"]:
                if sep in s_clean:
                    parts = s_clean.split(sep)
                    if len(parts) == 2 and all(p.isdigit() for p in parts):
                        return int(parts[0]), int(parts[1])
        return np.nan, np.nan

    goals = df["Result"].apply(parse_score)
    df["home_goals"] = goals.apply(lambda x: x[0])
    df["away_goals"] = goals.apply(lambda x: x[1])

    # On construit H / D / A pour pouvoir calculer result_code etc.
    def goals_to_result(row):
        hg, ag = row["home_goals"], row["away_goals"]
        if pd.isna(hg) or pd.isna(ag):
            return None
        if hg > ag:
            return "H"
        elif hg < ag:
            return "A"
        else:
            return "D"

    df["result"] = df.apply(goals_to_result, axis=1)

    # On garde les colonnes utiles
    df = df[["Date", "home", "away", "home_goals", "away_goals", "result", "round"]]

    return df


def load_future_season_features(
    future_filename: str = FUTURE_FILENAME,
):
    """
    Construit les features pour les 14 premières journées de 2025/2026.

    Étapes :
      1. Charger toutes les saisons historiques (2018–2025)
      2. Charger le fichier epl-2025-GMTStandardTime.csv
         et garder uniquement Round <= 14
      3. Concaténer historique + ces 14 journées
      4. Recalculer les formes (_add_basic_features)
      5. Extraire uniquement les matches de 2025/26 (14 journées)
         et renvoyer :
             - future_df (avec result_code réel)
             - X_future (features pour le modèle)
    """
    hist_df = _load_all_seasons()

    future_path = os.path.join(DATA_DIR, future_filename)
    if not os.path.exists(future_path):
        raise FileNotFoundError(f"Future season file not found: {future_path}")

    # 14 premières journées 2025/26
    future_raw = _load_future_matches(future_path)

    # On concatène historique + future (14 journées)
    combined_raw = pd.concat([hist_df, future_raw], ignore_index=True)

    # On calcule les features et result_code pour tout
    combined_feat = _add_basic_features(combined_raw)

    # Tous les matches 2025/26 sont après les saisons historiques
    future_start = future_raw["Date"].min()
    future_end = future_raw["Date"].max()

    # On récupère uniquement les matches 2025/26 (14 journées)
    future_df = combined_feat[
        (combined_feat["Date"] >= future_start) & (combined_feat["Date"] <= future_end)
    ].copy()

    # Pour être sûr d'avoir le même ordre que le CSV futur
    future_df = future_df.sort_values(["Date", "home", "away"]).reset_index(drop=True)

    # Features pour le modèle
    X_future = future_df[FEATURE_COLUMNS].values

    return future_df, X_future
