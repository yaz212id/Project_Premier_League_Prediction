"""
Data loading and feature creation for Premier League project.
"""

import os
from typing import Tuple, List

import numpy as np
import pandas as pd

# Folder where the raw CSVs live:
# data/raw/PL_2018_2019_data.csv, ..., PL_2024_2025_data.csv
DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "raw")

# Filenames we expect to find in DATA_DIR
SEASONS = [
    "PL_2018_2019_data.csv",
    "PL_2019_2020_data.csv",
    "PL_2020_2021_data.csv",
    "PL_2021_2022_data.csv",
    "PL_2022_2023_data.csv",
    "PL_2023_2024_data.csv",
    "PL_2024_2025_data.csv",
]


def _load_season(path: str) -> pd.DataFrame:
    """
    Load one season CSV and keep only the useful columns.

    Parameters
    ----------
    path : str
        Full path to a Premier League CSV file.

    Returns
    -------
    df : pd.DataFrame
        Cleaned dataframe with columns:
        Date, home, away, home_goals, away_goals, result
    """
    # Skip the first text row (it’s just a description in these files)
    df = pd.read_csv(path, skiprows=1)

    # Rename the original columns to something simpler
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
    Load and concatenate all seasons into one big dataframe.
    """
    dfs: List[pd.DataFrame] = []
    for fname in SEASONS:
        full_path = os.path.join(DATA_DIR, fname)
        if not os.path.exists(full_path):
            raise FileNotFoundError(f"Missing CSV file: {full_path}")
        dfs.append(_load_season(full_path))

    df_all = pd.concat(dfs, ignore_index=True)
    return df_all


def _add_basic_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add label (result_code) and 'form' features for home and away teams.

    - result_code: 0 = away win, 1 = draw, 2 = home win
    - home_form / away_form: average points in last 5 games
    - home_gd_form / away_gd_form: average goal difference in last 5 games
    """
    # --- 1) Date & label ------------------------------------------------
    df["Date"] = pd.to_datetime(df["Date"], dayfirst=True, errors="coerce")
    df = df.sort_values("Date").reset_index(drop=True)

    result_map = {"A": 0, "D": 1, "H": 2}
    df["result_code"] = df["result"].map(result_map)

    # --- 2) Points per game from each team’s perspective -----------------
    def result_to_points(res: str) -> tuple[int, int]:
        """Return (home_points, away_points) for a given match result."""
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

    # Goal difference from each team’s perspective
    df["home_gd"] = df["home_goals"] - df["away_goals"]
    df["away_gd"] = -df["home_gd"]

    # --- 3) Long format to compute rolling form -------------------------
    # One row per TEAM per MATCH (home & away together)
    home_long = df[["Date", "home", "home_points", "home_gd"]].rename(
        columns={"home": "team", "home_points": "points", "home_gd": "gd"}
    )
    away_long = df[["Date", "away", "away_points", "away_gd"]].rename(
        columns={"away": "team", "away_points": "points", "away_gd": "gd"}
    )

    long_df = pd.concat([home_long, away_long], ignore_index=True)
    long_df = long_df.sort_values(["team", "Date"])

    # Rolling average over the last 5 matches.
    # shift(1) ensures we only use PAST games for the current match.
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

    # First matches have no history -> NaN; replace with 0
    long_df[["form_pts", "form_gd"]] = long_df[["form_pts", "form_gd"]].fillna(0.0)

    # --- 4) Merge back to original df to get home/away features ---------
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

    # Safety: fill remaining NaNs with 0
    df[["home_form", "away_form", "home_gd_form", "away_gd_form"]] = df[
        ["home_form", "away_form", "home_gd_form", "away_gd_form"]
    ].fillna(0.0)

    return df


def load_and_split(
    test_start: str = "2024-08-01",
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Load Premier League data, build features, and split into train/test.

    Parameters
    ----------
    test_start : str
        All matches on or after this date go into the test set.

    Returns
    -------
    X_train, X_test, y_train, y_test : np.ndarray
    """
    df = _load_all_seasons()
    df = _add_basic_features(df)

    cutoff_date = pd.Timestamp(test_start)
    train_df = df[df["Date"] < cutoff_date].copy()
    test_df = df[df["Date"] >= cutoff_date].copy()

    # Our features: recent form and recent goal-diff form
    feature_cols = ["home_form", "away_form", "home_gd_form", "away_gd_form"]

    X_train = train_df[feature_cols].values
    y_train = train_df["result_code"].values

    X_test = test_df[feature_cols].values
    y_test = test_df["result_code"].values

    return X_train, X_test, y_train, y_test
