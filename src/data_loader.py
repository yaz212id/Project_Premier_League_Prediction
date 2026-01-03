"""
Data loading and feature engineering for the Premier League prediction project.

- Loads historical seasons (football-data style CSVs)
- Loads the 2025/26 schedule/results file
- Builds simple "form" features:
    * rolling mean of points over last N played matches
    * rolling mean of goal difference over last N played matches

This implementation is robust to:
- header issues (accidentally reading a data row as header)
- delimiter issues (comma/semicolon/tab)
"""

from __future__ import annotations

import os
from collections import defaultdict, deque
from typing import List, Tuple

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------
# Features used by every model
# ---------------------------------------------------------------------
FEATURE_COLUMNS: List[str] = [
    "home_form",
    "away_form",
    "home_gd_form",
    "away_gd_form",
]

THIS_DIR = os.path.dirname(__file__)
DATA_DIR = os.path.abspath(os.path.join(THIS_DIR, "..", "data", "raw"))

SEASONS: List[str] = [
    "PL_2018_2019_data.csv",
    "PL_2019_2020_data.csv",
    "PL_2020_2021_data.csv",
    "PL_2021_2022_data.csv",
    "PL_2022_2023_data.csv",
    "PL_2023_2024_data.csv",
    "PL_2024_2025_data.csv",  # test season
]

FUTURE_FILENAME = "epl-2025-GMTStandardTime.csv"


# ---------------------------------------------------------------------
# CSV helpers (robust reading)
# ---------------------------------------------------------------------
def _read_csv_flexible(path: str) -> pd.DataFrame:
    """
    Try reading a CSV with common delimiters.
    If everything fails, raise a helpful error (file may be corrupted).
    """
    errors = []

    # 1) Normal CSV (comma) - fastest and usually correct
    try:
        return pd.read_csv(path, encoding="utf-8-sig")
    except Exception as e:
        errors.append(("comma/default", str(e)))

    # 2) Semicolon
    try:
        return pd.read_csv(path, sep=";", engine="python", encoding="utf-8-sig")
    except Exception as e:
        errors.append(("semicolon", str(e)))

    # 3) Tab
    try:
        return pd.read_csv(path, sep="\t", engine="python", encoding="utf-8-sig")
    except Exception as e:
        errors.append(("tab", str(e)))

    # 4) Auto-sniff (slow but can rescue weird files)
    try:
        return pd.read_csv(path, sep=None, engine="python", encoding="utf-8-sig")
    except Exception as e:
        errors.append(("auto-sniff", str(e)))

    msg = "\n".join([f"- {mode}: {err}" for mode, err in errors])
    raise ValueError(
        f"Could not parse CSV: {path}\n{msg}\n\n"
        "Most common cause: the file is not a real CSV (e.g., saved webpage) or it was re-saved incorrectly.\n"
        "Fix: re-download the raw CSV and put it back into data/raw/."
    )


def _pick_col(df: pd.DataFrame, candidates: List[str]) -> str:
    for c in candidates:
        if c in df.columns:
            return c
    raise ValueError(f"Could not find any of columns {candidates}. Found: {list(df.columns)}")


# ---------------------------------------------------------------------
# Historical seasons (football-data like)
# ---------------------------------------------------------------------
def _standardize_historical_season(path: str) -> pd.DataFrame:
    """
    Return a clean DataFrame with columns:
      Date, home, away, home_goals, away_goals, result  (H/D/A)
    """
    df = _read_csv_flexible(path)

    # If header got messed up, try skipping the first row (ONLY if needed)
    needed_any = {"HomeTeam", "AwayTeam", "FTHG", "FTAG", "FTR", "Date"}
    if len(set(df.columns).intersection(needed_any)) == 0:
        df2 = _read_csv_flexible(path)
        # try pandas skiprows only if first read did not include expected columns
        try:
            df2 = pd.read_csv(path, encoding="utf-8-sig", skiprows=1)
            df = df2
        except Exception:
            # keep original df; the error will be raised below by _pick_col
            pass

    date_col = _pick_col(df, ["Date", "date"])
    home_col = _pick_col(df, ["HomeTeam", "Home Team", "home", "Home"])
    away_col = _pick_col(df, ["AwayTeam", "Away Team", "away", "Away"])
    hg_col = _pick_col(df, ["FTHG", "HomeGoals", "HG"])
    ag_col = _pick_col(df, ["FTAG", "AwayGoals", "AG"])
    res_col = _pick_col(df, ["FTR", "Result", "res"])

    out = df[[date_col, home_col, away_col, hg_col, ag_col, res_col]].copy()
    out.columns = ["Date", "home", "away", "home_goals", "away_goals", "result"]

    # Normalize result to H/D/A (some files already use H/D/A)
    out["result"] = out["result"].astype(str).str.strip()
    out["Date"] = pd.to_datetime(out["Date"], dayfirst=True, errors="coerce")

    out = out.dropna(subset=["Date", "home", "away"]).reset_index(drop=True)
    return out


def _load_all_seasons() -> pd.DataFrame:
    dfs: List[pd.DataFrame] = []
    for fname in SEASONS:
        full_path = os.path.join(DATA_DIR, fname)
        if not os.path.exists(full_path):
            raise FileNotFoundError(f"Missing CSV file: {full_path}")
        dfs.append(_standardize_historical_season(full_path))
    return pd.concat(dfs, ignore_index=True)


# ---------------------------------------------------------------------
# Future season file (schedule/results)
# ---------------------------------------------------------------------
def _parse_score_to_goals(val) -> Tuple[float, float]:
    if not isinstance(val, str):
        return (np.nan, np.nan)

    s = val.strip().replace(" ", "")
    for sep in ["-", "–", "—"]:
        if sep in s:
            parts = s.split(sep)
            if len(parts) == 2 and parts[0].isdigit() and parts[1].isdigit():
                return (float(parts[0]), float(parts[1]))
    return (np.nan, np.nan)


def _standardize_future_first14(path: str) -> pd.DataFrame:
    """
    Return a clean DataFrame for first 14 matchweeks with columns:
      Date, home, away, home_goals, away_goals, result, round
    """
    df = _read_csv_flexible(path)

    date_col = _pick_col(df, ["Date", "date", "Match Date"])
    home_col = _pick_col(df, ["Home Team", "HomeTeam", "Home"])
    away_col = _pick_col(df, ["Away Team", "AwayTeam", "Away"])
    round_col = _pick_col(df, ["Round Number", "Round", "Week", "Matchweek"])

    # score can be "Result" or similar
    score_col = None
    for c in ["Result", "Score", "FT", "Full Time Result"]:
        if c in df.columns:
            score_col = c
            break

    out = df[[date_col, home_col, away_col, round_col]].copy()
    out.columns = ["Date", "home", "away", "round"]
    out["Date"] = pd.to_datetime(out["Date"], dayfirst=True, errors="coerce")

    # keep only first 14
    out["round"] = pd.to_numeric(out["round"], errors="coerce")
    out = out.dropna(subset=["round", "Date", "home", "away"]).copy()
    out = out[out["round"] <= 14].copy()

    # goals/result (may be missing for unplayed games)
    out["home_goals"] = np.nan
    out["away_goals"] = np.nan
    out["result"] = None

    if score_col is not None:
        goals = df.loc[out.index, score_col].apply(_parse_score_to_goals)
        out["home_goals"] = goals.apply(lambda x: x[0])
        out["away_goals"] = goals.apply(lambda x: x[1])

        def goals_to_result(hg, ag):
            if pd.isna(hg) or pd.isna(ag):
                return None
            if hg > ag:
                return "H"
            if hg < ag:
                return "A"
            return "D"

        out["result"] = [
            goals_to_result(hg, ag) for hg, ag in zip(out["home_goals"], out["away_goals"])
        ]

    out = out[["Date", "home", "away", "home_goals", "away_goals", "result", "round"]]
    out = out.sort_values(["Date", "home", "away"]).reset_index(drop=True)
    return out


# ---------------------------------------------------------------------
# Feature engineering (form) - robust to missing future results
# ---------------------------------------------------------------------
def _add_form_features(df: pd.DataFrame, window: int = 5) -> pd.DataFrame:
    """
    Add:
      - result_code: 0=away win, 1=draw, 2=home win (NaN if unknown)
      - home_form, away_form: rolling mean of points over last `window` played matches
      - home_gd_form, away_gd_form: rolling mean of goal difference over last `window` played matches

    IMPORTANT:
    - if a match has no result, we still compute form from past matches,
      but we do NOT update the history (so we don't poison future features).
    """
    df = df.copy()
    df["Date"] = pd.to_datetime(df["Date"], dayfirst=True, errors="coerce")
    df = df.sort_values("Date").reset_index(drop=True)

    result_map = {"A": 0, "D": 1, "H": 2}

    pts_hist = defaultdict(lambda: deque(maxlen=window))
    gd_hist = defaultdict(lambda: deque(maxlen=window))

    home_form = []
    away_form = []
    home_gd_form = []
    away_gd_form = []
    result_code = []

    for _, row in df.iterrows():
        h = row["home"]
        a = row["away"]

        home_form.append(float(np.mean(pts_hist[h])) if len(pts_hist[h]) else 0.0)
        away_form.append(float(np.mean(pts_hist[a])) if len(pts_hist[a]) else 0.0)
        home_gd_form.append(float(np.mean(gd_hist[h])) if len(gd_hist[h]) else 0.0)
        away_gd_form.append(float(np.mean(gd_hist[a])) if len(gd_hist[a]) else 0.0)

        res = row.get("result", None)
        hg = row.get("home_goals", np.nan)
        ag = row.get("away_goals", np.nan)

        if res in result_map and (not pd.isna(hg)) and (not pd.isna(ag)):
            code = result_map[res]
            result_code.append(code)

            # points
            if code == 2:  # home win
                hp, ap = 3, 0
            elif code == 0:  # away win
                hp, ap = 0, 3
            else:  # draw
                hp, ap = 1, 1

            # goal difference
            hgd = float(hg - ag)
            agd = float(ag - hg)

            pts_hist[h].append(hp)
            pts_hist[a].append(ap)
            gd_hist[h].append(hgd)
            gd_hist[a].append(agd)
        else:
            result_code.append(np.nan)

    df["home_form"] = home_form
    df["away_form"] = away_form
    df["home_gd_form"] = home_gd_form
    df["away_gd_form"] = away_gd_form
    df["result_code"] = result_code

    return df


# ---------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------
def load_train_test_with_metadata(test_start: str = "2024-08-01"):
    """
    Returns:
      X_train, X_test, y_train, y_test, train_df, test_df
    """
    all_df = _load_all_seasons()
    all_df = _add_form_features(all_df, window=5)

    cutoff = pd.Timestamp(test_start)
    train_df = all_df[all_df["Date"] < cutoff].copy()
    test_df = all_df[all_df["Date"] >= cutoff].copy()

    X_train = train_df[FEATURE_COLUMNS].to_numpy()
    y_train = train_df["result_code"].astype(int).to_numpy()

    X_test = test_df[FEATURE_COLUMNS].to_numpy()
    y_test = test_df["result_code"].astype(int).to_numpy()

    return X_train, X_test, y_train, y_test, train_df, test_df


def load_future_season_features(future_filename: str = FUTURE_FILENAME):
    """
    Returns:
      future_df, X_future

    future_df has:
      - result_code (NaN if match not played / no score in file)
      - form features ready for prediction
    """
    hist_df = _load_all_seasons()

    future_path = os.path.join(DATA_DIR, future_filename)
    if not os.path.exists(future_path):
        raise FileNotFoundError(f"Future season file not found: {future_path}")

    future_raw = _standardize_future_first14(future_path)

    hist_df["__is_future__"] = False
    future_raw["__is_future__"] = True

    combined = pd.concat([hist_df, future_raw], ignore_index=True)
    combined = _add_form_features(combined, window=5)

    future_df = combined[combined["__is_future__"]].copy()
    future_df = future_df.drop(columns=["__is_future__"]).reset_index(drop=True)

    X_future = future_df[FEATURE_COLUMNS].to_numpy()
    return future_df, X_future
