"""
Evaluation helpers for Premier League models.

Includes:
- league table builder (robust to NaNs)
- Wilson 95% CI for accuracy
- McNemar exact test p-value vs baseline (no SciPy needed)
- clean accuracy summary plot
- logistic regression coefficient heatmap
"""

from __future__ import annotations

from collections import defaultdict
from math import comb, sqrt
from typing import Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score


def build_league_table(matches_df: pd.DataFrame, result_col: str) -> pd.DataFrame:
    """
    Build a league table from a match dataframe.

    Required columns:
      - home, away
      - result_col:
          0 = away win
          1 = draw
          2 = home win

    IMPORTANT:
    - rows with NaN in result_col are skipped (this fixes your NaN->int crash).
    - outputs are integers (P/W/D/L/Pts).
    """
    table = defaultdict(lambda: {"P": 0, "W": 0, "D": 0, "L": 0, "Pts": 0})

    for _, row in matches_df.iterrows():
        res = row.get(result_col)
        if pd.isna(res):
            continue

        home = row["home"]
        away = row["away"]
        res = int(res)

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
        else:  # draw
            table[home]["D"] += 1
            table[away]["D"] += 1
            table[home]["Pts"] += 1
            table[away]["Pts"] += 1

    df = pd.DataFrame.from_dict(table, orient="index")
    df.index.name = "Team"
    df = df.sort_values(by=["Pts", "W"], ascending=False)
    df.insert(0, "Pos", range(1, len(df) + 1))

    for c in ["Pos", "P", "W", "D", "L", "Pts"]:
        df[c] = df[c].astype(int)

    return df


def accuracy_ci_wilson(k: int, n: int, z: float = 1.96) -> Tuple[float, float]:
    """
    Wilson score interval for binomial proportion.
    """
    if n == 0:
        return (0.0, 0.0)

    phat = k / n
    denom = 1.0 + (z**2) / n
    center = (phat + (z**2) / (2 * n)) / denom
    half = (z * sqrt((phat * (1 - phat) / n) + (z**2 / (4 * n**2)))) / denom
    return (max(0.0, center - half), min(1.0, center + half))


def mcnemar_pvalue_vs_baseline(
    y_true: np.ndarray, y_pred_model: np.ndarray, y_pred_base: np.ndarray
) -> float:
    """
    McNemar exact test using discordant pairs:
      b = model correct, baseline wrong
      c = model wrong, baseline correct

    p-value = 2 * BinomialCDF(min(b,c); n=b+c, p=0.5)
    """
    model_correct = (y_pred_model == y_true)
    base_correct = (y_pred_base == y_true)

    b = int(np.sum(model_correct & (~base_correct)))
    c = int(np.sum((~model_correct) & base_correct))
    n = b + c

    if n == 0:
        return 1.0

    m = min(b, c)
    cdf = 0.0
    for i in range(0, m + 1):
        cdf += comb(n, i) * (0.5**n)

    return float(min(1.0, 2.0 * cdf))


def plot_accuracy_summary(summary_df: pd.DataFrame, save_path: str, title: str) -> None:
    """
    summary_df expected columns:
      model, accuracy, ci_low, ci_high, p_value (NaN allowed for baseline)
    """
    names = summary_df["model"].tolist()
    acc = summary_df["accuracy"].to_numpy()
    ci_low = summary_df["ci_low"].to_numpy()
    ci_high = summary_df["ci_high"].to_numpy()
    pvals = summary_df["p_value"].to_numpy()

    yerr = np.vstack([acc - ci_low, ci_high - acc])

    plt.figure(figsize=(10, 4))
    x = np.arange(len(names))

    plt.bar(x, acc, width=0.55)
    plt.errorbar(x, acc, yerr=yerr, fmt="none", capsize=4)

    plt.ylim(0, 1.0)
    plt.ylabel("Accuracy")
    plt.title(title)
    plt.xticks(x, names, rotation=18, ha="right")

    for i, pv in enumerate(pvals):
        if np.isnan(pv):
            continue
        plt.text(i, acc[i] + 0.02, f"p={pv:.3f}", ha="center", fontsize=9)

    plt.tight_layout()
    plt.savefig(save_path, bbox_inches="tight")
    plt.close()


def plot_logistic_regression_coefficients(
    model,
    feature_names: Optional[list[str]],
    save_path: str,
    title: str = "Logistic Regression coefficients (multiclass)",
) -> None:
    """
    Works with:
      - LogisticRegression
      - Pipeline(scaler -> logreg)
    """
    if hasattr(model, "named_steps") and "logreg" in model.named_steps:
        lr = model.named_steps["logreg"]
    else:
        lr = model

    if not hasattr(lr, "coef_"):
        return

    coef = lr.coef_
    n_classes, n_features = coef.shape

    if not feature_names or len(feature_names) != n_features:
        feature_names = [f"feature_{i}" for i in range(n_features)]

    plt.figure(figsize=(10, 3.2))
    im = plt.imshow(coef, aspect="auto")
    plt.colorbar(im, fraction=0.02, pad=0.02)

    plt.title(title)
    plt.yticks(range(n_classes), [f"class {i}" for i in range(n_classes)])
    plt.xticks(range(n_features), feature_names, rotation=25, ha="right")

    for i in range(n_classes):
        for j in range(n_features):
            plt.text(j, i, f"{coef[i, j]:.2f}", ha="center", va="center", fontsize=8)

    plt.tight_layout()
    plt.savefig(save_path, bbox_inches="tight")
    plt.close()


def simple_accuracy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(accuracy_score(y_true, y_pred))
