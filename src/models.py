"""
Model definitions for Premier League match prediction.

All comments are in English.
"""

from __future__ import annotations

from collections import Counter
from typing import Dict

import numpy as np
from sklearn.base import BaseEstimator
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


RANDOM_STATE = 42


def _balanced_class_weights(y: np.ndarray) -> Dict[int, float]:
    """
    Balanced weights: N / (K * n_c) for each class c
    """
    counter = Counter(y)
    classes = sorted(counter.keys())
    n = len(y)
    k = len(classes)
    return {c: n / (k * counter[c]) for c in classes}


def train_random_forest(
    X_train: np.ndarray,
    y_train: np.ndarray,
    random_state: int = 42,
    draw_class_weight: float = 1.0,
    draw_pred_boost: float = 1.0,
) -> BaseEstimator:
    """
    Train a RandomForestClassifier.

    Parameters
    ----------
    draw_class_weight:
        Multiplier applied to the draw class (class=1) in class_weight.
        Keep it at 1.0 for a neutral model.
    draw_pred_boost:
        Optional multiplier applied at prediction time on P(draw) if predict_proba exists.
        (Main script handles it if present.)
    """
    cw = _balanced_class_weights(y_train)
    if 1 in cw:
        cw[1] *= float(draw_class_weight)

    model = RandomForestClassifier(
        n_estimators=400,
        min_samples_split=4,
        min_samples_leaf=2,
        random_state=random_state,
        n_jobs=-1,
        class_weight=cw,
    )
    model.fit(X_train, y_train)

    # attach for optional use in main.py
    model.draw_pred_boost = float(draw_pred_boost)
    return model


def train_knn(
    X_train: np.ndarray,
    y_train: np.ndarray,
    n_neighbors: int = 25,
) -> BaseEstimator:
    """
    KNN with scaling + distance weighting.
    """
    model = Pipeline(
        steps=[
            ("scaler", StandardScaler()),
            ("knn", KNeighborsClassifier(n_neighbors=n_neighbors, weights="distance")),
        ]
    )
    model.fit(X_train, y_train)
    return model


def train_logistic_regression(
    X_train: np.ndarray,
    y_train: np.ndarray,
    random_state: int = 42,
    max_iter: int = 2000,
    draw_class_weight: float = 1.0,
    draw_pred_boost: float = 1.0,
) -> BaseEstimator:
    """
    Logistic regression (multiclass handled automatically by sklearn).
    We do NOT pass multi_class to avoid version issues.
    """
    cw = _balanced_class_weights(y_train)
    if 1 in cw:
        cw[1] *= float(draw_class_weight)

    model = Pipeline(
        steps=[
            ("scaler", StandardScaler()),
            (
                "logreg",
                LogisticRegression(
                    solver="lbfgs",
                    max_iter=max_iter,
                    class_weight=cw,
                    random_state=random_state,
                ),
            ),
        ]
    )
    model.fit(X_train, y_train)

    # attach for optional use in main.py
    model.draw_pred_boost = float(draw_pred_boost)
    return model


def train_gradient_boosting(
    X_train: np.ndarray,
    y_train: np.ndarray,
    random_state: int = 42,
) -> BaseEstimator:
    """
    Simple GradientBoostingClassifier baseline.
    """
    model = GradientBoostingClassifier(
        n_estimators=250,
        learning_rate=0.06,
        max_depth=3,
        random_state=random_state,
    )
    model.fit(X_train, y_train)
    return model
