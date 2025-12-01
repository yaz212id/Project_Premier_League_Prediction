"""Model definitions for Premier League match prediction."""

from typing import Any

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier


def train_random_forest(
    X_train: np.ndarray,
    y_train: np.ndarray,
    random_state: int = 42,
) -> RandomForestClassifier:
    """
    Train a Random Forest classifier on the training data.
    """
    model = RandomForestClassifier(
        n_estimators=200,   # number of trees
        max_depth=None,    # let trees grow fully
        random_state=random_state,
        n_jobs=-1,         # use all CPU cores
    )
    model.fit(X_train, y_train)
    return model


def train_knn(
    X_train: np.ndarray,
    y_train: np.ndarray,
    n_neighbors: int = 15,
) -> KNeighborsClassifier:
    """
    Train a K-Nearest Neighbors classifier.

    n_neighbors controls how many neighbours we look at.
    """
    model = KNeighborsClassifier(n_neighbors=n_neighbors)
    model.fit(X_train, y_train)
    return model


def train_logistic_regression(
    X_train: np.ndarray,
    y_train: np.ndarray,
    max_iter: int = 1000,
    random_state: int = 42,
) -> LogisticRegression:
    """
    Train a multinomial Logistic Regression classifier.

    We use multi_class='multinomial' because we have 3 classes:
    away win, draw, home win.
    """
    model = LogisticRegression(
        max_iter=max_iter,
        multi_class="multinomial",
        random_state=random_state,
    )
    model.fit(X_train, y_train)
    return model


