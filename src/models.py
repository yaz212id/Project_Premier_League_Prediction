"""Model definitions for Premier League match prediction."""

from typing import Any

import numpy as np
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier


def train_random_forest(
    X_train: np.ndarray,
    y_train: np.ndarray,
    random_state: int = 42,
) -> RandomForestClassifier:
    """
    Train a Random Forest classifier on the training data.

    Parameters
    ----------
    X_train : np.ndarray
        Matrix of training features (n_samples, n_features).
    y_train : np.ndarray
        Training labels (result_code: 0/1/2).
    random_state : int
        Seed for reproducibility.

    Returns
    -------
    model : RandomForestClassifier
        Fitted RandomForest model.
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

    Parameters
    ----------
    X_train : np.ndarray
        Training features.
    y_train : np.ndarray
        Training labels.
    n_neighbors : int
        Number of neighbours to use in KNN.

    Returns
    -------
    model : KNeighborsClassifier
        Fitted KNN model.
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
    away win (0), draw (1), home win (2).
    """
    model = LogisticRegression(
        max_iter=max_iter,
        multi_class="multinomial",
        random_state=random_state,
    )
    model.fit(X_train, y_train)
    return model


def train_gradient_boosting(
    X_train: np.ndarray,
    y_train: np.ndarray,
    n_estimators: int = 200,
    learning_rate: float = 0.05,
    max_depth: int = 3,
    random_state: int = 42,
) -> GradientBoostingClassifier:
    """
    Train a Gradient Boosting classifier.

    This is our "nonlinear" model alternative to Random Forest.

    Parameters
    ----------
    X_train : np.ndarray
        Training features.
    y_train : np.ndarray
        Training labels.
    n_estimators : int
        Number of boosting stages (trees).
    learning_rate : float
        Shrinks the contribution of each tree.
    max_depth : int
        Maximum depth of individual trees.
    random_state : int
        Seed for reproducibility.

    Returns
    -------
    model : GradientBoostingClassifier
        Fitted Gradient Boosting model.
    """
    model = GradientBoostingClassifier(
        n_estimators=n_estimators,
        learning_rate=learning_rate,
        max_depth=max_depth,
        random_state=random_state,
    )
    model.fit(X_train, y_train)
    return model

