"""
Model definitions for Premier League match prediction.

Objectif principal :
- Garder une bonne accuracy globale
- Mieux prendre en compte la classe 1 = "draw" (match nul)

Stratégies :
- class_weight pour RF & Logistic Regression (on booste les nuls)
- léger oversampling des nuls pour Gradient Boosting
- KNN avec standardisation + pondération par la distance
"""

from __future__ import annotations

from collections import Counter
from typing import Dict, Tuple

import numpy as np
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

# -------------------------------------------------------------------
# Hyperparamètres globaux
# -------------------------------------------------------------------
RANDOM_STATE = 42

# facteur pour sur-pondérer les nuls (classe 1)
# 1.0 = rien de spécial, 1.5–2.0 = on donne plus de poids aux nuls
DRAW_BOOST = 1.8


# -------------------------------------------------------------------
# Utilitaires : class weights & oversampling des nuls
# -------------------------------------------------------------------
def compute_class_weights(y_train: np.ndarray, draw_boost: float = DRAW_BOOST) -> Dict[int, float]:
    """
    Calcule des poids de classe "balanced" puis sur-pondère la classe 1 (draw).

    Retourne un dict du type {0: w_away, 1: w_draw, 2: w_home}.
    """
    counter = Counter(y_train)
    classes = sorted(counter.keys())
    n_classes = len(classes)
    n_samples = len(y_train)

    # balanced : N / (k * n_c)
    weights: Dict[int, float] = {}
    for c in classes:
        weights[c] = n_samples / (n_classes * counter[c])

    # on booste un peu les nuls
    if 1 in weights:
        weights[1] *= draw_boost

    return weights


def oversample_draws(
    X: np.ndarray,
    y: np.ndarray,
    factor: float = 1.5,
    random_state: int = RANDOM_STATE,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Duplique aléatoirement des matchs nuls pour que cette classe soit un peu plus présente.

    factor = 1.5 => on ajoute ~50 % de matchs nuls en plus.
    Si y contient très peu de nuls, l'effet est plus marqué.
    """
    rng = np.random.default_rng(random_state)

    draw_mask = (y == 1)
    n_draws = int(draw_mask.sum())
    if n_draws == 0 or factor <= 1.0:
        return X, y

    n_extra = int((factor - 1.0) * n_draws)
    if n_extra <= 0:
        return X, y

    X_draw = X[draw_mask]
    y_draw = y[draw_mask]

    idx = rng.integers(low=0, high=n_draws, size=n_extra, endpoint=False)
    X_extra = X_draw[idx]
    y_extra = y_draw[idx]

    X_balanced = np.vstack([X, X_extra])
    y_balanced = np.concatenate([y, y_extra])

    return X_balanced, y_balanced


# -------------------------------------------------------------------
# 1. Random Forest
# -------------------------------------------------------------------
def train_random_forest(
    X_train: np.ndarray,
    y_train: np.ndarray,
    random_state: int = RANDOM_STATE,
) -> RandomForestClassifier:
    """
    Train a Random Forest classifier.

    On utilise class_weight pour mieux prendre en compte la classe "draw".
    """
    class_weight = compute_class_weights(y_train)

    model = RandomForestClassifier(
        n_estimators=300,
        max_depth=None,
        min_samples_split=4,
        min_samples_leaf=2,
        random_state=random_state,
        n_jobs=-1,
        class_weight=class_weight,
    )
    model.fit(X_train, y_train)
    return model


# -------------------------------------------------------------------
# 2. KNN
# -------------------------------------------------------------------
def train_knn(
    X_train: np.ndarray,
    y_train: np.ndarray,
    n_neighbors: int = 25,
) -> Pipeline:
    """
    Train a K-Nearest Neighbors classifier.

    - Standardisation des features
    - weights="distance" : les voisins proches comptent plus que les lointains
    """
    model = Pipeline(
        steps=[
            ("scaler", StandardScaler()),
            (
                "knn",
                KNeighborsClassifier(
                    n_neighbors=n_neighbors,
                    weights="distance",
                    metric="minkowski",
                    p=2,
                ),
            ),
        ]
    )
    model.fit(X_train, y_train)
    return model


# -------------------------------------------------------------------
# 3. Logistic Regression
# -------------------------------------------------------------------
def train_logistic_regression(
    X_train: np.ndarray,
    y_train: np.ndarray,
    max_iter: int = 1000,
    random_state: int = RANDOM_STATE,
) -> Pipeline:
    """
    Multinomial Logistic Regression avec class_weight et standardisation.

    On donne plus de poids à la classe 1 (draw) via compute_class_weights.
    """
    class_weight = compute_class_weights(y_train)

    model = Pipeline(
        steps=[
            ("scaler", StandardScaler()),
            (
                "logreg",
                LogisticRegression(
                    max_iter=max_iter,
                    multi_class="multinomial",
                    solver="lbfgs",
                    class_weight=class_weight,
                    random_state=random_state,
                ),
            ),
        ]
    )
    model.fit(X_train, y_train)
    return model


# -------------------------------------------------------------------
# 4. Gradient Boosting
# -------------------------------------------------------------------
def train_gradient_boosting(
    X_train: np.ndarray,
    y_train: np.ndarray,
    n_estimators: int = 300,
    learning_rate: float = 0.05,
    max_depth: int = 3,
    random_state: int = RANDOM_STATE,
) -> GradientBoostingClassifier:
    """
    Train a Gradient Boosting classifier.

    Sklearn GradientBoostingClassifier ne gère pas class_weight, donc on fait
    un léger oversampling des matchs nuls avant l'entraînement.
    """
    X_balanced, y_balanced = oversample_draws(X_train, y_train, factor=1.6)

    model = GradientBoostingClassifier(
        n_estimators=n_estimators,
        learning_rate=learning_rate,
        max_depth=max_depth,
        random_state=random_state,
    )
    model.fit(X_balanced, y_balanced)
    return model
