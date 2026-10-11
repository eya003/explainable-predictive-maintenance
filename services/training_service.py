"""Création et entraînement à la demande des pipelines de classification."""

from __future__ import annotations

from typing import Any

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier


MODEL_NAMES = [
    "Logistic Regression",
    "Decision Tree",
    "KNN",
    "Random Forest",
    "XGBoost",
]

RECOMMENDED_PARAMS: dict[str, dict[str, Any]] = {
    "Logistic Regression": {"C": 1.0, "solver": "lbfgs", "class_weight": "balanced"},
    "Decision Tree": {
        "max_depth": None,
        "min_samples_split": 2,
        "min_samples_leaf": 2,
        "class_weight": "balanced",
    },
    "KNN": {"n_neighbors": 3, "weights": "distance", "p": 2},
    "Random Forest": {
        "n_estimators": 100,
        "max_depth": None,
        "min_samples_split": 5,
        "min_samples_leaf": 2,
        "max_features": "log2",
        "class_weight": "balanced",
    },
    "XGBoost": {
        "n_estimators": 200,
        "max_depth": 4,
        "learning_rate": 0.1,
        "subsample": 0.8,
        "scale_pos_weight": "calculé sur le train",
    },
}


def recommended_params_for_training(
    model_name: str, y_train: pd.Series
) -> dict[str, Any]:
    """Retourne les paramètres recommandés, avec le poids XGBoost calculé sur le train."""
    params = RECOMMENDED_PARAMS[model_name].copy()
    if model_name == "XGBoost":
        counts = y_train.value_counts()
        positive_count = int(counts.get(1, 0))
        if positive_count == 0:
            raise ValueError("Le jeu d'apprentissage ne contient aucune panne.")
        params["scale_pos_weight"] = float(counts.get(0, 0) / positive_count)
    return params


def split_dataset(
    data: pd.DataFrame,
    feature_columns: list[str],
    target_column: str,
    test_size: float,
    random_state: int = 42,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """Produit un split stratifié reproductible."""
    x = data.loc[:, feature_columns].copy()
    y = data[target_column].astype(int).copy()
    return train_test_split(
        x, y, test_size=test_size, random_state=random_state, stratify=y
    )


def _preprocessor(feature_columns: list[str], scale_numeric: bool) -> ColumnTransformer:
    categorical = [column for column in feature_columns if column == "Type"]
    numeric = [column for column in feature_columns if column != "Type"]
    transformers: list[tuple[str, Any, list[str]]] = []
    if numeric:
        transformers.append(
            ("numeric", StandardScaler() if scale_numeric else "passthrough", numeric)
        )
    if categorical:
        transformers.append(
            (
                "categorical",
                OneHotEncoder(handle_unknown="ignore", sparse_output=False),
                categorical,
            )
        )
    return ColumnTransformer(transformers=transformers, remainder="drop")


def build_pipeline(
    model_name: str, params: dict[str, Any], feature_columns: list[str]
) -> Pipeline:
    """Construit un pipeline complet pour empêcher toute fuite de données."""
    if model_name == "Logistic Regression":
        estimator = LogisticRegression(max_iter=2000, random_state=42, **params)
        scale_numeric = True
    elif model_name == "Decision Tree":
        estimator = DecisionTreeClassifier(random_state=42, **params)
        scale_numeric = False
    elif model_name == "KNN":
        estimator = KNeighborsClassifier(**params)
        scale_numeric = True
    elif model_name == "Random Forest":
        estimator = RandomForestClassifier(random_state=42, n_jobs=-1, **params)
        scale_numeric = False
    elif model_name == "XGBoost":
        estimator = XGBClassifier(
            random_state=42,
            eval_metric="logloss",
            n_jobs=-1,
            **params,
        )
        scale_numeric = False
    else:
        raise ValueError(f"Modèle inconnu : {model_name}")

    return Pipeline(
        steps=[
            ("preprocessor", _preprocessor(feature_columns, scale_numeric)),
            ("classifier", estimator),
        ]
    )


def train_model(
    model_name: str,
    params: dict[str, Any],
    feature_columns: list[str],
    x_train: pd.DataFrame,
    y_train: pd.Series,
) -> Pipeline:
    pipeline = build_pipeline(model_name, params, feature_columns)
    pipeline.fit(x_train, y_train)
    return pipeline
