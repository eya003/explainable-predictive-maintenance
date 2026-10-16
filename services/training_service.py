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


def automatic_scale_pos_weight(y_train: pd.Series) -> float:
    """Calcule le ratio négatifs/positifs sur le seul jeu d'apprentissage."""
    counts = y_train.value_counts()
    positive_count = int(counts.get(1, 0))
    if positive_count == 0:
        raise ValueError("Le jeu d'apprentissage ne contient aucune panne.")
    return float(counts.get(0, 0) / positive_count)


def validate_model_params(model_name: str, params: dict[str, Any]) -> None:
    """Valide les hyperparamètres exposés dans l'interface avant l'entraînement."""
    def positive(key: str) -> None:
        if float(params[key]) <= 0:
            raise ValueError(f"{key} doit être strictement positif.")

    if model_name == "Logistic Regression":
        positive("C")
        if params.get("solver") not in {"lbfgs", "liblinear", "saga"}:
            raise ValueError("Solver de régression logistique non pris en charge.")
        if int(params.get("max_iter", 2000)) < 1:
            raise ValueError("max_iter doit être supérieur ou égal à 1.")
        if params.get("class_weight") not in {"balanced", None}:
            raise ValueError("class_weight doit être 'balanced' ou None.")
    elif model_name == "Decision Tree":
        if params.get("max_depth") is not None and int(params["max_depth"]) < 1:
            raise ValueError("max_depth doit être None ou supérieur ou égal à 1.")
        if int(params.get("min_samples_split", 0)) < 2:
            raise ValueError("min_samples_split doit être supérieur ou égal à 2.")
        if int(params.get("min_samples_leaf", 0)) < 1:
            raise ValueError("min_samples_leaf doit être supérieur ou égal à 1.")
        if params.get("criterion", "gini") not in {"gini", "entropy", "log_loss"}:
            raise ValueError("Critère d'arbre non pris en charge.")
        if params.get("class_weight") not in {"balanced", None}:
            raise ValueError("class_weight doit être 'balanced' ou None.")
    elif model_name == "KNN":
        if int(params.get("n_neighbors", 0)) < 1:
            raise ValueError("n_neighbors doit être supérieur ou égal à 1.")
        if params.get("weights") not in {"uniform", "distance"}:
            raise ValueError("weights doit être 'uniform' ou 'distance'.")
        if int(params.get("p", 0)) not in {1, 2}:
            raise ValueError("p doit valoir 1 ou 2.")
    elif model_name == "Random Forest":
        if int(params.get("n_estimators", 0)) < 1:
            raise ValueError("n_estimators doit être supérieur ou égal à 1.")
        if params.get("max_depth") is not None and int(params["max_depth"]) < 1:
            raise ValueError("max_depth doit être None ou supérieur ou égal à 1.")
        if int(params.get("min_samples_split", 0)) < 2:
            raise ValueError("min_samples_split doit être supérieur ou égal à 2.")
        if int(params.get("min_samples_leaf", 0)) < 1:
            raise ValueError("min_samples_leaf doit être supérieur ou égal à 1.")
        if params.get("max_features") not in {"sqrt", "log2", None}:
            raise ValueError("max_features doit être 'sqrt', 'log2' ou None.")
        if params.get("criterion", "gini") not in {"gini", "entropy", "log_loss"}:
            raise ValueError("Critère de forêt non pris en charge.")
        if params.get("class_weight") not in {"balanced", None}:
            raise ValueError("class_weight doit être 'balanced' ou None.")
    elif model_name == "XGBoost":
        if int(params.get("n_estimators", 0)) < 1:
            raise ValueError("n_estimators doit être supérieur ou égal à 1.")
        if int(params.get("max_depth", -1)) < 0:
            raise ValueError("max_depth doit être supérieur ou égal à 0.")
        positive("learning_rate")
        for key in ("subsample", "colsample_bytree"):
            value = float(params.get(key, 1.0))
            if not 0 < value <= 1:
                raise ValueError(f"{key} doit être compris entre 0 exclu et 1 inclus.")
        positive("scale_pos_weight")
    else:
        raise ValueError(f"Modèle inconnu : {model_name}")


def resolve_training_params(
    model_name: str, params: dict[str, Any], y_train: pd.Series
) -> dict[str, Any]:
    """Résout les valeurs automatiques puis valide une configuration utilisateur."""
    resolved = params.copy()
    if model_name == "XGBoost" and resolved.get("scale_pos_weight") in {
        "automatic", "calculé sur le train"
    }:
        resolved["scale_pos_weight"] = automatic_scale_pos_weight(y_train)
    validate_model_params(model_name, resolved)
    return resolved


def build_session_model_record(
    *, model_name: str, pipeline: Pipeline, params: dict[str, Any], metrics: dict[str, Any],
    features: list[str], test_size: float, x_test: pd.DataFrame, y_test: pd.Series,
    configuration: tuple[Any, ...], trained_at: str,
    target_column: str = "Machine failure", threshold: float = 0.50,
) -> dict[str, Any]:
    """Construit les métadonnées communes consommées par les autres pages."""
    return {
        "model_name": model_name,
        "pipeline": pipeline,
        "metrics": metrics,
        "params": params.copy(),
        "threshold": float(threshold),
        "features": features.copy(),
        "test_size": float(test_size),
        "x_test": x_test,
        "y_test": y_test,
        "trained_at": trained_at,
        "configuration": configuration,
        "train_test_configuration": {
            "features": features.copy(),
            "target_column": target_column,
            "test_size": float(test_size),
            "random_state": 42,
        },
    }


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
        estimator = LogisticRegression(random_state=42, **{"max_iter": 2000, **params})
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
    validate_model_params(model_name, params)
    pipeline = build_pipeline(model_name, params, feature_columns)
    pipeline.fit(x_train, y_train)
    return pipeline
