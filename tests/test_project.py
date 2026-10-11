"""Tests d'intégration du projet sur le dataset AI4I réel."""

from __future__ import annotations

import numpy as np
import pytest
from fastapi.testclient import TestClient

from services.data_service import MODEL_FEATURES, TARGET_COLUMN, fetch_ai4i_data
from services.evaluation_service import evaluate_pipeline, metrics_at_threshold
from services.training_service import (
    MODEL_NAMES,
    recommended_params_for_training,
    split_dataset,
    train_model,
)


@pytest.fixture(scope="session")
def dataset():
    data = fetch_ai4i_data()
    assert data.shape == (10000, 7)
    assert set(data[TARGET_COLUMN].unique()) == {0, 1}
    return data


@pytest.fixture(scope="session")
def trained_models(dataset):
    x_train, x_test, y_train, y_test = split_dataset(
        dataset, MODEL_FEATURES, TARGET_COLUMN, 0.20
    )
    results = {}
    for name in MODEL_NAMES:
        params = recommended_params_for_training(name, y_train)
        pipeline = train_model(name, params, MODEL_FEATURES, x_train, y_train)
        metrics = evaluate_pipeline(pipeline, x_test, y_test)
        prediction = pipeline.predict(x_test)
        probabilities = pipeline.predict_proba(x_test)
        assert prediction.shape == (len(x_test),)
        assert probabilities.shape == (len(x_test), 2)
        for metric in ("precision", "recall", "f1", "roc_auc", "pr_auc"):
            assert 0 <= metrics[metric] <= 1
        results[name] = {"pipeline": pipeline, "metrics": metrics}
    assert set(results) == set(MODEL_NAMES)
    return results, x_train, x_test, y_train, y_test


def test_empty_selection_trains_nothing(dataset):
    selected_models = []
    trained = {}
    if selected_models:
        raise AssertionError("Ce bloc ne doit pas être exécuté sans sélection.")
    assert trained == {}


def test_five_models_share_test_set_and_metrics(trained_models):
    results, _, x_test, _, _ = trained_models
    assert len(results) == 5
    assert all(len(result["metrics"]["scores"]) == len(x_test) for result in results.values())


def test_threshold_changes_predictions_without_refit(trained_models):
    results, _, _, _, y_test = trained_models
    pipeline = results["XGBoost"]["pipeline"]
    scores = results["XGBoost"]["metrics"]["scores"]
    identity_before = id(pipeline)
    low = metrics_at_threshold(y_test, scores, 0.10)
    high = metrics_at_threshold(y_test, scores, 0.90)
    assert id(pipeline) == identity_before
    assert not np.array_equal(
        (scores >= low["threshold"]).astype(int),
        (scores >= high["threshold"]).astype(int),
    )


def _payload(row):
    return {
        "Type": row["Type"],
        "air_temperature": float(row["Air temperature"]),
        "process_temperature": float(row["Process temperature"]),
        "rotational_speed": int(row["Rotational speed"]),
        "torque": float(row["Torque"]),
        "tool_wear": int(row["Tool wear"]),
    }


def test_fastapi_reference_endpoints(dataset):
    import main

    normal = dataset.loc[dataset[TARGET_COLUMN] == 0].iloc[0]
    failure = dataset.loc[dataset[TARGET_COLUMN] == 1].iloc[0]
    with TestClient(main.app) as client:
        root = client.get("/")
        health = client.get("/health")
        assert root.status_code == 200
        assert health.status_code == 200
        assert health.json()["model_loaded"] is True
        for row in (normal, failure):
            prediction = client.post("/predict", json=_payload(row))
            assert prediction.status_code == 200
            assert 0 <= prediction.json()["failure_score"] <= 1
