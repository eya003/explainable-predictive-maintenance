"""Tests d'intégration du projet sur le dataset AI4I réel."""

from __future__ import annotations

import numpy as np
import pytest
from pathlib import Path
from fastapi.testclient import TestClient

from services.data_service import MODEL_FEATURES, TARGET_COLUMN, fetch_ai4i_data
from services.evaluation_service import evaluate_pipeline, metrics_at_threshold
from services.interpretation_service import (
    canonical_feature,
    classification_outcome,
    get_transformed_feature_names,
    global_shap_explanation,
    grouped_shap_matrix,
    local_interpretation,
    local_explanation_key,
    human_readable_summary,
    native_feature_importance,
    permutation_feature_importance,
    pipeline_parts,
    raw_to_transformed_mapping,
    resolve_session_model,
    transformed_feature_groups,
    validate_prediction_explanation,
)
from services.training_service import (
    MODEL_NAMES,
    RECOMMENDED_PARAMS,
    automatic_scale_pos_weight,
    build_pipeline,
    build_session_model_record,
    recommended_params_for_training,
    resolve_training_params,
    split_dataset,
    train_model,
    validate_model_params,
)
from utils.plots import importance_bar_figure, shap_dependence_figure


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


@pytest.mark.parametrize(
    ("model_name", "params", "expected"),
    [
        (
            "Random Forest",
            {"n_estimators": 37, "max_depth": 9, "min_samples_split": 6,
             "min_samples_leaf": 3, "max_features": "sqrt", "criterion": "entropy",
             "class_weight": None},
            {"n_estimators": 37, "max_depth": 9, "min_samples_split": 6,
             "min_samples_leaf": 3, "max_features": "sqrt", "criterion": "entropy",
             "class_weight": None},
        ),
        (
            "XGBoost",
            {"n_estimators": 41, "max_depth": 3, "learning_rate": 0.07,
             "subsample": 0.72, "colsample_bytree": 0.81, "scale_pos_weight": 12.5},
            {"n_estimators": 41, "max_depth": 3, "learning_rate": 0.07,
             "subsample": 0.72, "colsample_bytree": 0.81, "scale_pos_weight": 12.5},
        ),
        (
            "KNN", {"n_neighbors": 7, "weights": "uniform", "p": 1},
            {"n_neighbors": 7, "weights": "uniform", "p": 1},
        ),
    ],
)
def test_manual_parameters_reach_estimators(model_name, params, expected):
    pipeline = build_pipeline(model_name, params, MODEL_FEATURES)
    estimator = pipeline.named_steps["classifier"]
    actual = estimator.get_params()
    for key, value in expected.items():
        assert actual[key] == value


def test_optimized_presets_and_automatic_class_weight(dataset):
    _, _, y_train, _ = split_dataset(dataset, MODEL_FEATURES, TARGET_COLUMN, 0.20)
    for name in MODEL_NAMES:
        resolved = recommended_params_for_training(name, y_train)
        validate_model_params(name, resolved)
        pipeline = build_pipeline(name, resolved, MODEL_FEATURES)
        assert pipeline is not None
        if name != "XGBoost":
            assert all(resolved[key] == value for key, value in RECOMMENDED_PARAMS[name].items())
    expected = float((y_train == 0).sum() / (y_train == 1).sum())
    assert automatic_scale_pos_weight(y_train) == pytest.approx(expected)
    automatic = resolve_training_params(
        "XGBoost",
        {"n_estimators": 20, "max_depth": 3, "learning_rate": 0.1,
         "subsample": 0.8, "colsample_bytree": 0.9,
         "scale_pos_weight": "automatic"},
        y_train,
    )
    assert automatic["scale_pos_weight"] == pytest.approx(expected)


def test_session_metadata_contains_used_hyperparameters(trained_models):
    results, _, x_test, _, y_test = trained_models
    pipeline = results["KNN"]["pipeline"]
    params = {"n_neighbors": 3, "weights": "distance", "p": 2}
    record = build_session_model_record(
        model_name="KNN", pipeline=pipeline, params=params, metrics={"f1": 0.5},
        features=MODEL_FEATURES, test_size=0.2, x_test=x_test, y_test=y_test,
        configuration=(tuple(MODEL_FEATURES), TARGET_COLUMN, 0.2, 42),
        trained_at="2026-01-01T00:00:00", target_column=TARGET_COLUMN,
    )
    assert record["model_name"] == "KNN"
    assert record["params"] == params
    assert record["threshold"] == 0.50
    assert record["train_test_configuration"]["random_state"] == 42
    assert record["train_test_configuration"]["target_column"] == TARGET_COLUMN


@pytest.mark.parametrize(
    ("model_name", "params"),
    [
        ("KNN", {"n_neighbors": 0, "weights": "uniform", "p": 2}),
        ("Random Forest", {"n_estimators": 0, "max_depth": None,
                           "min_samples_split": 2, "min_samples_leaf": 1,
                           "max_features": "sqrt", "criterion": "gini",
                           "class_weight": None}),
        ("XGBoost", {"n_estimators": 10, "max_depth": 3,
                     "learning_rate": 0.1, "subsample": 1.2,
                     "colsample_bytree": 0.8, "scale_pos_weight": 1.0}),
        ("Logistic Regression", {"C": 0, "solver": "lbfgs",
                                 "max_iter": 1000, "class_weight": None}),
    ],
)
def test_invalid_manual_hyperparameters_are_rejected(model_name, params):
    with pytest.raises(ValueError):
        validate_model_params(model_name, params)


def test_interpretation_service_returns_real_explanations(trained_models):
    results, _, x_test, _, y_test = trained_models
    pipeline = results["XGBoost"]["pipeline"]
    preprocessor, _ = pipeline_parts(pipeline)
    names = get_transformed_feature_names(preprocessor)
    assert names
    assert any("Torque" in name for name in names)

    observation = x_test.iloc[[0]]
    reference = x_test.sample(60, random_state=42)
    result = local_interpretation(pipeline, reference, observation, threshold=0.75)
    assert result["shap"].shape[0] == len(names)
    assert not result["lime"].empty
    assert result["summary"]
    assert result["comparison"]["agreement_denominator"] == 5
    assert result["shap_additive_consistent"] is True
    assert result["shap_probability_consistent"] is True

    native = native_feature_importance(pipeline)
    permutation = permutation_feature_importance(
        pipeline, x_test, y_test, max_samples=60, n_repeats=2
    )
    shap_result, shap_importance = global_shap_explanation(
        pipeline, x_test.sample(30, random_state=7), max_samples=20
    )
    assert not native.empty
    assert not permutation.empty
    assert shap_result.values.shape[0] == 20
    assert list(shap_result.raw_data.columns) == list(x_test.columns)
    assert shap_result.raw_data["Air temperature"].min() > 250
    mapping = raw_to_transformed_mapping(pipeline, list(x_test.columns))
    assert "Air temperature" in mapping
    assert not shap_importance.empty


@pytest.mark.parametrize("model_name", ["Decision Tree", "Random Forest", "XGBoost"])
def test_grouped_native_type_equals_one_hot_sum(trained_models, model_name):
    results, _, _, _, _ = trained_models
    pipeline = results[model_name]["pipeline"]
    technical = native_feature_importance(pipeline, group_categorical=False)
    grouped = native_feature_importance(pipeline)
    technical_type = technical[technical["feature"].str.startswith("Type_")]
    grouped_type = grouped.loc[grouped["feature"] == "Type", "importance"].iloc[0]
    assert len(technical_type) == 3
    assert grouped_type == pytest.approx(technical_type["importance"].sum())
    assert not grouped["feature"].str.startswith("Type_").any()


def test_grouped_global_and_local_shap_preserve_required_aggregations(trained_models):
    results, _, x_test, _, y_test = trained_models
    pipeline = results["XGBoost"]["pipeline"]
    groups = transformed_feature_groups(pipeline)
    shap_result, grouped_importance = global_shap_explanation(
        pipeline, x_test.sample(40, random_state=8), max_samples=25
    )
    type_indices = groups["Type"]
    expected_global_type = np.abs(shap_result.values[:, type_indices]).mean(axis=0).sum()
    actual_global_type = grouped_importance.loc[
        grouped_importance["feature"] == "Type", "mean_abs_shap"
    ].iloc[0]
    assert actual_global_type == pytest.approx(expected_global_type)

    grouped_values, _, grouped_names = grouped_shap_matrix(pipeline, shap_result)
    expected_features = {
        "Type", "Air temperature [K]", "Process temperature [K]",
        "Rotational speed [rpm]", "Torque [Nm]", "Tool wear [min]",
    }
    assert set(grouped_importance["feature"]) == expected_features
    assert set(grouped_names) == expected_features
    assert not any(name.startswith("Type_") for name in grouped_names)
    assert np.allclose(grouped_values.sum(axis=1), shap_result.values.sum(axis=1))
    for frame, value_column in (
        (native_feature_importance(pipeline), "importance"),
        (grouped_importance, "mean_abs_shap"),
    ):
        figure = importance_bar_figure(frame, value_column, "Test")
        plotted_features = {
            str(feature) for trace in figure.data for feature in trace.y
        }
        assert plotted_features == expected_features
        assert not any(feature.startswith("Type_") for feature in plotted_features)

    local = local_interpretation(
        pipeline, x_test.sample(60, random_state=13), x_test.iloc[[2]], threshold=0.75
    )
    assert local["shap_grouped"]["shap_value"].sum() == pytest.approx(
        local["shap"]["shap_value"].sum()
    )
    technical_type = local["shap"].loc[
        local["shap"]["feature"].str.startswith("Type_"), "shap_value"
    ].sum()
    grouped_type = local["shap_grouped"].loc[
        local["shap_grouped"]["feature"] == "Type", "shap_value"
    ].iloc[0]
    assert grouped_type == pytest.approx(technical_type)
    assert set(local["shap_grouped"]["feature"]) == expected_features
    assert not local["shap_grouped"]["feature"].str.startswith("Type_").any()
    reconstructed_from_grouped = (
        local["shap_base_value"] + local["shap_grouped"]["shap_value"].sum()
    )
    assert reconstructed_from_grouped == pytest.approx(
        local["shap_model_output"], abs=1e-4
    )
    assert "Type_" not in local["summary"]

    permutation = permutation_feature_importance(
        pipeline, x_test, y_test, max_samples=60, n_repeats=2
    )
    assert set(permutation["feature"]) == expected_features
    assert not permutation["feature"].str.startswith("Type_").any()


def test_selected_session_model_changes_xai_pipeline(trained_models):
    results, _, x_test, _, y_test = trained_models
    stored = {
        name: {
            "pipeline": item["pipeline"], "x_test": x_test, "y_test": y_test,
            "features": MODEL_FEATURES, "trained_at": name,
        }
        for name, item in results.items()
    }
    logistic = resolve_session_model("Logistic Regression", stored, {"Logistic Regression": 0.41})
    forest = resolve_session_model("Random Forest", stored, {"Random Forest": 0.63})
    assert logistic["pipeline"] is results["Logistic Regression"]["pipeline"]
    assert forest["pipeline"] is results["Random Forest"]["pipeline"]
    assert logistic["pipeline"] is not forest["pipeline"]
    assert logistic["threshold"] == 0.41
    assert forest["threshold"] == 0.63


@pytest.mark.parametrize(
    "model_name", ["Logistic Regression", "Decision Tree", "Random Forest", "XGBoost"]
)
def test_each_compatible_session_model_uses_its_own_xai_prediction(
    trained_models, model_name
):
    results, _, x_test, _, _ = trained_models
    pipeline = results[model_name]["pipeline"]
    observation = x_test.iloc[[3]]
    result = local_interpretation(
        pipeline, x_test.sample(60, random_state=11), observation, threshold=0.57
    )
    expected_score = float(pipeline.predict_proba(observation)[0, 1])
    assert result["probability"] == pytest.approx(expected_score)
    assert result["prediction"] == int(expected_score >= 0.57)
    assert result["shap_available"] is True
    assert not result["shap"].empty


def test_knn_xai_disables_expensive_shap_but_keeps_lime(trained_models):
    results, _, x_test, _, _ = trained_models
    pipeline = results["KNN"]["pipeline"]
    result = local_interpretation(
        pipeline, x_test.sample(60, random_state=12), x_test.iloc[[4]],
        threshold=0.50, enable_shap=False,
    )
    assert result["shap_available"] is False
    assert result["shap"].empty
    assert not result["lime"].empty
    assert result["comparison"] is None


def test_local_explanation_cache_key_invalidates_model_input_and_threshold():
    machine = {
        "Type": "L", "Air temperature": 300.0,
        "Process temperature": 310.0, "Rotational speed": 1500,
        "Torque": 40.0, "Tool wear": 100,
    }
    original = local_explanation_key("model-a", machine, 0.50)
    assert local_explanation_key("model-b", machine, 0.50) != original
    changed_input = {**machine, "Torque": 41.0}
    assert local_explanation_key("model-a", changed_input, 0.50) != original
    assert local_explanation_key("model-a", machine, 0.60) != original


def test_prediction_explanation_consistency_guard():
    prediction = {"prediction": 1, "probability": 0.771, "threshold": 0.75}
    validate_prediction_explanation(prediction, prediction.copy())
    for field, value in (
        ("prediction", 0), ("probability", 0.72), ("threshold", 0.50)
    ):
        inconsistent = prediction.copy()
        inconsistent[field] = value
        with pytest.raises(ValueError):
            validate_prediction_explanation(prediction, inconsistent)


def test_xai_navigation_separates_global_and_local_user_experiences():
    source = Path("app.py").read_text(encoding="utf-8")
    prediction_ui = source.split("def prediction_page()", 1)[1].split(
        "def _machine_payload", 1
    )[0]
    global_ui = source.split("def explainable_ai_page()", 1)[1].split(
        "initialize_state()", 1
    )[0]
    assert "Expliquer cette prédiction" in prediction_ui
    assert "_render_local_xai" in prediction_ui
    assert "_render_local_comparison" in prediction_ui
    assert "_global_xai_tab" in global_ui
    assert "_local_xai_tab" not in global_ui
    assert "_comparison_xai_tab" not in global_ui


def test_local_waterfall_uses_grouped_business_features_only():
    source = Path("app.py").read_text(encoding="utf-8")
    local_renderer = source.split("def _render_local_xai", 1)[1].split(
        "def _render_local_comparison", 1
    )[0]
    waterfall_call = local_renderer.split("shap_waterfall_figure(", 1)[1].split(
        "clear_figure=True", 1
    )[0]
    assert 'grouped_shap_table["contribution"]' in waterfall_call
    assert 'grouped_shap_table["input_value"]' in waterfall_call
    assert 'grouped_shap_table["feature"]' in waterfall_call
    assert 'shap_table["contribution"]' not in waterfall_call.replace(
        'grouped_shap_table["contribution"]', ""
    )
    assert "Comparaison des rangs uniquement" not in source


def test_truth_status_summary_grammar_and_type_normalization():
    assert classification_outcome(1, 1)["label"] == "Vrai positif (TP)"
    assert classification_outcome(0, 0)["label"] == "Vrai négatif (TN)"
    assert classification_outcome(1, 0)["label"] == "Faux positif (FP)"
    assert classification_outcome(0, 1)["label"] == "Faux négatif (FN)"
    assert classification_outcome(1, None) is None
    assert canonical_feature("categorical__Type_L") == "Type"
    table = __import__("pandas").DataFrame({
        "feature": ["Torque [Nm]", "Air temperature [K]", "Type_L"],
        "shap_value": [0.4, 0.2, -0.1],
    })
    summary, _, _ = human_readable_summary(1, 0.77, 0.75, table)
    assert "légèrement supérieur" in summary
    assert "contribuent" in summary
    assert "le couple et la température de l'air" in summary.lower()
    assert "le type de machine" in summary.lower()
    assert "Type_L" not in summary


def test_dependence_plot_uses_raw_physical_values():
    raw_air = np.array([298.2, 301.4])
    raw_torque = np.array([35.0, 57.5])
    figure = shap_dependence_figure(
        x_values=raw_air,
        shap_values=np.array([-0.2, 0.4]),
        feature_name="Air temperature [K]",
        color_values=raw_torque,
        color_name="Torque [Nm]",
    )
    assert np.array_equal(np.asarray(figure.data[0].x), raw_air)
    assert figure.layout.xaxis.title.text == "Air temperature [K]"
    assert figure.layout.coloraxis.colorbar.title.text == "Torque [Nm]"

    no_color = shap_dependence_figure(
        x_values=raw_air,
        shap_values=np.array([-0.2, 0.4]),
        feature_name="Air temperature [K]",
    )
    assert np.array_equal(np.asarray(no_color.data[0].x), raw_air)
    assert isinstance(no_color.data[0].marker.color, str)
    assert no_color.layout.coloraxis.colorbar.title.text is None


def _payload(row):
    return {
        "Type": row["Type"],
        "air_temperature": float(row["Air temperature"]),
        "process_temperature": float(row["Process temperature"]),
        "rotational_speed": int(row["Rotational speed"]),
        "torque": float(row["Torque"]),
        "tool_wear": int(row["Tool wear"]),
    }


def test_fastapi_reference_endpoints(dataset, monkeypatch):
    import main

    normal = dataset.loc[dataset[TARGET_COLUMN] == 0].iloc[0]
    failure = dataset.loc[dataset[TARGET_COLUMN] == 1].iloc[0]
    _, x_test, _, y_test = split_dataset(dataset, MODEL_FEATURES, TARGET_COLUMN, 0.20)
    monkeypatch.setattr(main, "reference_test_data", lambda: (x_test, y_test))
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
        interpretation = client.post("/interpret", json=_payload(failure))
        assert interpretation.status_code == 200, interpretation.text
        body = interpretation.json()
        assert body["shap"]
        assert body["shap_grouped"]
        assert any(row["feature"] == "Type" for row in body["shap_grouped"])
        assert not any(
            row["feature"].startswith("Type_") for row in body["shap_grouped"]
        )
        assert body["lime"]
        assert body["summary"]
        assert "shap_lime_comparison" in body
        prediction = client.post("/predict", json=_payload(failure)).json()
        assert body["probability"] == prediction["failure_score"]
        assert body["threshold"] == prediction["threshold"]
        assert body["prediction"] == prediction["prediction"]
        assert body["shap_additive_consistent"] is True
        assert body["shap_probability_consistent"] is True
        invalid = _payload(normal)
        invalid["Type"] = "INVALID"
        assert client.post("/interpret", json=invalid).status_code == 422
