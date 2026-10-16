"""Services d'interprétation du comportement des pipelines de classification."""

from __future__ import annotations

from dataclasses import dataclass
from math import exp
from typing import Any

import numpy as np
import pandas as pd
import shap
from lime.lime_tabular import LimeTabularExplainer
from sklearn.inspection import permutation_importance
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder


DISPLAY_NAMES = {
    "Air temperature": "Air temperature [K]",
    "Process temperature": "Process temperature [K]",
    "Rotational speed": "Rotational speed [rpm]",
    "Torque": "Torque [Nm]",
    "Tool wear": "Tool wear [min]",
}

FRENCH_NAMES = {
    "Torque [Nm]": "le couple",
    "Process temperature [K]": "la température processus",
    "Air temperature [K]": "la température de l'air",
    "Rotational speed [rpm]": "la vitesse de rotation",
    "Tool wear [min]": "l'usure de l'outil",
    "Type": "le type de machine",
}


@dataclass
class ShapResult:
    values: np.ndarray
    base_values: np.ndarray
    data: np.ndarray
    feature_names: list[str]
    raw_data: pd.DataFrame | None = None

    def explanation(self) -> shap.Explanation:
        return shap.Explanation(
            values=self.values,
            base_values=self.base_values,
            data=self.data,
            feature_names=self.feature_names,
        )


def pipeline_parts(pipeline: Pipeline) -> tuple[Any, Any]:
    """Retourne preprocessing et classifieur pour les deux conventions du projet."""
    preprocessor = pipeline.named_steps.get("preprocessor") or pipeline.named_steps.get(
        "preprocessing"
    )
    classifier = pipeline.named_steps.get("classifier") or pipeline.named_steps.get("model")
    if preprocessor is None or classifier is None:
        raise ValueError("Pipeline incompatible : preprocessing ou classifieur introuvable.")
    return preprocessor, classifier


def friendly_name(name: str) -> str:
    cleaned = name.split("__", 1)[-1]
    return DISPLAY_NAMES.get(cleaned, cleaned)


def get_transformed_feature_names(preprocessor: Any) -> list[str]:
    """Récupère les noms réels produits par ColumnTransformer/OneHotEncoder."""
    try:
        raw_names = preprocessor.get_feature_names_out()
    except Exception as exc:
        raise ValueError("Impossible de récupérer les noms après preprocessing.") from exc
    return [friendly_name(str(name)) for name in raw_names]


def transform_frame(
    pipeline: Pipeline, data: pd.DataFrame
) -> tuple[np.ndarray, list[str]]:
    preprocessor, _ = pipeline_parts(pipeline)
    transformed = preprocessor.transform(data)
    if hasattr(transformed, "toarray"):
        transformed = transformed.toarray()
    return np.asarray(transformed, dtype=float), get_transformed_feature_names(preprocessor)


def _raw_columns(preprocessor: Any, columns: Any) -> list[str]:
    feature_names = list(getattr(preprocessor, "feature_names_in_", []))
    if isinstance(columns, slice):
        return feature_names[columns]
    values = list(columns) if not isinstance(columns, str) else [columns]
    if values and isinstance(values[0], (bool, np.bool_)):
        return [name for name, keep in zip(feature_names, values) if keep]
    if values and isinstance(values[0], (int, np.integer)):
        return [feature_names[int(index)] for index in values]
    return [str(value) for value in values]


def _one_hot_encoder(transformer: Any) -> OneHotEncoder | None:
    if isinstance(transformer, OneHotEncoder):
        return transformer
    if isinstance(transformer, Pipeline):
        for _, step in reversed(transformer.steps):
            if isinstance(step, OneHotEncoder):
                return step
    return None


def transformed_feature_groups(pipeline: Pipeline) -> dict[str, list[int]]:
    """Mappe chaque variable brute vers ses colonnes transformées ajustées."""
    preprocessor, _ = pipeline_parts(pipeline)
    groups: dict[str, list[int]] = {}
    output_indices = getattr(preprocessor, "output_indices_", {})
    for transformer_name, transformer, columns in preprocessor.transformers_:
        if transformer == "drop" or transformer_name not in output_indices:
            continue
        raw_columns = _raw_columns(preprocessor, columns)
        output_slice = output_indices[transformer_name]
        positions = list(range(output_slice.start, output_slice.stop))
        encoder = _one_hot_encoder(transformer)
        if encoder is None:
            if len(positions) == len(raw_columns):
                for raw_name, position in zip(raw_columns, positions):
                    groups[raw_name] = [position]
            continue
        cursor = 0
        drop_indices = getattr(encoder, "drop_idx_", None)
        for index, raw_name in enumerate(raw_columns):
            dropped = (
                drop_indices is not None
                and drop_indices[index] is not None
            )
            width = len(encoder.categories_[index]) - int(dropped)
            groups[raw_name] = positions[cursor : cursor + width]
            cursor += width
    ordered_names = list(getattr(preprocessor, "feature_names_in_", groups))
    return {name: groups[name] for name in ordered_names if name in groups}


def raw_to_transformed_mapping(
    pipeline: Pipeline, raw_columns: list[str]
) -> dict[str, int]:
    """Associe une variable brute à sa colonne transformée sans supposer son indice."""
    groups = transformed_feature_groups(pipeline)
    return {
        raw: groups[raw][0]
        for raw in raw_columns
        if raw in groups and len(groups[raw]) == 1
    }


def native_feature_importance(
    pipeline: Pipeline, group_categorical: bool = True
) -> pd.DataFrame:
    """Importance native du classifieur dans l'espace transformé."""
    preprocessor, classifier = pipeline_parts(pipeline)
    names = get_transformed_feature_names(preprocessor)
    if hasattr(classifier, "feature_importances_"):
        values = np.asarray(classifier.feature_importances_, dtype=float)
        frame = pd.DataFrame({"feature": names, "importance": values})
    elif hasattr(classifier, "coef_"):
        coefficients = np.asarray(classifier.coef_, dtype=float)
        signed = coefficients[0] if coefficients.ndim > 1 else coefficients
        frame = pd.DataFrame(
            {"feature": names, "coefficient": signed, "importance": np.abs(signed)}
        )
    else:
        raise ValueError("Ce modèle ne fournit pas d'importance native des variables.")
    if group_categorical:
        groups = transformed_feature_groups(pipeline)
        grouped_rows = []
        for raw_name, indices in groups.items():
            row = {
                "feature": friendly_name(raw_name),
                "importance": float(frame.iloc[indices]["importance"].sum()),
            }
            if "coefficient" in frame:
                row["coefficient"] = float(frame.iloc[indices]["coefficient"].sum())
            grouped_rows.append(row)
        frame = pd.DataFrame(grouped_rows)
    return frame.sort_values("importance", ascending=False).reset_index(drop=True)


def permutation_feature_importance(
    pipeline: Pipeline,
    x_test: pd.DataFrame,
    y_test: pd.Series,
    max_samples: int = 500,
    n_repeats: int = 5,
) -> pd.DataFrame:
    """Importance par permutation sur le pipeline complet et les colonnes brutes."""
    if len(x_test) > max_samples:
        sample = x_test.sample(max_samples, random_state=42)
        target = y_test.loc[sample.index]
    else:
        sample, target = x_test, y_test
    result = permutation_importance(
        pipeline,
        sample,
        target,
        scoring="average_precision",
        n_repeats=n_repeats,
        random_state=42,
        n_jobs=-1,
    )
    return (
        pd.DataFrame(
            {
                "feature": [friendly_name(column) for column in sample.columns],
                "importance_mean": result.importances_mean.astype(float),
                "importance_std": result.importances_std.astype(float),
            }
        )
        .sort_values("importance_mean", ascending=False)
        .reset_index(drop=True)
    )


def _positive_class_values(explanation: shap.Explanation) -> tuple[np.ndarray, np.ndarray]:
    values = np.asarray(explanation.values)
    base_values = np.asarray(explanation.base_values)
    if values.ndim == 3:
        values = values[:, :, 1]
        if base_values.ndim == 2:
            base_values = base_values[:, 1]
    if base_values.ndim == 0:
        base_values = np.repeat(float(base_values), values.shape[0])
    elif base_values.ndim > 1:
        base_values = base_values.reshape(base_values.shape[0], -1)[:, -1]
    return np.asarray(values, dtype=float), np.asarray(base_values, dtype=float)


def compute_shap_values(
    pipeline: Pipeline,
    background: pd.DataFrame,
    observations: pd.DataFrame,
) -> ShapResult:
    """Calcule les contributions SHAP pour la classe positive Panne (1)."""
    _, classifier = pipeline_parts(pipeline)
    background_values, names = transform_frame(pipeline, background)
    observed_values, _ = transform_frame(pipeline, observations)
    limited_background = shap.sample(
        background_values, min(100, len(background_values)), random_state=42
    )

    classifier_name = classifier.__class__.__name__
    if classifier_name == "KNeighborsClassifier":
        kernel_background = shap.sample(
            limited_background, min(25, len(limited_background)), random_state=42
        )
        explainer = shap.KernelExplainer(
            lambda values: classifier.predict_proba(values)[:, 1], kernel_background
        )
        raw_values = explainer.shap_values(observed_values, nsamples=100, silent=True)
        values = np.asarray(raw_values, dtype=float)
        if values.ndim == 3:
            values = values[:, :, -1]
        base_values = np.repeat(
            float(np.asarray(explainer.expected_value).reshape(-1)[-1]), len(observed_values)
        )
    elif classifier_name == "LogisticRegression":
        explainer = shap.LinearExplainer(
            classifier, limited_background, feature_names=names
        )
        values, base_values = _positive_class_values(explainer(observed_values))
    elif classifier_name in {
        "DecisionTreeClassifier", "RandomForestClassifier", "XGBClassifier"
    }:
        explainer = shap.TreeExplainer(
            classifier,
            limited_background,
            feature_names=names,
            feature_perturbation="interventional",
        )
        values, base_values = _positive_class_values(explainer(observed_values))
    else:
        explainer = shap.Explainer(classifier, limited_background, feature_names=names)
        values, base_values = _positive_class_values(explainer(observed_values))

    return ShapResult(values, base_values, observed_values, names)


def global_shap_explanation(
    pipeline: Pipeline,
    x_test: pd.DataFrame,
    max_samples: int = 300,
) -> tuple[ShapResult, pd.DataFrame]:
    sample = x_test.sample(n=min(max_samples, len(x_test)), random_state=42)
    background = x_test.sample(n=min(200, len(x_test)), random_state=7)
    result = compute_shap_values(pipeline, background, sample)
    result.raw_data = sample.reset_index(drop=True)
    transformed_importance = np.abs(result.values).mean(axis=0)
    importance = pd.DataFrame(
        [
            {
                "feature": friendly_name(raw_name),
                "mean_abs_shap": float(transformed_importance[indices].sum()),
            }
            for raw_name, indices in transformed_feature_groups(pipeline).items()
        ]
    ).sort_values("mean_abs_shap", ascending=False).reset_index(drop=True)
    return result, importance


def grouped_shap_matrix(
    pipeline: Pipeline, result: ShapResult
) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """Agrège les SHAP par variable brute tout en conservant la somme par ligne."""
    groups = transformed_feature_groups(pipeline)
    values = np.column_stack(
        [result.values[:, indices].sum(axis=1) for indices in groups.values()]
    )
    data_columns = []
    for raw_name, indices in groups.items():
        if result.raw_data is not None and raw_name in result.raw_data:
            series = result.raw_data[raw_name]
            if pd.api.types.is_numeric_dtype(series):
                data_columns.append(series.astype(float).to_numpy())
            else:
                data_columns.append(pd.Categorical(series).codes.astype(float))
        else:
            data_columns.append(result.data[:, indices[0]])
    data = np.column_stack(data_columns)
    names = [friendly_name(raw_name) for raw_name in groups]
    return values, data, names


def _raw_value(feature: str, observation: pd.DataFrame, transformed_value: float) -> Any:
    reverse_names = {display: raw for raw, display in DISPLAY_NAMES.items()}
    raw_name = reverse_names.get(feature, feature)
    if raw_name in observation.columns:
        value = observation.iloc[0][raw_name]
        return value.item() if hasattr(value, "item") else value
    if feature.startswith("Type_"):
        return int(str(observation.iloc[0]["Type"]) == feature.removeprefix("Type_"))
    return float(transformed_value)


def local_shap_explanation(
    pipeline: Pipeline,
    background: pd.DataFrame,
    observation: pd.DataFrame,
) -> tuple[ShapResult, pd.DataFrame]:
    result = compute_shap_values(pipeline, background, observation)
    table = pd.DataFrame(
        {
            "feature": result.feature_names,
            "input_value": [
                _raw_value(name, observation, result.data[0, index])
                for index, name in enumerate(result.feature_names)
            ],
            "shap_value": result.values[0].astype(float),
        }
    )
    table["effect"] = np.where(
        table["shap_value"] >= 0,
        "Augmente le score de panne",
        "Réduit le score de panne",
    )
    table["absolute_contribution"] = table["shap_value"].abs()
    return result, table.sort_values("absolute_contribution", ascending=False).reset_index(drop=True)


def group_local_shap(
    pipeline: Pipeline, shap_table: pd.DataFrame, observation: pd.DataFrame
) -> pd.DataFrame:
    """Produit la vue métier SHAP en sommant les modalités one-hot."""
    preprocessor, _ = pipeline_parts(pipeline)
    transformed_names = get_transformed_feature_names(preprocessor)
    contributions = shap_table.set_index("feature")["shap_value"]
    rows = []
    for raw_name, indices in transformed_feature_groups(pipeline).items():
        value = observation.iloc[0][raw_name]
        if hasattr(value, "item"):
            value = value.item()
        contribution = float(
            sum(float(contributions[transformed_names[index]]) for index in indices)
        )
        rows.append(
            {
                "feature": friendly_name(raw_name),
                "input_value": value,
                "shap_value": contribution,
                "effect": (
                    "Augmente le score de panne"
                    if contribution >= 0 else "Réduit le score de panne"
                ),
                "absolute_contribution": abs(contribution),
            }
        )
    return pd.DataFrame(rows).sort_values(
        "absolute_contribution", ascending=False
    ).reset_index(drop=True)


def _lime_matrix(data: pd.DataFrame) -> tuple[np.ndarray, list[str], dict[int, list[str]]]:
    columns = list(data.columns)
    matrix = data.copy()
    categorical_names: dict[int, list[str]] = {}
    if "Type" in columns:
        type_index = columns.index("Type")
        categories = sorted(str(value) for value in data["Type"].dropna().unique())
        for expected in ("L", "M", "H"):
            if expected not in categories:
                categories.append(expected)
        mapping = {value: index for index, value in enumerate(categories)}
        matrix["Type"] = matrix["Type"].astype(str).map(mapping).astype(float)
        categorical_names[type_index] = categories
    return matrix.astype(float).to_numpy(), columns, categorical_names


def local_lime_explanation(
    pipeline: Pipeline,
    reference_data: pd.DataFrame,
    observation: pd.DataFrame,
    num_features: int = 8,
) -> pd.DataFrame:
    """Explique une observation brute via predict_proba du pipeline complet."""
    matrix, columns, categorical_names = _lime_matrix(reference_data)
    type_index = columns.index("Type") if "Type" in columns else None
    encoded_observation = []
    for index, column in enumerate(columns):
        value = observation.iloc[0][column]
        if index in categorical_names:
            encoded_observation.append(float(categorical_names[index].index(str(value))))
        else:
            encoded_observation.append(float(value))
    encoded_observation = np.asarray(encoded_observation, dtype=float)

    def predict(values: np.ndarray) -> np.ndarray:
        frame = pd.DataFrame(values, columns=columns)
        if type_index is not None:
            categories = categorical_names[type_index]
            indices = np.clip(np.rint(frame["Type"]).astype(int), 0, len(categories) - 1)
            frame["Type"] = [categories[index] for index in indices]
        return pipeline.predict_proba(frame)

    explainer = LimeTabularExplainer(
        matrix,
        feature_names=[friendly_name(column) for column in columns],
        class_names=["Pas de panne", "Panne"],
        categorical_features=list(categorical_names),
        categorical_names=categorical_names,
        mode="classification",
        discretize_continuous=True,
        random_state=42,
    )
    explanation = explainer.explain_instance(
        encoded_observation,
        predict,
        labels=(1,),
        num_features=min(num_features, len(columns)),
        num_samples=1000,
    )
    table = pd.DataFrame(
        explanation.as_list(label=1), columns=["condition_or_feature", "lime_weight"]
    )
    table["lime_weight"] = table["lime_weight"].astype(float)
    table["effect"] = np.where(
        table["lime_weight"] >= 0,
        "Augmente le score de panne",
        "Réduit le score de panne",
    )
    table["absolute_contribution"] = table["lime_weight"].abs()
    return table.sort_values("absolute_contribution", ascending=False).reset_index(drop=True)


def canonical_feature(name: str) -> str:
    if "Type" in name:
        return "Type"
    for raw, display in DISPLAY_NAMES.items():
        if raw in name or display in name:
            return display
    return name.split("__", 1)[-1]


def french_feature(name: str) -> str:
    return FRENCH_NAMES.get(canonical_feature(name), canonical_feature(name))


def french_join(values: list[str]) -> str:
    unique = list(dict.fromkeys(values))
    if len(unique) < 2:
        return "".join(unique)
    if len(unique) == 2:
        return " et ".join(unique)
    return ", ".join(unique[:-1]) + " et " + unique[-1]


def compare_shap_lime(
    shap_table: pd.DataFrame, lime_table: pd.DataFrame, top_n: int = 5
) -> dict[str, Any]:
    shap_features = list(
        dict.fromkeys(canonical_feature(value) for value in shap_table.head(top_n)["feature"])
    )[:top_n]
    lime_features = list(
        dict.fromkeys(
            canonical_feature(value) for value in lime_table.head(top_n)["condition_or_feature"]
        )
    )[:top_n]
    common = [feature for feature in shap_features if feature in lime_features]
    count = len(common)
    level = (
        "très élevé" if count == 5 else
        "élevé" if count == 4 else
        "modéré" if count == 3 else "limité"
    )
    return {
        "top_shap_features": shap_features,
        "top_lime_features": lime_features,
        "common_features": common,
        "shap_only_features": [value for value in shap_features if value not in common],
        "lime_only_features": [value for value in lime_features if value not in common],
        "agreement_count": count,
        "agreement_denominator": top_n,
        "agreement_level": level,
        "agreement_text": (
            f"Accord {level} : {count}/{top_n} variables principales "
            "sont communes aux deux méthodes."
        ),
    }


def human_readable_summary(
    prediction: int,
    score: float,
    threshold: float,
    contribution_table: pd.DataFrame,
) -> tuple[str, list[str], list[str]]:
    value_column = "shap_value" if "shap_value" in contribution_table else "lime_weight"
    name_column = "feature" if "feature" in contribution_table else "condition_or_feature"
    positive = contribution_table[contribution_table[value_column] > 0]
    negative = contribution_table[contribution_table[value_column] < 0]
    positive_names = list(
        dict.fromkeys(canonical_feature(value) for value in positive[name_column])
    )[:3]
    negative_names = list(
        dict.fromkeys(canonical_feature(value) for value in negative[name_column])
    )[:2]
    relation = "supérieur" if prediction else "inférieur"
    if abs(score - threshold) <= 0.05:
        relation = "légèrement " + relation
    decision = "une panne" if prediction else "une absence de panne"
    sentences = [
        f"Le modèle prédit {decision} avec un score de {score:.3f}, "
        f"{relation} au seuil de {threshold:.2f}."
    ]
    factors = []
    if positive_names:
        names = [french_feature(value) for value in positive_names]
        verb = "contribue" if len(names) == 1 else "contribuent"
        factors.append(
            f"{french_join(names).capitalize()} {verb} à augmenter ce score"
        )
    if negative_names:
        names = [french_feature(value) for value in negative_names]
        verb = "contribue" if len(names) == 1 else "contribuent"
        factors.append(
            f"{french_join(names).capitalize()} {verb} à le réduire"
        )
    if factors:
        sentences.append("; tandis que ".join(factors) + ".")
    return " ".join(sentences), positive_names, negative_names


def classification_outcome(prediction: int, truth: int | None) -> dict[str, str] | None:
    """Retourne TP/TN/FP/FN uniquement lorsqu'une vérité terrain existe."""
    if truth is None:
        return None
    labels = {
        (1, 1): ("Vrai positif (TP)", "La panne réelle a été correctement signalée."),
        (0, 0): ("Vrai négatif (TN)", "L'absence réelle de panne a été correctement reconnue."),
        (1, 0): (
            "Faux positif (FP)",
            "Une panne est signalée alors que l'observation réelle n'est pas en panne.",
        ),
        (0, 1): (
            "Faux négatif (FN)",
            "La panne réelle n'a pas été signalée par le modèle.",
        ),
    }
    label, explanation = labels[(int(prediction), int(truth))]
    return {"label": label, "explanation": explanation}


def local_explanation_key(
    model_identity: str,
    observation: pd.DataFrame | pd.Series | dict[str, Any],
    threshold: float,
) -> tuple[str, tuple[tuple[str, Any], ...], float]:
    """Clé stable liant une explication locale au modèle, à l'entrée et au seuil."""
    if isinstance(observation, pd.DataFrame):
        values = observation.iloc[0].to_dict()
    elif isinstance(observation, pd.Series):
        values = observation.to_dict()
    else:
        values = dict(observation)
    normalized = []
    for name in sorted(values):
        value = values[name]
        if hasattr(value, "item"):
            value = value.item()
        normalized.append((str(name), value))
    return str(model_identity), tuple(normalized), float(threshold)


def validate_prediction_explanation(
    prediction: dict[str, Any], explanation: dict[str, Any], atol: float = 1e-6
) -> None:
    """Empêche l'affichage d'une explication issue d'une autre décision."""
    if int(prediction["prediction"]) != int(explanation["prediction"]):
        raise ValueError("La décision expliquée ne correspond pas à la prédiction affichée.")
    if not np.isclose(
        float(prediction["probability"]), float(explanation["probability"]), atol=atol
    ):
        raise ValueError("Le score expliqué ne correspond pas à la prédiction affichée.")
    if not np.isclose(
        float(prediction["threshold"]), float(explanation["threshold"]), atol=atol
    ):
        raise ValueError("Le seuil expliqué ne correspond pas à la prédiction affichée.")


def resolve_session_model(
    selection: str,
    trained_models: dict[str, dict[str, Any]],
    thresholds: dict[str, float],
) -> dict[str, Any]:
    """Résout explicitement pipeline, données et seuil du modèle sélectionné."""
    if selection not in trained_models:
        raise KeyError(f"Modèle de session indisponible : {selection}")
    stored = trained_models[selection]
    return {
        "name": selection,
        "pipeline": stored["pipeline"],
        "x_test": stored["x_test"],
        "y_test": stored["y_test"],
        "threshold": float(thresholds.get(selection, 0.50)),
        "features": stored.get("features", list(stored["x_test"].columns)),
        "trained_at": stored.get("trained_at", "session"),
    }


def _shap_output_diagnostics(
    pipeline: Pipeline,
    observation: pd.DataFrame,
    result: ShapResult,
    score: float,
) -> dict[str, Any]:
    _, classifier = pipeline_parts(pipeline)
    transformed, _ = transform_frame(pipeline, observation)
    reconstructed = float(result.base_values[0] + result.values[0].sum())
    classifier_name = classifier.__class__.__name__
    raw_output: float
    probability_consistent = False
    scale = "sortie du modèle"
    if classifier_name == "XGBClassifier":
        raw_output = float(classifier.predict(transformed, output_margin=True)[0])
        probability_consistent = bool(
            np.isclose(1 / (1 + exp(-reconstructed)), score, atol=1e-4)
        )
        if probability_consistent:
            scale = "log-odds"
    elif hasattr(classifier, "decision_function"):
        raw_output = float(
            np.asarray(classifier.decision_function(transformed)).reshape(-1)[0]
        )
        probability_consistent = bool(
            np.isclose(1 / (1 + exp(-reconstructed)), score, atol=1e-4)
        )
        if probability_consistent:
            scale = "log-odds"
    else:
        raw_output = score
        probability_consistent = bool(np.isclose(reconstructed, score, atol=1e-4))
        if probability_consistent:
            scale = "probabilité"
    return {
        "shap_reconstructed_output": reconstructed,
        "shap_model_output": raw_output,
        "shap_additive_consistent": bool(
            np.isclose(reconstructed, raw_output, atol=1e-4)
        ),
        "shap_probability_consistent": probability_consistent,
        "shap_output_scale": scale,
    }


def local_interpretation(
    pipeline: Pipeline,
    reference_data: pd.DataFrame,
    observation: pd.DataFrame,
    threshold: float,
    enable_shap: bool = True,
) -> dict[str, Any]:
    score = float(pipeline.predict_proba(observation)[0, 1])
    prediction = int(score >= threshold)
    background = reference_data.sample(n=min(200, len(reference_data)), random_state=42)
    lime_table = local_lime_explanation(pipeline, background, observation)
    shap_result: ShapResult | None = None
    shap_table = pd.DataFrame()
    grouped_shap_table = pd.DataFrame()
    comparison = None
    diagnostics: dict[str, Any] = {}
    if enable_shap:
        shap_result, shap_table = local_shap_explanation(pipeline, background, observation)
        grouped_shap_table = group_local_shap(pipeline, shap_table, observation)
        comparison = compare_shap_lime(grouped_shap_table, lime_table)
        diagnostics = _shap_output_diagnostics(pipeline, observation, shap_result, score)
    summary_source = grouped_shap_table if not grouped_shap_table.empty else lime_table
    summary, positive, negative = human_readable_summary(
        prediction, score, threshold, summary_source
    )
    return {
        "prediction": prediction,
        "label": "Panne" if prediction else "Pas de panne",
        "probability": score,
        "threshold": float(threshold),
        "shap_available": enable_shap,
        "shap_base_value": float(shap_result.base_values[0]) if shap_result else None,
        "shap": shap_table,
        "shap_grouped": grouped_shap_table,
        "lime": lime_table,
        "top_positive_factors": positive,
        "top_negative_factors": negative,
        "comparison": comparison,
        "summary": summary,
        **diagnostics,
    }
