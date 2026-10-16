"""API FastAPI du système de maintenance prédictive."""

from __future__ import annotations

import logging
from contextlib import asynccontextmanager
from functools import lru_cache
from pathlib import Path
from typing import Any, Literal

import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, ConfigDict, Field

from services.data_service import TARGET_COLUMN, fetch_ai4i_data
from services.interpretation_service import (
    global_shap_explanation,
    grouped_shap_matrix,
    local_interpretation,
    native_feature_importance,
    permutation_feature_importance,
    raw_to_transformed_mapping,
)
from services.training_service import split_dataset

LOGGER = logging.getLogger(__name__)
MODEL_PATH = Path(__file__).resolve().parent / "predictive_maintenance_model.pkl"
FEATURE_COLUMNS = [
    "Type",
    "Air temperature",
    "Process temperature",
    "Rotational speed",
    "Torque",
    "Tool wear",
]

model = None
threshold: float | None = None
model_error: str | None = None


def load_model() -> None:
    """Charge une seule fois l'artefact sauvegardé, sans entraîner de modèle."""
    global model, threshold, model_error

    if model is not None:
        return
    if not MODEL_PATH.is_file():
        model_error = f"Fichier modèle introuvable : {MODEL_PATH.name}"
        LOGGER.error(model_error)
        return

    try:
        saved_data = joblib.load(MODEL_PATH)
        if not isinstance(saved_data, dict) or "model" not in saved_data:
            raise ValueError("l'artefact doit contenir la clé 'model'")
        loaded_threshold = float(saved_data.get("threshold", 0.75))
        if not 0 <= loaded_threshold <= 1:
            raise ValueError("le seuil doit être compris entre 0 et 1")
        model = saved_data["model"]
        threshold = loaded_threshold
        model_error = None
    except Exception as exc:  # Le détail est journalisé, pas exposé au client.
        model = None
        threshold = None
        model_error = "Le modèle n'a pas pu être chargé. Vérifiez les versions des dépendances."
        LOGGER.exception("Échec du chargement du modèle : %s", exc)


@asynccontextmanager
async def lifespan(_: FastAPI):
    load_model()
    yield


app = FastAPI(
    title="Explainable Predictive Maintenance API",
    description="API de prédiction du risque de panne d'une machine industrielle.",
    version="1.0.0",
    lifespan=lifespan,
)


class MachineInput(BaseModel):
    """Mesures d'une machine à analyser."""

    model_config = ConfigDict(extra="forbid")

    Type: Literal["L", "M", "H"] = Field(description="Type de produit")
    air_temperature: float = Field(ge=250, le=400)
    process_temperature: float = Field(ge=250, le=450)
    rotational_speed: int = Field(ge=0, le=10000)
    torque: float = Field(ge=0, le=500)
    tool_wear: int = Field(ge=0, le=1000)


class PredictionOutput(BaseModel):
    prediction: int
    label: str
    failure_score: float
    threshold: float


class ContributionOutput(BaseModel):
    feature: str
    input_value: Any | None = None
    contribution: float
    effect: str


class LimeContributionOutput(BaseModel):
    feature: str
    contribution: float
    effect: str


class InterpretationOutput(BaseModel):
    prediction: int
    label: str
    probability: float
    threshold: float
    shap_available: bool = True
    shap_base_value: float | None = None
    shap: list[ContributionOutput]
    shap_grouped: list[ContributionOutput]
    lime: list[LimeContributionOutput]
    top_positive_factors: list[str]
    top_negative_factors: list[str]
    shap_lime_comparison: dict[str, Any]
    summary: str
    shap_reconstructed_output: float | None = None
    shap_model_output: float | None = None
    shap_additive_consistent: bool | None = None
    shap_probability_consistent: bool | None = None
    shap_output_scale: str | None = None


class GlobalInterpretationOutput(BaseModel):
    native_importance: list[dict[str, Any]]
    permutation_importance: list[dict[str, Any]]
    shap_importance: list[dict[str, Any]]
    feature_names: list[str]
    shap_values: list[list[float]]
    feature_values: list[list[float]]
    grouped_feature_names: list[str]
    grouped_shap_values: list[list[float]]
    grouped_feature_values: list[list[float]]
    raw_feature_names: list[str]
    raw_feature_values: list[list[Any]]
    raw_to_transformed: dict[str, int]
    sample_size: int


def machine_frame(payload: MachineInput) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "Type": payload.Type,
                "Air temperature": payload.air_temperature,
                "Process temperature": payload.process_temperature,
                "Rotational speed": payload.rotational_speed,
                "Torque": payload.torque,
                "Tool wear": payload.tool_wear,
            }
        ],
        columns=FEATURE_COLUMNS,
    )


@lru_cache(maxsize=1)
def reference_test_data() -> tuple[pd.DataFrame, pd.Series]:
    """Charge une fois le jeu de test stratifié utilisé comme référence XAI."""
    data = fetch_ai4i_data()
    _, x_test, _, y_test = split_dataset(
        data, FEATURE_COLUMNS, TARGET_COLUMN, test_size=0.20, random_state=42
    )
    return x_test, y_test


def _native_records(frame: pd.DataFrame) -> list[dict[str, Any]]:
    return [
        {
            key: (value.item() if hasattr(value, "item") else value)
            for key, value in row.items()
            if key != "absolute_contribution"
        }
        for row in frame.to_dict(orient="records")
    ]


@lru_cache(maxsize=1)
def cached_global_interpretation() -> dict[str, Any]:
    if model is None:
        raise RuntimeError("Modèle de référence indisponible.")
    x_test, y_test = reference_test_data()
    native = native_feature_importance(model)
    permutation = permutation_feature_importance(model, x_test, y_test)
    shap_result, shap_importance = global_shap_explanation(model, x_test, max_samples=300)
    grouped_values, grouped_data, grouped_names = grouped_shap_matrix(model, shap_result)
    return {
        "native_importance": _native_records(native),
        "permutation_importance": _native_records(permutation),
        "shap_importance": _native_records(shap_importance),
        "feature_names": shap_result.feature_names,
        "shap_values": shap_result.values.astype(float).tolist(),
        "feature_values": shap_result.data.astype(float).tolist(),
        "grouped_feature_names": grouped_names,
        "grouped_shap_values": grouped_values.astype(float).tolist(),
        "grouped_feature_values": grouped_data.astype(float).tolist(),
        "raw_feature_names": list(shap_result.raw_data.columns),
        "raw_feature_values": shap_result.raw_data.to_numpy().tolist(),
        "raw_to_transformed": raw_to_transformed_mapping(
            model, list(shap_result.raw_data.columns)
        ),
        "sample_size": int(len(shap_result.values)),
    }


@app.get("/")
def root() -> dict[str, str]:
    return {"message": "L'API de maintenance prédictive fonctionne."}


@app.get("/health")
def health() -> dict[str, object]:
    return {
        "status": "ok" if model is not None else "degraded",
        "model_loaded": model is not None,
        "threshold": threshold,
        **({"detail": model_error} if model_error else {}),
    }


@app.post("/predict", response_model=PredictionOutput)
def predict(payload: MachineInput) -> PredictionOutput:
    if model is None or threshold is None:
        raise HTTPException(status_code=503, detail=model_error or "Modèle indisponible.")

    machine = machine_frame(payload)

    try:
        failure_score = float(model.predict_proba(machine)[0, 1])
    except Exception as exc:
        LOGGER.exception("Erreur pendant la prédiction : %s", exc)
        raise HTTPException(
            status_code=500,
            detail="La prédiction a échoué. Vérifiez la compatibilité du modèle.",
        ) from exc

    prediction = int(failure_score >= threshold)
    return PredictionOutput(
        prediction=prediction,
        label="Panne" if prediction else "Pas de panne",
        failure_score=round(failure_score, 6),
        threshold=threshold,
    )


@app.post("/interpret", response_model=InterpretationOutput)
def interpret(payload: MachineInput) -> InterpretationOutput:
    """Explique localement la décision du XGBoost de référence avec SHAP et LIME."""
    if model is None or threshold is None:
        raise HTTPException(status_code=503, detail=model_error or "Modèle indisponible.")
    machine = machine_frame(payload)
    try:
        x_test, _ = reference_test_data()
        result = local_interpretation(model, x_test, machine, threshold)
    except Exception as exc:
        LOGGER.exception("Échec de l'interprétation locale : %s", exc)
        raise HTTPException(
            status_code=500,
            detail="L'interprétation n'a pas pu être calculée. Réessayez ultérieurement.",
        ) from exc

    return InterpretationOutput(
        prediction=result["prediction"],
        label=result["label"],
        probability=round(result["probability"], 6),
        threshold=result["threshold"],
        shap_available=result["shap_available"],
        shap_base_value=result["shap_base_value"],
        shap=[
            ContributionOutput(
                feature=str(row["feature"]),
                input_value=row["input_value"],
                contribution=float(row["shap_value"]),
                effect=str(row["effect"]),
            )
            for row in _native_records(result["shap"])
        ],
        shap_grouped=[
            ContributionOutput(
                feature=str(row["feature"]),
                input_value=row["input_value"],
                contribution=float(row["shap_value"]),
                effect=str(row["effect"]),
            )
            for row in _native_records(result["shap_grouped"])
        ],
        lime=[
            LimeContributionOutput(
                feature=str(row["condition_or_feature"]),
                contribution=float(row["lime_weight"]),
                effect=str(row["effect"]),
            )
            for row in _native_records(result["lime"])
        ],
        top_positive_factors=result["top_positive_factors"],
        top_negative_factors=result["top_negative_factors"],
        shap_lime_comparison=result["comparison"],
        summary=result["summary"],
        shap_reconstructed_output=result.get("shap_reconstructed_output"),
        shap_model_output=result.get("shap_model_output"),
        shap_additive_consistent=result.get("shap_additive_consistent"),
        shap_probability_consistent=result.get("shap_probability_consistent"),
        shap_output_scale=result.get("shap_output_scale"),
    )


@app.get("/interpret/global", response_model=GlobalInterpretationOutput)
def interpret_global() -> GlobalInterpretationOutput:
    """Retourne les explications globales mises en cache du modèle de référence."""
    if model is None:
        raise HTTPException(status_code=503, detail=model_error or "Modèle indisponible.")
    try:
        return GlobalInterpretationOutput(**cached_global_interpretation())
    except Exception as exc:
        LOGGER.exception("Échec de l'interprétation globale : %s", exc)
        raise HTTPException(
            status_code=500,
            detail="L'interprétation globale n'a pas pu être calculée.",
        ) from exc
