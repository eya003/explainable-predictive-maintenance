"""API FastAPI du système de maintenance prédictive."""

from __future__ import annotations

import logging
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Literal

import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, ConfigDict, Field

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
