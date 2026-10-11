"""Chargement et préparation du jeu de données AI4I 2020."""

from __future__ import annotations

import pandas as pd
from ucimlrepo import fetch_ucirepo


TARGET_COLUMN = "Machine failure"
MODEL_FEATURES = [
    "Type",
    "Air temperature",
    "Process temperature",
    "Rotational speed",
    "Torque",
    "Tool wear",
]


def fetch_ai4i_data() -> pd.DataFrame:
    """Télécharge AI4I 2020 depuis UCI et retourne features + cible."""
    dataset = fetch_ucirepo(id=601)
    features = dataset.data.features.copy()
    targets = dataset.data.targets.copy()
    if TARGET_COLUMN not in targets.columns:
        raise ValueError(f"La cible '{TARGET_COLUMN}' est absente des données UCI.")
    target = targets[TARGET_COLUMN].astype(int).rename(TARGET_COLUMN)
    return pd.concat([features.reset_index(drop=True), target.reset_index(drop=True)], axis=1)


def model_frame(data: pd.DataFrame) -> pd.DataFrame:
    """Sélectionne dans l'ordre les colonnes attendues par le pipeline."""
    missing = [column for column in MODEL_FEATURES if column not in data.columns]
    if missing:
        raise ValueError("Colonnes requises absentes : " + ", ".join(missing))
    return data.loc[:, MODEL_FEATURES].copy()
