"""Dashboard Streamlit du projet Explainable Predictive Maintenance."""

from __future__ import annotations

import os
from datetime import datetime
from typing import Any

import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st

from services.api_client import (
    APIClientError,
    request_global_interpretation,
    request_interpretation,
    request_prediction,
)
from services.data_service import MODEL_FEATURES, TARGET_COLUMN, fetch_ai4i_data
from services.evaluation_service import evaluate_pipeline, metrics_at_threshold
from services.interpretation_service import (
    classification_outcome,
    global_shap_explanation,
    grouped_shap_matrix,
    friendly_name,
    local_interpretation,
    local_explanation_key,
    native_feature_importance,
    permutation_feature_importance,
    pipeline_parts,
    raw_to_transformed_mapping,
    resolve_session_model,
    validate_prediction_explanation,
)
from services.training_service import (
    MODEL_NAMES,
    RECOMMENDED_PARAMS,
    build_session_model_record,
    recommended_params_for_training,
    resolve_training_params,
    split_dataset,
    train_model,
)
from utils.plots import (
    COLORS,
    confusion_matrix_figure,
    distribution_figure,
    precision_recall_curves_figure,
    roc_curves_figure,
    importance_bar_figure,
    lime_contribution_figure,
    shap_beeswarm_figure,
    shap_dependence_figure,
    shap_waterfall_figure,
)


st.set_page_config(
    page_title="Predictive Maintenance",
    page_icon="⚙️",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
    .block-container {padding-top: 2rem; padding-bottom: 3rem;}
    [data-testid="stMetric"] {background:#17212f; border:1px solid #334155;
      padding:1rem; border-radius:.65rem;}
    [data-testid="stMetricLabel"] {color:#cbd5e1 !important;}
    [data-testid="stMetricValue"] {color:#f8fafc !important;}
    [data-testid="stMetricDelta"] {color:#cbd5e1 !important;}
    div[data-baseweb="select"] > div, div[data-baseweb="input"] > div {
      background-color:#111827; border-color:#475569; color:#f8fafc;
    }
    [data-testid="stSlider"] p, [data-testid="stTabs"] button {color:#e2e8f0;}
    .stButton > button, .stFormSubmitButton > button {border-color:#64748b;}
    .hero {padding:2rem; border-radius:.8rem; color:white;
      background:linear-gradient(120deg,#17324d,#315b7d); margin-bottom:1.5rem;}
    .muted {color:#cbd5e1;}
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_data(show_spinner="Téléchargement du dataset AI4I 2020…", ttl=3600)
def load_data() -> pd.DataFrame:
    return fetch_ai4i_data()


@st.cache_data(show_spinner=False, ttl=3600)
def load_global_xai(api_url: str) -> dict[str, Any]:
    return request_global_interpretation(api_url)


def initialize_state() -> None:
    defaults = {
        "trained_models": {},
        "selected_features": MODEL_FEATURES.copy(),
        "target_column": TARGET_COLUMN,
        "test_size": 0.20,
        "xai_local_result": None,
        "xai_local_results": {},
        "xai_global_results": {},
        "prediction_result": None,
        "prediction_local_explanations": {},
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


def safe_data() -> pd.DataFrame | None:
    try:
        return load_data()
    except Exception:
        st.error(
            "Le dataset UCI n'a pas pu être chargé. Vérifiez la connexion Internet "
            "et l'installation de `ucimlrepo`, puis réessayez."
        )
        return None


def page_title(title: str, subtitle: str | None = None) -> None:
    st.title(title)
    if subtitle:
        st.caption(subtitle)


def home_page() -> None:
    data = safe_data()
    if data is None:
        return
    failures = int(data[TARGET_COLUMN].sum())
    failure_rate = failures / len(data)
    st.markdown(
        """
        <div class="hero">
          <h1>Explainable Predictive Maintenance System</h1>
        </div>
        """,
        unsafe_allow_html=True,
    )
    columns = st.columns(4)
    columns[0].metric("Dataset", "AI4I 2020")
    columns[1].metric("Observations", f"{len(data):,}".replace(",", " "))
    columns[2].metric("Variables explicatives", len(MODEL_FEATURES))
    columns[3].metric("Pannes", f"{failures} ({failure_rate:.2%})")
    summary = (
        f"{len(data):,} observations, {len(MODEL_FEATURES)} variables explicatives et "
        f"{failures} pannes ({failure_rate:.2%})."
    ).replace(",", " ")
    st.write(summary)


def data_page() -> None:
    page_title("Données")
    data = safe_data()
    if data is None:
        return
    failures = int(data[TARGET_COLUMN].sum())
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Lignes", f"{data.shape[0]:,}".replace(",", " "))
    c2.metric("Colonnes", data.shape[1])
    c3.metric("Valeurs manquantes", int(data.isna().sum().sum()))
    c4.metric("Doublons", int(data.duplicated().sum()))
    overview, stats, quality = st.tabs(["Aperçu", "Statistiques", "Qualité"])
    with overview:
        st.dataframe(data.head(20), use_container_width=True)
    with stats:
        st.dataframe(data.describe(include="all").transpose(), use_container_width=True)
    with quality:
        types = pd.DataFrame(
            {"Type": data.dtypes.astype(str), "Valeurs manquantes": data.isna().sum()}
        )
        st.dataframe(types, use_container_width=True)
    left, right = st.columns([1.35, 1])
    with left:
        st.plotly_chart(distribution_figure(data), use_container_width=True)
    with right:
        total = len(data)
        st.metric("Pas de panne", f"{(total - failures) / total:.2%}")
        st.metric("Panne", f"{failures / total:.2%}")
        st.write(f"{failures} pannes sur {total:,} observations, soit {failures / total:.2%}.".replace(",", " "))


def exploration_page() -> None:
    page_title("Exploration")
    data = safe_data()
    if data is None:
        return
    plot_data = data.copy()
    plot_data["Statut"] = plot_data[TARGET_COLUMN].map({0: "Pas de panne", 1: "Panne"})
    numeric_features = [column for column in MODEL_FEATURES if column != "Type"]
    feature = st.selectbox("Variable numérique", numeric_features)
    figure = px.box(
        plot_data, x="Statut", y=feature, color="Statut",
        color_discrete_map=COLORS, points="outliers",
        title=f"{feature} selon l'état de la machine",
    )
    st.plotly_chart(figure, use_container_width=True)
    no_failure = data.loc[data[TARGET_COLUMN] == 0, feature]
    failure = data.loc[data[TARGET_COLUMN] == 1, feature]
    no_q1, no_q3 = no_failure.quantile([0.25, 0.75])
    failure_q1, failure_q3 = failure.quantile([0.25, 0.75])
    overlap = max(no_q1, failure_q1) <= min(no_q3, failure_q3)
    st.write(
        f"Médiane sans panne : {no_failure.median():.2f} · avec panne : "
        f"{failure.median():.2f}. Les intervalles interquartiles "
        f"{'se chevauchent' if overlap else 'ne se chevauchent pas'}."
    )
    left, right = st.columns(2)
    with left:
        rates = (
            data.groupby("Type", as_index=False)[TARGET_COLUMN].mean()
            .rename(columns={TARGET_COLUMN: "Taux de panne"})
        )
        type_figure = px.bar(
            rates, x="Type", y="Taux de panne", text_auto=".2%",
            title="Taux de panne observé selon Type", color="Taux de panne",
            color_continuous_scale="Blues",
        )
        type_figure.update_yaxes(tickformat=".1%")
        st.plotly_chart(type_figure, use_container_width=True)
    with right:
        corr_columns = numeric_features + [TARGET_COLUMN]
        corr = data[corr_columns].corr(numeric_only=True)
        corr_figure = px.imshow(
            corr, text_auto=".2f", color_continuous_scale="RdBu_r", zmin=-1, zmax=1,
            title="Matrice de corrélation numérique",
        )
        st.plotly_chart(corr_figure, use_container_width=True)


def preparation_page() -> None:
    page_title("Préparation")
    data = safe_data()
    if data is None:
        return
    target = st.selectbox("Variable cible", [TARGET_COLUMN], index=0)
    features = st.multiselect(
        "Variables explicatives",
        MODEL_FEATURES,
        default=st.session_state.selected_features,
        placeholder="Sélectionnez une ou plusieurs variables",
    )
    test_size = st.slider(
        "Part du jeu de test", min_value=0.10, max_value=0.40,
        value=float(st.session_state.test_size), step=0.05, format="%.2f",
    )
    if not features:
        st.warning("Sélectionnez au moins une variable explicative.")
        return
    if st.button("Enregistrer la configuration", type="primary"):
        st.session_state.selected_features = features
        st.session_state.target_column = target
        st.session_state.test_size = test_size
        st.success("Configuration enregistrée. Les modèles existants ne sont pas modifiés.")
    train_count = int(round(len(data) * (1 - test_size)))
    c1, c2, c3 = st.columns(3)
    c1.metric("Apprentissage", train_count)
    c2.metric("Test", len(data) - train_count)
    c3.metric("Variables", len(features))
    split_summary = (
        f"{train_count:,} observations pour l'apprentissage et "
        f"{len(data) - train_count:,} pour le test · test_size={test_size:.2f} · "
        "split stratifié · random_state=42"
    ).replace(",", " ")
    st.write(split_summary)


def _depth_control(label: str, key: str, default: int | None) -> int | None:
    unlimited = st.checkbox(
        f"{label} illimitée", value=default is None, key=f"{key}_unlimited"
    )
    if unlimited:
        return None
    return st.slider(label, 1, 30, default or 8, key=key)


def _parameter_controls(name: str) -> dict[str, Any]:
    prefix = name.replace(" ", "_")
    if name == "Logistic Regression":
        return {
            "C": st.number_input("C", min_value=0.001, max_value=100.0, value=1.0, key=f"{prefix}_C"),
            "solver": st.selectbox("Algorithme d'optimisation", ["lbfgs", "liblinear", "saga"], key=f"{prefix}_solver"),
            "max_iter": st.number_input("Nombre maximal d'itérations", min_value=1, max_value=10000, value=1000, step=100, key=f"{prefix}_iterations"),
            "class_weight": st.selectbox("Pondération des classes", ["balanced", None], key=f"{prefix}_cw"),
        }
    if name == "Decision Tree":
        return {
            "max_depth": _depth_control("Profondeur maximale", f"{prefix}_depth", 8),
            "min_samples_split": st.number_input("Minimum pour diviser", min_value=2, max_value=100, value=2, key=f"{prefix}_split"),
            "min_samples_leaf": st.number_input("Minimum par feuille", min_value=1, max_value=100, value=1, key=f"{prefix}_leaf"),
            "criterion": st.selectbox("Critère", ["gini", "entropy", "log_loss"], key=f"{prefix}_criterion"),
            "class_weight": st.selectbox("Pondération des classes", ["balanced", None], key=f"{prefix}_cw"),
        }
    if name == "KNN":
        return {
            "n_neighbors": st.number_input("Nombre de voisins", min_value=1, max_value=200, value=5, key=f"{prefix}_neighbors"),
            "weights": st.selectbox("Pondération", ["uniform", "distance"], key=f"{prefix}_weights"),
            "p": st.selectbox("Distance p", [1, 2], index=1, key=f"{prefix}_p"),
        }
    if name == "Random Forest":
        return {
            "n_estimators": st.number_input("Nombre d'arbres", min_value=10, max_value=1000, value=100, step=10, key=f"{prefix}_trees"),
            "max_depth": _depth_control("Profondeur maximale", f"{prefix}_depth", 12),
            "min_samples_split": st.number_input("Minimum pour diviser", min_value=2, max_value=100, value=2, key=f"{prefix}_split"),
            "min_samples_leaf": st.number_input("Minimum par feuille", min_value=1, max_value=100, value=1, key=f"{prefix}_leaf"),
            "max_features": st.selectbox("Variables par division", ["sqrt", "log2", None], key=f"{prefix}_features"),
            "criterion": st.selectbox("Critère", ["gini", "entropy", "log_loss"], key=f"{prefix}_criterion"),
            "class_weight": st.selectbox("Pondération des classes", ["balanced", None], key=f"{prefix}_cw"),
        }
    scale_mode = st.radio(
        "scale_pos_weight", ["Automatique", "Manuel"], horizontal=True,
        help="Automatique = nombre de machines sans panne / nombre de pannes dans le train.",
        key=f"{prefix}_spw_mode",
    )
    params = {
        "n_estimators": st.number_input("Nombre d'arbres", min_value=1, max_value=1000, value=200, step=10, key=f"{prefix}_trees"),
        "max_depth": st.number_input("Profondeur maximale", min_value=0, max_value=30, value=4, key=f"{prefix}_depth"),
        "learning_rate": st.number_input("Taux d'apprentissage", min_value=0.001, max_value=2.0, value=0.1, step=0.01, key=f"{prefix}_rate"),
        "subsample": st.number_input("Sous-échantillonnage", min_value=0.01, max_value=1.0, value=0.8, step=0.05, key=f"{prefix}_sample"),
        "colsample_bytree": st.number_input("Part des variables par arbre", min_value=0.01, max_value=1.0, value=1.0, step=0.05, key=f"{prefix}_columns"),
    }
    params["scale_pos_weight"] = (
        "automatic" if scale_mode == "Automatique"
        else st.number_input("Poids manuel de la classe positive", min_value=0.001, max_value=1000.0, value=1.0, key=f"{prefix}_spw")
    )
    return params


def modeling_page() -> None:
    page_title("Modélisation")
    data = safe_data()
    if data is None:
        return
    features = st.session_state.selected_features
    st.caption(
        f"Configuration active : {len(features)} variable(s), test_size={st.session_state.test_size:.2f}."
    )
    selected = st.multiselect(
        "Modèles à entraîner", MODEL_NAMES,
        placeholder="Sélectionnez un ou plusieurs modèles",
    )
    mode = st.radio(
        "Mode de configuration",
        ["Paramètres optimisés recommandés", "Configuration manuelle"],
        horizontal=True,
    )
    recommended = mode == "Paramètres optimisés recommandés"
    parameters: dict[str, dict[str, Any]] = {}
    for name in selected:
        with st.expander(f"⚙ {name}", expanded=True):
            if recommended:
                parameters[name] = RECOMMENDED_PARAMS[name].copy()
                st.json(parameters[name])
            else:
                parameters[name] = _parameter_controls(name)
    if selected:
        st.subheader("Configuration sélectionnée")
        for name in selected:
            st.markdown(f"**{name}**")
            st.json(parameters[name], expanded=False)
    if st.button("Entraîner les modèles sélectionnés", type="primary", disabled=not selected):
        try:
            x_train, x_test, y_train, y_test = split_dataset(
                data, features, st.session_state.target_column,
                st.session_state.test_size,
            )
        except Exception:
            st.error("Impossible de préparer les données avec cette configuration.")
            return
        configuration = (
            tuple(features), st.session_state.target_column,
            float(st.session_state.test_size), 42,
        )
        stale = [
            name for name, result in st.session_state.trained_models.items()
            if result.get("configuration") != configuration
        ]
        for name in stale:
            del st.session_state.trained_models[name]
        if stale:
            st.info(
                "Les modèles d'une ancienne configuration ont été retirés de la comparaison : "
                + ", ".join(stale)
            )
        progress = st.progress(0, text="Initialisation de l'entraînement…")
        successes = 0
        for index, name in enumerate(selected, start=1):
            progress.progress((index - 1) / len(selected), text=f"Entraînement : {name}")
            try:
                requested_params = (
                    recommended_params_for_training(name, y_train)
                    if recommended else parameters[name]
                )
                effective_params = resolve_training_params(name, requested_params, y_train)
                pipeline = train_model(name, effective_params, features, x_train, y_train)
                evaluation = evaluate_pipeline(pipeline, x_test, y_test)
                st.session_state.trained_models[name] = build_session_model_record(
                    model_name=name, pipeline=pipeline, params=effective_params,
                    metrics=evaluation, features=features,
                    test_size=st.session_state.test_size, x_test=x_test, y_test=y_test,
                    configuration=configuration,
                    target_column=st.session_state.target_column,
                    trained_at=datetime.now().isoformat(timespec="seconds"),
                )
                successes += 1
            except ValueError as exc:
                st.error(f"Configuration invalide pour {name} : {exc}")
            except Exception:
                st.error(f"L'entraînement de {name} a échoué. Vérifiez ses paramètres.")
        progress.progress(1.0, text="Entraînement terminé")
        if successes:
            st.success(f"{successes} modèle(s) entraîné(s) et conservé(s) dans la session.")
    trained = list(st.session_state.trained_models)
    st.info(
        "Modèles disponibles dans la session : " + (", ".join(trained) if trained else "aucun")
    )


def comparison_page() -> None:
    page_title("Comparaison")
    results = st.session_state.trained_models
    if not results:
        st.warning("Entraînez au moins un modèle depuis la page Modélisation.")
        return
    rows = []
    for name, result in results.items():
        metrics = result["metrics"]
        rows.append(
            {"Modèle": name, "Précision panne": metrics["precision"], "Rappel panne": metrics["recall"],
             "F1 panne": metrics["f1"], "ROC-AUC": metrics["roc_auc"],
             "PR-AUC (Average Precision)": metrics["pr_auc"]}
        )
    comparison = pd.DataFrame(rows).set_index("Modèle")
    st.dataframe(comparison.style.format("{:.3f}"), use_container_width=True)
    best_f1 = comparison["F1 panne"].idxmax()
    best_recall = comparison["Rappel panne"].idxmax()
    best_pr_auc = comparison["PR-AUC (Average Precision)"].idxmax()
    st.write(
        f"Meilleur F1-score : {best_f1} ({comparison.loc[best_f1, 'F1 panne']:.3f}) · "
        f"meilleur rappel : {best_recall} ({comparison.loc[best_recall, 'Rappel panne']:.3f}) · "
        f"meilleure PR-AUC : {best_pr_auc} "
        f"({comparison.loc[best_pr_auc, 'PR-AUC (Average Precision)']:.3f})."
    )
    with st.expander("Voir les hyperparamètres"):
        for name, result in results.items():
            st.markdown(f"**{name}**")
            st.json(result.get("params", {}), expanded=False)
    metric = st.selectbox("Métrique du graphique", comparison.columns.tolist())
    bar = px.bar(
        comparison.reset_index(), x="Modèle", y=metric, color="Modèle", text_auto=".3f",
        title=f"Comparaison — {metric}",
    )
    bar.update_layout(showlegend=False)
    st.plotly_chart(bar, use_container_width=True)
    left, right = st.columns(2)
    with left:
        st.plotly_chart(roc_curves_figure(results), use_container_width=True)
    with right:
        st.plotly_chart(precision_recall_curves_figure(results), use_container_width=True)
    st.subheader("Matrices de confusion au seuil 0,50")
    matrix_columns = st.columns(min(2, len(results)))
    for index, (name, result) in enumerate(results.items()):
        with matrix_columns[index % len(matrix_columns)]:
            st.plotly_chart(
                confusion_matrix_figure(result["metrics"]["confusion_matrix"], name),
                use_container_width=True,
            )


def evaluation_page() -> None:
    page_title("Évaluation")
    results = st.session_state.trained_models
    if not results:
        st.warning("Aucun modèle entraîné n'est disponible dans cette session.")
        return
    name = st.selectbox("Modèle", list(results))
    result = results[name]
    threshold = st.slider(
        "Seuil de décision", 0.0, 1.0,
        float(result.get("threshold", 0.50)), 0.01, key=f"threshold_{name}"
    )
    result["threshold"] = float(threshold)
    metrics = metrics_at_threshold(result["y_test"], result["metrics"]["scores"], threshold)
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Précision panne", f"{metrics['precision']:.3f}")
    c2.metric("Rappel panne", f"{metrics['recall']:.3f}")
    c3.metric("F1-score", f"{metrics['f1']:.3f}")
    c4.metric("Seuil", f"{threshold:.2f}")
    st.write(
        f"Au seuil {threshold:.2f} : précision {metrics['precision']:.1%}, "
        f"rappel {metrics['recall']:.1%}, F1-score {metrics['f1']:.1%}."
    )
    st.plotly_chart(
        confusion_matrix_figure(metrics["confusion_matrix"], f"{name} — seuil {threshold:.2f}"),
        use_container_width=True,
    )


def prediction_page() -> None:
    page_title("Prédiction")
    trained = st.session_state.trained_models
    choices = ["Modèle de référence pré-entraîné (API)"] + list(trained)
    choice = st.selectbox("Modèle utilisé", choices)
    is_reference = choice == choices[0]
    if is_reference:
        threshold = 0.75
        model_identity = "reference::xgboost"
    else:
        threshold = st.slider(
            "Seuil de décision", 0.0, 1.0,
            float(trained[choice].get("threshold", 0.50)), 0.01,
            key=f"prediction_threshold_{choice}"
        )
        trained[choice]["threshold"] = float(threshold)
        model_identity = f"session::{choice}::{trained[choice].get('trained_at', 'session')}"
    with st.form("machine_form"):
        c1, c2 = st.columns(2)
        machine_type = c1.selectbox("Type", ["L", "M", "H"])
        air = c1.number_input("Air temperature", 250.0, 400.0, 300.0, 0.1)
        process = c1.number_input("Process temperature", 250.0, 450.0, 310.0, 0.1)
        speed = c2.number_input("Rotational speed", 0, 10000, 1500, 10)
        torque = c2.number_input("Torque", 0.0, 500.0, 40.0, 0.1)
        wear = c2.number_input("Tool wear", 0, 1000, 100, 1)
        submitted = st.form_submit_button("Analyser la machine", type="primary")
    machine = pd.DataFrame([{
        "Type": machine_type, "Air temperature": air,
        "Process temperature": process, "Rotational speed": speed,
        "Torque": torque, "Tool wear": wear,
    }])
    payload = _machine_payload(machine.iloc[0])
    current_key = local_explanation_key(model_identity, machine, float(threshold))
    if submitted:
        try:
            if is_reference:
                response = request_prediction(
                    os.getenv("FASTAPI_URL", "http://127.0.0.1:8000"), payload
                )
                score = float(response["failure_score"])
                decision = int(response["prediction"])
                used_threshold = float(response["threshold"])
            else:
                stored = trained[choice]
                score = float(
                    stored["pipeline"].predict_proba(machine[stored["features"]])[:, 1][0]
                )
                used_threshold = float(threshold)
                decision = int(score >= used_threshold)
        except APIClientError as exc:
            st.error(str(exc))
            return
        except Exception:
            st.error("La prédiction n'a pas pu être calculée avec ce modèle.")
            return
        current_key = local_explanation_key(model_identity, machine, used_threshold)
        st.session_state.prediction_result = {
            "key": current_key,
            "model_identity": model_identity,
            "model_name": "XGBoost de référence" if is_reference else choice,
            "reference": is_reference,
            "machine": machine,
            "payload": payload,
            "probability": score,
            "prediction": decision,
            "label": "Panne" if decision else "Pas de panne",
            "threshold": used_threshold,
        }
    prediction = st.session_state.get("prediction_result")
    if not prediction or prediction.get("key") != current_key:
        return
    score = float(prediction["probability"])
    decision = int(prediction["prediction"])
    used_threshold = float(prediction["threshold"])
    c1, c2, c3 = st.columns(3)
    st.caption(f"Modèle utilisé : **{choice}**")
    c1.metric("Décision", "Panne" if decision else "Pas de panne")
    c2.metric("Score de risque du modèle", f"{score:.3f}")
    c3.metric("Seuil utilisé", f"{used_threshold:.2f}")
    operator = ">=" if score >= used_threshold else "<"
    st.write(
        f"Score = {score:.3f} {operator} seuil = {used_threshold:.2f} → "
        f"{'Panne' if decision else 'Pas de panne'}"
    )
    if decision:
        st.error("Une panne est signalée : une vérification de la machine est recommandée.")
    else:
        st.success("Aucune panne n'est signalée pour les mesures saisies.")
    if st.button(
        "Expliquer cette prédiction", type="primary",
        key=f"explain_prediction_{abs(hash(current_key))}",
    ):
        with st.spinner("Calcul de l'explication locale SHAP/LIME…"):
            try:
                if prediction["reference"]:
                    explanation = request_interpretation(
                        os.getenv("FASTAPI_URL", "http://127.0.0.1:8000"),
                        prediction["payload"],
                    )
                else:
                    stored = trained[choice]
                    local = local_interpretation(
                        stored["pipeline"], stored["x_test"][stored["features"]],
                        prediction["machine"][stored["features"]], used_threshold,
                        enable_shap=choice != "KNN",
                    )
                    explanation = _serialize_local(local)
                validate_prediction_explanation(prediction, explanation)
                explanation.update({
                    "true_label": None,
                    "model_key": prediction["model_identity"],
                    "model_name": prediction["model_name"],
                })
                st.session_state.prediction_local_explanations[current_key] = explanation
            except (APIClientError, ValueError) as exc:
                st.error(str(exc))
                return
    explanation = st.session_state.prediction_local_explanations.get(current_key)
    if explanation:
        st.divider()
        st.header("Pourquoi cette prédiction ?")
        _render_local_xai(explanation, prediction["model_name"], show_metrics=False)
        _render_local_comparison(explanation)


def _machine_payload(machine: pd.Series | dict[str, Any]) -> dict[str, Any]:
    return {
        "Type": str(machine["Type"]),
        "air_temperature": float(machine["Air temperature"]),
        "process_temperature": float(machine["Process temperature"]),
        "rotational_speed": int(machine["Rotational speed"]),
        "torque": float(machine["Torque"]),
        "tool_wear": int(machine["Tool wear"]),
    }


REFERENCE_XAI = "XGBoost — modèle de référence"


def _xai_context(selection: str) -> dict[str, Any]:
    if selection == REFERENCE_XAI:
        return {"key": "reference", "name": "XGBoost", "reference": True, "threshold": 0.75}
    thresholds = {
        name: float(
            st.session_state.get(
                f"threshold_{name}", result.get("threshold", 0.50)
            )
        )
        for name, result in st.session_state.trained_models.items()
    }
    context = resolve_session_model(selection, st.session_state.trained_models, thresholds)
    context.update({"key": f"session::{selection}::{context['trained_at']}", "reference": False})
    return context


def _records(frame: pd.DataFrame) -> list[dict[str, Any]]:
    return [
        {key: value.item() if hasattr(value, "item") else value for key, value in row.items()}
        for row in frame.to_dict(orient="records")
    ]


def _session_global_xai(context: dict[str, Any]) -> dict[str, Any]:
    pipeline, x_test, y_test = context["pipeline"], context["x_test"], context["y_test"]
    try:
        native = _records(native_feature_importance(pipeline))
        native_message = None
    except ValueError:
        native = []
        native_message = "KNN ne possède pas d'importance native des variables."
    permutation = _records(permutation_feature_importance(pipeline, x_test, y_test))
    if context["name"] == "KNN":
        return {
            "native_importance": native, "native_message": native_message,
            "permutation_importance": permutation, "shap_available": False,
            "shap_message": (
                "SHAP est désactivé pour KNN : KernelExplainer est une méthode "
                "agnostique trop coûteuse pour ce calcul interactif."
            ),
        }
    shap_result, shap_importance = global_shap_explanation(pipeline, x_test, max_samples=300)
    grouped_values, grouped_data, grouped_names = grouped_shap_matrix(pipeline, shap_result)
    raw = shap_result.raw_data
    return {
        "native_importance": native, "native_message": native_message,
        "permutation_importance": permutation, "shap_available": True,
        "shap_importance": _records(shap_importance),
        "feature_names": shap_result.feature_names,
        "shap_values": shap_result.values.tolist(),
        "feature_values": shap_result.data.tolist(),
        "grouped_feature_names": grouped_names,
        "grouped_shap_values": grouped_values.tolist(),
        "grouped_feature_values": grouped_data.tolist(),
        "raw_feature_names": list(raw.columns),
        "raw_feature_values": raw.to_numpy().tolist(),
        "raw_to_transformed": raw_to_transformed_mapping(pipeline, list(raw.columns)),
    }


def _global_xai_tab(api_url: str, context: dict[str, Any]) -> None:
    model_label = "XGBoost de référence" if context["reference"] else context["name"]
    st.subheader(f"Analyse globale — {model_label}")
    if st.button("Calculer l'explication globale", type="primary", key=f"global_{context['key']}"):
        with st.spinner("Calcul des importances et des valeurs SHAP…"):
            try:
                result = load_global_xai(api_url) if context["reference"] else _session_global_xai(context)
                st.session_state.xai_global_results[context["key"]] = result
            except (APIClientError, ValueError) as exc:
                st.error(str(exc))
                return
    result = st.session_state.xai_global_results.get(context["key"])
    if not result:
        st.info("Lancez le calcul pour afficher les explications globales de ce modèle.")
        return
    permutation = pd.DataFrame(result["permutation_importance"])
    left, right = st.columns(2)
    with left:
        if result.get("native_importance"):
            native_frame = pd.DataFrame(result["native_importance"])
            native_value = "importance"
            native_title = (
                "Amplitude groupée des coefficients — Logistic Regression"
                if context["name"] == "Logistic Regression"
                else f"Importance native groupée — {context['name']}"
            )
            st.plotly_chart(
                importance_bar_figure(native_frame, native_value, native_title),
                use_container_width=True,
            )
        else:
            st.info(result.get("native_message", "Ce modèle ne fournit pas d'importance native."))
    with right:
        st.plotly_chart(
            importance_bar_figure(pd.DataFrame(result["permutation_importance"]), "importance_mean", "Importance par permutation — Average Precision", "importance_std"),
            use_container_width=True,
        )
    if not result.get("shap_available", True):
        if not permutation.empty:
            top_permutation = permutation.iloc[0]
            st.write(
                f"Importance par permutation maximale : {top_permutation['feature']} "
                f"({top_permutation['importance_mean']:.4f})."
            )
        st.info(result["shap_message"])
        return
    values = np.asarray(result["shap_values"], dtype=float)
    transformed = np.asarray(result["feature_values"], dtype=float)
    names = list(result["feature_names"])
    shap_importance = pd.DataFrame(result["shap_importance"])
    st.plotly_chart(importance_bar_figure(shap_importance, "mean_abs_shap", "Importance SHAP moyenne absolue"), use_container_width=True)
    top_permutation = permutation.iloc[0]
    top_shap = shap_importance.iloc[0]
    st.write(
        f"Permutation : {top_permutation['feature']} arrive en tête "
        f"({top_permutation['importance_mean']:.4f}) · SHAP moyen absolu : "
        f"{top_shap['feature']} ({top_shap['mean_abs_shap']:.4f})."
    )
    st.subheader("Résumé SHAP groupé — variables originales")
    grouped_values = np.asarray(result["grouped_shap_values"], dtype=float)
    grouped_data = np.asarray(result["grouped_feature_values"], dtype=float)
    grouped_names = list(result["grouped_feature_names"])
    st.pyplot(
        shap_beeswarm_figure(grouped_values, grouped_data, grouped_names),
        clear_figure=True,
    )
    raw_names = list(result["raw_feature_names"])
    numeric_names = [name for name in raw_names if name != "Type" and name in result["raw_to_transformed"]]
    if not numeric_names:
        st.info("Aucune variable numérique compatible avec le graphique de dépendance.")
        return
    default_x = numeric_names.index("Air temperature") if "Air temperature" in numeric_names else 0
    x_name = st.selectbox("Variable sur l'axe x", numeric_names, index=default_x, format_func=friendly_name, key=f"dep_x_{context['key']}")
    default_color = numeric_names.index("Torque") if "Torque" in numeric_names else 0
    color_name = st.selectbox("Variable de couleur", numeric_names, index=default_color, format_func=friendly_name, key=f"dep_color_{context['key']}")
    raw = pd.DataFrame(result["raw_feature_values"], columns=raw_names)
    shap_index = int(result["raw_to_transformed"][x_name])
    figure = shap_dependence_figure(
        x_values=raw[x_name].astype(float).to_numpy(),
        shap_values=values[:, shap_index],
        feature_name=friendly_name(x_name),
        color_values=(
            raw[color_name].astype(float).to_numpy() if color_name else None
        ),
        color_name=friendly_name(color_name) if color_name else None,
    )
    st.plotly_chart(
        figure,
        use_container_width=True,
    )


def _serialize_local(result: dict[str, Any]) -> dict[str, Any]:
    shap_rows = [] if result["shap"].empty else [
        {"feature": row["feature"], "input_value": row["input_value"], "contribution": float(row["shap_value"]), "effect": row["effect"]}
        for row in _records(result["shap"])
    ]
    lime_rows = [
        {"feature": row["condition_or_feature"], "contribution": float(row["lime_weight"]), "effect": row["effect"]}
        for row in _records(result["lime"])
    ]
    grouped_rows = [] if result["shap_grouped"].empty else [
        {"feature": row["feature"], "input_value": row["input_value"], "contribution": float(row["shap_value"]), "effect": row["effect"]}
        for row in _records(result["shap_grouped"])
    ]
    return {
        **{key: value for key, value in result.items() if key not in {"shap", "shap_grouped", "lime", "comparison"}},
        "shap": shap_rows, "shap_grouped": grouped_rows, "lime": lime_rows,
        "shap_lime_comparison": result["comparison"],
    }


def _render_local_xai(
    result: dict[str, Any], model_name: str, show_metrics: bool = True
) -> None:
    if show_metrics:
        c1, c2, c3 = st.columns(3)
        c1.metric("Décision", result["label"])
        c2.metric("Score du modèle", f"{result['probability']:.3f}")
        c3.metric("Seuil", f"{result['threshold']:.2f}")
    outcome = classification_outcome(result["prediction"], result.get("true_label"))
    if outcome:
        st.info(f"**{outcome['label']}** — {outcome['explanation']}")
    shap_table = pd.DataFrame(result["shap"])
    grouped_shap_table = pd.DataFrame(result.get("shap_grouped", []))
    lime_table = pd.DataFrame(result["lime"])
    grouped_shap_display = grouped_shap_table.copy()
    technical_shap_display = shap_table.copy()
    for display_table in (grouped_shap_display, technical_shap_display):
        if "input_value" in display_table:
            display_table["input_value"] = display_table["input_value"].map(str)
    if result.get("shap_available", True) and not grouped_shap_table.empty:
        st.subheader("Explication SHAP")
        st.caption(f"Modèle expliqué : {model_name}")
        st.pyplot(
            shap_waterfall_figure(
                grouped_shap_table["contribution"].to_numpy(float),
                float(result["shap_base_value"]),
                grouped_shap_table["input_value"].to_numpy(),
                grouped_shap_table["feature"].tolist(),
            ),
            clear_figure=True,
        )
        st.dataframe(grouped_shap_display, use_container_width=True, hide_index=True)
        factors_left, factors_right = st.columns(2)
        factors_left.markdown("**Facteurs qui augmentent le score**")
        factors_left.write(", ".join(result.get("top_positive_factors", [])) or "Aucun")
        factors_right.markdown("**Facteurs qui réduisent le score**")
        factors_right.write(", ".join(result.get("top_negative_factors", [])) or "Aucun")
        with st.expander("Détail technique des colonnes transformées"):
            st.dataframe(technical_shap_display, use_container_width=True, hide_index=True)
        if result.get("shap_additive_consistent"):
            scale = result.get("shap_output_scale", "sortie du modèle")
            st.caption(f"Additivité SHAP vérifiée sur l'échelle {scale}.")
        else:
            st.caption("Les contributions SHAP sont affichées sur l'échelle propre à l'explainer ; elles ne doivent pas être additionnées directement au score de risque.")
    else:
        st.info(
            "SHAP local n'est pas activé pour KNN en raison du coût de KernelSHAP. "
            "L'explication locale LIME reste disponible."
        )
    st.subheader("Explication LIME")
    lime_plot = lime_table.rename(columns={"feature": "condition_or_feature", "contribution": "lime_weight"})
    st.plotly_chart(lime_contribution_figure(lime_plot), use_container_width=True)
    st.dataframe(lime_table, use_container_width=True, hide_index=True)
    st.subheader("Explication simplifiée")
    st.info(result["summary"])


def _render_local_comparison(result: dict[str, Any]) -> None:
    comparison = result.get("shap_lime_comparison")
    if not comparison:
        return
    st.subheader("Cohérence SHAP / LIME")
    st.metric(
        "Accord Top-5",
        f"{comparison['agreement_count']}/{comparison['agreement_denominator']}",
    )
    st.write(comparison["agreement_text"])
    left, right = st.columns(2)
    with left:
        st.markdown("**Top 5 SHAP**")
        st.dataframe(pd.DataFrame({
            "Rang": range(1, len(comparison["top_shap_features"]) + 1),
            "Variable": comparison["top_shap_features"],
        }), hide_index=True)
    with right:
        st.markdown("**Top 5 LIME**")
        st.dataframe(pd.DataFrame({
            "Rang": range(1, len(comparison["top_lime_features"]) + 1),
            "Variable": comparison["top_lime_features"],
        }), hide_index=True)
    st.write(
        "**Variables communes :**",
        ", ".join(comparison["common_features"]) or "Aucune",
    )


def _local_xai_tab(api_url: str, context: dict[str, Any]) -> None:
    st.subheader(f"Explication locale — {context['name']}")
    data = safe_data()
    if data is None:
        return
    if context["reference"]:
        _, x_test, _, y_test = split_dataset(data, MODEL_FEATURES, TARGET_COLUMN, 0.20, 42)
        features = MODEL_FEATURES
    else:
        x_test, y_test, features = context["x_test"], context["y_test"], context["features"]
    source = st.radio("Machine à expliquer", ["Observation du jeu de test", "Saisie manuelle"], horizontal=True, key=f"source_{context['key']}")
    observation = None
    payload = None
    true_label = None
    submitted = False
    if source == "Observation du jeu de test":
        position = st.number_input("Numéro dans le jeu de test", 0, len(x_test) - 1, 0, 1, key=f"position_{context['key']}")
        observation = x_test.iloc[[int(position)]][features]
        true_label = int(y_test.iloc[int(position)])
        st.dataframe(observation, use_container_width=True)
        if context["reference"]:
            payload = _machine_payload(observation.iloc[0])
        submitted = st.button("Expliquer cette observation", type="primary", key=f"local_{context['key']}")
    else:
        with st.form(f"manual_{context['key']}"):
            left, right = st.columns(2)
            machine = {
                "Type": left.selectbox("Type", ["L", "M", "H"]),
                "Air temperature": left.number_input("Air temperature [K]", 250.0, 400.0, 300.0, 0.1),
                "Process temperature": left.number_input("Process temperature [K]", 250.0, 450.0, 310.0, 0.1),
                "Rotational speed": right.number_input("Rotational speed [rpm]", 0, 10000, 1500, 10),
                "Torque": right.number_input("Torque [Nm]", 0.0, 500.0, 40.0, 0.1),
                "Tool wear": right.number_input("Tool wear [min]", 0, 1000, 100, 1),
            }
            submitted = st.form_submit_button("Expliquer cette machine", type="primary")
        observation = pd.DataFrame([machine])[features]
        payload = _machine_payload(machine)
    if submitted:
        with st.spinner("Calcul de l'explication pour le modèle sélectionné…"):
            try:
                if context["reference"]:
                    result = request_interpretation(api_url, payload)
                else:
                    enable_shap = context["name"] != "KNN"
                    result = _serialize_local(local_interpretation(context["pipeline"], x_test[features], observation, context["threshold"], enable_shap=enable_shap))
                result.update({"true_label": true_label, "model_key": context["key"], "model_name": context["name"]})
                st.session_state.xai_local_results[context["key"]] = result
            except (APIClientError, ValueError) as exc:
                st.error(str(exc))
                return
    result = st.session_state.xai_local_results.get(context["key"])
    if result:
        _render_local_xai(result, context["name"])
    else:
        st.info("Sélectionnez ou saisissez une machine, puis lancez son explication.")


def _comparison_xai_tab(context: dict[str, Any]) -> None:
    st.subheader(f"SHAP vs LIME — {context['name']}")
    result = st.session_state.xai_local_results.get(context["key"])
    if not result:
        st.info("Calculez d'abord une explication locale pour ce modèle.")
        return
    if not result.get("shap_lime_comparison"):
        st.info("Comparaison indisponible pour KNN puisque SHAP est désactivé.")
        return
    _render_local_comparison(result)


def explainable_ai_page() -> None:
    page_title("Explainable AI")
    choices = [REFERENCE_XAI] + list(st.session_state.trained_models)
    selection = st.selectbox("Modèle à expliquer", choices, key="xai_model_selection")
    context = _xai_context(selection)
    st.caption(f"Modèle actif : **{context['name']}** · seuil de décision : **{context['threshold']:.2f}**")
    api_url = os.getenv("FASTAPI_URL", "http://127.0.0.1:8000")
    _global_xai_tab(api_url, context)


initialize_state()
PAGES = {
    "Accueil": home_page,
    "Données": data_page,
    "Exploration": exploration_page,
    "Préparation": preparation_page,
    "Modélisation": modeling_page,
    "Comparaison": comparison_page,
    "Évaluation": evaluation_page,
    "Prédiction": prediction_page,
    "Explainable AI": explainable_ai_page,
}
with st.sidebar:
    st.title("Maintenance prédictive")
    page = st.radio("Navigation", list(PAGES), label_visibility="collapsed")
    st.divider()
    count = len(st.session_state.trained_models)
    st.metric("Modèles en session", count)
    st.caption("Référence API : XGBoost pré-entraîné · seuil 0,75")

PAGES[page]()
