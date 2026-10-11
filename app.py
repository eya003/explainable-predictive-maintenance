"""Dashboard Streamlit du projet Explainable Predictive Maintenance."""

from __future__ import annotations

import os
from datetime import datetime
from typing import Any

import pandas as pd
import plotly.express as px
import streamlit as st

from services.api_client import APIClientError, request_prediction
from services.data_service import MODEL_FEATURES, TARGET_COLUMN, fetch_ai4i_data
from services.evaluation_service import evaluate_pipeline, metrics_at_threshold
from services.training_service import (
    MODEL_NAMES,
    RECOMMENDED_PARAMS,
    recommended_params_for_training,
    split_dataset,
    train_model,
)
from utils.plots import (
    COLORS,
    confusion_matrix_figure,
    distribution_figure,
    precision_recall_curves_figure,
    roc_curves_figure,
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


def initialize_state() -> None:
    defaults = {
        "trained_models": {},
        "selected_features": MODEL_FEATURES.copy(),
        "target_column": TARGET_COLUMN,
        "test_size": 0.20,
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


def page_title(title: str, subtitle: str) -> None:
    st.title(title)
    st.caption(subtitle)


def home_page() -> None:
    st.markdown(
        """
        <div class="hero">
          <h1>Explainable Predictive Maintenance System</h1>
          <p>Anticiper le risque de panne à partir des conditions de fonctionnement
          d'une machine industrielle.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )
    columns = st.columns(4)
    columns[0].metric("Dataset", "AI4I 2020")
    columns[1].metric("Observations", "10 000")
    columns[2].metric("Variables candidates", "6")
    columns[3].metric("Taux de panne", "≈ 3,39 %")
    st.subheader("Une classification fortement déséquilibrée")
    st.write(
        "La classe positive — **Machine failure** — est rare. L'exactitude seule serait "
        "donc trompeuse : la précision, le rappel, le F1-score et la PR-AUC sont suivis "
        "pour comparer les modèles de façon pertinente."
    )
    st.info(
        "Le modèle XGBoost pré-entraîné constitue la référence de l'API. "
        "Le dashboard permet en parallèle d'entraîner cinq familles de modèles à la demande."
    )


def data_page() -> None:
    page_title("Données", "Chargement automatique depuis le dépôt UCI (identifiant 601).")
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
        st.caption("La faible fréquence des pannes justifie la stratification du découpage.")


def exploration_page() -> None:
    page_title("Exploration", "Relations descriptives entre mesures et état de la machine.")
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
    st.caption(
        "Les écarts observés décrivent une association dans cet échantillon ; "
        "ils ne démontrent pas un lien de causalité."
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
        st.caption("Le taux varie selon le type, sans permettre à lui seul une conclusion causale.")
    with right:
        corr_columns = numeric_features + [TARGET_COLUMN]
        corr = data[corr_columns].corr(numeric_only=True)
        corr_figure = px.imshow(
            corr, text_auto=".2f", color_continuous_scale="RdBu_r", zmin=-1, zmax=1,
            title="Matrice de corrélation numérique",
        )
        st.plotly_chart(corr_figure, use_container_width=True)
        st.caption("La corrélation mesure une association linéaire, pas une causalité.")


def preparation_page() -> None:
    page_title("Préparation", "Configuration du jeu d'apprentissage et du jeu de test.")
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
    st.write(
        "Le découpage utilise `stratify=y` et un état aléatoire fixe. "
        "L'encodage et la standardisation sont ajustés uniquement sur l'apprentissage, dans un pipeline."
    )


def _depth_control(label: str, key: str, default: int | None) -> int | None:
    unlimited = st.checkbox(
        f"{label} illimitée", value=default is None, key=f"{key}_unlimited"
    )
    if unlimited:
        return None
    return st.slider(label, 1, 30, default or 8, key=key)


def _parameter_controls(name: str) -> dict[str, Any]:
    defaults: dict[str, Any] = {}
    prefix = name.replace(" ", "_")
    if name == "Logistic Regression":
        return {
            "C": st.number_input("C", 0.001, 100.0, float(defaults.get("C", 1.0)), key=f"{prefix}_C"),
            "solver": st.selectbox("Algorithme d'optimisation", ["lbfgs", "liblinear", "saga"], key=f"{prefix}_solver"),
            "class_weight": st.selectbox("Pondération des classes", ["balanced", None], key=f"{prefix}_cw"),
        }
    if name == "Decision Tree":
        return {
            "max_depth": _depth_control("Profondeur maximale", f"{prefix}_depth", defaults.get("max_depth", 8)),
            "min_samples_split": st.slider("Minimum pour diviser", 2, 30, int(defaults.get("min_samples_split", 2)), key=f"{prefix}_split"),
            "min_samples_leaf": st.slider("Minimum par feuille", 1, 20, int(defaults.get("min_samples_leaf", 1)), key=f"{prefix}_leaf"),
            "class_weight": st.selectbox("Pondération des classes", ["balanced", None], key=f"{prefix}_cw"),
        }
    if name == "KNN":
        return {
            "n_neighbors": st.slider("Nombre de voisins", 1, 30, int(defaults.get("n_neighbors", 5)), key=f"{prefix}_neighbors"),
            "weights": st.selectbox("Pondération", ["distance", "uniform"], key=f"{prefix}_weights"),
            "p": st.selectbox("Distance p", [1, 2], index=1, key=f"{prefix}_p"),
        }
    if name == "Random Forest":
        return {
            "n_estimators": st.slider("Nombre d'arbres", 50, 500, int(defaults.get("n_estimators", 100)), 25, key=f"{prefix}_trees"),
            "max_depth": _depth_control("Profondeur maximale", f"{prefix}_depth", defaults.get("max_depth", 12)),
            "min_samples_split": st.slider("Minimum pour diviser", 2, 30, int(defaults.get("min_samples_split", 2)), key=f"{prefix}_split"),
            "min_samples_leaf": st.slider("Minimum par feuille", 1, 20, int(defaults.get("min_samples_leaf", 1)), key=f"{prefix}_leaf"),
            "max_features": st.selectbox("Variables par division", ["sqrt", "log2", None], key=f"{prefix}_features"),
            "class_weight": st.selectbox("Pondération des classes", ["balanced", None], key=f"{prefix}_cw"),
        }
    return {
        "n_estimators": st.slider("Nombre d'arbres", 50, 500, int(defaults.get("n_estimators", 200)), 25, key=f"{prefix}_trees"),
        "max_depth": st.slider("Profondeur maximale", 1, 15, int(defaults.get("max_depth", 4)), key=f"{prefix}_depth"),
        "learning_rate": st.number_input("Taux d'apprentissage", 0.01, 1.0, float(defaults.get("learning_rate", 0.1)), 0.01, key=f"{prefix}_rate"),
        "subsample": st.slider("Sous-échantillonnage", 0.5, 1.0, float(defaults.get("subsample", 0.8)), 0.05, key=f"{prefix}_sample"),
        "scale_pos_weight": st.number_input("Poids de la classe positive", 1.0, 100.0, float(defaults.get("scale_pos_weight", 28.52)), 0.5, key=f"{prefix}_spw"),
    }


def modeling_page() -> None:
    page_title("Modélisation", "Entraînement manuel de cinq pipelines configurables.")
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
    recommended = st.toggle("Utiliser les paramètres optimisés recommandés", value=True)
    parameters: dict[str, dict[str, Any]] = {}
    for name in selected:
        with st.expander(name, expanded=True):
            if recommended:
                parameters[name] = RECOMMENDED_PARAMS[name].copy()
                st.caption("Paramètres recommandés actifs")
                st.json(parameters[name])
            else:
                parameters[name] = _parameter_controls(name)
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
                effective_params = (
                    recommended_params_for_training(name, y_train)
                    if recommended else parameters[name]
                )
                pipeline = train_model(name, effective_params, features, x_train, y_train)
                evaluation = evaluate_pipeline(pipeline, x_test, y_test)
                st.session_state.trained_models[name] = {
                    "pipeline": pipeline,
                    "metrics": evaluation,
                    "params": effective_params,
                    "features": features.copy(),
                    "test_size": st.session_state.test_size,
                    "x_test": x_test,
                    "y_test": y_test,
                    "trained_at": datetime.now().isoformat(timespec="seconds"),
                    "configuration": configuration,
                }
                successes += 1
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
    page_title("Comparaison", "Comparaison des modèles entraînés sur leur jeu de test.")
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
    page_title("Évaluation", "Effet du seuil sur un modèle entraîné, sans réentraînement.")
    results = st.session_state.trained_models
    if not results:
        st.warning("Aucun modèle entraîné n'est disponible dans cette session.")
        return
    name = st.selectbox("Modèle", list(results))
    result = results[name]
    st.info("Classe d'intérêt : **Panne (1)**")
    threshold = st.slider("Seuil de décision", 0.0, 1.0, 0.50, 0.01, key=f"threshold_{name}")
    metrics = metrics_at_threshold(result["y_test"], result["metrics"]["scores"], threshold)
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Précision panne", f"{metrics['precision']:.3f}")
    c2.metric("Rappel panne", f"{metrics['recall']:.3f}")
    c3.metric("F1-score", f"{metrics['f1']:.3f}")
    c4.metric("Seuil", f"{threshold:.2f}")
    st.plotly_chart(
        confusion_matrix_figure(metrics["confusion_matrix"], f"{name} — seuil {threshold:.2f}"),
        use_container_width=True,
    )
    with st.expander("Comprendre les métriques"):
        st.write("**Précision** : parmi les machines signalées en panne, part réellement en panne.")
        st.write("**Rappel** : parmi toutes les vraies pannes, part correctement détectée.")
        st.write("**F1-score** : compromis harmonique entre précision et rappel.")
    st.caption(
        "Le curseur recalcule seulement la décision à partir des scores déjà produits ; "
        "le pipeline n'est pas réentraîné."
    )


def prediction_page() -> None:
    page_title("Prédiction", "Analyse manuelle d'une nouvelle machine.")
    trained = st.session_state.trained_models
    choices = ["Modèle de référence pré-entraîné (API)"] + list(trained)
    choice = st.selectbox("Modèle utilisé", choices)
    if choice == choices[0]:
        st.caption("La requête suit le flux Streamlit → HTTP → FastAPI → XGBoost de référence.")
        threshold = None
    else:
        threshold = st.slider("Seuil de décision", 0.0, 1.0, 0.50, 0.01, key=f"prediction_threshold_{choice}")
        st.caption("Ce modèle a été entraîné dans la session Streamlit courante.")
    with st.form("machine_form"):
        c1, c2 = st.columns(2)
        machine_type = c1.selectbox("Type", ["L", "M", "H"])
        air = c1.number_input("Air temperature", 250.0, 400.0, 300.0, 0.1)
        process = c1.number_input("Process temperature", 250.0, 450.0, 310.0, 0.1)
        speed = c2.number_input("Rotational speed", 0, 10000, 1500, 10)
        torque = c2.number_input("Torque", 0.0, 500.0, 40.0, 0.1)
        wear = c2.number_input("Tool wear", 0, 1000, 100, 1)
        submitted = st.form_submit_button("Analyser la machine", type="primary")
    if not submitted:
        return
    payload = {
        "Type": machine_type, "air_temperature": air,
        "process_temperature": process, "rotational_speed": speed,
        "torque": torque, "tool_wear": wear,
    }
    try:
        if choice == choices[0]:
            response = request_prediction(os.getenv("FASTAPI_URL", "http://127.0.0.1:8000"), payload)
            score = float(response["failure_score"])
            decision = int(response["prediction"])
            used_threshold = float(response["threshold"])
        else:
            result = trained[choice]
            machine = pd.DataFrame([{
                "Type": machine_type, "Air temperature": air,
                "Process temperature": process, "Rotational speed": speed,
                "Torque": torque, "Tool wear": wear,
            }])
            score = float(result["pipeline"].predict_proba(machine[result["features"]])[:, 1][0])
            used_threshold = float(threshold)
            decision = int(score >= used_threshold)
    except APIClientError as exc:
        st.error(str(exc))
        return
    except Exception:
        st.error("La prédiction n'a pas pu être calculée avec ce modèle.")
        return
    c1, c2, c3 = st.columns(3)
    st.caption(f"Modèle utilisé : **{choice}**")
    c1.metric("Décision", "Panne" if decision else "Pas de panne")
    c2.metric("Score de risque du modèle", f"{score:.3f}")
    c3.metric("Seuil utilisé", f"{used_threshold:.2f}")
    if decision:
        st.error("Une panne est signalée : une vérification de la machine est recommandée.")
    else:
        st.success("Aucune panne n'est signalée pour les mesures saisies.")
    st.caption("Le score est un score de risque du modèle, pas une probabilité nécessairement calibrée.")


def explainable_ai_placeholder_page() -> None:
    st.title("Explainable AI")
    st.info("🚧 Module en cours de développement — bientôt disponible.")
    st.markdown(
        """
        Cette extension permettra à terme de mieux comprendre les décisions produites
        par les modèles de maintenance prédictive.
        """
    )


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
    "Explainable AI": explainable_ai_placeholder_page,
}
with st.sidebar:
    st.title("Maintenance prédictive")
    st.caption("Dashboard académique")
    page = st.radio("Navigation", list(PAGES), label_visibility="collapsed")
    st.divider()
    count = len(st.session_state.trained_models)
    st.metric("Modèles en session", count)
    st.caption("Référence API : XGBoost pré-entraîné · seuil 0,75")

PAGES[page]()
