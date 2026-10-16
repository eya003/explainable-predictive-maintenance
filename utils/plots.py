"""Figures Plotly communes au dashboard."""

from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go


COLORS = {"Pas de panne": "#315b7d", "Panne": "#c84b4b"}


def confusion_matrix_figure(matrix: np.ndarray, title: str) -> go.Figure:
    labels = [[f"TN<br>{matrix[0, 0]}", f"FP<br>{matrix[0, 1]}"],
              [f"FN<br>{matrix[1, 0]}", f"TP<br>{matrix[1, 1]}"]]
    figure = go.Figure(
        data=go.Heatmap(
            z=matrix,
            x=["Prédit : pas de panne", "Prédit : panne"],
            y=["Réel : pas de panne", "Réel : panne"],
            colorscale="Blues",
            showscale=False,
            text=labels,
            texttemplate="%{text}",
            hovertemplate="%{y}<br>%{x}<br>Effectif : %{z}<extra></extra>",
        )
    )
    figure.update_layout(title=title, height=410, margin=dict(l=20, r=20, t=60, b=20))
    figure.update_yaxes(autorange="reversed")
    return figure


def distribution_figure(data: pd.DataFrame) -> go.Figure:
    counts = data["Machine failure"].value_counts().reindex([0, 1], fill_value=0)
    frame = pd.DataFrame(
        {"Statut": ["Pas de panne", "Panne"], "Observations": counts.values}
    )
    return px.bar(
        frame,
        x="Statut",
        y="Observations",
        color="Statut",
        color_discrete_map=COLORS,
        text_auto=True,
        title="Distribution de la variable cible",
    )


def roc_curves_figure(results: dict[str, dict]) -> go.Figure:
    figure = go.Figure()
    for name, result in results.items():
        metrics = result["metrics"]
        figure.add_trace(
            go.Scatter(
                x=metrics["fpr"], y=metrics["tpr"], mode="lines",
                name=f"{name} (AUC={metrics['roc_auc']:.3f})",
            )
        )
    figure.add_trace(
        go.Scatter(
            x=[0, 1], y=[0, 1], mode="lines", name="Aléatoire",
            line=dict(color="#888", dash="dash"),
        )
    )
    figure.update_layout(
        title="Courbes ROC", xaxis_title="Taux de faux positifs",
        yaxis_title="Taux de vrais positifs", height=480,
    )
    return figure


def precision_recall_curves_figure(results: dict[str, dict]) -> go.Figure:
    figure = go.Figure()
    for name, result in results.items():
        metrics = result["metrics"]
        figure.add_trace(
            go.Scatter(
                x=metrics["pr_recall"], y=metrics["pr_precision"], mode="lines",
                name=f"{name} (AP={metrics['pr_auc']:.3f})",
            )
        )
    figure.update_layout(
        title="Courbes précision-rappel — AP dans la légende", xaxis_title="Rappel",
        yaxis_title="Précision", height=480,
    )
    return figure


def importance_bar_figure(
    data: pd.DataFrame,
    value_column: str,
    title: str,
    error_column: str | None = None,
) -> go.Figure:
    figure = px.bar(
        data.sort_values(value_column),
        x=value_column,
        y="feature",
        error_x=error_column,
        orientation="h",
        title=title,
        color=value_column,
        color_continuous_scale="Blues",
    )
    figure.update_layout(coloraxis_showscale=False, yaxis_title="Variable", height=430)
    return figure


def lime_contribution_figure(data: pd.DataFrame) -> go.Figure:
    frame = data.head(10).sort_values("lime_weight").copy()
    frame["Sens"] = np.where(
        frame["lime_weight"] >= 0, "Augmente le score", "Réduit le score"
    )
    figure = px.bar(
        frame,
        x="lime_weight",
        y="condition_or_feature",
        color="Sens",
        orientation="h",
        color_discrete_map={"Augmente le score": "#c84b4b", "Réduit le score": "#315b7d"},
        title="Contributions locales LIME",
    )
    figure.update_layout(xaxis_title="Contribution LIME", yaxis_title="Condition")
    return figure


def shap_beeswarm_figure(values: np.ndarray, data: np.ndarray, names: list[str]):
    import matplotlib.pyplot as plt
    import shap

    explanation = shap.Explanation(values=values, data=data, feature_names=names)
    plt.figure()
    shap.plots.beeswarm(explanation, max_display=15, show=False)
    return plt.gcf()


def shap_waterfall_figure(
    values: np.ndarray,
    base_value: float,
    data: np.ndarray,
    names: list[str],
):
    import matplotlib.pyplot as plt
    import shap

    explanation = shap.Explanation(
        values=values, base_values=base_value, data=data, feature_names=names
    )
    plt.figure()
    shap.plots.waterfall(explanation, max_display=15, show=False)
    return plt.gcf()


def shap_dependence_figure(
    x_values: np.ndarray,
    shap_values: np.ndarray,
    feature_name: str,
    color_values: np.ndarray | None = None,
    color_name: str | None = None,
) -> go.Figure:
    """Dépendance SHAP avec axes exprimés dans les unités physiques brutes."""
    frame = pd.DataFrame({feature_name: x_values, "Valeur SHAP": shap_values})
    color_column = None
    hover_data = {feature_name: True, "Valeur SHAP": ":.4f"}
    if color_values is not None and color_name:
        frame[color_name] = color_values
        color_column = color_name
        hover_data[color_name] = True
    figure = px.scatter(
        frame,
        x=feature_name,
        y="Valeur SHAP",
        color=color_column,
        color_continuous_scale="RdBu_r",
        title=f"Dépendance SHAP — {feature_name}",
        hover_data=hover_data,
    )
    figure.update_layout(height=470)
    return figure
