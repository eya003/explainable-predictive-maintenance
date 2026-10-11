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
