# Explainable Predictive Maintenance System

Application de maintenance prédictive destinée à estimer le risque de panne d'une machine
industrielle et à comparer plusieurs modèles de classification supervisée.

Le projet associe un dashboard Streamlit, une API FastAPI et un modèle XGBoost de référence
pré-entraîné.

## Démonstration et dépôt

- **Streamlit :** https://explainable-predictive-maintenance-pew4gacfaf4x8lvtmumjgg.streamlit.app/
- **FastAPI :** https://explainable-predictive-maintenance-api.onrender.com/docs
- **GitHub :** https://github.com/eya003/explainable-predictive-maintenance.git

## Objectif et dataset

La cible est `Machine failure` dans le dataset **AI4I 2020 Predictive Maintenance** du dépôt
UCI. Le dataset est téléchargé automatiquement avec `ucimlrepo` (identifiant 601).

- 10 000 observations ;
- 339 pannes, soit environ 3,39 % des observations ;
- 6 variables explicatives ;
- une cible binaire : `0 = Pas de panne`, `1 = Panne`.

Variables utilisées :

- `Type`
- `Air temperature`
- `Process temperature`
- `Rotational speed`
- `Torque`
- `Tool wear`

Le fort déséquilibre des classes rend l'accuracy insuffisante. L'application suit donc la
précision de la classe panne, le rappel, le F1-score, le ROC-AUC et la PR-AUC calculée avec
`average_precision_score`.

## Préparation et modèles

Le pipeline applique :

1. un découpage train/test avec `stratify=y` et `random_state=42` ;
2. l'encodage One-Hot de `Type` ;
3. la standardisation des variables numériques lorsque le modèle l'exige ;
4. le preprocessing et le classifieur dans un même pipeline afin d'éviter la fuite de données ;
5. l'évaluation des modèles sur le même jeu de test ;
6. l'ajustement interactif du seuil sans réentraînement.

Les cinq modèles disponibles sont :

- Logistic Regression
- Decision Tree
- KNN
- Random Forest
- XGBoost

Les modèles sont entraînés uniquement après une action explicite de l'utilisateur et restent
disponibles dans `st.session_state`. Pour XGBoost, `scale_pos_weight` est calculé à partir du jeu
d'entraînement. Le modèle XGBoost sauvegardé dans `predictive_maintenance_model.pkl` reste le
modèle de référence de l'API, avec un seuil de décision de `0.75`.

## Fonctionnalités Streamlit

Le dashboard contient huit pages :

1. Accueil
2. Données
3. Exploration
4. Préparation
5. Modélisation
6. Comparaison
7. Évaluation
8. Prédiction

La page Comparaison fournit le tableau des métriques, les graphiques comparatifs, les courbes
ROC et précision-rappel et les matrices de confusion. La page Évaluation recalcule en direct la
précision, le rappel, le F1-score et la matrice de confusion lorsque le seuil change.

La page Prédiction accepte les mesures d'une nouvelle machine et peut utiliser soit un modèle
entraîné dans la session, soit le XGBoost de référence via FastAPI.

## Architecture de prédiction

```text
Utilisateur
   ↓
Streamlit
   ↓ HTTP
FastAPI
   ↓
XGBoost de référence
   ↓
Prédiction : Panne / Pas de panne
```

## Structure du projet

```text
explainable-predictive-maintenance/
├── app.py
├── main.py
├── predictive_maintenance_model.pkl
├── requirements.txt
├── README.md
├── notebook/
│   └── explainable-predictive-maintenance.ipynb
├── services/
│   ├── api_client.py
│   ├── data_service.py
│   ├── evaluation_service.py
│   └── training_service.py
├── utils/
│   └── plots.py
├── tests/
│   └── test_project.py
└── predictive_maintenance_best_screenshots/
    └── predictive_maintenance_screenshots/
```

## Installation locale

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
```

`scikit-learn==1.6.1` est épinglé pour rester compatible avec le fichier modèle fourni.

## Lancement

Terminal 1 — FastAPI :

```powershell
uvicorn main:app --reload
```

Documentation Swagger locale : `http://127.0.0.1:8000/docs`.

Endpoints disponibles :

```text
GET  /
GET  /health
POST /predict
```

Terminal 2 — Streamlit :

```powershell
streamlit run app.py
```

Par défaut, Streamlit appelle `http://127.0.0.1:8000`. En déploiement, définir :

```text
FASTAPI_URL=https://explainable-predictive-maintenance-api.onrender.com
```

## Exemple de prédiction API

```json
{
  "Type": "L",
  "air_temperature": 300.0,
  "process_temperature": 310.0,
  "rotational_speed": 1500,
  "torque": 40.0,
  "tool_wear": 100
}
```

La réponse contient la décision, le libellé, le score du modèle et le seuil utilisé. Le score
n'est pas présenté comme une probabilité nécessairement calibrée.

## Tests

```powershell
python -m compileall .
python -m pytest -q
```

La suite vérifie le téléchargement du dataset, le split stratifié, l'entraînement des cinq
modèles, `predict`, `predict_proba`, les métriques, l'effet du seuil et les endpoints `/`,
`/health` et `/predict`.

## Captures

Les captures sont stockées dans
`predictive_maintenance_best_screenshots/predictive_maintenance_screenshots/`.

- [Accueil](predictive_maintenance_best_screenshots/predictive_maintenance_screenshots/01_home.png)
- [Données](predictive_maintenance_best_screenshots/predictive_maintenance_screenshots/02_data.png)
- [Exploration](predictive_maintenance_best_screenshots/predictive_maintenance_screenshots/03_exploration.png)
- [Préparation](predictive_maintenance_best_screenshots/predictive_maintenance_screenshots/04_preparation.png)
- [Modélisation](predictive_maintenance_best_screenshots/predictive_maintenance_screenshots/05_modeling_complete.png)
- [Comparaison](predictive_maintenance_best_screenshots/predictive_maintenance_screenshots/06_comparison_table.png)
- [Courbes ROC et PR](predictive_maintenance_best_screenshots/predictive_maintenance_screenshots/07_roc_pr_curves.png)
- [Matrices de confusion](predictive_maintenance_best_screenshots/predictive_maintenance_screenshots/08_confusion_matrices.png)
- [Évaluation du seuil](predictive_maintenance_best_screenshots/predictive_maintenance_screenshots/09_evaluation_threshold.png)
- [Prédiction API](predictive_maintenance_best_screenshots/predictive_maintenance_screenshots/10_prediction_api.png)
- [Documentation FastAPI](predictive_maintenance_best_screenshots/predictive_maintenance_screenshots/15_fastapi_docs.png)

## Déploiement

- FastAPI est déployée comme Web Service sur Render.
- Streamlit est déployé sur Streamlit Community Cloud.
- `FASTAPI_URL` relie le dashboard public à l'API Render.

## Limites et améliorations possibles

Le dataset est fortement déséquilibré et les scores ne sont pas nécessairement calibrés. Des
extensions possibles sont la calibration, l'évaluation sur des données industrielles réelles ou
temporelles, la détection de dérive et le monitoring des prédictions déployées.

## Technologies

`Python` · `Pandas` · `NumPy` · `scikit-learn` · `XGBoost` · `FastAPI` · `Uvicorn` ·
`Streamlit` · `Plotly`
