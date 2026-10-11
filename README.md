# Predictive Maintenance System

Système de maintenance prédictive permettant d'estimer le risque de panne d'une machine industrielle, de comparer plusieurs modèles de classification et d'utiliser un modèle de référence à travers une application web.

Le projet combine un **dashboard Streamlit**, une **API FastAPI** et un modèle **XGBoost pré-entraîné**.

## Live demo

- **Application Streamlit :** https://explainable-predictive-maintenance-pew4gacfaf4x8lvtmumjgg.streamlit.app/
- **Documentation FastAPI :** https://explainable-predictive-maintenance-api.onrender.com/docs
- **Dépôt GitHub :** https://github.com/eya003/explainable-predictive-maintenance

## Objectif

L'objectif est de prédire la variable **Machine failure** à partir des conditions de fonctionnement d'une machine industrielle.

Le dataset est fortement déséquilibré : les pannes représentent environ **3,39 %** des observations. Pour cette raison, l'évaluation ne repose pas uniquement sur l'accuracy. Les modèles sont comparés avec la **précision**, le **rappel**, le **F1-score**, le **ROC-AUC** et la **PR-AUC**.

## Dataset

Le projet utilise le **AI4I 2020 Predictive Maintenance Dataset** disponible sur le dépôt UCI.

- **10 000 observations**
- **339 pannes**
- **6 variables explicatives**
- **1 variable cible :** `Machine failure`

Variables utilisées :

- `Type`
- `Air temperature`
- `Process temperature`
- `Rotational speed`
- `Torque`
- `Tool wear`

## Préparation des données

Le pipeline applique :

1. un découpage apprentissage/test stratifié ;
2. l'encodage de la variable catégorielle `Type` ;
3. la standardisation des variables numériques lorsque nécessaire ;
4. l'entraînement des modèles sur le même jeu d'apprentissage ;
5. l'évaluation sur le même jeu de test ;
6. l'étude de l'effet du seuil de décision sans réentraînement.

Le découpage utilisé par défaut est de **80 % pour l'apprentissage** et **20 % pour le test**.

## Modèles comparés

Cinq modèles de classification sont disponibles dans le dashboard :

- Logistic Regression
- Decision Tree
- KNN
- Random Forest
- XGBoost

Pour les modèles compatibles, le déséquilibre des classes est pris en compte avec `class_weight="balanced"`. Pour XGBoost, `scale_pos_weight` est calculé à partir des données d'apprentissage.

## Résultats

| Modèle | Précision panne | Rappel panne | F1 panne | ROC-AUC | PR-AUC |
|---|---:|---:|---:|---:|---:|
| Logistic Regression | 0.142 | 0.824 | 0.242 | 0.907 | 0.382 |
| Decision Tree | 0.701 | 0.691 | 0.696 | 0.842 | 0.578 |
| KNN | 0.619 | 0.382 | 0.473 | 0.788 | 0.401 |
| Random Forest | **0.815** | 0.647 | **0.721** | 0.969 | 0.803 |
| XGBoost | 0.615 | **0.824** | 0.704 | **0.970** | **0.810** |

**Random Forest** obtient la meilleure précision et le meilleur F1-score.

**XGBoost** obtient le meilleur rappel, ROC-AUC et PR-AUC. Il est utilisé comme **modèle de référence dans l'API** avec un seuil de décision fixé à **0,75**.

## Fonctionnalités du dashboard

Le dashboard Streamlit permet de :

- consulter le dataset ;
- visualiser les distributions et corrélations ;
- choisir les variables explicatives ;
- modifier la taille du jeu de test ;
- entraîner les cinq modèles ;
- comparer les performances ;
- afficher les courbes ROC et précision-rappel ;
- consulter les matrices de confusion ;
- modifier le seuil de décision ;
- effectuer une prédiction sur une nouvelle machine ;
- utiliser le modèle XGBoost de référence à travers l'API FastAPI.

## Architecture

```text
Utilisateur
   |
   v
Dashboard Streamlit
   |
   | HTTP
   v
API FastAPI / Render
   |
   v
Modèle XGBoost pré-entraîné
   |
   v
Prédiction : Panne / Pas de panne
```

Le dashboard peut également entraîner les cinq modèles directement dans la session Streamlit afin de comparer leurs performances.


## Structure du projet

```text
explainable-predictive-maintenance/
|
|-- app.py
|-- main.py
|-- predictive_maintenance_model.pkl
|-- requirements.txt
|-- README.md
|-- .gitignore
|
|-- notebook/
|   `-- explainable-predictive-maintenance.ipynb
|
|-- services/
|   |-- api_client.py
|   |-- data_service.py
|   |-- evaluation_service.py
|   `-- training_service.py
|
|-- utils/
|   `-- plots.py
|
|-- tests/
|   `-- test_project.py
|
`-- screenshots/
```

## Installation locale

### 1. Cloner le dépôt

```bash
git clone https://github.com/eya003/explainable-predictive-maintenance.git
cd explainable-predictive-maintenance
```

### 2. Créer un environnement virtuel

```bash
python -m venv .venv
```

Sous Windows PowerShell :

```powershell
.venv\Scripts\Activate.ps1
```

### 3. Installer les dépendances

```bash
pip install -r requirements.txt
```

## Lancer FastAPI

```bash
uvicorn main:app --reload
```

Documentation Swagger locale :

```text
http://127.0.0.1:8000/docs
```

Endpoints disponibles :

```text
GET  /
GET  /health
POST /predict
```

## Lancer Streamlit

Dans un deuxième terminal :

```bash
streamlit run app.py
```

Par défaut, le dashboard peut utiliser l'API locale :

```text
http://127.0.0.1:8000
```

Pour le déploiement public, l'adresse FastAPI est fournie avec la variable d'environnement :

```text
FASTAPI_URL=https://explainable-predictive-maintenance-api.onrender.com
```

## Tests

Lancer les tests avec :

```bash
python -m pytest -q
```

Les tests vérifient notamment :

- le chargement des données ;
- l'entraînement des modèles ;
- `predict` et `predict_proba` ;
- le calcul des métriques ;
- les endpoints `/health` et `/predict`.

## Déploiement

- **Streamlit Community Cloud** héberge le dashboard.
- **Render** héberge l'API FastAPI.
- La variable `FASTAPI_URL` permet au dashboard public de communiquer avec l'API déployée.

## Limites

Le dataset est fortement déséquilibré et le nombre de pannes reste faible par rapport aux observations normales.

Le score retourné par le modèle n'est pas nécessairement une probabilité calibrée. Le choix du seuil de décision influence directement le compromis entre les faux positifs et les pannes non détectées.

Les résultats sont obtenus sur le dataset AI4I 2020 et doivent être validés sur d'autres données avant une utilisation industrielle réelle.

## Améliorations possibles

Des améliorations futures peuvent inclure :

- la calibration des scores ;
- l'évaluation sur des données industrielles réelles ou temporelles ;
- le suivi de la dérive des données ;
- le monitoring des prédictions déployées ;
- l'étude d'autres méthodes de traitement du déséquilibre des classes.

## Technologies

`Python` · `Pandas` · `NumPy` · `scikit-learn` · `XGBoost` · `FastAPI` · `Uvicorn` · `Streamlit` · `Plotly` · `Matplotlib` · `ucimlrepo`
