# credit-scoring-mlops

API de prédiction de scoring crédit, mise en service avec MLflow et Docker.
Projet **P2** du module DevOps et MLOps (Master SDIA, ENSET Mohammedia).

> Objectif : une chaîne reproductible, versionnée, containerisée et automatisée,
> qu'un tiers peut relancer depuis un clone du dépôt sans intervention des auteurs.

## Équipe

| Membre | GitHub | Responsabilités |
|---|---|---|
| Achraf El Makaoui | [@achrafelmakaoui](https://github.com/achrafelmakaoui) | Structure du projet, données et DVC, API FastAPI, Docker et Compose |
| Houdaifa Bahou | [@bahouhoudaifa](https://github.com/bahouhoudaifa) | Entraînement et suivi MLflow, registre de modèles, supervision Prometheus/Grafana, intégration continue |

Chaque membre écrit les tests de son propre code et relit les pull requests de l'autre.

## Architecture

```mermaid
flowchart LR
    T[trainer<br/>download, prepare, train, evaluate] -->|runs, modèles, alias champion| M[(MLflow<br/>registre + volume)]
    M -->|models:/credit-scoring@champion| A[API FastAPI<br/>:8000]
    C[client] -->|POST /predict| A
```

Trois services Docker (`mlflow`, `trainer`, `api`). Prometheus et Grafana s'y ajoutent à la phase 6.


## Stack

- Données : German Credit (UCI Statlog), versionnées avec DVC (remote DagsHub)
- Modèles : scikit-learn (régression logistique, forêt aléatoire, gradient boosting)
- Suivi et registre : MLflow
- Service : FastAPI + Uvicorn
- Supervision : Prometheus + Grafana
- Tests et qualité : pytest, ruff, GitHub Actions
- Environnement : Python 3.11, Docker Desktop

## Installation locale (développement)

Prérequis : Git, Python 3.11, Docker Desktop.

```powershell
git clone https://github.com/achrafelmakaoui/credit-scoring-mlops.git
cd credit-scoring-mlops
py -3.11 -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements/dev.txt
copy .env.example .env
```

## Relancer le pipeline

Prérequis : Docker Desktop. Aucune autre installation, aucun secret.

```powershell
git clone https://github.com/achrafelmakaoui/credit-scoring-mlops.git
cd credit-scoring-mlops
copy .env.example .env
docker compose up --build
```

Équivalents : `.\run.ps1` (Windows) ou `make up` (Linux/macOS).

- Interface MLflow : http://localhost:5000
- API et documentation : http://localhost:8000/docs
- Réinitialiser (supprime le registre) : `docker compose down -v`

Le conteneur `trainer` rejoue les mêmes étapes que `dvc.yaml` en téléchargeant les données depuis la source publique UCI. Le versionnage DVC (remote DagsHub) reste le workflow de développement.

## Lancer l'API en local

Prérequis : serveur MLflow démarré et modèle `champion` enregistré (`dvc repro`).

```powershell
uvicorn src.api.main:app --port 8000
```

Documentation interactive : http://127.0.0.1:8000/docs

| Endpoint | Rôle |
|---|---|
| `POST /predict` | Probabilité de défaut pour une demande |
| `POST /predict/batch` | Jusqu'à 1000 demandes |
| `GET /health` | Liveness |
| `GET /ready` | Readiness (modèle chargé) |
| `GET /model-info` | Version du champion en service |
| `POST /model/reload` | Recharge le champion après un changement d'alias |

## Variables d'environnement

Voir [`.env.example`](.env.example). Le fichier `.env` n'est jamais commité.

| Variable | Rôle |
|---|---|
| `MLFLOW_TRACKING_URI` | Adresse du serveur MLflow |
| `MLFLOW_EXPERIMENT_NAME` | Nom de l'expérience MLflow |
| `MODEL_NAME` / `MODEL_ALIAS` | Modèle chargé par l'API depuis le registre |
| `DAGSHUB_USER` / `DAGSHUB_TOKEN` | Accès au remote DVC |
| `API_PORT` | Port de l'API |
| `GRAFANA_ADMIN_USER` / `GRAFANA_ADMIN_PASSWORD` | Accès à Grafana |

## Structure du dépôt

```
src/data/        préparation des données
src/features/    transformations (ColumnTransformer)
src/models/      entraînement, évaluation, promotion dans le registre
src/api/         application FastAPI, schémas, métriques
tests/           tests unitaires, API et validation du modèle
docker/          Dockerfiles multi-étapes
monitoring/      configuration Prometheus et tableaux de bord Grafana
docs/            guide de déploiement, schémas
```

## Contribution

Voir [CONTRIBUTING.md](CONTRIBUTING.md).

## Limites connues

_À compléter._

## Licences

- Jeu de données : Statlog (German Credit Data), UCI Machine Learning Repository, licence CC BY 4.0.
