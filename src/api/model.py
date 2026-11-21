import os
from dataclasses import dataclass

import mlflow
import mlflow.sklearn
from mlflow import MlflowClient

DEFAULT_TRACKING_URI = "http://127.0.0.1:5000"
DEFAULT_MODEL_NAME = "credit-scoring"
DEFAULT_ALIAS = "champion"


@dataclass(frozen=True)
class ModelInfo:
    name: str
    alias: str
    version: str
    run_id: str


def load_champion():
    """Retourne (modèle, infos) pour la version pointée par l'alias champion."""
    name = os.environ.get("MODEL_NAME", DEFAULT_MODEL_NAME)
    alias = os.environ.get("MODEL_ALIAS", DEFAULT_ALIAS)
    mlflow.set_tracking_uri(os.environ.get("MLFLOW_TRACKING_URI", DEFAULT_TRACKING_URI))

    version = MlflowClient().get_model_version_by_alias(name, alias)
    model = mlflow.sklearn.load_model(f"models:/{name}/{version.version}")
    info = ModelInfo(name=name, alias=alias, version=str(version.version), run_id=version.run_id)
    return model, info
