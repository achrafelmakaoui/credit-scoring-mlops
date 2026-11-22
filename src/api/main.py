import logging
import os
from contextlib import asynccontextmanager

import pandas as pd
from fastapi import FastAPI, HTTPException, Request

from src.api.model import load_champion
from src.api.schemas import (
    BatchPrediction,
    BatchRequest,
    CreditApplication,
    ModelInfoResponse,
    Prediction,
)
from src.features.build import ALL_FEATURES, NUMERIC_FEATURES

logger = logging.getLogger("credit-scoring-api")

DEFAULT_THRESHOLD = 0.5


def decision_threshold() -> float:
    return float(os.environ.get("DECISION_THRESHOLD", DEFAULT_THRESHOLD))


def try_load(app: FastAPI) -> bool:
    try:
        app.state.model, app.state.model_info = load_champion()
    except Exception:
        logger.exception("Impossible de charger le modèle champion")
        return False
    logger.info("Modèle chargé : version %s", app.state.model_info.version)
    return True


@asynccontextmanager
async def lifespan(app: FastAPI):
    app.state.model = None
    app.state.model_info = None
    try_load(app)
    yield


app = FastAPI(
    title="Credit Scoring API",
    description="Prédit la probabilité de défaut de paiement avec le modèle champion MLflow.",
    version="0.1.0",
    lifespan=lifespan,
)


def require_model(request: Request):
    if request.app.state.model is None:
        raise HTTPException(status_code=503, detail="Modèle non chargé")
    return request.app.state.model, request.app.state.model_info


def to_frame(applications: list[CreditApplication]) -> pd.DataFrame:
    frame = pd.DataFrame([a.model_dump() for a in applications])[ALL_FEATURES]
    frame[NUMERIC_FEATURES] = frame[NUMERIC_FEATURES].astype("float64")
    return frame


def score(model, version: str, applications: list[CreditApplication]) -> list[Prediction]:
    probabilities = model.predict_proba(to_frame(applications))[:, 1]
    threshold = decision_threshold()
    return [
        Prediction(
            default_probability=round(float(p), 4),
            prediction=int(p >= threshold),
            model_version=version,
        )
        for p in probabilities
    ]


@app.get("/health", tags=["supervision"])
def health():
    return {"status": "ok"}


@app.get("/ready", tags=["supervision"])
def ready(request: Request):
    _, info = require_model(request)
    return {"status": "ready", "model_version": info.version}


@app.get("/model-info", response_model=ModelInfoResponse, tags=["modèle"])
def model_info(request: Request):
    _, info = require_model(request)
    return ModelInfoResponse(
        model_name=info.name,
        alias=info.alias,
        model_version=info.version,
        run_id=info.run_id,
        decision_threshold=decision_threshold(),
    )


@app.post("/predict", response_model=Prediction, tags=["prédiction"])
def predict(application: CreditApplication, request: Request):
    model, info = require_model(request)
    return score(model, info.version, [application])[0]


@app.post("/predict/batch", response_model=BatchPrediction, tags=["prédiction"])
def predict_batch(batch: BatchRequest, request: Request):
    model, info = require_model(request)
    return BatchPrediction(
        predictions=score(model, info.version, batch.applications),
        model_version=info.version,
    )


@app.post("/model/reload", tags=["modèle"])
def reload_model(request: Request):
    if not try_load(request.app):
        raise HTTPException(
            status_code=502, detail="Rechargement impossible : ancien modèle conservé"
        )
    return {"status": "reloaded", "model_version": request.app.state.model_info.version}
