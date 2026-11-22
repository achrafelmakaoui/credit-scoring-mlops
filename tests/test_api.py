import numpy as np
import pytest
from fastapi.testclient import TestClient

from src.api import main
from src.api.model import ModelInfo
from src.api.schemas import EXAMPLE, CreditApplication
from src.features.build import ALL_FEATURES, NUMERIC_FEATURES

INFO = ModelInfo(name="credit-scoring", alias="champion", version="7", run_id="abc123")


class FakeModel:
    def predict_proba(self, x):
        assert list(x.columns) == ALL_FEATURES
        assert all(x[col].dtype == "float64" for col in NUMERIC_FEATURES)
        p = np.full(len(x), 0.8)
        return np.column_stack([1 - p, p])


@pytest.fixture
def client(monkeypatch):
    monkeypatch.setattr(main, "load_champion", lambda: (FakeModel(), INFO))
    with TestClient(main.app) as test_client:
        yield test_client


@pytest.fixture
def client_without_model(monkeypatch):
    def fail():
        raise RuntimeError("registre indisponible")

    monkeypatch.setattr(main, "load_champion", fail)
    with TestClient(main.app) as test_client:
        yield test_client


def test_schema_matches_training_features():
    assert set(CreditApplication.model_fields) == set(ALL_FEATURES)


def test_health_is_always_ok(client_without_model):
    assert client_without_model.get("/health").json() == {"status": "ok"}


def test_ready_when_model_loaded(client):
    response = client.get("/ready")
    assert response.status_code == 200
    assert response.json()["model_version"] == "7"


def test_not_ready_without_model(client_without_model):
    assert client_without_model.get("/ready").status_code == 503
    assert client_without_model.post("/predict", json=EXAMPLE).status_code == 503


def test_predict_returns_probability_and_label(client):
    response = client.post("/predict", json=EXAMPLE)
    assert response.status_code == 200
    body = response.json()
    assert body["default_probability"] == 0.8
    assert body["prediction"] == 1
    assert body["model_version"] == "7"


def test_threshold_is_configurable(client, monkeypatch):
    monkeypatch.setenv("DECISION_THRESHOLD", "0.9")
    assert client.post("/predict", json=EXAMPLE).json()["prediction"] == 0


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("age", 5),
        ("age", 150),
        ("credit_amount", -100),
        ("duration_months", 0),
        ("housing", "A999"),
        ("purpose", "autre"),
    ],
)
def test_invalid_values_return_422(client, field, value):
    response = client.post("/predict", json={**EXAMPLE, field: value})
    assert response.status_code == 422


def test_missing_field_returns_422(client):
    body = {k: v for k, v in EXAMPLE.items() if k != "age"}
    assert client.post("/predict", json=body).status_code == 422


def test_unknown_field_returns_422(client):
    assert client.post("/predict", json={**EXAMPLE, "salary": 1000}).status_code == 422


def test_batch_prediction(client):
    response = client.post("/predict/batch", json={"applications": [EXAMPLE, EXAMPLE]})
    assert response.status_code == 200
    assert len(response.json()["predictions"]) == 2


def test_empty_batch_returns_422(client):
    assert client.post("/predict/batch", json={"applications": []}).status_code == 422


def test_model_info(client):
    body = client.get("/model-info").json()
    assert body["model_version"] == "7"
    assert body["alias"] == "champion"


def test_reload_keeps_old_model_on_failure(client, monkeypatch):
    def fail():
        raise RuntimeError("registre indisponible")

    monkeypatch.setattr(main, "load_champion", fail)
    assert client.post("/model/reload").status_code == 502
    assert client.get("/ready").status_code == 200
