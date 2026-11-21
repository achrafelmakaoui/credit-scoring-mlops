import numpy as np
import pandas as pd
import pytest

from src.features.build import ALL_FEATURES, CATEGORICAL_FEATURES, NUMERIC_FEATURES
from src.models.train import build_pipeline, compute_metrics

SMALL_PARAMS = {
    "logistic_regression": {"C": 1.0, "max_iter": 200},
    "random_forest": {"n_estimators": 10, "max_depth": 3, "min_samples_leaf": 1},
    "gradient_boosting": {"n_estimators": 10, "learning_rate": 0.1, "max_depth": 2},
}


def make_xy(n: int = 120, seed: int = 0):
    rng = np.random.default_rng(seed)
    x = pd.DataFrame({col: rng.normal(size=n) for col in NUMERIC_FEATURES})
    for col in CATEGORICAL_FEATURES:
        x[col] = rng.choice(["A", "B", "C"], size=n)
    y = pd.Series([0, 1] * (n // 2))
    return x[ALL_FEATURES], y


@pytest.mark.parametrize("name", list(SMALL_PARAMS))
def test_pipeline_outputs_valid_probabilities(name):
    x, y = make_xy()
    pipeline = build_pipeline(name, SMALL_PARAMS[name], seed=0).fit(x, y)
    proba = pipeline.predict_proba(x)
    assert proba.shape == (len(x), 2)
    assert ((proba >= 0) & (proba <= 1)).all()
    assert proba.sum(axis=1) == pytest.approx(1.0)


def test_pipeline_handles_unseen_category():
    x, y = make_xy()
    pipeline = build_pipeline("logistic_regression", SMALL_PARAMS["logistic_regression"], 0)
    pipeline.fit(x, y)
    row = x.iloc[:1].copy()
    row["purpose"] = "Z"
    assert pipeline.predict(row).shape == (1,)


def test_unknown_model_raises():
    with pytest.raises(ValueError):
        build_pipeline("svm", {}, seed=0)


def test_compute_metrics_perfect_predictions():
    y = pd.Series([0, 0, 1, 1])
    metrics = compute_metrics(y, y, y.astype(float))
    assert metrics["roc_auc"] == 1.0
    assert metrics["recall"] == 1.0
    assert metrics["f1"] == 1.0
