import json
import os
from pathlib import Path

import mlflow
import mlflow.sklearn
import pandas as pd
import yaml
from matplotlib.figure import Figure
from mlflow.models import infer_signature
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.pipeline import Pipeline

from src.data.prepare import TARGET
from src.features.build import ALL_FEATURES, NUMERIC_FEATURES, build_preprocessor

TRAIN_PATH = Path("data/processed/train.csv")
TEST_PATH = Path("data/processed/test.csv")
METRICS_PATH = Path("reports/metrics.json")
DEFAULT_TRACKING_URI = "http://127.0.0.1:5000"
DEFAULT_EXPERIMENT = "credit-scoring"
SKOPS_TRUSTED_TYPES = ["sklearn.tree._tree.Tree"]


def load_params(path: str = "params.yaml") -> dict:
    with open(path, encoding="utf-8") as handle:
        return yaml.safe_load(handle)["train"]


def load_data():
    train = pd.read_csv(TRAIN_PATH)
    test = pd.read_csv(TEST_PATH)
    for df in (train, test):
        df[NUMERIC_FEATURES] = df[NUMERIC_FEATURES].astype("float64")
    return train[ALL_FEATURES], train[TARGET], test[ALL_FEATURES], test[TARGET]


def build_pipeline(name: str, params: dict, seed: int) -> Pipeline:
    if name == "logistic_regression":
        estimator = LogisticRegression(class_weight="balanced", random_state=seed, **params)
    elif name == "random_forest":
        estimator = RandomForestClassifier(
            class_weight="balanced", random_state=seed, n_jobs=-1, **params
        )
    elif name == "gradient_boosting":
        estimator = GradientBoostingClassifier(random_state=seed, **params)
    else:
        raise ValueError(f"Modèle inconnu : {name}")
    return Pipeline([("preprocess", build_preprocessor()), ("model", estimator)])


def compute_metrics(y_true, y_pred, y_score) -> dict:
    """La classe positive est le défaut de paiement (1)."""
    return {
        "roc_auc": float(roc_auc_score(y_true, y_score)),
        "recall": float(recall_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred)),
        "accuracy": float(accuracy_score(y_true, y_pred)),
    }


def roc_figure(y_true, y_score, auc: float) -> Figure:
    fpr, tpr, _ = roc_curve(y_true, y_score)
    fig = Figure(figsize=(5, 4))
    ax = fig.subplots()
    ax.plot(fpr, tpr, label=f"AUC = {auc:.3f}")
    ax.plot([0, 1], [0, 1], "--", color="gray")
    ax.set_xlabel("Taux de faux positifs")
    ax.set_ylabel("Taux de vrais positifs")
    ax.legend(loc="lower right")
    fig.tight_layout()
    return fig


def confusion_figure(y_true, y_pred) -> Figure:
    matrix = confusion_matrix(y_true, y_pred)
    labels = ["bon payeur", "défaut"]
    fig = Figure(figsize=(4, 4))
    ax = fig.subplots()
    ax.imshow(matrix, cmap="Blues")
    ax.set_xticks([0, 1], labels=labels)
    ax.set_yticks([0, 1], labels=labels)
    ax.set_xlabel("Prédit")
    ax.set_ylabel("Réel")
    for i in range(2):
        for j in range(2):
            ax.text(j, i, str(matrix[i, j]), ha="center", va="center")
    fig.tight_layout()
    return fig


def main() -> None:
    params = load_params()
    seed = params["random_seed"]

    mlflow.set_tracking_uri(os.environ.get("MLFLOW_TRACKING_URI", DEFAULT_TRACKING_URI))
    mlflow.set_experiment(os.environ.get("MLFLOW_EXPERIMENT_NAME", DEFAULT_EXPERIMENT))

    x_train, y_train, x_test, y_test = load_data()
    all_metrics = {}

    for name, model_params in params["models"].items():
        pipeline = build_pipeline(name, model_params, seed)
        with mlflow.start_run(run_name=name):
            mlflow.log_params({"model": name, "random_seed": seed, **model_params})
            mlflow.log_params({"train_rows": len(x_train), "test_rows": len(x_test)})

            pipeline.fit(x_train, y_train)
            y_score = pipeline.predict_proba(x_test)[:, 1]
            y_pred = pipeline.predict(x_test)

            metrics = compute_metrics(y_test, y_pred, y_score)
            mlflow.log_metrics(metrics)
            mlflow.log_figure(roc_figure(y_test, y_score, metrics["roc_auc"]), "roc_curve.png")
            mlflow.log_figure(confusion_figure(y_test, y_pred), "confusion_matrix.png")

            # Traces pour rejouer l'exécution : hyperparamètres et empreintes des données
            for trace in ("params.yaml", "dvc.lock"):
                if Path(trace).exists():
                    mlflow.log_artifact(trace)

            signature = infer_signature(x_train, pipeline.predict(x_train))
            mlflow.sklearn.log_model(
                pipeline,
                name="model",
                signature=signature,
                input_example=x_train.head(5),
                skops_trusted_types=SKOPS_TRUSTED_TYPES,
            )

            all_metrics[name] = metrics
            print(f"{name}: " + ", ".join(f"{k}={v:.3f}" for k, v in metrics.items()))

    METRICS_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(METRICS_PATH, "w", encoding="utf-8") as handle:
        json.dump(all_metrics, handle, indent=2)


if __name__ == "__main__":
    main()
