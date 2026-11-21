import json
import os
from pathlib import Path

import mlflow
import yaml
from mlflow import MlflowClient
from mlflow.exceptions import MlflowException

REPORT_PATH = Path("reports/promotion.json")
DEFAULT_TRACKING_URI = "http://127.0.0.1:5000"
DEFAULT_EXPERIMENT = "credit-scoring"
DEFAULT_MODEL_NAME = "credit-scoring"
DEFAULT_ALIAS = "champion"


def load_params(path: str = "params.yaml") -> dict:
    with open(path, encoding="utf-8") as handle:
        return yaml.safe_load(handle)["evaluate"]


def decide(candidate_auc: float, champion_auc: float | None, min_auc: float) -> tuple[bool, str]:
    """Règle de promotion : seuil minimal d'AUC, puis meilleur que le champion actuel."""
    if candidate_auc < min_auc:
        return False, f"AUC {candidate_auc:.3f} sous le seuil minimal {min_auc:.2f}"
    if champion_auc is None:
        return True, "aucun champion en place"
    if candidate_auc > champion_auc:
        return True, f"AUC {candidate_auc:.3f} supérieure au champion ({champion_auc:.3f})"
    return False, f"AUC {candidate_auc:.3f} ne dépasse pas le champion ({champion_auc:.3f})"


def get_champion(client: MlflowClient, name: str, alias: str):
    try:
        return client.get_model_version_by_alias(name, alias)
    except MlflowException:
        return None


def best_run(client: MlflowClient, experiment_id: str):
    runs = client.search_runs(
        experiment_ids=[experiment_id],
        filter_string="tags.model_uri LIKE 'models:%'",
        order_by=["metrics.roc_auc DESC"],
        max_results=1,
    )
    return runs[0] if runs else None


def get_or_register_version(client: MlflowClient, run, name: str):
    for version in client.search_model_versions(f"name='{name}'"):
        if version.run_id == run.info.run_id:
            return version
    version = mlflow.register_model(run.data.tags["model_uri"], name)
    client.set_model_version_tag(
        name, version.version, "roc_auc", f"{run.data.metrics['roc_auc']:.4f}"
    )
    client.set_model_version_tag(name, version.version, "model", run.data.params.get("model", ""))
    return version


def main() -> None:
    min_auc = load_params()["min_auc"]
    model_name = os.environ.get("MODEL_NAME", DEFAULT_MODEL_NAME)
    alias = os.environ.get("MODEL_ALIAS", DEFAULT_ALIAS)
    experiment_name = os.environ.get("MLFLOW_EXPERIMENT_NAME", DEFAULT_EXPERIMENT)

    mlflow.set_tracking_uri(os.environ.get("MLFLOW_TRACKING_URI", DEFAULT_TRACKING_URI))
    client = MlflowClient()

    experiment = client.get_experiment_by_name(experiment_name)
    if experiment is None:
        raise SystemExit(
            f"Expérience introuvable : {experiment_name}. Lancez d'abord l'entraînement."
        )
    run = best_run(client, experiment.experiment_id)
    if run is None:
        raise SystemExit("Aucune exécution avec un modèle taggué. Relancez l'entraînement.")

    candidate_auc = run.data.metrics["roc_auc"]
    version = get_or_register_version(client, run, model_name)
    champion = get_champion(client, model_name, alias)
    champion_auc = None
    if champion is not None:
        champion_auc = client.get_run(champion.run_id).data.metrics["roc_auc"]

    promote, reason = decide(candidate_auc, champion_auc, min_auc)
    if champion is not None and champion.version == version.version:
        promote, reason = False, "ce modèle est déjà le champion"

    if promote:
        if champion is not None:
            client.set_model_version_tag(
                model_name, version.version, "replaced_version", champion.version
            )
        client.set_registered_model_alias(model_name, alias, version.version)

    report = {
        "candidate_version": version.version,
        "candidate_model": run.data.params.get("model"),
        "candidate_roc_auc": round(candidate_auc, 4),
        "champion_version_before": champion.version if champion else None,
        "promoted": promote,
        "reason": reason,
    }
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(REPORT_PATH, "w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2, ensure_ascii=False)

    status = "PROMU" if promote else "NON PROMU"
    print(f"Version {version.version} ({report['candidate_model']}) : {status} - {reason}")


if __name__ == "__main__":
    main()
