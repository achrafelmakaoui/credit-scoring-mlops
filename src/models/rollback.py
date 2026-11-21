import argparse
import os

import mlflow
from mlflow import MlflowClient

from src.models.evaluate import (
    DEFAULT_ALIAS,
    DEFAULT_MODEL_NAME,
    DEFAULT_TRACKING_URI,
    get_champion,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--version",
        help="version cible (par défaut : la version remplacée par le champion actuel)",
    )
    args = parser.parse_args()

    model_name = os.environ.get("MODEL_NAME", DEFAULT_MODEL_NAME)
    alias = os.environ.get("MODEL_ALIAS", DEFAULT_ALIAS)
    mlflow.set_tracking_uri(os.environ.get("MLFLOW_TRACKING_URI", DEFAULT_TRACKING_URI))
    client = MlflowClient()

    champion = get_champion(client, model_name, alias)
    if champion is None:
        raise SystemExit("Aucun champion en place : rien à restaurer.")

    target = args.version or champion.tags.get("replaced_version")
    if not target:
        raise SystemExit("Aucune version précédente connue. Précisez --version N.")

    client.get_model_version(model_name, target)  # échoue si la version n'existe pas
    client.set_registered_model_alias(model_name, alias, target)
    client.set_model_version_tag(model_name, target, "rolled_back_from", champion.version)
    print(f"Alias '{alias}' : version {champion.version} -> version {target}")


if __name__ == "__main__":
    main()
