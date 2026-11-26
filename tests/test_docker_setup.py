from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parent.parent


def load_compose() -> dict:
    return yaml.safe_load((ROOT / "docker-compose.yml").read_text(encoding="utf-8"))


def test_expected_services_exist():
    assert {"mlflow", "trainer", "api"} <= set(load_compose()["services"])


def test_api_waits_for_training_to_finish():
    api = load_compose()["services"]["api"]
    assert api["depends_on"]["trainer"]["condition"] == "service_completed_successfully"


def test_dockerfiles_run_as_non_root():
    for name in ("mlflow", "train", "api"):
        text = (ROOT / "docker" / f"Dockerfile.{name}").read_text(encoding="utf-8")
        assert "USER app" in text, f"Dockerfile.{name} tourne en root"
