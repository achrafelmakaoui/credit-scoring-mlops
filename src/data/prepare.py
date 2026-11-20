from pathlib import Path

import pandas as pd
import yaml
from sklearn.model_selection import train_test_split

RAW_PATH = Path("data/raw/german_credit.csv")
OUT_DIR = Path("data/processed")
TARGET = "default"


def load_params(path: str = "params.yaml") -> dict:
    with open(path, encoding="utf-8") as handle:
        return yaml.safe_load(handle)["prepare"]


def clean(df: pd.DataFrame) -> pd.DataFrame:
    """Supprime doublons et lignes incomplètes, crée la cible binaire (1 = défaut)."""
    df = df.drop_duplicates().dropna().copy()
    df[TARGET] = (df["class"] == 2).astype(int)
    return df.drop(columns=["class"])


def split(df: pd.DataFrame, test_size: float, seed: int):
    return train_test_split(df, test_size=test_size, random_state=seed, stratify=df[TARGET])


def main() -> None:
    params = load_params()
    df = clean(pd.read_csv(RAW_PATH))
    train, test = split(df, params["test_size"], params["random_seed"])
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    train.to_csv(OUT_DIR / "train.csv", index=False)
    test.to_csv(OUT_DIR / "test.csv", index=False)
    print(f"train: {len(train)} lignes, test: {len(test)} lignes")


if __name__ == "__main__":
    main()