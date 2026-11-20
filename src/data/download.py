import io
import urllib.request
import zipfile
from pathlib import Path

import pandas as pd

URL = "https://archive.ics.uci.edu/static/public/144/statlog+german+credit+data.zip"
OUTPUT = Path("data/raw/german_credit.csv")

COLUMNS = [
    "checking_status",
    "duration_months",
    "credit_history",
    "purpose",
    "credit_amount",
    "savings",
    "employment_since",
    "installment_rate",
    "personal_status_sex",
    "other_debtors",
    "residence_since",
    "property",
    "age",
    "other_installment_plans",
    "housing",
    "existing_credits",
    "job",
    "num_dependents",
    "telephone",
    "foreign_worker",
    "class",
]


def download(url: str = URL, output: Path = OUTPUT) -> Path:
    with urllib.request.urlopen(url, timeout=60) as response:
        payload = response.read()
    with zipfile.ZipFile(io.BytesIO(payload)) as archive, archive.open("german.data") as handle:
        df = pd.read_csv(handle, sep=" ", header=None, names=COLUMNS)
    output.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output, index=False)
    return output


if __name__ == "__main__":
    path = download()
    print(f"Données écrites dans {path}")
