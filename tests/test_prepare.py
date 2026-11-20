import pandas as pd
import pytest

from src.data.prepare import TARGET, clean, split


def make_df(n: int = 100) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "duration_months": pd.Series(range(n), dtype="float"),
            "class": [1] * 70 + [2] * 30,
        }
    )


def test_clean_creates_binary_target():
    df = clean(make_df())
    assert set(df[TARGET].unique()) == {0, 1}
    assert df[TARGET].sum() == 30
    assert "class" not in df.columns


def test_clean_removes_missing_rows():
    raw = make_df()
    raw.loc[3, "duration_months"] = float("nan")
    df = clean(raw)
    assert len(df) == 99
    assert df.isna().sum().sum() == 0


def test_split_is_stratified():
    train, test = split(clean(make_df()), test_size=0.2, seed=42)
    assert len(test) == 20
    assert test[TARGET].mean() == pytest.approx(0.3, abs=0.01)
    assert set(train.index).isdisjoint(test.index)


def test_split_is_reproducible():
    df = clean(make_df())
    _, test_a = split(df, test_size=0.2, seed=42)
    _, test_b = split(df, test_size=0.2, seed=42)
    assert list(test_a.index) == list(test_b.index)