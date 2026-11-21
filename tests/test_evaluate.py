from src.models.evaluate import decide


def test_rejects_candidate_below_threshold():
    promote, reason = decide(0.65, None, min_auc=0.70)
    assert not promote
    assert "seuil" in reason


def test_promotes_first_candidate_above_threshold():
    promote, _ = decide(0.75, None, min_auc=0.70)
    assert promote


def test_promotes_candidate_better_than_champion():
    promote, _ = decide(0.82, 0.80, min_auc=0.70)
    assert promote


def test_keeps_champion_when_candidate_is_not_better():
    assert not decide(0.78, 0.80, min_auc=0.70)[0]
    assert not decide(0.80, 0.80, min_auc=0.70)[0]


def test_threshold_applies_even_if_champion_is_worse():
    assert not decide(0.60, 0.50, min_auc=0.70)[0]
