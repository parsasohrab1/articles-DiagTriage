import numpy as np

from diagtriage.conformal import calibrate_classwise, calibrate_marginal, prediction_sets, triage


def _synthetic(n, rng):
    y = rng.integers(0, 3, n)
    logits = rng.normal(0, 1, (n, 3))
    logits[np.arange(n), y] += 1.5
    P = np.exp(logits) / np.exp(logits).sum(1, keepdims=True)
    return P, y


def test_marginal_coverage_close_to_target():
    rng = np.random.default_rng(0)
    Pc, yc = _synthetic(3000, rng)
    Pt, yt = _synthetic(6000, rng)
    s = prediction_sets(Pt, calibrate_marginal(Pc, yc, 0.1))
    assert abs(s[np.arange(len(yt)), yt].mean() - 0.9) < 0.02


def test_classwise_coverage_per_class():
    rng = np.random.default_rng(1)
    Pc, yc = _synthetic(6000, rng)
    Pt, yt = _synthetic(9000, rng)
    s = prediction_sets(Pt, calibrate_classwise(Pc, yc, 0.1))
    for c in range(3):
        m = yt == c
        assert abs(s[m, c].mean() - 0.9) < 0.03


def test_triage_flags_non_singletons():
    sets = np.array([[1, 0, 0], [1, 1, 0], [0, 0, 0]], dtype=bool)
    assert list(triage(sets)) == [0, 1, 1]
