from __future__ import annotations

import numpy as np


def _q(scores: np.ndarray, alpha: float) -> float:
    n = len(scores)
    k = int(np.ceil((n + 1) * (1 - alpha)))
    return float(np.sort(scores)[k - 1]) if k <= n else float("inf")


def calibrate_marginal(P_cal: np.ndarray, y_cal: np.ndarray, alpha: float) -> float:
    """Threshold on the nonconformity 1 - p(true class), pooled over classes."""
    return _q(1 - P_cal[np.arange(len(y_cal)), y_cal], alpha)


def calibrate_classwise(P_cal: np.ndarray, y_cal: np.ndarray, alpha: float) -> np.ndarray:
    """Mondrian (class-conditional) thresholds: one per true class."""
    th = np.full(P_cal.shape[1], np.inf)
    for c in range(P_cal.shape[1]):
        m = y_cal == c
        if m.sum():
            th[c] = _q(1 - P_cal[m, c], alpha)
    return th


def prediction_sets(P: np.ndarray, thresholds) -> np.ndarray:
    """Boolean (n, n_classes): class c is in the set when 1 - p_c <= threshold (per class if array)."""
    th = np.asarray(thresholds)
    return (1 - P) <= th


def triage(sets: np.ndarray) -> np.ndarray:
    """0 = automatic (singleton set), 1 = refer to expert (empty or multi-label set)."""
    return (sets.sum(axis=1) != 1).astype(int)
