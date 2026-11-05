from __future__ import annotations

import numpy as np
import pandas as pd

FEATURES = [
    "LB",
    "AC.1",
    "FM.1",
    "UC.1",
    "DL.1",
    "DS.1",
    "DP.1",
    "ASTV",
    "MSTV",
    "ALTV",
    "MLTV",
    "Width",
    "Min",
    "Max",
    "Nmax",
    "Nzeros",
    "Mode",
    "Mean",
    "Median",
    "Variance",
    "Tendency",
]


def load_ctg(path: str) -> tuple[np.ndarray, np.ndarray]:
    """UCI Cardiotocography (2126 fetal records); y = NSP 0 normal, 1 suspect, 2 pathologic."""
    d = pd.read_excel(path, sheet_name="Data", header=1).dropna(subset=["NSP"])
    return d[FEATURES].to_numpy(float), d["NSP"].to_numpy(int) - 1
