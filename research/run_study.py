"""UCI Cardiotocography (n=2126, NSP normal/suspect/pathologic): conformal triage.

usage: python research/run_study.py /path/to/CTG.xls
50 random splits: 50 % train, 25 % calibration, 25 % test (stratified by class).
"""

import json
import sys
import warnings
from pathlib import Path

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from diagtriage.conformal import calibrate_classwise, calibrate_marginal, prediction_sets, triage
from diagtriage.data import load_ctg

warnings.filterwarnings("ignore")
ALPHAS = (0.05, 0.1)
REPS = 50
PATH_CLASS = 2


def summarize(sets, y, P):
    n = len(y)
    cov = sets[np.arange(n), y]
    refer = triage(sets).astype(bool)
    auto = ~refer
    pred = sets.argmax(1)
    path = y == PATH_CLASS
    return {
        "coverage": float(cov.mean()),
        "coverage_pathologic": float(cov[path].mean()),
        "referral_rate": float(refer.mean()),
        "accuracy_on_automatic": float((pred[auto] == y[auto]).mean()) if auto.any() else None,
        "pathologic_auto_missed": float(((pred == 0) & auto & path).sum() / max(path.sum(), 1)),
        "pathologic_referred_or_correct": float((refer | (pred == PATH_CLASS))[path].mean()),
    }


def main(path: str):
    X, y = load_ctg(path)
    out = {
        "note": "real UCI CTG records; classes: normal 1655, suspect 295, pathologic 176; "
        "50 random splits 50/25/25",
        "n": len(y),
        "models": {},
    }
    mk = {
        "logistic": lambda: make_pipeline(StandardScaler(), LogisticRegression(max_iter=3000)),
        "forest": lambda: RandomForestClassifier(300, min_samples_leaf=2, random_state=0),
    }
    for mname, make in mk.items():
        acc = {"argmax": []}
        for a in ALPHAS:
            acc[f"marginal_{a}"] = []
            acc[f"classwise_{a}"] = []
        base_acc = []
        for r in range(REPS):
            Xtr, Xr, ytr, yr = train_test_split(X, y, test_size=0.5, stratify=y, random_state=r)
            Xc, Xt, yc, yt = train_test_split(Xr, yr, test_size=0.5, stratify=yr, random_state=r)
            m = make().fit(Xtr, ytr)
            Pc, Pt = m.predict_proba(Xc), m.predict_proba(Xt)
            base_acc.append(float((Pt.argmax(1) == yt).mean()))
            onehot = np.eye(3, dtype=bool)[Pt.argmax(1)]
            acc["argmax"].append(summarize(onehot, yt, Pt))
            for a in ALPHAS:
                acc[f"marginal_{a}"].append(
                    summarize(prediction_sets(Pt, calibrate_marginal(Pc, yc, a)), yt, Pt)
                )
                acc[f"classwise_{a}"].append(
                    summarize(prediction_sets(Pt, calibrate_classwise(Pc, yc, a)), yt, Pt)
                )
        res = {"accuracy_argmax": float(np.mean(base_acc)), "rules": {}}
        for rule, lst in acc.items():
            res["rules"][rule] = {
                k: float(np.mean([d[k] for d in lst if d[k] is not None])) for k in lst[0]
            }
        out["models"][mname] = res
        print(mname, round(res["accuracy_argmax"], 3))
        for rule, v in res["rules"].items():
            print("  ", rule, {k: round(x, 3) for k, x in v.items()})
    Path(__file__).with_name("results.json").write_text(json.dumps(out, indent=1))


if __name__ == "__main__":
    main(sys.argv[1])
