"""
GLM binary-relevance baseline on the controlled injected dataset.

Fits one logistic regression per injected typology on the tabular design matrix
(src/features.py) and reports per-typology test AUC/AP + the multi-label suite.
This is the reference line the GNN (ring) and Mamba (velocity/temporal) must beat.

Expectation: the tabular baseline should NAIL `geo` (distance feature) and
`velocity` (rolling 1h count), partly catch `temporal`/`category`, and WHIFF on
`ring` — no single row reveals a ring, so that headroom belongs to the GNN.

Uses sklearn L2-logistic here because `geo` is perfectly separable by distance,
which makes a plain statsmodels MLE Logit raise PerfectSeparationError. The
statsmodels BinaryRelevanceGLM in src/models/glm.py is reserved for the
LR-test admission gate when GNN/Mamba scalars are added.

Run from the project root:  python run_glm_baseline.py
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, ".")
from src.features import build_features  # noqa: E402
from src.inject import typology_dummies, is_cross_border, TYPOLOGY_COL, TYPOLOGIES  # noqa: E402
from src.evaluation import multi_label_report  # noqa: E402

OUT = Path("data/processed")


def load(split: str) -> pd.DataFrame:
    df = pd.read_parquet(OUT / f"injected_{split}.parquet")
    df["trans_dt"] = pd.to_datetime(df["trans_date_trans_time"])  # build_features needs it
    return df


def main() -> None:
    tr, te = load("train"), load("test")

    Xtr = build_features(tr)
    Xte = build_features(te).reindex(columns=Xtr.columns, fill_value=0.0)
    Ytr = typology_dummies(tr)[TYPOLOGIES]
    Yte = typology_dummies(te)[TYPOLOGIES]

    scaler = StandardScaler().fit(Xtr)
    Xtr_s, Xte_s = scaler.transform(Xtr), scaler.transform(Xte)

    scores = np.zeros((len(te), len(TYPOLOGIES)))
    for i, typ in enumerate(TYPOLOGIES):
        clf = LogisticRegression(max_iter=1000, class_weight="balanced")
        clf.fit(Xtr_s, Ytr[typ].to_numpy())
        scores[:, i] = clf.predict_proba(Xte_s)[:, 1]
    preds = (scores >= 0.5).astype(int)

    # Isolated detectability: each signature's SOLO rows vs legit, so a typology
    # can't borrow another's signal through an overlap (e.g. geo+ring via distance).
    # This is the honest "can representation X recover signature Y?" grid.
    typ_te = te[TYPOLOGY_COL].fillna("").to_numpy()
    legit = typ_te == ""
    print(f"\nGLM baseline — isolated AUC (signature solo rows vs legit)   "
          f"[{Xtr.shape[1]} tabular features]")
    print(f"{'typology':10s} {'AUC':>7s} {'solo+':>8s}   expectation")
    notes = {"ring": "needs graph -> GNN", "velocity": "tabular (rolling count)",
             "temporal": "weak; -> Mamba", "category": "weak; rare-for-card",
             "geo": "tabular (distance)"}
    for i, typ in enumerate(TYPOLOGIES):
        solo = typ_te == typ
        mask = solo | legit
        auc = roc_auc_score(solo[mask].astype(int), scores[mask, i])
        print(f"{typ:10s} {auc:7.3f} {int(solo.sum()):8,d}   {notes[typ]}")

    rep = multi_label_report(Yte.to_numpy(), preds, scores, label_names=TYPOLOGIES)
    cb = is_cross_border(te).to_numpy()
    print(f"\nmulti-label view (incl. overlaps): mean_auc={rep['mean_auc']:.3f}  "
          f"LRAP={rep['label_ranking_ap']:.3f}  cross_border test rows={int(cb.sum()):,}")


if __name__ == "__main__":
    main()
