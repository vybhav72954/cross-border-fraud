"""
GNN ring-detection step of the controlled benchmark.

The tabular GLM baseline whiffs on `ring` (isolated AUC ~0.58) because no single
row reveals a ring -- a ring is a TIME-WINDOWED merchant fan-in (cards_per_ring
distinct cards at one merchant within a short window), and per-row tabular
features can't see merchant-side structure. This script shows the graph view
recovers it, two ways, then runs the LR-test admission gate.

  [1] structural: windowed merchant-node degree (merch_win_cards / _txns).
      The honest graph signal -- what one bipartite message-passing step computes.
  [2] learned:    RingSAGE (GraphSAGE over a card<->merchant-time-bucket graph)
      derives the fan-in end-to-end from raw node features, no degree handed in.
  [3] gate:       LR test on the `ring` label, compact classical M0 vs M0+GNN,
      via the production BinaryRelevanceGLM.admit_extension.

Run from the project root:  python scripts/03_gnn_ring.py
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

sys.path.insert(0, ".")
from src.features import build_features  # noqa: E402
from src.inject import typology_dummies, TYPOLOGY_COL  # noqa: E402
from src.models.gnn import merchant_window_features, RingSAGE  # noqa: E402
from src.models.glm import BinaryRelevanceGLM  # noqa: E402

OUT = Path("data/processed")
WINDOW_H = 2.0  # match inject_ring's window_hours
CLASSICAL = ["vel_1h", "log_amt", "merch_dist_km",
             "hour_sin", "hour_cos", "age", "log_city_pop"]
TABULAR_RING_AUC = 0.582  # from scripts/01_glm_baseline.py -- the line to beat


def load(split: str) -> pd.DataFrame:
    df = pd.read_parquet(OUT / f"injected_{split}.parquet")
    df["trans_dt"] = pd.to_datetime(df["trans_date_trans_time"])
    return df


def isolated_auc(score: np.ndarray, typ: np.ndarray) -> float:
    """AUC for ring SOLO rows vs legit only (no overlap signal-borrowing)."""
    solo, legit = typ == "ring", typ == ""
    mask = solo | legit
    return roc_auc_score(solo[mask].astype(int), score[mask])


def main() -> None:
    tr, te = load("train"), load("test")
    typ_tr = tr[TYPOLOGY_COL].fillna("").to_numpy()
    typ_te = te[TYPOLOGY_COL].fillna("").to_numpy()
    ring_tr = typology_dummies(tr)["ring"].to_numpy()

    print("== GNN ring detection (controlled benchmark) ==")
    print(f"train rows {len(tr):,} | ring rows {int(ring_tr.sum()):,} | "
          f"window +/-{WINDOW_H}h")

    # [1] structural: windowed merchant-node degree -------------------------
    fan_tr = merchant_window_features(tr, window_hours=WINDOW_H)
    fan_te = merchant_window_features(te, window_hours=WINDOW_H)
    print("\n[1] windowed merchant fan-in (structural)  "
          "isolated ring AUC (solo vs legit)")
    for col in ["merch_win_cards", "merch_win_txns"]:
        auc = isolated_auc(fan_te[col].to_numpy(), typ_te)
        print(f"    {col:18s} {auc:.3f}")
    print(f"    {'tabular baseline':18s} {TABULAR_RING_AUC:.3f}   <- line to beat")

    # [2] learned: RingSAGE --------------------------------------------------
    print("\n[2] RingSAGE (GraphSAGE, card<->merchant-time-bucket)")
    sage = RingSAGE(window_hours=WINDOW_H, epochs=60, seed=0).fit(tr, ring_tr)
    sage_te = sage.score(te)
    print(f"    isolated ring AUC  {isolated_auc(sage_te, typ_te):.3f}")

    # [3] LR-test admission gate on the `ring` label ------------------------
    print("\n[3] LR-test gate (ring label, full train): classical vs classical+GNN")
    Xtr = build_features(tr)
    X_base = Xtr[CLASSICAL].reset_index(drop=True)
    X_ext = fan_tr.reset_index(drop=True)  # the windowed-degree scalars
    y = pd.DataFrame({"ring": ring_tr})
    try:
        res = BinaryRelevanceGLM(maxiter=100).admit_extension(X_base, X_ext, y, "ring")
        print(f"    G2={res['G2']:.1f}  df={res['df']}  p={res['p_value']:.2e}  "
              f"admitted={res['admitted']}")
    except Exception as exc:  # perfect/quasi separation => infinite evidence
        print(f"    classical+GNN separates ring (separation: {type(exc).__name__}) "
              f"-> admitted (overwhelming evidence)")


if __name__ == "__main__":
    main()
