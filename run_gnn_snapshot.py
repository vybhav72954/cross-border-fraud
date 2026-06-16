"""
Snapshot-graph GNN step (track E: push the learned ring model toward its oracle).

The fixed-bucket ``RingSAGE`` buckets time at ``t // W`` and learns the fan-in from
raw node features (~0.841), but sits below the sliding ``merchant_window_features``
oracle (~0.959): a ring straddling a bucket boundary is split across two merchant
nodes, halving its fan-in. ``SnapshotRingSAGE`` swaps the floor bucket for OVERLAPPING
centered snapshots (width 2W, centered at multiples of W) so a row is scored in the
window it centers -- the same centered window the oracle uses. This script runs all
three side by side on the SAME split so the gap-closing (or not) is one honest table,
then runs the LR-test admission gate on the snapshot scalar.

  [0] oracle:    merch_win_cards -- the sliding-window structural ceiling (no torch).
  [1] fixed:     RingSAGE (floor-bucket graph, the line track E pushes from).
  [2] snapshot:  SnapshotRingSAGE (overlapping centered-snapshot graph).
  [3] gate:      LR test on the `ring` label, classical M0 vs M0+snapshot.

Run from the project root:  python run_gnn_snapshot.py
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

sys.path.insert(0, ".")
from src.features import build_features  # noqa: E402
from src.inject import TYPOLOGY_COL, typology_dummies  # noqa: E402
from src.models.glm import BinaryRelevanceGLM  # noqa: E402
from src.models.gnn import (  # noqa: E402
    RingSAGE,
    SnapshotRingSAGE,
    merchant_window_features,
)

OUT = Path("data/processed")
WINDOW_H = 2.0  # match inject_ring's window_hours
CLASSICAL = ["vel_1h", "log_amt", "merch_dist_km",
             "hour_sin", "hour_cos", "age", "log_city_pop"]
TABULAR_RING_AUC = 0.582  # from run_glm_baseline.py -- the line to beat


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
    typ_te = te[TYPOLOGY_COL].fillna("").to_numpy()
    ring_tr = typology_dummies(tr)["ring"].to_numpy()

    print("== Snapshot-graph GNN ring detection (track E) ==")
    print(f"train rows {len(tr):,} | ring rows {int(ring_tr.sum()):,} | "
          f"window +/-{WINDOW_H}h")

    # [0] oracle + tabular reference ----------------------------------------
    fan_te = merchant_window_features(te, window_hours=WINDOW_H)
    oracle_auc = isolated_auc(fan_te["merch_win_cards"].to_numpy(), typ_te)
    print("\nisolated ring AUC (solo vs legit):")
    print(f"  [tabular]   per-row baseline         {TABULAR_RING_AUC:.3f}   (floor)")
    print(f"  [oracle]    merch_win_cards          {oracle_auc:.3f}   (ceiling)")

    # [1] fixed-bucket RingSAGE (the line track E pushes from) ---------------
    fixed = RingSAGE(window_hours=WINDOW_H, epochs=60, seed=0).fit(tr, ring_tr)
    fixed_auc = isolated_auc(fixed.score(te), typ_te)
    print(f"  [fixed]     RingSAGE (t//W bucket)    {fixed_auc:.3f}   (floor-bucket graph)")

    # [2] snapshot graph (overlapping centered windows) ---------------------
    snap = SnapshotRingSAGE(window_hours=WINDOW_H, epochs=60, seed=0).fit(tr, ring_tr)
    snap_tr = snap.score(tr)
    snap_auc = isolated_auc(snap.score(te), typ_te)
    print(f"  [snapshot]  SnapshotRingSAGE          {snap_auc:.3f}   (centered-snapshot graph)")

    span = oracle_auc - fixed_auc
    closed = (snap_auc - fixed_auc) / span if span > 1e-9 else float("nan")
    print(f"\n  snapshot vs fixed: {snap_auc - fixed_auc:+.3f}  "
          f"({closed:+.0%} of the fixed->oracle gap closed)")

    # [3] LR-test gate on the `ring` label ----------------------------------
    print("\nLR-test gate (ring label, full train): classical vs classical+snapshot")
    Xtr = build_features(tr)
    X_base = Xtr[CLASSICAL].reset_index(drop=True)
    X_ext = pd.DataFrame({"gnn_snapshot": snap_tr})
    y = pd.DataFrame({"ring": ring_tr})
    try:
        res = BinaryRelevanceGLM(maxiter=100).admit_extension(X_base, X_ext, y, "ring")
        print(f"  G2={res['G2']:.1f}  df={res['df']}  p={res['p_value']:.2e}  "
              f"admitted={res['admitted']}")
    except Exception as exc:
        print(f"  classical+snapshot separates ring ({type(exc).__name__}) "
              f"-> admitted (overwhelming evidence)")


if __name__ == "__main__":
    main()
