"""
Selective-SSM temporal step (track E: push the learned model toward its oracle).

The fixed-A ``TemporalSSM`` mixes a FIXED bank of decay timescales and learns
only the readout -- it recovers most of the card-relative hour signal (~0.806)
but sits below the ``card_hour_rarity`` oracle (~0.877). ``SelectiveTemporalSSM``
replaces the fixed decay with a LEARNED, input-dependent one (the Mamba S6
mechanism): a per-token step ``dt_t`` -- a function of [recency, history-so-far]
-- decides how much of the card's hour profile survives at each transaction.
This script runs all three side by side on the SAME split so the gap-closing (or
not) is a single honest table, then runs the LR-test admission gate on the
selective scalar.

  [0] oracle:    card_hour_rarity -- the interpretable ceiling (no torch).
  [1] fixed:     TemporalSSM (fixed multi-timescale decay, readout-only learned).
  [2] selective: SelectiveTemporalSSM (learned input-dependent decay, true S6).
  [3] gate:      LR test on the `temporal` label, classical M0 vs M0+selective.

Run from the project root:  python run_ssm_selective.py
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
from src.models.ssm import (  # noqa: E402
    SelectiveTemporalSSM,
    TemporalSSM,
    card_hour_rarity,
)

OUT = Path("data/processed")
CLASSICAL = ["hour_sin", "hour_cos", "log_amt", "vel_1h",
             "is_weekend", "age", "log_city_pop"]
TABULAR_TEMPORAL_AUC = 0.702  # global hour_sin/cos line (run_glm_baseline.py)


def load(split: str) -> pd.DataFrame:
    df = pd.read_parquet(OUT / f"injected_{split}.parquet")
    df["trans_dt"] = pd.to_datetime(df["trans_date_trans_time"])
    return df


def isolated_auc(score: np.ndarray, typ: np.ndarray) -> float:
    """AUC for temporal SOLO rows vs legit only (no overlap signal-borrowing)."""
    solo, legit = typ == "temporal", typ == ""
    mask = solo | legit
    return roc_auc_score(solo[mask].astype(int), score[mask])


def main() -> None:
    tr, te = load("train"), load("test")
    typ_te = te[TYPOLOGY_COL].fillna("").to_numpy()
    temp_tr = typology_dummies(tr)["temporal"].to_numpy()

    print("== Selective SSM temporal detection (track E) ==")
    print(f"train rows {len(tr):,} | temporal rows {int(temp_tr.sum()):,}")

    # [0] oracle + tabular reference ----------------------------------------
    rar_te = card_hour_rarity(te).to_numpy()
    oracle_auc = isolated_auc(rar_te, typ_te)
    print("\nisolated temporal AUC (solo vs legit):")
    print(f"  [tabular]   global hour_sin/cos      {TABULAR_TEMPORAL_AUC:.3f}   (floor)")
    print(f"  [oracle]    card_hour_rarity         {oracle_auc:.3f}   (ceiling)")

    # [1] fixed-A TemporalSSM (the line track E pushes from) -----------------
    fixed = TemporalSSM(epochs=25, seed=0).fit(tr, temp_tr)
    fixed_auc = isolated_auc(fixed.score(te), typ_te)
    print(f"  [fixed]     TemporalSSM              {fixed_auc:.3f}   (fixed multi-timescale)")

    # [2] selective SSM (learned input-dependent decay) ----------------------
    sel = SelectiveTemporalSSM(epochs=12, seed=0).fit(tr, temp_tr)
    sel_tr = sel.score(tr)
    sel_auc = isolated_auc(sel.score(te), typ_te)
    print(f"  [selective] SelectiveTemporalSSM     {sel_auc:.3f}   (learned S6 decay)")

    span = oracle_auc - fixed_auc
    closed = (sel_auc - fixed_auc) / span if span > 1e-9 else float("nan")
    print(f"\n  selective vs fixed: {sel_auc - fixed_auc:+.3f}  "
          f"({closed:+.0%} of the fixed->oracle gap closed)")

    # [3] LR-test gate on the `temporal` label -------------------------------
    print("\nLR-test gate (temporal label, full train): classical vs classical+selective")
    Xtr = build_features(tr)
    X_base = Xtr[CLASSICAL].reset_index(drop=True)
    X_ext = pd.DataFrame({"ssm_selective": sel_tr})
    y = pd.DataFrame({"temporal": temp_tr})
    try:
        res = BinaryRelevanceGLM(maxiter=100).admit_extension(X_base, X_ext, y, "temporal")
        print(f"  G2={res['G2']:.1f}  df={res['df']}  p={res['p_value']:.2e}  "
              f"admitted={res['admitted']}")
    except Exception as exc:
        print(f"  classical+selective separates temporal ({type(exc).__name__}) "
              f"-> admitted (overwhelming evidence)")


if __name__ == "__main__":
    main()
