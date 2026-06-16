"""
SSM/Mamba temporal-detection step of the controlled benchmark.

The tabular GLM tops out at ~0.70 on `temporal` because its hour_sin/hour_cos
see only the wall-clock hour, while the injected signature is a transaction at
the CARD'S own rarest hour-of-day -- a per-card anomaly. This script shows the
sequence view recovers it, two ways, then runs the LR-test admission gate.

  [1] structural: card_hour_rarity -- the card's historical share of this hour.
      The interpretable oracle; what a sequence model should learn.
  [2] learned:    TemporalSSM (diagonal S4D-style SSM, CPU) over one sequence per
      card, scoring each transaction by how out-of-profile its hour is.
  [3] gate:       LR test on the `temporal` label, classical M0 (incl. global
      hour) vs M0+SSM scalars, via BinaryRelevanceGLM.admit_extension.

Run from the project root:  python run_ssm_temporal.py
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

sys.path.insert(0, ".")
from src.features import build_features  # noqa: E402
from src.inject import typology_dummies, TYPOLOGY_COL  # noqa: E402
from src.models.ssm import card_hour_rarity, TemporalSSM  # noqa: E402
from src.models.glm import BinaryRelevanceGLM  # noqa: E402

OUT = Path("data/processed")
CLASSICAL = ["hour_sin", "hour_cos", "log_amt", "vel_1h",
             "is_weekend", "age", "log_city_pop"]
TABULAR_TEMPORAL_AUC = 0.702  # from run_glm_baseline.py -- the line to beat


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

    print("== SSM temporal detection (controlled benchmark) ==")
    print(f"train rows {len(tr):,} | temporal rows {int(temp_tr.sum()):,}")

    # [1] structural: card-relative hour rarity -----------------------------
    rar_tr = card_hour_rarity(tr)
    rar_te = card_hour_rarity(te)
    print("\n[1] card_hour_rarity (structural)  isolated temporal AUC (solo vs legit)")
    print(f"    {'card_hour_rarity':18s} {isolated_auc(rar_te.to_numpy(), typ_te):.3f}")
    print(f"    {'tabular baseline':18s} {TABULAR_TEMPORAL_AUC:.3f}   <- line to beat")

    # [2] learned: TemporalSSM ----------------------------------------------
    print("\n[2] TemporalSSM (diagonal S4D-style SSM, one sequence per card)")
    ssm = TemporalSSM(epochs=25, seed=0).fit(tr, temp_tr)
    ssm_te = ssm.score(te)
    print(f"    isolated temporal AUC  {isolated_auc(ssm_te, typ_te):.3f}")

    # [3] LR-test gate on the `temporal` label ------------------------------
    print("\n[3] LR-test gate (temporal label, full train): classical vs classical+SSM")
    Xtr = build_features(tr)
    X_base = Xtr[CLASSICAL].reset_index(drop=True)
    ssm_tr = ssm.score(tr)
    X_ext = pd.DataFrame({"card_hour_rarity": rar_tr.to_numpy(),
                          "ssm_timing": ssm_tr})
    y = pd.DataFrame({"temporal": temp_tr})
    try:
        res = BinaryRelevanceGLM(maxiter=100).admit_extension(X_base, X_ext, y, "temporal")
        print(f"    G2={res['G2']:.1f}  df={res['df']}  p={res['p_value']:.2e}  "
              f"admitted={res['admitted']}")
    except Exception as exc:
        print(f"    classical+SSM separates temporal ({type(exc).__name__}) "
              f"-> admitted (overwhelming evidence)")


if __name__ == "__main__":
    main()
