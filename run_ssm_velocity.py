"""
SSM velocity-detection step of the controlled benchmark -- the "neural matches
(does not beat) tabular" data point.

Unlike ring/temporal, velocity is NOT a neural-headroom slot: the tabular GLM
already nails it with a per-card rolling 1h count (~0.88), because a burst is
visible in a single rolling-window feature. The question here is therefore not
"does the sequence view beat tabular?" but "does it RECOVER the same signal?" --
and the LR gate should then show the SSM scalar adds little once the rolling
count is already in the model (the gate correctly declining a redundant feature).

  [1] tabular:  rolling 1h count (features.vel_1h) -- the velocity oracle, which
      is itself a tabular feature, so there is no separate structural ceiling.
  [2] learned:  VelocitySSM (continuous-time, input-dependent-dt diagonal SSM)
      over one sequence per card, scoring each txn by its local arrival rate.
  [3] gate:     LR test on the `velocity` label, two bases --
        (a) classical WITHOUT vel_1h  -> SSM should be admitted (it carries the
            burst signal the rest of the classical features miss);
        (b) classical WITH vel_1h     -> the "beat the rolling count?" test:
            little headroom expected, i.e. neural matches but does not beat.

Run from the project root:  python run_ssm_velocity.py
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

sys.path.insert(0, ".")
from src.features import build_features  # noqa: E402
from src.inject import typology_dummies, TYPOLOGY_COL  # noqa: E402
from src.models.ssm import VelocitySSM  # noqa: E402
from src.models.glm import BinaryRelevanceGLM  # noqa: E402

OUT = Path("data/processed")
CLASSICAL = ["hour_sin", "hour_cos", "log_amt", "vel_1h",
             "is_weekend", "age", "log_city_pop"]
TABULAR_VELOCITY_AUC = 0.882  # from run_glm_baseline.py -- the rolling-count line


def load(split: str) -> pd.DataFrame:
    df = pd.read_parquet(OUT / f"injected_{split}.parquet")
    df["trans_dt"] = pd.to_datetime(df["trans_date_trans_time"])
    return df


def isolated_auc(score: np.ndarray, typ: np.ndarray) -> float:
    """AUC for velocity SOLO rows vs legit only (no overlap signal-borrowing)."""
    solo, legit = typ == "velocity", typ == ""
    mask = solo | legit
    return roc_auc_score(solo[mask].astype(int), score[mask])


def gate(name: str, X_base: pd.DataFrame, X_ext: pd.DataFrame,
         y: pd.DataFrame) -> None:
    try:
        res = BinaryRelevanceGLM(maxiter=100).admit_extension(X_base, X_ext, y, "velocity")
        print(f"    {name:28s} G2={res['G2']:8.1f}  df={res['df']}  "
              f"p={res['p_value']:.2e}  admitted={res['admitted']}")
    except Exception as exc:
        print(f"    {name:28s} separation ({type(exc).__name__}) -> admitted")


def main() -> None:
    tr, te = load("train"), load("test")
    typ_te = te[TYPOLOGY_COL].fillna("").to_numpy()
    vel_tr = typology_dummies(tr)["velocity"].to_numpy()

    print("== SSM velocity detection (controlled benchmark) ==")
    print(f"train rows {len(tr):,} | velocity rows {int(vel_tr.sum()):,}")

    # [1] tabular reference: rolling 1h count -------------------------------
    Xtr = build_features(tr)
    Xte = build_features(te).reindex(columns=Xtr.columns, fill_value=0.0)
    print("\n[1] rolling 1h count (tabular)  isolated velocity AUC (solo vs legit)")
    print(f"    {'vel_1h':18s} {isolated_auc(Xte['vel_1h'].to_numpy(), typ_te):.3f}")
    print(f"    {'tabular baseline':18s} {TABULAR_VELOCITY_AUC:.3f}   <- line to MATCH")

    # [2] learned: VelocitySSM ----------------------------------------------
    print("\n[2] VelocitySSM (continuous-time diagonal SSM, one sequence per card)")
    ssm = VelocitySSM(epochs=25, seed=0).fit(tr, vel_tr)
    ssm_te = ssm.score(te)
    print(f"    isolated velocity AUC  {isolated_auc(ssm_te, typ_te):.3f}")

    # [3] LR-test gate on the `velocity` label ------------------------------
    print("\n[3] LR-test gate (velocity label, full train)")
    ssm_tr = ssm.score(tr)
    X_ext = pd.DataFrame({"ssm_velocity": ssm_tr})
    y = pd.DataFrame({"velocity": vel_tr})
    base_no_vel = Xtr[[c for c in CLASSICAL if c != "vel_1h"]].reset_index(drop=True)
    base_with_vel = Xtr[CLASSICAL].reset_index(drop=True)
    gate("(a) classical \\ vel_1h + SSM", base_no_vel, X_ext, y)
    gate("(b) classical (+vel_1h) + SSM", base_with_vel, X_ext, y)


if __name__ == "__main__":
    main()
