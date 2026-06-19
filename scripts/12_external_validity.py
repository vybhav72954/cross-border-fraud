"""
Track F — IEEE-CIS external-validity fold.

The controlled injection is Sparkov-specific and does NOT transfer to anonymized
IEEE-CIS (no merchant id / lat-long / clean card key), so this is NOT a bake-off
re-run. It asks the two questions that CAN be asked externally:

  PREMISE CHECK -- does any single representation cleanly separate REAL ``isFraud``?
      On the PLANTED signatures the matched representation hits 0.84-0.96 in
      isolation. If on real fraud every representation is only modest and
      overlapping, real fraud is entangled across typologies -- which is exactly
      why the project plants clean signatures to measure recovery.

  REPRESENTATION TRANSFER -- do the per-uid sequence SSMs (velocity / temporal),
      built only from a timestamp + amount + the surrogate uid, carry signal on
      real ``isFraud`` ABOVE a tabular GLM, via the SAME LR-gate? Decisive bar is
      held-out ΔAUC (effect size), not raw p -- per RESULTS.md §9.2, at n~10^5 the
      LR test admits on negligible side-effects.

Validates premise + representation relevance, NOT the which-representation-recovers
-which thesis (that needs the answer key, which only injection provides).

Run from project root (after placing train_transaction.csv under data/raw/ieee/):
    python scripts/12_external_validity.py
    python scripts/12_external_validity.py --nrows 100000 --no-ssm   # fast premise smoke
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler

# Windows consoles default to cp1252; write UTF-8 so non-ASCII output (Δ, §, ->)
# in help text / progress lines can't raise UnicodeEncodeError.
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

sys.path.insert(0, ".")
from src.external import (  # noqa: E402
    load_ieee, time_split, build_ieee_features,
)
from src.models.ssm import (  # noqa: E402
    card_hour_rarity, card_rate_states, TemporalSSM, VelocitySSM,
)
from src.models.glm import BinaryRelevanceGLM  # noqa: E402

OUT = Path("results")   # json output (IEEE inputs read via src/external.py)

# Planted-signature reference points (RESULTS.md bake-off grid) -- the contrast
# that makes the premise check legible: clean separation on planted vs entangled
# on real fraud.
PLANTED = {
    "ring (GNN oracle)": 0.959, "temporal (oracle)": 0.877,
    "temporal (SSM)": 0.806, "velocity (SSM)": 0.913,
}


def fit_auc(X_tr: np.ndarray, y_tr: np.ndarray, X_te: np.ndarray, y_te: np.ndarray) -> float:
    """Standardised, unweighted logit (per §9 — balancing wrecks calibration);
    return held-out isFraud AUC."""
    sc = StandardScaler().fit(X_tr)
    lr = LogisticRegression(max_iter=3000, C=1.0).fit(sc.transform(X_tr), y_tr)
    return float(roc_auc_score(y_te, lr.predict_proba(sc.transform(X_te))[:, 1]))


def auc(y: np.ndarray, s: np.ndarray) -> float:
    return float(roc_auc_score(y, s))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--nrows", type=int, default=None, help="cap rows (smoke)")
    ap.add_argument("--frac", type=float, default=0.8, help="train fraction (time-ordered)")
    ap.add_argument("--epochs", type=int, default=25, help="SSM readout epochs")
    ap.add_argument("--no-ssm", action="store_true", help="premise check only (skip torch SSMs)")
    args = ap.parse_args()

    print("== IEEE-CIS external-validity fold (track F) ==")
    df = load_ieee(nrows=args.nrows)
    n_uid = df["cc_num"].nunique()
    seq = df.groupby("cc_num").size()
    print(f"rows {len(df):,} | isFraud {df['isFraud'].mean()*100:.2f}% | "
          f"uids {n_uid:,} | singleton-uid {float((seq == 1).mean())*100:.1f}% | "
          f"median seq {int(seq.median())}")

    tr, te = time_split(df, frac=args.frac)
    y_tr = tr["isFraud"].to_numpy().astype(int)
    y_te = te["isFraud"].to_numpy().astype(int)
    print(f"time-split: train {len(tr):,} ({y_tr.mean()*100:.2f}% fraud) | "
          f"test {len(te):,} ({y_te.mean()*100:.2f}% fraud)")

    results: dict = {"meta": {"rows": int(len(df)), "n_uid": int(n_uid),
                              "frac": args.frac, "test_fraud_rate": float(y_te.mean())}}

    # ── representation scalars on the held-out test split ────────────────────
    rar_te = card_hour_rarity(te).to_numpy()                      # temporal oracle
    rate_te = np.log1p(card_rate_states(te)).max(axis=1)          # velocity oracle
    rar_tr = card_hour_rarity(tr).to_numpy()
    rate_tr = np.log1p(card_rate_states(tr)).max(axis=1)

    ssm_scores = {}
    if not args.no_ssm:
        print("\nfitting per-uid sequence SSMs on real isFraud ...")
        tssm = TemporalSSM(epochs=args.epochs, seed=0).fit(tr, y_tr)
        vssm = VelocitySSM(epochs=args.epochs, seed=0).fit(tr, y_tr)
        ssm_scores = {
            "temporal_ssm": (tssm.score(tr), tssm.score(te)),
            "velocity_ssm": (vssm.score(tr), vssm.score(te)),
        }

    # ── [1] premise check: per-representation isFraud AUC (test) ─────────────
    print("\n[1] premise check -- single-representation isFraud AUC (held-out test)")
    Xf_tr = build_ieee_features(tr, include_counts=True)
    Xf_te = build_ieee_features(te, include_counts=True)
    Xf_te = Xf_te.reindex(columns=Xf_tr.columns, fill_value=0.0)  # align dummies
    tab_full = fit_auc(Xf_tr.to_numpy(), y_tr, Xf_te.to_numpy(), y_te)

    premise = {
        "tabular_M0_full (amt+hour+ProductCD+C1..14)": tab_full,
        "card_hour_rarity (temporal oracle)": auc(y_te, rar_te),
        "max_decayed_rate (velocity oracle)": auc(y_te, rate_te),
    }
    for name, (_, s_te) in ssm_scores.items():
        premise[f"{name} (learned)"] = auc(y_te, s_te)
    for name, a in premise.items():
        print(f"    {name:48s} {a:.3f}")
    print("    --- planted-signature reference (RESULTS.md bake-off) ---")
    for name, a in PLANTED.items():
        print(f"    {name:48s} {a:.3f}")
    results["premise_auc"] = premise
    results["planted_reference"] = PLANTED

    if args.no_ssm:
        _dump(results)
        return

    # ── [2] LR-gate + held-out ΔAUC: do the SSM scalars add over tabular? ────
    print("\n[2] representation transfer -- LR-gate + held-out ΔAUC on isFraud")
    ext_tr = pd.DataFrame({k: v[0] for k, v in ssm_scores.items()})
    ext_te_np = np.column_stack([ssm_scores[k][1] for k in ssm_scores])
    y_df = pd.DataFrame({"isFraud": y_tr})

    results["gate"], results["heldout"] = {}, {}
    for tag, with_counts in [("M0_min", False), ("M0_full", True)]:
        Xb_tr = build_ieee_features(tr, include_counts=with_counts)
        Xb_te = build_ieee_features(te, include_counts=with_counts)
        Xb_te = Xb_te.reindex(columns=Xb_tr.columns, fill_value=0.0)
        base_r = Xb_tr.reset_index(drop=True)

        # gate (statsmodels LR test on the train split)
        try:
            g = BinaryRelevanceGLM(maxiter=100).admit_extension(
                base_r, ext_tr.reset_index(drop=True), y_df, "isFraud")
            gate = {"G2": g["G2"], "df": g["df"], "p_value": g["p_value"],
                    "admitted": g["admitted"]}
        except Exception as exc:  # separation -> overwhelming evidence
            gate = {"error": type(exc).__name__, "admitted": True}

        # decisive bar: held-out ΔAUC (effect size, per §9.2)
        a0 = fit_auc(Xb_tr.to_numpy(), y_tr, Xb_te.to_numpy(), y_te)
        a1 = fit_auc(np.column_stack([Xb_tr.to_numpy(), ext_tr.to_numpy()]), y_tr,
                     np.column_stack([Xb_te.to_numpy(), ext_te_np]), y_te)
        results["gate"][tag] = gate
        results["heldout"][tag] = {"auc_base": a0, "auc_with_ssm": a1, "delta": a1 - a0}
        g_str = (f"G2={gate['G2']:.1f} df={gate['df']} p={gate['p_value']:.2e} "
                 f"admitted={gate['admitted']}" if "G2" in gate
                 else f"separation -> admitted")
        print(f"    {tag}: {g_str}")
        print(f"        held-out isFraud AUC  base={a0:.3f}  +SSM={a1:.3f}  Δ={a1-a0:+.3f}")

    _dump(results)


def _dump(results: dict) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    path = OUT / "external_validity_results.json"
    path.write_text(json.dumps(results, indent=2))
    print(f"\nwrote {path}")


if __name__ == "__main__":
    main()
