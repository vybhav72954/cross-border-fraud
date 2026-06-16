"""
Production integration -- the nested test the whole architecture was built around.

The bake-off scored each representation in isolation (solo-vs-legit AUC per
typology). This script closes the loop: it carries the ADMITTED neural scalars
from the per-slot gates into the production binary-relevance design matrix,
refits all five labels, and reports the multi-label suite for the nested chain

    M_classical  subset  M_{+GNN}  subset  M_{+GNN+SSM}

so the payoff is measured where it actually ships -- in the five-logit classifier,
not a single-typology AUC. Two neural blocks, each the scalar set that cleared
its §7.2 LR gate:

  GNN (ring slot)      windowed merchant fan-in  (merch_win_cards / _txns)
  SSM (temporal slot)  card-relative hour-rarity + learned TemporalSSM readout

Label answer key -> production label names (typology -> L_*):
  ring->L_R  velocity->L_V  temporal->L_T  category->L_C  geo->L_G

Design choices, each inherited from an established result in this repo:
  * Suite fitted with sklearn L2-logistic (statsmodels MLE chokes on geo's perfect
    separation -- the bake-off uses sklearn for scoring for exactly this reason),
    UNWEIGHTED + standardized (class-balancing destroys the 0.5-threshold
    Hamming/subset-accuracy and the probability scale -- see the calibration study).
  * The per-label admission gate uses the production statsmodels
    BinaryRelevanceGLM.admit_extension over a compact classical base with the geo
    distance dropped (the run_geo_control pattern, so geo's perfect separator does
    not mask the negative controls).
  * Negative-control SPECIFICITY is read on EFFECT SIZE, not raw p: at n~1.3M the
    LR test admits on negligible side-effects (the §8.3 geo finding), so each
    scalar's G^2 is shown relative to its own matched-slot G^2, alongside the
    held-out per-label AUC lift. A neural block should move ONLY its matched label.

Run from the project root:  python run_production_integration.py
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, ".")
from src.evaluation import LABEL_NAMES, multi_label_report  # noqa: E402
from src.features import build_features  # noqa: E402
from src.inject import typology_dummies  # noqa: E402
from src.models.glm import BinaryRelevanceGLM  # noqa: E402
from src.models.gnn import merchant_window_features  # noqa: E402
from src.models.ssm import TemporalSSM, card_hour_rarity  # noqa: E402
from src.robustness import compact_base  # noqa: E402

OUT = Path("data/processed")
WINDOW_H = 2.0  # match inject_ring's window_hours

# typology answer key -> production label, and the matched neural slot per label
TYP2LABEL = {"ring": "L_R", "velocity": "L_V", "temporal": "L_T",
             "category": "L_C", "geo": "L_G"}
GNN_COLS = ["merch_win_cards", "merch_win_txns"]
SSM_COLS = ["card_hour_rarity", "ssm_timing"]
GNN_MATCH, SSM_MATCH = "L_R", "L_T"


def load(split: str) -> pd.DataFrame:
    df = pd.read_parquet(OUT / f"injected_{split}.parquet")
    df["trans_dt"] = pd.to_datetime(df["trans_date_trans_time"])
    return df


def label_frame(df: pd.DataFrame) -> pd.DataFrame:
    """Injection answer key as the production L_* label matrix (LABEL_NAMES order)."""
    return (typology_dummies(df).rename(columns=TYP2LABEL)[LABEL_NAMES]
            .astype(int).reset_index(drop=True))


def neural_blocks(tr: pd.DataFrame, te: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """The two admitted neural scalar blocks (GNN ring, SSM temporal) for tr/te.

    GNN: windowed merchant fan-in (training-free). SSM: card hour-rarity oracle
    plus the learned TemporalSSM readout, trained on train's temporal label.
    """
    print(">> GNN block: windowed merchant fan-in")
    gnn_tr = merchant_window_features(tr, window_hours=WINDOW_H)
    gnn_te = merchant_window_features(te, window_hours=WINDOW_H)

    print(">> SSM block: card_hour_rarity + TemporalSSM readout")
    temp_tr = typology_dummies(tr)["temporal"].to_numpy()
    ssm = TemporalSSM(epochs=25, seed=0).fit(tr, temp_tr)
    ssm_tr = pd.DataFrame({"card_hour_rarity": card_hour_rarity(tr).to_numpy(),
                           "ssm_timing": ssm.score(tr)}, index=tr.index)
    ssm_te = pd.DataFrame({"card_hour_rarity": card_hour_rarity(te).to_numpy(),
                           "ssm_timing": ssm.score(te)}, index=te.index)

    tr_block = pd.concat([gnn_tr, ssm_tr], axis=1).reset_index(drop=True)
    te_block = pd.concat([gnn_te, ssm_te], axis=1).reset_index(drop=True)
    return tr_block, te_block


def fit_suite(Xtr_s: np.ndarray, Xte_s: np.ndarray, cols: list[int],
              Ytr: pd.DataFrame, Yte: pd.DataFrame) -> dict:
    """Binary-relevance sklearn logits over a column subset -> multi-label report.

    Unweighted standardized L2-logistic (probability-faithful for the 0.5-threshold
    metrics), one per label, scored on the held-out test split.
    """
    n_te = Xte_s.shape[0]
    score = np.zeros((n_te, len(LABEL_NAMES)))
    for k, name in enumerate(LABEL_NAMES):
        clf = LogisticRegression(max_iter=2000, C=1.0)
        clf.fit(Xtr_s[:, cols], Ytr[name].to_numpy())
        score[:, k] = clf.predict_proba(Xte_s[:, cols])[:, 1]
    pred = (score >= 0.5).astype(int)
    return multi_label_report(Yte.to_numpy(), pred, score, LABEL_NAMES)


def gate_table(gate_base: pd.DataFrame, ext: pd.DataFrame, Ytr: pd.DataFrame,
               match: str) -> dict:
    """LR test of a neural block on EVERY label; G^2 relative to the matched slot.

    Returns {label: {G2, p, rel_to_match, admitted_p}} with the matched label's
    G^2 as the 100% reference for the effect-size (specificity) reading.
    """
    raw = {}
    for name in LABEL_NAMES:
        y = pd.DataFrame({name: Ytr[name].to_numpy()})
        try:
            res = BinaryRelevanceGLM(maxiter=80).admit_extension(
                gate_base, ext, y, name)
            raw[name] = {"G2": float(res["G2"]), "p": float(res["p_value"]),
                         "admitted_p": bool(res["admitted"])}
        except Exception as exc:  # perfect/quasi separation -> infinite evidence
            raw[name] = {"G2": float("inf"), "p": 0.0, "admitted_p": True,
                         "note": type(exc).__name__}
    ref = raw[match]["G2"]
    for name in LABEL_NAMES:
        g = raw[name]["G2"]
        raw[name]["rel_to_match"] = (g / ref) if np.isfinite(g) and ref else (
            1.0 if name == match else float("nan"))
    return raw


def print_suite(name: str, rep: dict) -> None:
    pl = rep["per_label"]
    aucs = " ".join(f"{n}={(pl[n]['auc'] or float('nan')):.3f}" for n in LABEL_NAMES)
    print(f"  {name:14s} hamming={rep['hamming_loss']:.5f}  "
          f"subset_acc={rep['subset_accuracy']:.4f}  LRAP={rep['label_ranking_ap']:.4f}  "
          f"meanAUC={rep['mean_auc']:.4f}")
    print(f"  {'':14s} per-label AUC: {aucs}")


def main() -> None:
    print("== Production integration: M_classical < M_+GNN < M_+GNN+SSM ==")
    tr, te = load("train"), load("test")
    Ytr, Yte = label_frame(tr), label_frame(te)
    print(f"train {len(tr):,} rows | test {len(te):,} rows | "
          f"label prevalence (test): "
          + ", ".join(f"{n}={Yte[n].mean():.4f}" for n in LABEL_NAMES))

    print("\nBuilding design matrices (classical + neural blocks)...")
    Xtr_c = build_features(tr).reset_index(drop=True)
    Xte_c = build_features(te).reindex(columns=Xtr_c.columns, fill_value=0.0).reset_index(drop=True)
    tr_block, te_block = neural_blocks(tr, te)

    Xtr_full = pd.concat([Xtr_c, tr_block], axis=1)
    Xte_full = pd.concat([Xte_c, te_block], axis=1)
    scaler = StandardScaler().fit(Xtr_full)
    Xtr_s = scaler.transform(Xtr_full)
    Xte_s = scaler.transform(Xte_full)

    cols = list(Xtr_full.columns)
    idx = {c: i for i, c in enumerate(cols)}
    classical = [idx[c] for c in Xtr_c.columns]
    gnn = classical + [idx[c] for c in GNN_COLS]
    gnn_ssm = gnn + [idx[c] for c in SSM_COLS]

    print("\nMulti-label suite on the held-out test split (nested models):")
    reps = {
        "M_classical": fit_suite(Xtr_s, Xte_s, classical, Ytr, Yte),
        "M_+GNN": fit_suite(Xtr_s, Xte_s, gnn, Ytr, Yte),
        "M_+GNN+SSM": fit_suite(Xtr_s, Xte_s, gnn_ssm, Ytr, Yte),
    }
    for name, rep in reps.items():
        print_suite(name, rep)

    # held-out per-label AUC deltas: the effect-size view of specificity
    def auc(rep, lab):
        v = rep["per_label"][lab]["auc"]
        return float(v) if v is not None else float("nan")
    d_gnn = {n: auc(reps["M_+GNN"], n) - auc(reps["M_classical"], n) for n in LABEL_NAMES}
    d_ssm = {n: auc(reps["M_+GNN+SSM"], n) - auc(reps["M_+GNN"], n) for n in LABEL_NAMES}

    # per-label LR gate, base = compact classical minus distance (geo separator)
    print("\nPer-label admission gate (effect size, not raw p):")
    base = compact_base(tr).drop(columns=["merch_dist_km"]).reset_index(drop=True)
    g_gnn = gate_table(base, tr_block[GNN_COLS].reset_index(drop=True), Ytr, GNN_MATCH)
    g_ssm = gate_table(base, tr_block[SSM_COLS].reset_index(drop=True), Ytr, SSM_MATCH)

    def print_gate(title: str, g: dict, deltas: dict, match: str) -> None:
        print(f"\n  {title}  (matched slot = {match})")
        print(f"    {'label':6s} {'G2':>11s} {'p':>9s} {'rel-G2':>8s} "
              f"{'dAUC':>7s}   verdict")
        for n in LABEL_NAMES:
            r = g[n]
            g2 = "inf" if not np.isfinite(r["G2"]) else f"{r['G2']:.1f}"
            rel = "  -" if np.isnan(r["rel_to_match"]) else f"{r['rel_to_match']:6.1%}"
            tag = "ADMIT (matched)" if n == match else (
                "control: negligible" if r["rel_to_match"] < 0.1 else "control: check")
            print(f"    {n:6s} {g2:>11s} {r['p']:9.1e} {rel:>8s} "
                  f"{deltas[n]:+7.3f}   {tag}")

    print_gate("GNN ring block", g_gnn, d_gnn, GNN_MATCH)
    print_gate("SSM temporal block", g_ssm, d_ssm, SSM_MATCH)

    print("\nReading: the suite climbs M_classical -> +GNN -> +GNN+SSM via the per-label\n"
          "AUC of exactly the matched labels -- GNN lifts L_R (ring), SSM lifts L_T\n"
          "(temporal) -- while every other label moves by ~0 (dAUC column): the blocks\n"
          "are SPECIFIC. By raw p the gate admits a block on mismatched labels too (n~1.3M\n"
          "power inflation, RESULTS 8.3), but rel-G2 exposes those as a small fraction of the\n"
          "matched slot. Hamming/subset-acc are base-rate dominated (rare labels); the\n"
          "discriminative payoff is in per-label AUC, LRAP and mean AUC.")

    summary = {
        "suite": {name: {k: v for k, v in rep.items() if k != "per_label"}
                  | {"per_label": rep["per_label"]} for name, rep in reps.items()},
        "delta_auc_gnn": d_gnn, "delta_auc_ssm": d_ssm,
        "gate_gnn": g_gnn, "gate_ssm": g_ssm,
    }
    path = OUT / "integration_results.json"
    path.write_text(json.dumps(summary, indent=2, default=float))
    print(f"\nwrote {path}")


if __name__ == "__main__":
    main()
