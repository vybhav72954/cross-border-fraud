"""State-space lineage bake-off (track E) -- the sequence-model showcase.

Puts the whole S4->Mamba lineage head-to-head against RNN / TCN / Transformer
baselines on the two SEQUENCE slots of the controlled benchmark (temporal,
velocity), all on the SAME per-card stream, same readout protocol, same isolated
solo-vs-legit AUC, same LR-gate. The point is not "Mamba wins" -- it is a
mechanistic map of WHERE each inductive bias pays off:

  * temporal  = a STATIONARY per-card hour distribution -> long, near-uniform
    memory helps; input-dependent selectivity has little to exploit.
  * velocity  = a bursty, content-dependent inter-arrival pattern -> selectivity
    (input-dependent dt) is exactly the right bias.

Reference rows: ``tabular`` (global logreg floor), ``oracle`` (the hand-crafted
card-relative feature = ceiling), ``s4d_fixed`` (the precomputed-state SSM from
ssm.py). Learned rows: gru, lstm, tcn, transformer, lru, s5, dss, mamba_s6.

CPU-tractable: the learned models do full BPTT, so the zoo runs on a CARD
SUBSAMPLE (every model on the SAME subsample -> apples-to-apples). The production
extractors in scripts/04 and 05 use the full 1.3M rows; this isolates the
ARCHITECTURE comparison, not the headline number.

Run from the project root:
    python scripts/13_sequence_zoo.py                  # default 150-card subsample
    python scripts/13_sequence_zoo.py --quick          # fast smoke (80 cards, 3 ep)
    python scripts/13_sequence_zoo.py --cards 300 --epochs 12
"""
import argparse
import json
import sys
import time
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

sys.path.insert(0, ".")
from src.features import build_features  # noqa: E402
from src.inject import TYPOLOGY_COL, typology_dummies  # noqa: E402
from src.models.glm import BinaryRelevanceGLM  # noqa: E402
from src.models.sequence import ARCHITECTURES, SequenceModel  # noqa: E402
from src.models.ssm import (  # noqa: E402
    TemporalSSM,
    VelocitySSM,
    card_hour_rarity,
    card_rate_states,
)

OUT = Path("data/processed")
RES = Path("results")
RES.mkdir(exist_ok=True)

CLASSICAL = ["hour_sin", "hour_cos", "log_amt", "vel_1h",
             "is_weekend", "age", "log_city_pop"]
LEARNED = ["gru", "lstm", "tcn", "transformer", "lru", "s5", "dss", "mamba_s6"]
FAMILY = {"tabular": "ref", "oracle": "ref", "s4d_fixed": "ref",
          "gru": "rnn", "lstm": "rnn", "tcn": "conv", "transformer": "attn",
          "lru": "ssm", "s5": "ssm", "dss": "ssm", "mamba_s6": "ssm"}
COLORS = {"ref": "#9ca3af", "rnn": "#1f77b4", "conv": "#2ca02c",
          "attn": "#d62728", "ssm": "#7c3aed"}


def load(split):
    df = pd.read_parquet(OUT / f"injected_{split}.parquet")
    df["trans_dt"] = pd.to_datetime(df["trans_date_trans_time"])
    return df


def subsample(tr, te, n_cards, seed):
    shared = np.intersect1d(tr["cc_num"].unique(), te["cc_num"].unique())
    rng = np.random.default_rng(seed)
    pick = rng.choice(shared, size=min(n_cards, len(shared)), replace=False)
    return (tr[tr["cc_num"].isin(pick)].reset_index(drop=True),
            te[te["cc_num"].isin(pick)].reset_index(drop=True))


def isolated_auc(score, typ, slot):
    solo, legit = typ == slot, typ == ""
    m = solo | legit
    return float(roc_auc_score(solo[m].astype(int), score[m]))


def tabular_scores(tr, te):
    Xtr, Xte = build_features(tr)[CLASSICAL], build_features(te)[CLASSICAL]
    sc = StandardScaler().fit(Xtr)
    return sc.transform(Xtr), sc.transform(Xte)


def gate(base_tr, score_tr, slot, y_tr):
    """LR test: does this model's scalar add over the classical base for `slot`?"""
    base = pd.DataFrame(base_tr, columns=CLASSICAL).reset_index(drop=True)
    ext = pd.DataFrame({"seq_score": np.asarray(score_tr)})
    y = pd.DataFrame({slot: y_tr})
    try:
        r = BinaryRelevanceGLM(maxiter=100).admit_extension(base, ext, y, slot)
        return {"G2": round(r["G2"], 1), "df": r["df"],
                "p": r["p_value"], "admitted": bool(r["admitted"])}
    except Exception as exc:  # separation -> overwhelming evidence
        return {"G2": float("inf"), "df": 1, "p": 0.0, "admitted": True,
                "note": type(exc).__name__}


def run_slot(slot, tr, te, args):
    print(f"\n{'=' * 64}\nSLOT: {slot}\n{'=' * 64}")
    y_tr = typology_dummies(tr)[slot].to_numpy()
    typ_te = te[TYPOLOGY_COL].fillna("").to_numpy()
    n_pos_tr = int(y_tr.sum())
    n_solo_te = int((typ_te == slot).sum())
    print(f"train rows {len(tr):,} | {slot} train {n_pos_tr:,} | "
          f"test solo {n_solo_te:,} | test legit {int((typ_te=='').sum()):,}")

    Xtab_tr, Xtab_te = tabular_scores(tr, te)
    rows = []

    # ── reference: tabular floor, oracle ceiling, fixed S4D ────────────────
    lr = LogisticRegression(max_iter=2000).fit(Xtab_tr, y_tr)
    tab_te = lr.predict_proba(Xtab_te)[:, 1]
    rows.append(("tabular", isolated_auc(tab_te, typ_te, slot), 0, 0.0, None))

    if slot == "temporal":
        orc_tr, orc_te = card_hour_rarity(tr).to_numpy(), card_hour_rarity(te).to_numpy()
        fixed = TemporalSSM(epochs=25, seed=args.seed)
    else:
        # the 10-min decay channel (RATE_DECAYS[0]) is the burst detector; the
        # long channels just track overall activity, so use the short one.
        orc_tr = np.log1p(card_rate_states(tr))[:, 0]
        orc_te = np.log1p(card_rate_states(te))[:, 0]
        fixed = VelocitySSM(epochs=25, seed=args.seed)
    rows.append(("oracle", isolated_auc(orc_te, typ_te, slot), 0, 0.0,
                 gate(Xtab_tr, orc_tr, slot, y_tr)))

    t0 = time.perf_counter()
    fixed.fit(tr, y_tr)
    rows.append(("s4d_fixed", isolated_auc(fixed.score(te), typ_te, slot),
                 0, time.perf_counter() - t0, gate(Xtab_tr, fixed.score(tr), slot, y_tr)))

    # ── learned lineage + baselines ───────────────────────────────────────
    for arch in LEARNED:
        m = SequenceModel(arch, slot, epochs=args.epochs, n_state=args.n_state,
                          hidden=args.hidden, max_seq=args.max_seq,
                          batch_cards=args.batch_cards, seed=args.seed).fit(tr, y_tr)
        auc = isolated_auc(m.score(te), typ_te, slot)
        rows.append((arch, auc, m.n_params(), m.train_seconds,
                     gate(Xtab_tr, m.score(tr), slot, y_tr)))
        print(f"  {arch:12s} AUC={auc:.3f}  params={m.n_params():>6,}  "
              f"{m.train_seconds:5.1f}s")

    return rows


def to_frame(results):
    recs = []
    for slot, rows in results.items():
        for name, auc, params, secs, g in rows:
            recs.append({"slot": slot, "model": name, "family": FAMILY[name],
                         "auc_iso": round(auc, 3), "n_params": params,
                         "train_sec": round(secs, 1),
                         "lr_G2": (g or {}).get("G2"), "lr_df": (g or {}).get("df"),
                         "lr_p": (g or {}).get("p"),
                         "admitted": (g or {}).get("admitted")})
    return pd.DataFrame(recs)


def plot(df):
    slots = df["slot"].unique()
    fig, axes = plt.subplots(1, len(slots), figsize=(7.5 * len(slots), 5),
                             squeeze=False)
    for ax, slot in zip(axes[0], slots):
        sub = df[df.slot == slot]
        x = np.arange(len(sub))
        ax.bar(x, sub["auc_iso"], color=[COLORS[f] for f in sub["family"]], alpha=0.9)
        for ref, ls in [("tabular", ":"), ("oracle", "--")]:
            v = sub[sub.model == ref]["auc_iso"]
            if len(v):
                ax.axhline(float(v.iloc[0]), ls=ls, c="grey", lw=1.2,
                           label=f"{ref} ({float(v.iloc[0]):.3f})")
        ax.set_xticks(x)
        ax.set_xticklabels(sub["model"], rotation=45, ha="right", fontsize=8)
        ax.set_ylim(0.45, 1.02)
        ax.axhline(0.5, c="k", lw=0.6, alpha=0.4)
        ax.set_ylabel("isolated AUC (solo vs legit)")
        ax.set_title(f"{slot} slot -- state-space lineage vs baselines")
        for xi, (a, fam) in enumerate(zip(sub["auc_iso"], sub["family"])):
            ax.text(xi, a + 0.008, f"{a:.2f}", ha="center", fontsize=7)
        ax.legend(fontsize=8, loc="lower right")
    fig.tight_layout()
    p = RES / "sequence_zoo.png"
    fig.savefig(p, dpi=130)
    plt.close(fig)
    print(f"\n  -> {p}")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--cards", type=int, default=150)
    ap.add_argument("--epochs", type=int, default=8)
    ap.add_argument("--n-state", type=int, default=32, dest="n_state")
    ap.add_argument("--hidden", type=int, default=64)
    ap.add_argument("--max-seq", type=int, default=384, dest="max_seq")
    ap.add_argument("--batch-cards", type=int, default=128, dest="batch_cards")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--quick", action="store_true", help="80 cards, 3 epochs")
    ap.add_argument("--slots", nargs="+", default=["temporal", "velocity"])
    args = ap.parse_args()
    if args.quick:
        args.cards, args.epochs = 80, 3
    assert set(ARCHITECTURES) >= set(LEARNED)

    print("== sequence-model zoo (state-space lineage bake-off) ==")
    tr_full, te_full = load("train"), load("test")
    tr, te = subsample(tr_full, te_full, args.cards, args.seed)
    print(f"card subsample: {tr['cc_num'].nunique()} cards | "
          f"train {len(tr):,} rows | test {len(te):,} rows")

    t0 = time.perf_counter()
    results = {slot: run_slot(slot, tr, te, args) for slot in args.slots}
    df = to_frame(results)

    print("\n" + "=" * 64 + "\nRESULTS (isolated solo-vs-legit AUC)\n" + "=" * 64)
    for slot in args.slots:
        print(f"\n[{slot}]")
        s = df[df.slot == slot].sort_values("auc_iso", ascending=False)
        print(s[["model", "family", "auc_iso", "n_params", "train_sec",
                 "lr_G2", "admitted"]].to_string(index=False))

    df.to_csv(RES / "sequence_zoo.csv", index=False)
    (RES / "sequence_zoo.json").write_text(json.dumps(
        {"args": vars(args), "rows": df.to_dict(orient="records")}, indent=2, default=str))
    plot(df)
    print(f"\nsaved results/sequence_zoo.{{csv,json,png}}  "
          f"(total {time.perf_counter() - t0:.0f}s)")


if __name__ == "__main__":
    main()
