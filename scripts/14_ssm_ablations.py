"""State-space design ablations (track E) -- the SSM-theory companion to the zoo.

Isolates three knobs of the diagonal SSM engine (``_DiagSSM``) so the choices
behind the lineage are evidence, not folklore. All on the same per-card stream,
same isolated solo-vs-legit AUC, on a CPU card subsample.

  A1 init           HiPPO-LegS diagonal init vs random init (does structured A help?)
  A2 discretisation zero-order-hold vs bilinear/Tukey (a = exp(dt A) vs Cayley)
  A3 state-dim      N in {4,8,16,32,64} (how much state does the signal need?)

Run from the project root:
    python scripts/14_ssm_ablations.py
    python scripts/14_ssm_ablations.py --quick
"""
import argparse
import json
import sys
import time
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

sys.path.insert(0, ".")
from src.inject import TYPOLOGY_COL, typology_dummies  # noqa: E402
from src.models.sequence import SequenceModel  # noqa: E402

OUT = Path("data/processed")
RES = Path("results")
RES.mkdir(exist_ok=True)
SLOT_COLOR = {"temporal": "#2ca02c", "velocity": "#1f77b4"}


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


def fit_diag(slot, tr, te, y_tr, typ_te, args, **kw):
    """Train a parametrised diagonal SSM and return its isolated AUC + train time."""
    m = SequenceModel("diag", slot, n_state=kw.pop("n_state", args.n_state),
                      hidden=args.hidden, epochs=args.epochs, max_seq=args.max_seq,
                      batch_cards=args.batch_cards, seed=args.seed, **kw).fit(tr, y_tr)
    return isolated_auc(m.score(te), typ_te, slot), m.train_seconds


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
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--slots", nargs="+", default=["temporal", "velocity"])
    ap.add_argument("--dims", nargs="+", type=int, default=[4, 8, 16, 32, 64])
    args = ap.parse_args()
    if args.quick:
        args.cards, args.epochs, args.dims = 80, 3, [8, 32]

    print("== SSM design ablations ==")
    tr, te = subsample(load("train"), load("test"), args.cards, args.seed)
    print(f"card subsample: {tr['cc_num'].nunique()} cards | "
          f"train {len(tr):,} | test {len(te):,}")

    recs = []
    t0 = time.perf_counter()
    for slot in args.slots:
        y_tr = typology_dummies(tr)[slot].to_numpy()
        typ_te = te[TYPOLOGY_COL].fillna("").to_numpy()
        print(f"\n[{slot}]")

        # A1 init: HiPPO vs random (complex, zoh, non-selective) ------------
        for hippo in (True, False):
            auc, secs = fit_diag(slot, tr, te, y_tr, typ_te, args,
                                 complex_state=True, hippo=hippo, selective=False)
            recs.append({"study": "init", "slot": slot,
                         "setting": "hippo" if hippo else "random",
                         "auc": round(auc, 3), "sec": round(secs, 1)})
            print(f"  init={'hippo' if hippo else 'random':6s}  AUC={auc:.3f}")

        # A2 discretisation: ZOH vs bilinear (complex, hippo) --------------
        for disc in ("zoh", "bilinear"):
            auc, secs = fit_diag(slot, tr, te, y_tr, typ_te, args,
                                 complex_state=True, hippo=True, selective=False, disc=disc)
            recs.append({"study": "disc", "slot": slot, "setting": disc,
                         "auc": round(auc, 3), "sec": round(secs, 1)})
            print(f"  disc={disc:8s}  AUC={auc:.3f}")

        # A3 state-dim sweep (complex, hippo, zoh) -------------------------
        for n in args.dims:
            auc, secs = fit_diag(slot, tr, te, y_tr, typ_te, args,
                                 n_state=n, complex_state=True, hippo=True, selective=False)
            recs.append({"study": "state_dim", "slot": slot, "setting": str(n),
                         "auc": round(auc, 3), "sec": round(secs, 1)})
            print(f"  N={n:<3d}        AUC={auc:.3f}")

    df = pd.DataFrame(recs)
    df.to_csv(RES / "ssm_ablations.csv", index=False)
    (RES / "ssm_ablations.json").write_text(json.dumps(recs, indent=2))

    fig, axes = plt.subplots(1, 3, figsize=(16, 4.6))
    for ax, study, title in zip(
            axes, ["init", "disc", "state_dim"],
            ["A1 init: HiPPO vs random", "A2 discretisation: ZOH vs bilinear",
             "A3 state-dim sweep"]):
        sub = df[df.study == study]
        if study == "state_dim":
            for slot in args.slots:
                s = sub[sub.slot == slot]
                xs = [int(v) for v in s["setting"]]
                ax.plot(xs, s["auc"], "o-", color=SLOT_COLOR[slot], label=slot)
            ax.set_xscale("log", base=2)
            ax.set_xlabel("state dimension N")
        else:
            settings = sub["setting"].unique()
            x = np.arange(len(settings))
            w = 0.38
            for i, slot in enumerate(args.slots):
                s = sub[sub.slot == slot].set_index("setting").reindex(settings)
                ax.bar(x + (i - 0.5) * w, s["auc"], w, color=SLOT_COLOR[slot], label=slot)
            ax.set_xticks(x)
            ax.set_xticklabels(settings)
        ax.set_ylim(0.45, 1.02)
        ax.axhline(0.5, c="k", lw=0.6, alpha=0.4)
        ax.set_ylabel("isolated AUC")
        ax.set_title(title)
        ax.legend(fontsize=8)
        ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(RES / "ssm_ablations.png", dpi=130)
    plt.close(fig)
    print(f"\nsaved results/ssm_ablations.{{csv,json,png}}  "
          f"(total {time.perf_counter() - t0:.0f}s)")


if __name__ == "__main__":
    main()
