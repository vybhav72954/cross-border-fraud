"""G3 cross-dataset replication: sequence zoo on BankSim / PaySim (track G).

Re-runs the S4->Mamba lineage bake-off (scripts/13_sequence_zoo.py) on an
alternative base dataset to ask: does the Mamba-S6 / fixed-bank bracket
observed on Sparkov (RESULTS.md §13 / §10.1) replicate when the base
dataset changes?

Dataset suitability (key findings documented here):
  BankSim  -- 3,761 persistent customers, median 17 txns each, 594k rows.
              Daily ``step`` -> temporal EXCLUDED (hour is always 0, degenerate).
              Velocity slot valid: injected 5-txn bursts are still time-concentrated
              (inter-arrival << 1 day) and detectable via the decayed-rate state.
              FULL learned zoo runs here -- BankSim is the primary replication base.
  PaySim   -- ``nameOrig`` is unique per transaction (near-singletons, median 1 txn).
              Oracles / fixed SSM generalise (velocity oracle 0.951, temporal 0.998 --
              already confirmed in G2 validation). Learned zoo SKIPPED: per-entity
              sequences are too short (<2 txns) to train on.
              Exposes a benchmark PREREQUISITE: learned sequence models require a
              minimum per-entity transaction history; datasets that lack it demonstrate
              scope, not failure of the protocol.

Tabular baseline uses 5 schema-aware features (no Sparkov demographics):
    hour_sin / hour_cos / log_amt / vel_1h / is_weekend

Reference (Sparkov §13 150-card subsample numbers -- the apples-to-apples match
for this subsample learned-zoo run; the oracle is the same 10-min decayed-rate
``card_rate_states[:, 0]`` feature §13 uses, NOT §12's max-over-timescales):
    temporal  s4d_fixed=0.781  oracle=0.879  mamba_s6=0.668  (fixed bank dominates)
    velocity  s4d_fixed=0.902  oracle=0.894  mamba_s6=0.891  (mamba_s6 ties TCN, leads SSMs)

The ``ring`` slot (GNN half) is opt-in via ``--slots`` since it uses a dedicated
per-split injection (whole rings per half) rather than the velocity/temporal
entity-subsample, and runs the schema-driven ``merchant_window_features`` oracle +
``RingSAGE``. Combine slots to write one CSV with both representation halves.

Run from the project root:
    python scripts/15_g3_replication.py                      # BankSim, velocity zoo
    python scripts/15_g3_replication.py --dataset paysim     # oracle + fixed only
    python scripts/15_g3_replication.py --slots velocity ring  # both halves -> one CSV
    python scripts/15_g3_replication.py --slots ring         # GNN ring half only
    python scripts/15_g3_replication.py --quick --slots ring # ring smoke (3 ep)
    python scripts/15_g3_replication.py --cards 500          # larger BankSim subsample
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
from src.adapters import ADAPTERS, PAYSIM_FILE, BANKSIM_FILE  # noqa: E402
from src.features import cross_features  # noqa: E402
from src.inject import (  # noqa: E402
    DEFAULT_COUNTS, TYPOLOGY_COL, build_controlled_dataset, inject_ring,
    legit_background, typology_dummies,
)
from src.models.glm import BinaryRelevanceGLM  # noqa: E402
from src.models.gnn import RingSAGE, merchant_window_features  # noqa: E402
from src.models.sequence import ARCHITECTURES, SequenceModel  # noqa: E402
from src.models.ssm import (  # noqa: E402
    TemporalSSM, VelocitySSM, card_hour_rarity, card_rate_states,
)
from src.schema import Schema  # noqa: E402

RAW = Path("data/raw")
RES = Path("results")
RES.mkdir(exist_ok=True)

DATASET_PATHS = {
    "banksim": RAW / "banksim" / BANKSIM_FILE,
    "paysim":  RAW / "paysim"  / PAYSIM_FILE,
}

# BankSim: velocity valid; temporal excluded (daily step -> hour always 0).
# PaySim:  both slots valid for oracle/fixed, but learned zoo skipped (singletons).
DATASET_SLOTS = {
    "banksim": ["velocity"],
    "paysim":  ["velocity", "temporal"],
}

# PaySim subsampling is skipped for the learned zoo; cap raw rows to keep it fast.
PAYSIM_ROW_CAP = 300_000

# For the LR-gate baseline DataFrame
CROSS_FEATURES = ["hour_sin", "hour_cos", "log_amt", "vel_1h", "is_weekend"]

LEARNED_ARCHS = ["gru", "lstm", "tcn", "transformer", "lru", "s5", "dss", "mamba_s6"]
FAMILY = {"tabular": "ref", "oracle": "ref", "s4d_fixed": "ref", "ringsage": "gnn",
          "gru": "rnn", "lstm": "rnn", "tcn": "conv", "transformer": "attn",
          "lru": "ssm", "s5": "ssm", "dss": "ssm", "mamba_s6": "ssm"}
COLORS = {"ref": "#9ca3af", "rnn": "#1f77b4", "conv": "#2ca02c",
          "attn": "#d62728", "ssm": "#7c3aed", "gnn": "#e377c2"}

# Sparkov §13 150-card-subsample reference numbers (the velocity/temporal AUC
# columns of RESULTS.md §13.1) -- the apples-to-apples reference for this
# subsample learned-zoo run. The oracle here is the same 10-min decayed-rate
# `card_rate_states[:, 0]` feature §13 uses (NOT §12's max-over-timescales
# `max_decayed_rate`), so the comparison is like-for-like.
SPARKOV_REF = {
    "temporal": {"tabular": 0.647, "oracle": 0.879, "s4d_fixed": 0.781,
                 "gru": 0.638, "lstm": 0.637, "tcn": 0.633, "transformer": 0.594,
                 "lru": 0.665, "s5": 0.661, "dss": 0.657, "mamba_s6": 0.668},
    "velocity": {"tabular": 0.901, "oracle": 0.894, "s4d_fixed": 0.902,
                 "gru": 0.749, "lstm": 0.670, "tcn": 0.891, "transformer": 0.887,
                 "lru": 0.887, "s5": 0.872, "dss": 0.870, "mamba_s6": 0.891},
    # ring references are FULL-DATA Sparkov (§3 bake-off / §10.2) -- ring is not in
    # the §13 sequence zoo, so there is no subsample column for it.
    "ring": {"tabular": 0.582, "oracle": 0.959, "ringsage": 0.841},
}


# ── data loading & preparation ───────────────────────────────────────────────

def load_adapt(dataset: str):
    """Load raw CSV + adapt to (df, schema). No injection."""
    path = DATASET_PATHS[dataset]
    print(f"  loading {path.name} ...", flush=True)
    raw = pd.read_csv(path)
    if dataset == "paysim" and len(raw) > PAYSIM_ROW_CAP:
        raw = raw.iloc[:PAYSIM_ROW_CAP].copy()
        print(f"  PaySim capped to {PAYSIM_ROW_CAP:,} rows (full file has singletons "
              f"-> learned zoo not run regardless)")
    df, schema = ADAPTERS[dataset](raw)
    print(f"  rows after adapt: {len(df):,}  |  "
          f"unique entities: {df[schema.entity].nunique():,}")
    return df, schema


def load_and_inject(dataset: str, seed: int = 0):
    """Load raw CSV, adapt, inject. Returns (df_full, schema) -- unsplit."""
    df, schema = load_adapt(dataset)
    print(f"  injecting controlled typologies ...")
    full = build_controlled_dataset(df, seed=seed, schema=schema)
    n_inj = (full[TYPOLOGY_COL] != "").sum()
    print(f"  injected dataset: {len(full):,} rows  |  {n_inj:,} injected "
          f"({n_inj/len(full)*100:.2f}%)")
    return full, schema


def ring_dataset(adapted: pd.DataFrame, schema: Schema, seed: int,
                 n_rings: int, frac: float = 0.8):
    """Ring train/test where every ring event is WHOLE within one half.

    A ring spans ``cards_per_ring`` random entities, so the velocity per-entity
    subsample would shatter it and a post-injection split would scatter a ring's
    rows across train/test (destroying the test-frame fan-in). Instead we split
    the legit background chronologically, then inject ring INTO EACH HALF
    independently -- the same per-split injection protocol Sparkov uses for its
    pre-split ``injected_{train,test}`` parquets. Returns (tr, te)."""
    df_sorted = adapted.sort_values(schema.time).reset_index(drop=True)
    cut = int(len(df_sorted) * frac)
    base_tr = legit_background(df_sorted.iloc[:cut], schema)
    base_te = legit_background(df_sorted.iloc[cut:], schema)
    rng = np.random.default_rng(seed)
    n_te = max(1, int(round(n_rings * (1 - frac) / frac)))
    tr = inject_ring(base_tr, n_rings, schema=schema, rng=rng).reset_index(drop=True)
    te = inject_ring(base_te, n_te, schema=schema, rng=rng).reset_index(drop=True)
    return tr, te


def entity_split(df: pd.DataFrame, schema: Schema, frac: float = 0.8):
    """Per-entity chronological train/test split.

    Splitting per entity ensures every entity with >=2 txns appears in both
    halves. Singleton entities go to train only (they cannot form a valid test
    pair). This avoids the leak-free ordering problem: for datasets like PaySim
    where most entities are singletons a global time split produces an empty
    intersection and the subsample step fails.
    """
    dt = pd.to_datetime(df[schema.time])
    train_idx, test_idx = [], []
    for _, grp in df.groupby(schema.entity, sort=False):
        n = len(grp)
        if n < 2:
            train_idx.extend(grp.index)
            continue
        sorted_idx = grp.index[np.argsort(dt[grp.index].to_numpy())]
        cut = max(1, int(n * frac))
        train_idx.extend(sorted_idx[:cut])
        test_idx.extend(sorted_idx[cut:])
    tr = df.loc[train_idx].reset_index(drop=True)
    te = df.loc[test_idx].reset_index(drop=True)
    return tr, te


def global_split(df: pd.DataFrame, schema: Schema, frac: float = 0.8):
    """Global chronological train/test split (fallback for PaySim singletons)."""
    df = df.sort_values(schema.time).reset_index(drop=True)
    cut = int(len(df) * frac)
    return df.iloc[:cut].reset_index(drop=True), df.iloc[cut:].reset_index(drop=True)


def subsample(tr, te, n_cards, seed, schema):
    """Restrict to entities present in BOTH splits, then pick a random subset."""
    shared = np.intersect1d(
        tr[schema.entity].unique(), te[schema.entity].unique(),
    )
    if len(shared) == 0:
        return tr, te  # PaySim singleton fallback: use full data
    rng = np.random.default_rng(seed)
    pick = rng.choice(shared, size=min(n_cards, len(shared)), replace=False)
    return (tr[tr[schema.entity].isin(pick)].reset_index(drop=True),
            te[te[schema.entity].isin(pick)].reset_index(drop=True))


# ── schema-aware cross-dataset tabular features ──────────────────────────────
# `cross_features` (hour_sin/hour_cos/log_amt/vel_1h/is_weekend) now lives in
# src.features so the GLM baseline and this runner share one schema-driven path.

def tabular_scores(tr, te, schema):
    Xtr = cross_features(tr, schema)
    Xte = cross_features(te, schema)
    sc = StandardScaler().fit(Xtr)
    return sc.transform(Xtr), sc.transform(Xte)


# ── evaluation helpers ────────────────────────────────────────────────────────

def isolated_auc(score, typ, slot):
    solo, legit = typ == slot, typ == ""
    m = solo | legit
    if m.sum() == 0 or solo[m].sum() == 0:
        return float("nan")
    return float(roc_auc_score(solo[m].astype(int), score[m]))


def gate(base_tr, score_tr, slot, y_tr):
    base = pd.DataFrame(base_tr, columns=CROSS_FEATURES).reset_index(drop=True)
    ext = pd.DataFrame({"seq_score": np.asarray(score_tr)})
    y = pd.DataFrame({slot: y_tr})
    try:
        r = BinaryRelevanceGLM(maxiter=100).admit_extension(base, ext, y, slot)
        return {"G2": round(r["G2"], 1), "df": r["df"],
                "p": r["p_value"], "admitted": bool(r["admitted"])}
    except Exception as exc:
        return {"G2": float("inf"), "df": 1, "p": 0.0, "admitted": True,
                "note": type(exc).__name__}


# ── per-slot runner ───────────────────────────────────────────────────────────

def run_slot(slot, tr, te, schema, args, run_learned: bool):
    print(f"\n{'=' * 64}\nSLOT: {slot}\n{'=' * 64}")
    y_tr = typology_dummies(tr)[slot].to_numpy()
    typ_te = te[TYPOLOGY_COL].fillna("").to_numpy()
    print(f"train rows {len(tr):,} | {slot} train pos {int(y_tr.sum()):,} | "
          f"test solo {int((typ_te==slot).sum()):,} | "
          f"test legit {int((typ_te=='').sum()):,}")

    Xtab_tr, Xtab_te = tabular_scores(tr, te, schema)
    rows = []

    # ── tabular floor ────────────────────────────────────────────────────────
    lr = LogisticRegression(max_iter=2000).fit(Xtab_tr, y_tr)
    tab_te = lr.predict_proba(Xtab_te)[:, 1]
    rows.append(("tabular", isolated_auc(tab_te, typ_te, slot), 0, 0.0, None))

    # ── oracle + fixed SSM ──────────────────────────────────────────────────
    if slot == "temporal":
        orc_tr = card_hour_rarity(tr, schema).to_numpy()
        orc_te = card_hour_rarity(te, schema).to_numpy()
        fixed = TemporalSSM(epochs=25, seed=args.seed, schema=schema)
    else:
        orc_tr = np.log1p(card_rate_states(tr, schema=schema))[:, 0]
        orc_te = np.log1p(card_rate_states(te, schema=schema))[:, 0]
        fixed = VelocitySSM(epochs=25, seed=args.seed, schema=schema)

    rows.append(("oracle", isolated_auc(orc_te, typ_te, slot), 0, 0.0,
                 gate(Xtab_tr, orc_tr, slot, y_tr)))

    t0 = time.perf_counter()
    fixed.fit(tr, y_tr)
    rows.append(("s4d_fixed", isolated_auc(fixed.score(te), typ_te, slot),
                 0, time.perf_counter() - t0,
                 gate(Xtab_tr, fixed.score(tr), slot, y_tr)))

    if not run_learned:
        print("  [learned zoo skipped: singleton entities -- see script header]")
        return rows

    # ── learned lineage + baselines ─────────────────────────────────────────
    for arch in LEARNED_ARCHS:
        m = SequenceModel(arch, slot, epochs=args.epochs, n_state=args.n_state,
                          hidden=args.hidden, max_seq=args.max_seq,
                          batch_cards=args.batch_cards, seed=args.seed,
                          schema=schema).fit(tr, y_tr)
        auc = isolated_auc(m.score(te), typ_te, slot)
        rows.append((arch, auc, m.n_params(), m.train_seconds,
                     gate(Xtab_tr, m.score(tr), slot, y_tr)))
        print(f"  {arch:12s} AUC={auc:.3f}  params={m.n_params():>6,}  "
              f"{m.train_seconds:5.1f}s")

    return rows


# ── ring slot (GNN) ─────────────────────────────────────────────────────────

def run_ring_slot(tr, te, schema, args):
    """Ring slot: windowed merchant fan-in oracle + learned RingSAGE, scored
    isolated ring-solo-vs-legit AUC, gated against the cross-features tabular base.
    Threads the adapter ``schema`` through the (now schema-driven) gnn extractors."""
    import torch

    print(f"\n{'=' * 64}\nSLOT: ring\n{'=' * 64}")
    y_tr = typology_dummies(tr)["ring"].to_numpy()
    typ_te = te[TYPOLOGY_COL].fillna("").to_numpy()
    print(f"train rows {len(tr):,} | ring train pos {int(y_tr.sum()):,} | "
          f"test solo {int((typ_te=='ring').sum()):,} | "
          f"test legit {int((typ_te=='').sum()):,}")

    Xtab_tr, Xtab_te = tabular_scores(tr, te, schema)
    rows = []

    # tabular floor (cross-features have no merchant-side signal -> ~chance)
    lr = LogisticRegression(max_iter=2000).fit(Xtab_tr, y_tr)
    rows.append(("tabular", isolated_auc(lr.predict_proba(Xtab_te)[:, 1], typ_te, "ring"),
                 0, 0.0, None))

    # oracle: windowed merchant fan-in (distinct entities at the target +/-2h)
    fan_tr = merchant_window_features(tr, window_hours=2.0, schema=schema,
                                      show_progress=False)["merch_win_cards"].to_numpy()
    fan_te = merchant_window_features(te, window_hours=2.0, schema=schema,
                                      show_progress=False)["merch_win_cards"].to_numpy()
    rows.append(("oracle", isolated_auc(fan_te, typ_te, "ring"), 0, 0.0,
                 gate(Xtab_tr, fan_tr, "ring", y_tr)))

    # learned: RingSAGE (GraphSAGE over the (entity, target-time-bucket) graph)
    ring_epochs = 3 if args.quick else 60
    t0 = time.perf_counter()
    sage = RingSAGE(window_hours=2.0, epochs=ring_epochs, seed=args.seed,
                    schema=schema).fit(tr, y_tr)
    n_params = sum(p.numel() for p in sage._model.parameters())
    rows.append(("ringsage", isolated_auc(sage.score(te), typ_te, "ring"),
                 n_params, time.perf_counter() - t0,
                 gate(Xtab_tr, sage.score(tr), "ring", y_tr)))
    print(f"  {'ringsage':12s} AUC={rows[-1][1]:.3f}  params={n_params:>6,}  "
          f"{rows[-1][3]:5.1f}s")
    return rows


# ── output ────────────────────────────────────────────────────────────────────

def to_frame(results):
    recs = []
    for slot, rows in results.items():
        for name, auc, params, secs, g in rows:
            recs.append({"slot": slot, "model": name, "family": FAMILY.get(name, "?"),
                         "auc_iso": round(auc, 3) if not np.isnan(auc) else float("nan"),
                         "n_params": params, "train_sec": round(secs, 1),
                         "lr_G2": (g or {}).get("G2"),
                         "lr_admitted": (g or {}).get("admitted")})
    return pd.DataFrame(recs)


def print_comparison(df, dataset, slots):
    ref_map = {"banksim": "BankSim", "paysim": "PaySim"}
    ds_label = ref_map.get(dataset, dataset)
    print("\n" + "=" * 64)
    print(f"RESULTS vs SPARKOV §13 -- {ds_label}")
    print("=" * 64)
    for slot in slots:
        sub = df[df.slot == slot].set_index("model")["auc_iso"]
        ref = SPARKOV_REF.get(slot, {})
        print(f"\n[{slot}]  {'model':12s}  {'here':>7}  {'Sparkov':>7}  delta")
        for name in ["tabular", "oracle", "s4d_fixed", "ringsage", "gru", "lstm",
                     "tcn", "transformer", "lru", "s5", "dss", "mamba_s6"]:
            if name not in sub.index:
                continue
            here = sub[name]
            spk = ref.get(name, float("nan"))
            delta = "" if np.isnan(spk) else f"{here - spk:+.3f}"
            print(f"  {'':2s}{name:12s}  {here:7.3f}  {spk:7.3f}  {delta}")


def plot(df, dataset, slots):
    fig, axes = plt.subplots(1, len(slots), figsize=(7.5 * len(slots), 5),
                             squeeze=False)
    for ax, slot in zip(axes[0], slots):
        sub = df[df.slot == slot]
        x = np.arange(len(sub))
        ax.bar(x, sub["auc_iso"], color=[COLORS.get(f, "#aaa") for f in sub["family"]],
               alpha=0.9)
        for ref_name, ls in [("tabular", ":"), ("oracle", "--")]:
            v = sub[sub.model == ref_name]["auc_iso"]
            if len(v):
                ax.axhline(float(v.iloc[0]), ls=ls, c="grey", lw=1.2,
                           label=f"{ref_name} ({float(v.iloc[0]):.3f})")
        ax.set_xticks(x)
        ax.set_xticklabels(sub["model"], rotation=45, ha="right", fontsize=8)
        ax.set_ylim(0.45, 1.02)
        ax.axhline(0.5, c="k", lw=0.6, alpha=0.4)
        ax.set_ylabel("isolated AUC (solo vs legit)")
        ax.set_title(f"{dataset} / {slot} -- G3 replication")
        for xi, (a, _) in enumerate(zip(sub["auc_iso"], sub["family"])):
            ax.text(xi, a + 0.008, f"{a:.2f}", ha="center", fontsize=7)
        ax.legend(fontsize=8, loc="lower right")
    fig.tight_layout()
    p = RES / f"g3_{dataset}_replication.png"
    fig.savefig(p, dpi=130)
    plt.close(fig)
    print(f"\n  -> {p}")


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dataset", choices=["banksim", "paysim"], default="banksim")
    ap.add_argument("--cards", type=int, default=500,
                    help="entity subsample size (BankSim only; PaySim uses full data)")
    ap.add_argument("--epochs", type=int, default=8)
    ap.add_argument("--n-state", type=int, default=32, dest="n_state")
    ap.add_argument("--hidden", type=int, default=64)
    ap.add_argument("--max-seq", type=int, default=256, dest="max_seq")
    ap.add_argument("--batch-cards", type=int, default=256, dest="batch_cards")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--quick", action="store_true",
                    help="80 cards, 3 epochs (BankSim smoke test)")
    ap.add_argument("--slots", nargs="+", default=None,
                    help="override slot list (default: per-dataset)")
    args = ap.parse_args()

    if args.quick:
        args.cards, args.epochs = 80, 3

    slots = args.slots or DATASET_SLOTS[args.dataset]
    run_learned = args.dataset == "banksim"

    print(f"\n== G3 cross-dataset replication: {args.dataset} ==")
    print(f"   slots: {slots}  |  learned zoo: {run_learned}")
    if not run_learned:
        print("   [PaySim: singleton entities -- learned zoo requires per-entity history]")

    adapted, schema = load_adapt(args.dataset)
    non_ring = [s for s in slots if s != "ring"]
    results = {}
    t0 = time.perf_counter()

    # velocity / temporal: post-injection split (+ entity subsample on BankSim)
    if non_ring:
        print("  injecting controlled typologies ...")
        full = build_controlled_dataset(adapted, seed=args.seed, schema=schema)
        n_inj = (full[TYPOLOGY_COL] != "").sum()
        print(f"  injected dataset: {len(full):,} rows | {n_inj:,} injected "
              f"({n_inj/len(full)*100:.2f}%)")
        if args.dataset == "banksim":
            tr_full, te_full = entity_split(full, schema)
            tr, te = subsample(tr_full, te_full, args.cards, args.seed, schema)
            print(f"entity subsample: {tr[schema.entity].nunique()} entities | "
                  f"train {len(tr):,} rows | test {len(te):,} rows")
        else:
            tr, te = global_split(full, schema)
            print(f"global split: train {len(tr):,} | test {len(te):,} rows")
            print("  (no entity subsample: all intersecting entities used for oracle/fixed AUC)")
        for slot in non_ring:
            results[slot] = run_slot(slot, tr, te, schema, args, run_learned)

    # ring: dedicated per-split injection (whole rings per half; no entity subsample)
    if "ring" in slots:
        n_rings = 30 if args.quick else DEFAULT_COUNTS["ring"]
        print(f"\n  ring: per-split injection ({n_rings} train rings) -- whole rings per half")
        rtr, rte = ring_dataset(adapted, schema, args.seed, n_rings)
        print(f"  ring train {len(rtr):,} rows | test {len(rte):,} rows")
        results["ring"] = run_ring_slot(rtr, rte, schema, args)

    results = {s: results[s] for s in slots}  # restore requested order
    df_out = to_frame(results)

    print_comparison(df_out, args.dataset, slots)

    tag = args.dataset
    df_out.to_csv(RES / f"g3_{tag}_replication.csv", index=False)
    (RES / f"g3_{tag}_replication.json").write_text(json.dumps(
        {"dataset": args.dataset, "args": vars(args),
         "rows": df_out.to_dict(orient="records"),
         "sparkov_ref": SPARKOV_REF},
        indent=2, default=str,
    ))
    plot(df_out, tag, slots)
    print(f"\nsaved results/g3_{tag}_replication.{{csv,json,png}}  "
          f"(total {time.perf_counter() - t0:.0f}s)")


if __name__ == "__main__":
    main()
