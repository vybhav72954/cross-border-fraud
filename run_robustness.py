"""
Robustness / statistical-rigor suite (CLAUDE.md roadmap track D).

Runs four studies on the controlled benchmark and writes tables + figures:

  D1 degradation   per-typology matched-detector AUC vs overlap depth
  D2 multiseed     isolated AUC mean +/- std across injection seeds + LR stability
  D3 calibration   Hosmer-Lemeshow + reliability curves for the per-label logits
  D4 sensitivity   isolated AUC under +/-20% moves of each injection knob

Reuses the production injectors/oracles via src/robustness.py -- nothing here
reimplements a detector. Figures -> figures/, numeric summary -> data/processed/
robustness_results.json.

Run from the project root:
    python run_robustness.py                 # all four studies
    python run_robustness.py degradation     # one study (degradation|multiseed|
                                             #   calibration|sensitivity)
"""
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

sys.path.insert(0, ".")
import src.robustness as rb  # noqa: E402
from src.inject import TYPOLOGIES  # noqa: E402

FIG = Path("figures")
FIG.mkdir(exist_ok=True)
OUT = Path("data/processed")
COLORS = {"ring": "#d62728", "velocity": "#1f77b4", "temporal": "#2ca02c",
          "category": "#9467bd", "geo": "#ff7f0e"}


# ── D1 ───────────────────────────────────────────────────────────────────────

def run_degradation(base) -> dict:
    print("\n" + "=" * 70)
    print("D1  CROSS-BORDER DEGRADATION  (matched-detector AUC vs overlap depth)")
    print("=" * 70)
    df = rb.study_degradation(base)
    piv = df.pivot(index="typology", columns="depth", values="auc").reindex(TYPOLOGIES)
    npos = df.pivot(index="typology", columns="depth", values="n_pos").reindex(TYPOLOGIES)
    print("\nAUC by overlap depth (depth = # signatures on the event):")
    print(piv.to_string(float_format=lambda x: f"{x:.3f}", na_rep="  -  "))
    print("\n(n positive rows per cell)")
    print(npos.to_string(float_format=lambda x: f"{int(x)}", na_rep="  -  "))

    fig, ax = plt.subplots(figsize=(7, 4.5))
    for t in TYPOLOGIES:
        sub = df[df.typology == t].sort_values("depth")
        if len(sub):
            ax.plot(sub.depth, sub.auc, "o-", label=t, color=COLORS[t], lw=2)
    ax.set_xlabel("overlap depth (number of co-occurring signatures)")
    ax.set_ylabel("matched-detector AUC (typology rows vs legit)")
    ax.set_title("Cross-border degradation: does each representation still\n"
                 "recover its signature as typologies overlap?")
    ax.set_xticks([1, 2, 3, 4])
    ax.set_ylim(0.45, 1.02)
    ax.axhline(0.5, ls=":", c="grey", lw=1)
    ax.legend(title="typology", fontsize=8)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(FIG / "robustness_d1_degradation.png", dpi=130)
    plt.close(fig)
    print(f"\n  -> {FIG / 'robustness_d1_degradation.png'}")
    return {"auc_by_depth": piv.where(piv.notna(), None).to_dict(),
            "n_pos_by_depth": npos.where(npos.notna(), None).astype("object").to_dict()}


# ── D2 ───────────────────────────────────────────────────────────────────────

def run_multiseed(base, seeds) -> dict:
    print("\n" + "=" * 70)
    print(f"D2  MULTI-SEED VARIANCE  (isolated AUC across injection seeds {seeds})")
    print("=" * 70)
    df = rb.study_multiseed(base, seeds)
    auc_cols = [f"auc_{t}" for t in TYPOLOGIES]
    mean, std = df[auc_cols].mean(), df[auc_cols].std(ddof=1)
    print("\nIsolated solo-vs-legit AUC  (mean +/- std over seeds):")
    print(f"{'typology':10s} {'mean':>7s} {'std':>7s}")
    summary = {}
    for t in TYPOLOGIES:
        m, s = mean[f"auc_{t}"], std[f"auc_{t}"]
        print(f"{t:10s} {m:7.3f} {s:7.4f}")
        summary[t] = {"mean": float(m), "std": float(s)}
    print("\nLR-test gate stability (oracle scalar over classical base):")
    for t in ["ring", "temporal"]:
        g2 = df[f"lrG2_{t}"].replace([np.inf], np.nan)
        adm = df[f"lradm_{t}"].all()
        print(f"  {t:10s} G2 mean={g2.mean():.0f} std={g2.std(ddof=1):.0f}  "
              f"admitted all seeds={adm}")
        summary[f"lr_{t}"] = {"G2_mean": float(g2.mean()),
                              "G2_std": float(g2.std(ddof=1)),
                              "admitted_all": bool(adm)}

    fig, ax = plt.subplots(figsize=(7, 4.2))
    x = np.arange(len(TYPOLOGIES))
    ax.bar(x, [mean[f"auc_{t}"] for t in TYPOLOGIES],
           yerr=[std[f"auc_{t}"] for t in TYPOLOGIES], capsize=5,
           color=[COLORS[t] for t in TYPOLOGIES], alpha=0.85)
    ax.set_xticks(x)
    ax.set_xticklabels(TYPOLOGIES)
    ax.set_ylabel("isolated AUC (mean +/- std)")
    ax.set_ylim(0.5, 1.02)
    ax.set_title(f"Multi-seed stability of the bake-off ({len(seeds)} injection seeds)")
    for xi, t in zip(x, TYPOLOGIES):
        ax.text(xi, mean[f"auc_{t}"] + std[f"auc_{t}"] + 0.01,
                f"{mean[f'auc_{t}']:.3f}", ha="center", fontsize=8)
    ax.grid(alpha=0.3, axis="y")
    fig.tight_layout()
    fig.savefig(FIG / "robustness_d2_multiseed.png", dpi=130)
    plt.close(fig)
    print(f"\n  -> {FIG / 'robustness_d2_multiseed.png'}")
    return summary


# ── D3 ───────────────────────────────────────────────────────────────────────

def run_calibration() -> dict:
    print("\n" + "=" * 70)
    print("D3  CALIBRATION  (Hosmer-Lemeshow + reliability curves, per label)")
    print("=" * 70)
    tr = pd.read_parquet(OUT / "injected_train.parquet")
    te = pd.read_parquet(OUT / "injected_test.parquet")
    tr["trans_dt"] = pd.to_datetime(tr[rb.TIME_COL])
    te["trans_dt"] = pd.to_datetime(te[rb.TIME_COL])
    tbl, curves = rb.study_calibration(tr, te)
    print("\nHosmer-Lemeshow (test split; small p = miscalibrated):")
    print(tbl.to_string(index=False,
          float_format=lambda x: f"{x:.4f}"))

    fig, ax = plt.subplots(figsize=(5.5, 5.5))
    ax.plot([0, 1], [0, 1], "k:", lw=1, label="perfect")
    for t in TYPOLOGIES:
        pred, obs = curves[t]
        ax.plot(pred, obs, "o-", label=t, color=COLORS[t], lw=1.6, ms=4)
    ax.set_xlabel("mean predicted probability (decile bin)")
    ax.set_ylabel("observed frequency")
    ax.set_title("Reliability curves -- per-label logit\n(unweighted; admitted GLM design matrix)")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)
    ax.set_aspect("equal", "box")
    fig.tight_layout()
    fig.savefig(FIG / "robustness_d3_calibration.png", dpi=130)
    plt.close(fig)

    # zoom on the low-probability regime where rare-label mass lives
    fig2, ax2 = plt.subplots(figsize=(5.5, 5.5))
    lim = max(float(tbl["prevalence"].max()), float(tbl["mean_pred"].max())) * 3
    lim = min(max(lim, 0.05), 1.0)
    ax2.plot([0, lim], [0, lim], "k:", lw=1)
    for t in TYPOLOGIES:
        pred, obs = curves[t]
        ax2.plot(pred, obs, "o-", label=t, color=COLORS[t], lw=1.6, ms=4)
    ax2.set_xlim(0, lim)
    ax2.set_ylim(0, lim)
    ax2.set_xlabel("mean predicted probability (decile bin)")
    ax2.set_ylabel("observed frequency")
    ax2.set_title(f"Reliability curves (zoom 0-{lim:.2f})")
    ax2.legend(fontsize=8)
    ax2.grid(alpha=0.3)
    ax2.set_aspect("equal", "box")
    fig2.tight_layout()
    fig2.savefig(FIG / "robustness_d3_calibration_zoom.png", dpi=130)
    plt.close(fig2)
    print(f"\n  -> {FIG / 'robustness_d3_calibration.png'}  (+ _zoom)")
    return tbl.set_index("typology").to_dict(orient="index")


# ── D4 ───────────────────────────────────────────────────────────────────────

def run_sensitivity(base) -> dict:
    print("\n" + "=" * 70)
    print("D4  THRESHOLD / OVERLAP SENSITIVITY  (isolated AUC under +/-20%)")
    print("=" * 70)
    df = rb.study_sensitivity(base)
    piv = df.pivot(index="param", columns="setting", values="auc")[["-20%", "base", "+20%"]]
    piv["typology"] = df.groupby("param")["typology"].first()
    piv["max|delta|"] = (piv[["-20%", "+20%"]].sub(piv["base"], axis=0)).abs().max(axis=1)
    print("\nIsolated AUC under +/-20% moves of each injection knob:")
    print(piv.to_string(float_format=lambda x: f"{x:.3f}"))

    fig, ax = plt.subplots(figsize=(7.5, 4.5))
    params = piv.index.tolist()
    y = np.arange(len(params))
    for col, mk, lbl in [("-20%", "v", "-20%"), ("base", "o", "base"),
                         ("+20%", "^", "+20%")]:
        ax.scatter(piv[col], y, marker=mk, s=70, label=lbl, zorder=3)
    for yi, p in zip(y, params):
        ax.plot([piv.loc[p, "-20%"], piv.loc[p, "+20%"]], [yi, yi],
                c="grey", lw=1, zorder=1)
    ax.set_yticks(y)
    ax.set_yticklabels([f"{p}\n({t})" for p, t in zip(params, piv["typology"])],
                       fontsize=8)
    ax.set_xlabel("isolated AUC")
    ax.set_title("Threshold sensitivity: isolated AUC under +/-20% knob moves")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3, axis="x")
    fig.tight_layout()
    fig.savefig(FIG / "robustness_d4_sensitivity.png", dpi=130)
    plt.close(fig)
    print(f"\n  -> {FIG / 'robustness_d4_sensitivity.png'}")
    return piv.drop(columns="typology").to_dict(orient="index")


# ── main ─────────────────────────────────────────────────────────────────────

def main() -> None:
    which = sys.argv[1] if len(sys.argv) > 1 else "all"
    seeds = [0, 1, 2, 3]
    need_base = which in ("all", "degradation", "multiseed", "sensitivity")
    base = None
    if need_base:
        print("loading legit background (train) ...")
        base = rb.load_legit("train")
        print(f"  legit rows: {len(base):,}")

    results = {}
    if which in ("all", "degradation"):
        results["degradation"] = run_degradation(base)
    if which in ("all", "multiseed"):
        results["multiseed"] = run_multiseed(base, seeds)
    if which in ("all", "calibration"):
        results["calibration"] = run_calibration()
    if which in ("all", "sensitivity"):
        results["sensitivity"] = run_sensitivity(base)

    path = OUT / "robustness_results.json"
    prev = {}
    if path.exists():
        prev = json.loads(path.read_text())
    prev.update(results)
    path.write_text(json.dumps(prev, indent=2, default=str))
    print(f"\n[done] numeric summary -> {path}")


if __name__ == "__main__":
    main()
