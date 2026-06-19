"""
Multi-model tabular bake-off with cross-validation + classic test metrics.

The conventional ML-project evaluation layer, run on the controlled injected
dataset's tabular design matrix (src/features.py):

  1. CROSS-VALIDATION model comparison. Stratified k-fold CV (binary relevance,
     one model per typology) across logistic regression / random forest /
     histogram GB / XGBoost / LightGBM. Reports mean +/- std AUC and average
     precision per (model, typology). This is the "multiple models, cross-
     validated" table the GNN/SSM scripts deliberately skip (they answer a
     different, representation-recovery question).

  2. HELD-OUT TEST metrics at the true prevalence. The best model by mean CV AUC
     is refit and scored on the FULL test split; per-typology precision / recall
     / F1 and confusion matrices are reported (the classic threshold-0.5 metrics
     that src/evaluation.py's AUC/AP/ranking suite omits).

Outputs (mirrors the per-phase artifacts of the other projects in this portfolio):
  results/benchmark_cv_results.csv     per-fold long CV results
  results/benchmark_cv_summary.csv     mean +/- std per (model, typology)
  results/benchmark_test_metrics.csv   per-typology test metrics, all models
  results/benchmark_results.json       headline summary
  results/benchmark_cv_auc.png         grouped-bar model comparison
  results/benchmark_confusion.png      confusion matrices (best model)

The CV grid is fit on a subsample that keeps every injected (positive) row plus a
random legit sample (default 120k total) so the 5x5x5 fit grid is tractable; AUC/
AP are rank-based and so robust to the deflated negative side. Test metrics use
the full, unsampled test split at the real <2% prevalence.

Run from the project root:  python scripts/02_model_benchmark.py
                            python scripts/02_model_benchmark.py --sample 0 --folds 5
"""
import argparse
import json
import sys
import warnings
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import seaborn as sns  # noqa: E402
from sklearn.preprocessing import StandardScaler  # noqa: E402

sys.path.insert(0, ".")
from src.benchmark import (  # noqa: E402
    aggregate_cv,
    build_model_zoo,
    cross_validate_models,
    fit_score_test,
    isolated_auc,
    subsample_keep_positives,
)
from src.evaluation import (  # noqa: E402
    confusion_matrices,
    multi_label_report,
    per_label_threshold_metrics,
)
from src.features import build_features  # noqa: E402
from src.inject import TYPOLOGIES, TYPOLOGY_COL, typology_dummies  # noqa: E402

# LightGBM's sklearn wrapper emits this benign warning when fit/predict mix
# array and frame internally; the data here is consistent (numpy throughout).
warnings.filterwarnings("ignore", message="X does not have valid feature names")

OUT = Path("data/processed")   # parquet inputs
RES = Path("results")          # figures + tables + json outputs
SEED = 0

# Answer-key typology -> production label, for cross-reference in the tables.
TYP_TO_LABEL = {"ring": "L_R", "velocity": "L_V", "temporal": "L_T",
                "category": "L_C", "geo": "L_G"}


def load(split: str) -> pd.DataFrame:
    df = pd.read_parquet(OUT / f"injected_{split}.parquet")
    df["trans_dt"] = pd.to_datetime(df["trans_date_trans_time"])
    return df


def plot_cv_bars(summary: pd.DataFrame, path: Path, mean_col: str = "auc_mean",
                 std_col: str = "auc_std", subtitle: str = "") -> None:
    models = sorted(summary["model"].unique())
    s = summary.set_index(["model", "label"])
    x = np.arange(len(TYPOLOGIES))
    width = 0.8 / len(models)
    fig, ax = plt.subplots(figsize=(11, 6))
    for j, model in enumerate(models):
        means = [s.loc[(model, t), mean_col] for t in TYPOLOGIES]
        errs = np.nan_to_num([s.loc[(model, t), std_col] for t in TYPOLOGIES])
        ax.bar(x + j * width - 0.4 + width / 2, means, width, label=model,
               yerr=errs, capsize=2, error_kw={"lw": 0.8})
    ax.set_ylim(0.4, 1.02)
    ax.axhline(0.5, ls="--", lw=0.8, color="grey")
    ax.set_xticks(x)
    ax.set_xticklabels([f"{t}\n({TYP_TO_LABEL[t]})" for t in TYPOLOGIES])
    ax.set_xlabel("typology (answer key)")
    ax.set_ylabel("cross-validated AUC (mean +/- std)")
    title = f"Tabular model comparison - {summary['fold_n'].iloc[0]}-fold stratified CV"
    ax.set_title(title + (f"\n{subtitle}" if subtitle else ""))
    ax.legend(title="model", bbox_to_anchor=(1.01, 1), loc="upper left")
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def plot_confusions(cms: dict[str, np.ndarray], model: str, path: Path) -> None:
    fig, axes = plt.subplots(1, len(TYPOLOGIES), figsize=(3.1 * len(TYPOLOGIES), 3.2))
    for ax, typ in zip(axes, TYPOLOGIES, strict=True):
        cm = cms[typ]
        norm = cm / cm.sum(axis=1, keepdims=True).clip(min=1)
        annot = np.array([[f"{cm[r, c]:,}\n{norm[r, c]:.2f}" for c in range(2)]
                          for r in range(2)])
        sns.heatmap(norm, annot=annot, fmt="", cmap="Blues", vmin=0, vmax=1,
                    cbar=False, ax=ax, annot_kws={"fontsize": 8})
        ax.set_title(f"{typ} ({TYP_TO_LABEL[typ]})")
        ax.set_xlabel("predicted")
        ax.set_ylabel("true")
        ax.set_xticklabels([0, 1])
        ax.set_yticklabels([0, 1], rotation=0)
    fig.suptitle(f"Confusion matrices @0.5 — {model} (full test, count + "
                 f"row-normalised)", y=1.04)
    fig.tight_layout()
    fig.savefig(path, dpi=130, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--sample", type=int, default=120_000,
                   help="train subsample size for the CV grid (0 = all rows)")
    p.add_argument("--folds", type=int, default=5)
    args = p.parse_args()

    RES.mkdir(exist_ok=True)
    tr, te = load("train"), load("test")

    Xtr_full = build_features(tr)
    Xte = build_features(te).reindex(columns=Xtr_full.columns, fill_value=0.0)
    Ytr = typology_dummies(tr)[TYPOLOGIES]
    Yte = typology_dummies(te)[TYPOLOGIES]

    scaler = StandardScaler().fit(Xtr_full)
    idx = subsample_keep_positives(Ytr, args.sample, seed=SEED)
    Xtr_s = scaler.transform(Xtr_full.iloc[idx])
    Ytr_s = Ytr.iloc[idx].reset_index(drop=True)
    Xte_s = scaler.transform(Xte)

    # per-row typology tag for the isolated solo-vs-legit view (answer key)
    tags_tr_s = tr[TYPOLOGY_COL].fillna("").to_numpy()[idx]
    tags_te = te[TYPOLOGY_COL].fillna("").to_numpy()

    zoo = build_model_zoo(seed=SEED)
    print(f"Models: {', '.join(zoo)}")
    print(f"CV grid: {len(zoo)} models x {len(TYPOLOGIES)} typologies x "
          f"{args.folds} folds on {len(idx):,} rows "
          f"({int(Ytr.iloc[idx].to_numpy().any(axis=1).sum()):,} positive)\n")

    # ── 1. cross-validated model comparison ───────────────────────────────────
    cv = cross_validate_models(zoo, Xtr_s, Ytr_s, TYPOLOGIES,
                               n_splits=args.folds, seed=SEED, tags=tags_tr_s)
    summary = aggregate_cv(cv)
    summary["fold_n"] = args.folds
    cv.to_csv(RES / "benchmark_cv_results.csv", index=False)
    summary.to_csv(RES / "benchmark_cv_summary.csv", index=False)

    def make_pivot(value: str) -> pd.DataFrame:
        pv = summary.pivot(index="model", columns="label", values=value)[TYPOLOGIES]
        pv["mean"] = pv.mean(axis=1)
        return pv

    iso_pivot = make_pivot("auc_iso_mean").sort_values("mean", ascending=False)
    ml_pivot = make_pivot("auc_mean").reindex(iso_pivot.index)
    best = iso_pivot.index[0]

    hdr = "  ".join(f"{t}={TYP_TO_LABEL[t]}" for t in TYPOLOGIES)
    print(f"Cross-validated AUC - ISOLATED (each signature's solo rows vs legit)"
          f"\n   {hdr}")
    print(iso_pivot.round(3).to_string())
    print("\nCross-validated AUC - MULTI-LABEL (full set; overlaps let a typology"
          " borrow another's signal)")
    print(ml_pivot.round(3).to_string())
    print(f"\nBest model by mean isolated CV AUC: {best} "
          f"({iso_pivot.loc[best, 'mean']:.3f})")
    plot_cv_bars(summary, RES / "benchmark_cv_auc.png",
                 mean_col="auc_iso_mean", std_col="auc_iso_std",
                 subtitle="isolated solo-vs-legit AUC")

    # ── 2. held-out test metrics at true prevalence ───────────────────────────
    test_scores = fit_score_test(zoo, Xtr_s, Ytr_s, Xte_s, TYPOLOGIES)

    test_rows = []
    for mname, scores in test_scores.items():
        preds = (scores >= 0.5).astype(int)
        rep = multi_label_report(Yte.to_numpy(), preds, scores, label_names=TYPOLOGIES)
        thr = per_label_threshold_metrics(Yte.to_numpy(), preds, label_names=TYPOLOGIES)
        for j, typ in enumerate(TYPOLOGIES):
            test_rows.append({
                "model": mname, "typology": typ, "label": TYP_TO_LABEL[typ],
                "auc_ml": rep["per_label"][typ]["auc"],
                "auc_iso": isolated_auc(scores[:, j], tags_te, typ),
                "ap": rep["per_label"][typ]["ap"],
                "precision": thr.loc[typ, "precision"],
                "recall": thr.loc[typ, "recall"],
                "f1": thr.loc[typ, "f1"],
                "support": int(thr.loc[typ, "support"]),
            })
    test_df = pd.DataFrame(test_rows)
    test_df.to_csv(RES / "benchmark_test_metrics.csv", index=False)

    print(f"\nHeld-out test - {best} (full split, threshold 0.5)   "
          f"auc_ml=multi-label  auc_iso=solo-vs-legit:")
    bt = test_df[test_df["model"] == best].set_index("typology")[
        ["label", "auc_ml", "auc_iso", "precision", "recall", "f1", "support"]]
    print(bt.round(3).to_string())

    best_preds = (test_scores[best] >= 0.5).astype(int)
    cms = confusion_matrices(Yte.to_numpy(), best_preds, label_names=TYPOLOGIES)
    plot_confusions(cms, best, RES / "benchmark_confusion.png")

    best_test = test_df[test_df["model"] == best]
    summary_json = {
        "models": list(zoo),
        "folds": args.folds,
        "cv_sample_rows": len(idx),
        "best_model": best,
        "best_mean_cv_auc_isolated": float(iso_pivot.loc[best, "mean"]),
        "cv_auc_isolated_by_model": {m: float(iso_pivot.loc[m, "mean"])
                                     for m in iso_pivot.index},
        "cv_auc_multilabel_by_model": {m: float(ml_pivot.loc[m, "mean"])
                                       for m in ml_pivot.index},
        "test_auc_isolated_best_model": {
            r["typology"]: r["auc_iso"] for _, r in best_test.iterrows()},
        "test_auc_multilabel_best_model": {
            r["typology"]: r["auc_ml"] for _, r in best_test.iterrows()},
        "test_f1_best_model": {
            r["typology"]: r["f1"] for _, r in best_test.iterrows()},
    }
    (RES / "benchmark_results.json").write_text(json.dumps(summary_json, indent=2))
    print("\nSaved to results/: benchmark_cv_results.csv, benchmark_cv_summary.csv, "
          "benchmark_test_metrics.csv, benchmark_results.json")
    print("            benchmark_cv_auc.png, benchmark_confusion.png")


if __name__ == "__main__":
    main()
