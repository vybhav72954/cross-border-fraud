"""
Multi-model tabular bake-off with k-fold cross-validation.

Compares standard off-the-shelf tabular classifiers — logistic regression,
random forest, histogram gradient boosting, XGBoost, LightGBM — on the
binary-relevance task (one model per label), with stratified k-fold CV. This is
the conventional ML model-comparison layer that sits *alongside* the
representation-recovery benchmark: the GNN/SSM scripts answer "which architecture
recovers which planted signature?", this answers "how do standard tabular models
compare to each other, cross-validated?".

AUC / average precision are the CV headline (threshold-free and rank-based, so
robust to the negative subsampling used to keep the 5-model x 5-label x k-fold
grid tractable). Classic threshold metrics (precision / recall / F1, confusion
matrices) are reported by the caller on the FULL held-out test set at the true
prevalence — see ``run_model_benchmark.py``.
"""
from __future__ import annotations

from typing import Callable

import numpy as np
import pandas as pd
from sklearn.ensemble import (
    HistGradientBoostingClassifier,
    RandomForestClassifier,
)
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold


def build_model_zoo(seed: int = 0) -> dict[str, Callable]:
    """Map model name -> zero-arg factory returning a fresh estimator.

    A factory (not an instance) so every CV fold/label gets an unfitted clone.
    XGBoost and LightGBM are added only when importable, so the bake-off degrades
    gracefully to the three always-present sklearn models if either is missing.
    """
    zoo: dict[str, Callable] = {
        "logreg": lambda: LogisticRegression(
            max_iter=1000, class_weight="balanced"),
        "random_forest": lambda: RandomForestClassifier(
            n_estimators=150, n_jobs=-1, class_weight="balanced_subsample",
            random_state=seed),
        "hist_gb": lambda: HistGradientBoostingClassifier(
            max_iter=200, random_state=seed),
    }
    try:
        from xgboost import XGBClassifier
        zoo["xgboost"] = lambda: XGBClassifier(
            n_estimators=300, max_depth=6, learning_rate=0.1, tree_method="hist",
            n_jobs=-1, eval_metric="logloss", random_state=seed)
    except ImportError:
        pass
    try:
        from lightgbm import LGBMClassifier
        zoo["lightgbm"] = lambda: LGBMClassifier(
            n_estimators=300, learning_rate=0.1, n_jobs=-1,
            class_weight="balanced", random_state=seed, verbose=-1)
    except ImportError:
        pass
    return zoo


def subsample_keep_positives(
    Y: pd.DataFrame, n: int, seed: int = 0
) -> np.ndarray:
    """Positional row indices: keep EVERY positive row (any label set) and add a
    random sample of legit rows up to a total of ~``n``.

    A plain random subsample would mostly drop the rarest labels (ring is ~0.1%
    of rows); keeping all positives preserves enough of each signature to fit and
    fold. ``n <= 0`` returns all rows (no subsampling). AUC/AP are rank-based, so
    deflating the negative side does not bias the CV ranking.
    """
    yy = Y.to_numpy().astype(bool)
    any_pos = yy.any(axis=1)
    pos = np.where(any_pos)[0]
    neg = np.where(~any_pos)[0]
    if n <= 0 or n >= len(pos) + len(neg):
        idx = np.arange(len(Y))
    else:
        rng = np.random.default_rng(seed)
        n_neg = max(0, n - len(pos))
        if n_neg < len(neg):
            neg = rng.choice(neg, size=n_neg, replace=False)
        idx = np.concatenate([pos, neg])
    rng = np.random.default_rng(seed + 1)
    rng.shuffle(idx)
    return idx


def isolated_auc(score: np.ndarray, tags: np.ndarray, typ: str) -> float:
    """AUC for one typology scored on its SOLO rows vs legit only.

    The project's contamination-free detectability measure: restrict to rows
    tagged exactly ``typ`` (single-signature) or ``""`` (legit), so a typology
    cannot borrow another's signal through an overlap event (e.g. geo+ring
    letting ring inherit geo's distance). Returns NaN if the slice is single-
    class. Mirrors the isolated AUC in ``run_glm_baseline.py``.
    """
    solo = tags == typ
    legit = tags == ""
    m = solo | legit
    n_pos = int(solo[m].sum())
    if n_pos == 0 or n_pos == int(m.sum()):
        return float("nan")
    return roc_auc_score(solo[m].astype(int), score[m])


def cross_validate_models(
    models: dict[str, Callable],
    X: np.ndarray,
    Y: pd.DataFrame,
    label_names: list[str],
    n_splits: int = 5,
    seed: int = 0,
    tags: np.ndarray | None = None,
    progress: bool = True,
) -> pd.DataFrame:
    """Stratified k-fold CV per (label, model) — binary relevance.

    For each label, the same StratifiedKFold split (on that label's binary
    target) is reused across all models, so models are compared on identical
    folds. Returns a tidy long DataFrame with one row per (model, label, fold)
    and columns ``auc``, ``ap`` (multi-label view). If ``tags`` (the per-row
    "+"-joined typology string, aligned to ``X``) is given, also adds ``auc_iso``
    — the isolated solo-vs-legit AUC on the validation fold.
    """
    X = np.asarray(X)
    rows = []
    for label in label_names:
        y = Y[label].to_numpy().astype(int)
        skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
        for fold, (tr, va) in enumerate(skf.split(X, y)):
            for mname, factory in models.items():
                clf = factory()
                clf.fit(X[tr], y[tr])
                s = clf.predict_proba(X[va])[:, 1]
                row = {"model": mname, "label": label, "fold": fold,
                       "auc": roc_auc_score(y[va], s),
                       "ap": average_precision_score(y[va], s)}
                if tags is not None:
                    row["auc_iso"] = isolated_auc(s, tags[va], label)
                rows.append(row)
            if progress:
                print(f"  cv {label:9s} fold {fold + 1}/{n_splits} done",
                      flush=True)
    return pd.DataFrame(rows)


def aggregate_cv(cv: pd.DataFrame) -> pd.DataFrame:
    """Collapse the per-fold long frame to mean +/- std per (model, label)."""
    aggs = {"auc_mean": ("auc", "mean"), "auc_std": ("auc", "std"),
            "ap_mean": ("ap", "mean"), "ap_std": ("ap", "std")}
    if "auc_iso" in cv.columns:
        aggs["auc_iso_mean"] = ("auc_iso", "mean")
        aggs["auc_iso_std"] = ("auc_iso", "std")
    return cv.groupby(["model", "label"]).agg(**aggs).reset_index()


def fit_score_test(
    models: dict[str, Callable],
    X_train: np.ndarray,
    Y_train: pd.DataFrame,
    X_test: np.ndarray,
    label_names: list[str],
) -> dict[str, np.ndarray]:
    """Fit one model per label on the (sub)sampled train and score the full test.

    Returns ``{model_name: (n_test, K) probability matrix}``.
    """
    X_train, X_test = np.asarray(X_train), np.asarray(X_test)
    out: dict[str, np.ndarray] = {}
    for mname, factory in models.items():
        scores = np.zeros((X_test.shape[0], len(label_names)))
        for i, label in enumerate(label_names):
            clf = factory()
            clf.fit(X_train, Y_train[label].to_numpy().astype(int))
            scores[:, i] = clf.predict_proba(X_test)[:, 1]
        out[mname] = scores
    return out
