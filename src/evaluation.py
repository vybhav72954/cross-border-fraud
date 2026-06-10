"""
Multi-label evaluation metrics.

Standard single-label metrics (accuracy, AUC alone) are insufficient for
multi-label problems. This module reports the full suite from the TDD.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import chi2
from sklearn.metrics import (
    hamming_loss,
    accuracy_score,
    label_ranking_average_precision_score,
    roc_auc_score,
    average_precision_score,
)

LABEL_NAMES = ["L_V", "L_G", "L_C", "L_R", "L_T"]


def multi_label_report(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_score: np.ndarray,
    label_names: list[str] = LABEL_NAMES,
) -> dict:
    """Compute the full multi-label evaluation suite.

    Parameters
    ----------
    y_true:  (n, K) binary ground-truth label matrix
    y_pred:  (n, K) binary predicted label matrix (threshold 0.5)
    y_score: (n, K) predicted probability matrix (pre-threshold)

    Returns
    -------
    dict with scalar metrics and per-label breakdowns
    """
    report = {
        "hamming_loss": hamming_loss(y_true, y_pred),
        "subset_accuracy": accuracy_score(y_true, y_pred),
        "label_ranking_ap": label_ranking_average_precision_score(y_true, y_score),
        "mean_auc": roc_auc_score(y_true, y_score, average="macro"),
        "label_cardinality_true": float(y_true.sum(axis=1).mean()),
        "label_cardinality_pred": float(y_pred.sum(axis=1).mean()),
    }

    per_label: dict[str, dict] = {}
    for i, name in enumerate(label_names):
        if y_true[:, i].sum() == 0:
            per_label[name] = {"auc": None, "ap": None, "prevalence": 0.0}
            continue
        per_label[name] = {
            "auc": roc_auc_score(y_true[:, i], y_score[:, i]),
            "ap": average_precision_score(y_true[:, i], y_score[:, i]),
            "prevalence": float(y_true[:, i].mean()),
        }

    report["per_label"] = per_label
    return report


def lr_test(ll_full: float, ll_reduced: float, df_diff: int) -> dict:
    """Likelihood-ratio test for nested GLMs.

    Parameters
    ----------
    ll_full:    log-likelihood of the fuller model (M1)
    ll_reduced: log-likelihood of the restricted model (M0)
    df_diff:    number of additional parameters in M1

    Returns
    -------
    dict with G2 statistic, p-value, and admission decision
    """
    g2 = 2.0 * (ll_full - ll_reduced)
    p = float(chi2.sf(g2, df_diff))
    return {
        "G2": g2,
        "df": df_diff,
        "p_value": p,
        "admitted": bool(p < 0.05),
    }


def hosmer_lemeshow(y_true: np.ndarray, y_prob: np.ndarray, g: int = 10) -> dict:
    """Hosmer-Lemeshow calibration test for a single binary outcome."""
    d = pd.DataFrame({"y": y_true, "p": y_prob})
    d["grp"] = pd.qcut(d["p"], q=g, duplicates="drop", labels=False)
    agg = d.groupby("grp").agg(n=("y", "size"), obs=("y", "sum"), pbar=("p", "mean"))
    exp = agg["n"] * agg["pbar"]
    hl = float((((agg["obs"] - exp) ** 2) / (exp * (1 - agg["pbar"]))).sum())
    dof = len(agg) - 2
    return {"HL": hl, "df": dof, "p_value": float(chi2.sf(hl, dof))}


def label_cooccurrence_matrix(y: np.ndarray, label_names: list[str] = LABEL_NAMES) -> pd.DataFrame:
    """Pairwise label co-occurrence rates as a symmetric DataFrame."""
    k = y.shape[1]
    mat = np.zeros((k, k))
    for i in range(k):
        for j in range(k):
            both = ((y[:, i] == 1) & (y[:, j] == 1)).sum()
            mat[i, j] = both / len(y)
    return pd.DataFrame(mat, index=label_names, columns=label_names)


def cross_border_stats(df: pd.DataFrame) -> dict:
    """Summary statistics for cross-border (|L|≥2) transactions."""
    cb = df["cross_border"] if "cross_border" in df.columns else (
        df[["L_V", "L_G", "L_C", "L_R", "L_T"]].sum(axis=1) >= 2
    )
    return {
        "n_cross_border": int(cb.sum()),
        "pct_cross_border": float(cb.mean() * 100),
        "pct_of_fraud": float(
            cb[df["is_fraud"] == 1].mean() * 100
            if "is_fraud" in df.columns else np.nan
        ),
    }
