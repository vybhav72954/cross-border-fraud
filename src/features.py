"""
Feature engineering for the GLM design matrix.

Produces a clean X matrix from raw Sparkov columns + derived label columns.
All transformations are deterministic — no fitted scalers stored here.
Categorical variables are one-hot encoded with explicit reference categories.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from typing import Optional


# Reference categories (baseline for dummy encoding — chosen as most common class)
REF_CATEGORY = "grocery_pos"
REF_STATE = "TX"
REF_GENDER = "M"
REF_AMT_BIN = "10-50"

AMT_BINS = [0, 10, 50, 100, 250, 500, 1000, np.inf]
AMT_LABELS = ["0-10", "10-50", "50-100", "100-250", "250-500", "500-1000", "1000+"]


def build_features(df: pd.DataFrame,
                   include_geo_scalars: bool = True,
                   include_velocity_scalar: bool = True) -> pd.DataFrame:
    """Build GLM design matrix from raw + label-annotated Sparkov DataFrame.

    Returns a DataFrame of float columns ready for sm.add_constant().
    No intercept is added here.

    Parameters
    ----------
    df:
        Output of derive_labels() — must contain trans_dt, all raw Sparkov
        columns, and the L_* label columns.
    include_geo_scalars:
        Include haversine distance and a home_state_match indicator.
    include_velocity_scalar:
        Include per-card rolling 1-hour transaction count as a raw velocity
        feature (separate from the L_V label).
    """
    out = pd.DataFrame(index=df.index)

    # ── Transaction-level scalars ─────────────────────────────────────────────
    out["log_amt"] = np.log1p(df["amt"])

    amt_bin = pd.cut(df["amt"], bins=AMT_BINS, labels=AMT_LABELS, right=False)
    amt_dummies = pd.get_dummies(amt_bin, prefix="amt", drop_first=False).astype(float)
    if f"amt_{REF_AMT_BIN}" in amt_dummies.columns:
        amt_dummies = amt_dummies.drop(columns=[f"amt_{REF_AMT_BIN}"])
    out = pd.concat([out, amt_dummies], axis=1)

    out["hour"] = df["trans_dt"].dt.hour
    out["hour_sin"] = np.sin(2 * np.pi * out["hour"] / 24)
    out["hour_cos"] = np.cos(2 * np.pi * out["hour"] / 24)
    out["dow"] = df["trans_dt"].dt.dayofweek
    out["is_weekend"] = (out["dow"] >= 5).astype(float)
    out = out.drop(columns=["hour", "dow"])

    # ── Cardholder demographics ───────────────────────────────────────────────
    dob = pd.to_datetime(df["dob"])
    out["age"] = (df["trans_dt"] - dob).dt.days / 365.25

    gender_dummies = pd.get_dummies(df["gender"], prefix="gender", drop_first=False).astype(float)
    if f"gender_{REF_GENDER}" in gender_dummies.columns:
        gender_dummies = gender_dummies.drop(columns=[f"gender_{REF_GENDER}"])
    out = pd.concat([out, gender_dummies], axis=1)

    out["log_city_pop"] = np.log1p(df["city_pop"])

    # ── Merchant category ─────────────────────────────────────────────────────
    cat_dummies = pd.get_dummies(df["category"], prefix="cat", drop_first=False).astype(float)
    ref_col = f"cat_{REF_CATEGORY}"
    if ref_col in cat_dummies.columns:
        cat_dummies = cat_dummies.drop(columns=[ref_col])
    out = pd.concat([out, cat_dummies], axis=1)

    # ── Geography ─────────────────────────────────────────────────────────────
    state_dummies = pd.get_dummies(df["state"], prefix="state", drop_first=False).astype(float)
    ref_state_col = f"state_{REF_STATE}"
    if ref_state_col in state_dummies.columns:
        state_dummies = state_dummies.drop(columns=[ref_state_col])
    out = pd.concat([out, state_dummies], axis=1)

    if include_geo_scalars:
        from src.labels import _haversine_series
        out["merch_dist_km"] = _haversine_series(
            df["lat"], df["long"], df["merch_lat"], df["merch_long"]
        )
        out["home_state_match"] = (
            df["state"] == df["merchant"].apply(_merchant_state_stub)
        ).astype(float)

    # ── Per-card velocity scalar ──────────────────────────────────────────────
    if include_velocity_scalar:
        df_s = df.sort_values(["cc_num", "trans_dt"]).set_index("trans_dt")
        counts = (
            df_s.groupby("cc_num", group_keys=False)["trans_num"]
            .rolling("60min", closed="both")
            .count()
        )
        counts.index = counts.index.droplevel(0)
        df_s["_vel1h"] = counts
        df_s = df_s.reset_index()
        vel = df_s.set_index(df.sort_values(["cc_num", "trans_dt"]).index)["_vel1h"]
        out["vel_1h"] = vel.reindex(df.index).fillna(1).astype(float)

    return out.astype(float)


def _merchant_state_stub(merchant_name: str) -> str:
    """Placeholder — Sparkov merchant names don't encode state.
    Returns empty string; home_state_match will be 0 for all rows.
    Replace with a merchant-state lookup if one is derived later."""
    return ""


def label_matrix(df: pd.DataFrame) -> pd.DataFrame:
    """Return the K×N label matrix as a float DataFrame."""
    from src.labels import LABEL_COLS
    return df[LABEL_COLS].astype(float)


def severity_vector(df: pd.DataFrame) -> pd.Series:
    """Return label cardinality as an ordered integer series."""
    return df["label_cardinality"].astype(int)


def velocity_count_vector(df: pd.DataFrame, window_min: int = 10) -> pd.Series:
    """Per-card transaction count in a rolling window — Poisson response."""
    sorted_index = df.sort_values(["cc_num", "trans_dt"]).index
    df_s = df.sort_values(["cc_num", "trans_dt"]).set_index("trans_dt")
    window = f"{window_min}min"
    counts = (
        df_s.groupby("cc_num", group_keys=False)["trans_num"]
        .rolling(window, closed="both")
        .count()
    )
    counts.index = counts.index.droplevel(0)
    # counts are in (cc_num, trans_dt)-sorted order; map back to original rows
    # via the sorted index before reindexing (reset_index(drop=True) would align
    # sorted-order values to original positions — the alignment bug this fixes).
    vel = pd.Series(counts.to_numpy(), index=sorted_index)
    return vel.reindex(df.index).fillna(1).astype(int)
