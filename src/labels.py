"""
Rule-based label derivation for cross-border fraud detection.

All thresholds are fixed constants declared here. Labels are derived from
raw Sparkov columns only — no model output informs any label. Run this
module before any model fitting.

Label meanings:
  L_V  Velocity burst          — confirmed-fraud subtype (gated on is_fraud=1)
  L_G  Geographic anomaly      — anomaly precursor (fires on legitimate too)
  L_C  Category anomaly        — anomaly precursor
  L_R  Ring membership         — confirmed-fraud subtype (gated on is_fraud=1)
  L_T  Temporal anomaly        — anomaly precursor
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from math import radians, cos, sin, asin, sqrt

# ── Thresholds (fixed; sensitivity ±20% reported separately) ──────────────────
VELOCITY_WINDOW_MIN: int = 30       # rolling window in minutes
VELOCITY_MIN_TXN: int = 3           # transactions in window to fire L_V
GEO_DISTANCE_KM: float = 500.0      # km from home to merchant to fire L_G
RING_WINDOW_HOURS: int = 24         # ± hours around a transaction for ring check
RING_MIN_CARDS: int = 3             # distinct cc_nums at same merchant in window
TEMPORAL_IQR_MULTIPLIER: float = 1.5
CATEGORY_LOOKBACK_DAYS: int = 30
COLD_START_MIN_DAYS: int = 7        # use global prior below this history length

LABEL_COLS = ["L_V", "L_G", "L_C", "L_R", "L_T"]


# ── Haversine ─────────────────────────────────────────────────────────────────

def _haversine_series(lat1: pd.Series, lon1: pd.Series,
                      lat2: pd.Series, lon2: pd.Series) -> pd.Series:
    """Vectorized great-circle distance in km."""
    R = 6371.0
    lat1r, lon1r, lat2r, lon2r = map(
        lambda s: np.radians(s.values), [lat1, lon1, lat2, lon2]
    )
    dlat = lat2r - lat1r
    dlon = lon2r - lon1r
    a = np.sin(dlat / 2) ** 2 + np.cos(lat1r) * np.cos(lat2r) * np.sin(dlon / 2) ** 2
    return pd.Series(2 * R * np.arcsin(np.sqrt(a)), index=lat1.index)


# ── Individual label derivers ──────────────────────────────────────────────────

def _derive_L_V(df: pd.DataFrame) -> pd.Series:
    """Velocity burst: ≥VELOCITY_MIN_TXN txns from same cc_num in VELOCITY_WINDOW_MIN minutes.
    Gated on is_fraud=1."""
    df_s = df.sort_values(["cc_num", "trans_dt"]).copy()
    df_s = df_s.set_index("trans_dt")
    window = f"{VELOCITY_WINDOW_MIN}min"
    # count transactions (including self) in rolling window per card
    counts = (
        df_s.groupby("cc_num", group_keys=False)["trans_num"]
        .rolling(window, closed="both")
        .count()
    )
    counts.index = counts.index.droplevel(0)
    df_s["_vel"] = counts
    df_s = df_s.reset_index()
    df_s["L_V"] = ((df_s["_vel"] >= VELOCITY_MIN_TXN) & (df_s["is_fraud"] == 1)).astype(np.int8)
    return df_s.set_index(df.sort_values(["cc_num", "trans_dt"]).index)["L_V"].reindex(df.index)


def _derive_L_G(df: pd.DataFrame) -> pd.Series:
    """Geographic anomaly: haversine(home, merchant) > GEO_DISTANCE_KM.
    Not gated — legitimate travellers can also be far from home."""
    dist = _haversine_series(df["lat"], df["long"], df["merch_lat"], df["merch_long"])
    return (dist > GEO_DISTANCE_KM).astype(np.int8)


def _derive_L_C(df: pd.DataFrame) -> pd.Series:
    """Category anomaly: transaction category ≠ modal category over prior CATEGORY_LOOKBACK_DAYS.
    Not gated. Uses global modal category as cold-start prior."""
    global_mode = df["category"].mode().iloc[0]
    df_s = df.sort_values(["cc_num", "trans_dt"]).copy()
    results: list[tuple[int, int]] = []

    for cc, g in df_s.groupby("cc_num", sort=False):
        g = g.reset_index()
        flags = []
        for i, row in g.iterrows():
            cutoff = row["trans_dt"] - pd.Timedelta(days=CATEGORY_LOOKBACK_DAYS)
            history = g[(g["trans_dt"] < row["trans_dt"]) & (g["trans_dt"] >= cutoff)]
            span_days = (row["trans_dt"] - g["trans_dt"].min()).days
            if span_days < COLD_START_MIN_DAYS or len(history) == 0:
                modal = global_mode
            else:
                modal = history["category"].mode().iloc[0]
            flags.append((g.at[i, "index"], int(row["category"] != modal)))
        results.extend(flags)

    idx, vals = zip(*results) if results else ([], [])
    return pd.Series(vals, index=idx, dtype=np.int8).reindex(df.index).fillna(0).astype(np.int8)


def _derive_L_R(df: pd.DataFrame) -> pd.Series:
    """Ring membership: ≥RING_MIN_CARDS distinct fraudulent cc_nums at same merchant
    in ±RING_WINDOW_HOURS. Gated on is_fraud=1."""
    fraud = df[df["is_fraud"] == 1][["trans_dt", "cc_num", "merchant"]].copy()
    if fraud.empty:
        return pd.Series(0, index=df.index, dtype=np.int8)

    # For each fraudulent transaction find distinct cc_nums at same merchant in ±window
    fraud = fraud.sort_values("trans_dt").reset_index(names="orig_idx")
    window = pd.Timedelta(hours=RING_WINDOW_HOURS)

    ring_orig_idxs: set[int] = set()
    merchant_groups = fraud.groupby("merchant", sort=False)

    for _, mg in merchant_groups:
        if len(mg) < RING_MIN_CARDS:
            continue
        mg = mg.sort_values("trans_dt").reset_index(drop=True)
        times = mg["trans_dt"].values
        cards = mg["cc_num"].values
        orig_idxs = mg["orig_idx"].values
        for i in range(len(mg)):
            t = times[i]
            mask = (times >= t - window) & (times <= t + window)
            if len(set(cards[mask])) >= RING_MIN_CARDS:
                ring_orig_idxs.add(orig_idxs[i])

    result = pd.Series(0, index=df.index, dtype=np.int8)
    result.loc[list(ring_orig_idxs)] = 1
    return result


def _derive_L_T(df: pd.DataFrame) -> pd.Series:
    """Temporal anomaly: transaction hour outside Tukey 1.5×IQR of cardholder's
    hour-of-day distribution. Not gated. Uses global IQR as cold-start prior."""
    df_s = df.copy()
    df_s["_hour"] = df_s["trans_dt"].dt.hour

    global_q1, global_q3 = df_s["_hour"].quantile([0.25, 0.75])
    global_iqr = global_q3 - global_q1
    global_lo = global_q1 - TEMPORAL_IQR_MULTIPLIER * global_iqr
    global_hi = global_q3 + TEMPORAL_IQR_MULTIPLIER * global_iqr

    results: dict[int, int] = {}
    for cc, g in df_s.groupby("cc_num", sort=False):
        span_days = (g["trans_dt"].max() - g["trans_dt"].min()).days
        if span_days < COLD_START_MIN_DAYS or len(g) < 4:
            lo, hi = global_lo, global_hi
        else:
            q1, q3 = g["_hour"].quantile([0.25, 0.75])
            iqr = q3 - q1
            lo = q1 - TEMPORAL_IQR_MULTIPLIER * iqr
            hi = q3 + TEMPORAL_IQR_MULTIPLIER * iqr
        for idx, row in g.iterrows():
            results[idx] = int(row["_hour"] < lo or row["_hour"] > hi)

    return pd.Series(results, dtype=np.int8).reindex(df.index).fillna(0).astype(np.int8)


# ── Main entry point ───────────────────────────────────────────────────────────

def derive_labels(df: pd.DataFrame) -> pd.DataFrame:
    """Derive all five fraud labels from raw Sparkov columns.

    Adds columns L_V, L_G, L_C, L_R, L_T, label_cardinality, cross_border.
    Input must contain the full Sparkov schema. Returns a copy with labels appended.

    This function must be called before any model fitting.
    Thresholds are module-level constants; see sensitivity analysis for ±20% variants.
    """
    required = {"trans_date_trans_time", "cc_num", "merchant", "category",
                "lat", "long", "merch_lat", "merch_long", "is_fraud", "trans_num"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    out = df.copy()
    out["trans_dt"] = pd.to_datetime(out["trans_date_trans_time"])

    print("Deriving L_V (velocity burst)...")
    out["L_V"] = _derive_L_V(out)

    print("Deriving L_G (geographic anomaly)...")
    out["L_G"] = _derive_L_G(out)

    print("Deriving L_C (category anomaly)...")
    out["L_C"] = _derive_L_C(out)

    print("Deriving L_R (ring membership)...")
    out["L_R"] = _derive_L_R(out)

    print("Deriving L_T (temporal anomaly)...")
    out["L_T"] = _derive_L_T(out)

    out["label_cardinality"] = out[LABEL_COLS].sum(axis=1).astype(np.int8)
    out["cross_border"] = (out["label_cardinality"] >= 2).astype(np.int8)

    _report_prevalence(out)
    return out


def _report_prevalence(df: pd.DataFrame) -> None:
    n = len(df)
    print("\nLabel prevalence:")
    for col in LABEL_COLS:
        pct = df[col].sum() / n * 100
        print(f"  {col}: {df[col].sum():,} ({pct:.2f}%)")
    print(f"  cross_border (|L|≥2): {df['cross_border'].sum():,} ({df['cross_border'].mean()*100:.2f}%)")
