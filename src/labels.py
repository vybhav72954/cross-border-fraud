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
from tqdm.auto import tqdm

# ── Thresholds ────────────────────────────────────────────────────────────────
# Tuned to the Sparkov train distribution (see threshold diagnostic, 2026-06-10);
# sensitivity ±20% reported separately. Rationale per label noted inline.
VELOCITY_WINDOW_MIN: int = 60       # 30→60: (3 txns/30min) caught 7% of fraud; 60min catches 17%
VELOCITY_MIN_TXN: int = 3
GEO_DISTANCE_KM: float = 120.0      # 500→120 (~p95): max home-merchant dist is 152 km on Sparkov
RING_WINDOW_HOURS: int = 72         # 24→72: (3 cards/±24h) caught 0.2% of fraud; ±72h catches 3.1%
RING_MIN_CARDS: int = 3
TEMPORAL_IQR_MULTIPLIER: float = 1.0  # 1.5→1.0: k=1.5 fired 0.03%; k=1.0 fires 1.1% at 4.4x fraud-lift
CATEGORY_LOOKBACK_DAYS: int = 30
CATEGORY_RARITY_THRESHOLD: float = 0.0  # fire when category's window share <= this; 0.0 = unseen
COLD_START_MIN_DAYS: int = 7

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
    """Category anomaly: the current category is rare in the cardholder's prior
    CATEGORY_LOOKBACK_DAYS of activity — its share of that window is
    <= CATEGORY_RARITY_THRESHOLD (default 0.0 → category unseen in the window).
    Not gated. A transaction with no prior history in the window does not fire,
    since there is no basis to judge rarity.

    (Replaces the original ``!= modal category`` rule, which fired on ~89% of
    transactions — see threshold diagnostic.)"""
    cats_arr = np.array(sorted(df["category"].unique()))
    cidx = {c: i for i, c in enumerate(cats_arr)}
    window = f"{CATEGORY_LOOKBACK_DAYS}D"
    df_s = df.sort_values(["cc_num", "trans_dt"])
    out = pd.Series(0, index=df.index, dtype=np.int8)

    for _, g in tqdm(df_s.groupby("cc_num", sort=False), desc="  L_C per card",
                     unit="card", ncols=80):
        # Category counts in the trailing window [t - lookback, t), per row;
        # closed="left" excludes the current row → matches the strict `< t` cutoff.
        onehot = (pd.get_dummies(g["category"])
                  .reindex(columns=cats_arr, fill_value=0).astype(np.int32))
        onehot.index = g["trans_dt"].to_numpy()
        counts = onehot.rolling(window, closed="left").sum().fillna(0).to_numpy()
        total = counts.sum(axis=1)
        idx = np.array([cidx[c] for c in g["category"].to_numpy()])
        cur = counts[np.arange(len(g)), idx]
        with np.errstate(invalid="ignore", divide="ignore"):
            share = np.where(total > 0, cur / total, np.nan)

        # Fire only with prior history (total > 0) and a rare/unseen category.
        flag = (total > 0) & (share <= CATEGORY_RARITY_THRESHOLD)
        out.loc[g.index] = flag.astype(np.int8)

    return out


def _derive_L_R(df: pd.DataFrame) -> pd.Series:
    """Ring membership: ≥RING_MIN_CARDS distinct fraudulent cc_nums at same merchant
    in ±RING_WINDOW_HOURS. Gated on is_fraud=1."""
    fraud = df[df["is_fraud"] == 1][["trans_dt", "cc_num", "merchant"]].copy()
    if fraud.empty:
        return pd.Series(0, index=df.index, dtype=np.int8)

    fraud = fraud.sort_values("trans_dt").reset_index(names="orig_idx")
    window = pd.Timedelta(hours=RING_WINDOW_HOURS)

    ring_orig_idxs: set[int] = set()
    merchant_groups = list(fraud.groupby("merchant", sort=False))

    for _, mg in tqdm(merchant_groups, desc="  L_R per merchant", unit="merch", ncols=80):
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
    hour = df["trans_dt"].dt.hour

    gq1, gq3 = df["trans_dt"].dt.hour.quantile([0.25, 0.75])
    giqr = gq3 - gq1
    global_lo = gq1 - TEMPORAL_IQR_MULTIPLIER * giqr
    global_hi = gq3 + TEMPORAL_IQR_MULTIPLIER * giqr

    by_card = hour.groupby(df["cc_num"])
    q1 = df["cc_num"].map(by_card.quantile(0.25))
    q3 = df["cc_num"].map(by_card.quantile(0.75))
    iqr = q3 - q1
    lo = q1 - TEMPORAL_IQR_MULTIPLIER * iqr
    hi = q3 + TEMPORAL_IQR_MULTIPLIER * iqr

    span = df.groupby("cc_num")["trans_dt"].agg(["min", "max", "size"])
    cold = (span["max"] - span["min"]).dt.days < COLD_START_MIN_DAYS
    use_global = df["cc_num"].map(cold | (span["size"] < 4))
    lo = lo.where(~use_global, global_lo)
    hi = hi.where(~use_global, global_hi)

    return ((hour < lo) | (hour > hi)).astype(np.int8)


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

    steps = [
        ("L_V", "Velocity burst  ", _derive_L_V),
        ("L_G", "Geographic anom.", _derive_L_G),
        ("L_C", "Category anomaly", _derive_L_C),
        ("L_R", "Ring membership ", _derive_L_R),
        ("L_T", "Temporal anomaly", _derive_L_T),
    ]

    for col, desc, fn in tqdm(steps, desc="Deriving labels", unit="label", ncols=80):
        tqdm.write(f"\n[{desc}]")
        out[col] = fn(out)

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
