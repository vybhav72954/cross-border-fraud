"""
IEEE-CIS external-validity fold (track F) — data prep + adapters.

WHY THIS IS DIFFERENT FROM THE BAKE-OFF.  The controlled INJECTION is
Sparkov-specific: it needs a merchant id (ring fan-in), lat/long (geo distance),
and per-card category history. IEEE-CIS is fully anonymized and has none of those
(no merchant id, no coordinates, a coarse 5-value ``ProductCD``, no clean
cardholder key), so the typologies CANNOT be re-injected and the bake-off cannot
be re-run. What DOES transfer is the *representational machinery* — the per-card
sequence SSMs (velocity / temporal) need only a timestamp, an amount, and a card
key. This module prepares IEEE-CIS so those exact extractors (``src.models.ssm``)
run on it unchanged, to test whether the representations carry signal on REAL
``isFraud`` (via the same LR-gate), not whether they recover a planted answer key.

HEURISTIC CARD KEY ("uid").  IEEE-CIS has no cardholder column. We use the
community-standard surrogate

    day  = TransactionDT / 86400
    uid  = card1 _ addr1 _ floor(day - D1)

where ``D1`` ≈ days-since-card-began, so ``day - D1`` is ~constant per physical
card and pins transactions of the same card together. This is a documented
approximation, not a ground-truth id; rows missing any component get a unique
singleton uid (they carry no sequence signal). State this caveat in any writeup.

Only ``train_transaction.csv`` is used — it is the only file with public
``isFraud`` labels (the competition test labels are private). We make our own
time-ordered split inside it.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

IEEE_DIR = Path("data/raw/ieee")
TRAIN_TXN = "train_transaction.csv"

# TransactionDT origin established by the Kaggle community (holidays/day-of-week
# line up at this anchor). Hour-of-day is robust to it regardless, as long as the
# anchor sits at midnight, so the temporal signature does not depend on this date.
REF_START = pd.Timestamp("2017-12-01")

# Columns we actually need: target, time, amount, product, the three uid
# ingredients, and Vesta's count features C1..C14 (engineered velocity-like
# tabular features — the stringent "already-counted" baseline the SSM must beat).
_C_COLS = [f"C{i}" for i in range(1, 15)]
_USECOLS = (["TransactionID", "isFraud", "TransactionDT", "TransactionAmt",
             "ProductCD", "card1", "addr1", "D1"] + _C_COLS)


def _intstr(col: pd.Series) -> pd.Series:
    """NA-safe integer-like string (NaN -> '<NA>', overwritten by the bad mask)."""
    return col.round().astype("Int64").astype(str)


def build_uid(df: pd.DataFrame) -> pd.Series:
    """Community-standard surrogate cardholder key (see module docstring).

    Rows missing ``card1``/``addr1``/``D1`` can't be grouped to a card, so they
    get a unique ``u<TransactionID>`` singleton uid instead of polluting a shared
    group with unrelated transactions.
    """
    day = df["TransactionDT"] / 86400.0
    anchor = np.floor(day - df["D1"])  # NaN where D1 missing
    uid = _intstr(df["card1"]) + "_" + _intstr(df["addr1"]) + "_" + \
        anchor.astype("Int64").astype(str)
    bad = df["card1"].isna() | df["addr1"].isna() | df["D1"].isna()
    return uid.where(~bad, "u" + df["TransactionID"].astype(str))


def load_ieee(raw_dir: Path = IEEE_DIR, nrows: int | None = None) -> pd.DataFrame:
    """Load ``train_transaction.csv`` and attach the SSM-adapter columns.

    Adds ``cc_num`` (uid), ``trans_date_trans_time`` (datetime from
    ``TransactionDT``), and ``amt`` (= ``TransactionAmt``) so the unchanged
    extractors in ``src.models.ssm`` run directly on the frame.
    """
    path = raw_dir / TRAIN_TXN
    if not path.exists():
        raise FileNotFoundError(
            f"{path} not found.\n"
            "Download IEEE-CIS (competition 'ieee-fraud-detection') and place\n"
            f"train_transaction.csv under {raw_dir}/ .\n"
            "  kaggle competitions download -c ieee-fraud-detection\n"
            "Only train_transaction.csv is required (it has the public isFraud)."
        )
    df = pd.read_csv(path, usecols=lambda c: c in _USECOLS, nrows=nrows)
    df["cc_num"] = build_uid(df)
    df["trans_date_trans_time"] = (
        REF_START + pd.to_timedelta(df["TransactionDT"], unit="s")
    )
    df["amt"] = df["TransactionAmt"].astype(float)
    return df


def time_split(df: pd.DataFrame, frac: float = 0.8) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Time-ordered holdout: earliest ``frac`` -> train, latest -> test.

    A realistic temporal split (no future leakage); per-card SSM states are
    computed within each split independently, exactly as the bake-off does
    (``card_hour_rarity(tr)`` / ``card_hour_rarity(te)`` are separate calls).
    """
    order = np.argsort(df["TransactionDT"].to_numpy(), kind="stable")
    cut = int(len(df) * frac)
    tr = df.iloc[order[:cut]].copy()
    te = df.iloc[order[cut:]].copy()
    return tr, te


def build_ieee_features(df: pd.DataFrame, include_counts: bool = True) -> pd.DataFrame:
    """Tabular design matrix from IEEE-CIS native columns (no intercept).

    ``include_counts`` toggles Vesta's C1..C14 — the engineered count features.
    M0 with counts is the stringent baseline (the velocity-like signal is already
    tabulated); M0 without is the lenient one. The gap between the SSM's lift over
    each tells whether the per-uid sequence adds anything beyond Vesta's counts.
    """
    out = pd.DataFrame(index=df.index)
    out["log_amt"] = np.log1p(df["TransactionAmt"].astype(float))

    dt = df["trans_date_trans_time"]
    hour = dt.dt.hour
    out["hour_sin"] = np.sin(2 * np.pi * hour / 24)
    out["hour_cos"] = np.cos(2 * np.pi * hour / 24)
    out["is_weekend"] = (dt.dt.dayofweek >= 5).astype(float)

    pcd = pd.get_dummies(df["ProductCD"], prefix="pcd", drop_first=True).astype(float)
    out = pd.concat([out, pcd], axis=1)

    if include_counts:
        for c in _C_COLS:
            if c in df.columns:
                out[f"log_{c}"] = np.log1p(df[c].fillna(0.0).clip(lower=0.0))
    return out.astype(float)
