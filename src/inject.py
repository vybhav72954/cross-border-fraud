"""
Synthetic fraud injection for the controlled multi-typology benchmark.

Static Sparkov fraud carries no usable ring / geographic / sequence structure
(see threshold diagnostic), so instead of *detecting* unknown fraud we build a
*controlled* dataset: take the legitimate transactions as background and inject
fraud with KNOWN typology signatures. Every injected row carries its
ground-truth typology, so detectors can be scored against an answer key.

Typologies (one injector each, with the representation that should catch it):
  ring      K distinct cards hit one merchant in a short window   (graph    -> GNN)
  velocity  one card, many transactions in a short window         (sequence -> Mamba)
  temporal  a transaction at one of the card's rarest hours       (sequence -> Mamba)
  category  a transaction in one of the card's rarest categories  (tabular  -> GLM)
  geo       a merchant placed implausibly far from the card's home (tabular  -> GLM)

All injectors sample identity/merchant fields from the *legit* rows only, so
they compose safely when chained. v1 injects one typology per event (no
overlap); the overlap knob that creates `cross_border` ground truth is next.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

TYPOLOGIES = ["ring", "velocity", "temporal", "category", "geo"]

TYPOLOGY_COL = "typology"   # "" for legit background, else the typology name
EVENT_COL = "inj_event"     # groups rows belonging to the same injected event

DEFAULT_COUNTS = {"ring": 300, "velocity": 800, "temporal": 4000,
                  "category": 4000, "geo": 4000}


def legit_background(df: pd.DataFrame) -> pd.DataFrame:
    """Legit-only base. Injected fraud becomes the sole source of is_fraud=1,
    so the typology answer key is exact."""
    base = df.loc[df["is_fraud"] == 0].copy()
    base[TYPOLOGY_COL] = ""
    base[EVENT_COL] = pd.NA
    return base


def _pools(base: pd.DataFrame):
    """Sampling pools drawn from legit rows only (safe under chaining):
    per-card identity template, per-merchant location/category, time span."""
    legit = base.loc[base[TYPOLOGY_COL] == ""]
    template = legit.drop_duplicates("cc_num", keep="first").set_index("cc_num")
    merch_loc = legit.drop_duplicates("merchant", keep="first").set_index("merchant")
    t_min = legit["trans_date_trans_time"].min()
    t_max = legit["trans_date_trans_time"].max()
    return legit, template, merch_loc, template.index.to_numpy(), merch_loc.index.to_numpy(), t_min, t_max


def _finalize(base: pd.DataFrame, rows: list[pd.Series]) -> pd.DataFrame:
    injected = pd.DataFrame(rows, columns=base.columns)
    return pd.concat([base, injected], ignore_index=True)


def _row(template: pd.DataFrame, c, t: pd.Timestamp, amt: float,
         typology: str, event: str, trans_num: str,
         merchant=None, merch_loc: pd.DataFrame | None = None,
         category=None, merch_lat=None, merch_long=None) -> pd.Series:
    """Build one injected fraud row from a card's real identity template."""
    row = template.loc[c].copy()
    row["cc_num"] = c
    if merchant is not None:
        row["merchant"] = merchant
        m = merch_loc.loc[merchant]
        row["category"] = m["category"] if category is None else category
        row["merch_lat"] = m["merch_lat"] if merch_lat is None else merch_lat
        row["merch_long"] = m["merch_long"] if merch_long is None else merch_long
    if category is not None:
        row["category"] = category
    if merch_lat is not None:
        row["merch_lat"] = merch_lat
    if merch_long is not None:
        row["merch_long"] = merch_long
    row["trans_date_trans_time"] = t
    row["unix_time"] = int(t.timestamp())
    row["amt"] = round(float(amt), 2)
    row["is_fraud"] = 1
    row[TYPOLOGY_COL] = typology
    row[EVENT_COL] = event
    row["trans_num"] = trans_num
    return row


def inject_ring(base: pd.DataFrame, n_rings: int, cards_per_ring: int = 5,
                window_hours: float = 2.0,
                rng: np.random.Generator | None = None) -> pd.DataFrame:
    """`cards_per_ring` distinct cards transacting at one shared merchant inside
    a `window_hours` window — the card->merchant fan-in a graph model should see
    and a tabular model cannot. Knobs: cards_per_ring, window_hours."""
    rng = rng if rng is not None else np.random.default_rng()
    _, template, merch_loc, cards, merchants, t_min, t_max = _pools(base)
    span_s = (t_max - t_min - pd.Timedelta(hours=window_hours)).total_seconds()

    rows = []
    for r in range(n_rings):
        m = rng.choice(merchants)
        ring_cards = rng.choice(cards, size=cards_per_ring, replace=False)
        t0 = t_min + pd.Timedelta(seconds=float(rng.uniform(0, span_s)))
        for c in ring_cards:
            t = t0 + pd.Timedelta(hours=float(rng.uniform(0, window_hours)))
            rows.append(_row(template, c, t, rng.uniform(200, 1000), "ring",
                             f"ring_{r:04d}", f"INJ_ring_{r:04d}_{c}",
                             merchant=m, merch_loc=merch_loc))
    return _finalize(base, rows)


def inject_velocity(base: pd.DataFrame, n_events: int, txn_per_burst: int = 5,
                    window_minutes: float = 20.0,
                    rng: np.random.Generator | None = None) -> pd.DataFrame:
    """One card firing `txn_per_burst` transactions inside `window_minutes` — a
    rapid-succession burst a sequence model should catch. Knobs:
    txn_per_burst, window_minutes."""
    rng = rng if rng is not None else np.random.default_rng()
    _, template, merch_loc, cards, merchants, t_min, t_max = _pools(base)
    span_s = (t_max - t_min - pd.Timedelta(minutes=window_minutes)).total_seconds()

    rows = []
    for e in range(n_events):
        c = rng.choice(cards)
        t0 = t_min + pd.Timedelta(seconds=float(rng.uniform(0, span_s)))
        for j in range(txn_per_burst):
            t = t0 + pd.Timedelta(minutes=float(rng.uniform(0, window_minutes)))
            rows.append(_row(template, c, t, rng.uniform(50, 500), "velocity",
                             f"velocity_{e:05d}", f"INJ_velocity_{e:05d}_{j}",
                             merchant=rng.choice(merchants), merch_loc=merch_loc))
    return _finalize(base, rows)


def inject_temporal(base: pd.DataFrame, n_events: int, rarest_k: int = 5,
                    rng: np.random.Generator | None = None) -> pd.DataFrame:
    """A transaction at one of the card's `rarest_k` least-used hours-of-day —
    out-of-distribution timing relative to the card's own history. Knob:
    rarest_k (smaller = rarer = stronger signal)."""
    rng = rng if rng is not None else np.random.default_rng()
    legit, template, merch_loc, cards, merchants, t_min, t_max = _pools(base)
    n_days = max((t_max - t_min).days, 1)
    hour_counts = (legit.assign(_h=legit["trans_date_trans_time"].dt.hour)
                   .groupby(["cc_num", "_h"]).size())
    all_hours = pd.RangeIndex(24)

    rows = []
    for e in range(n_events):
        c = rng.choice(cards)
        hc = hour_counts.loc[c].reindex(all_hours, fill_value=0)
        target_hour = int(rng.choice(hc.nsmallest(rarest_k).index.to_numpy()))
        day = t_min.normalize() + pd.Timedelta(days=int(rng.integers(0, n_days)))
        t = day + pd.Timedelta(hours=target_hour, minutes=int(rng.integers(0, 60)))
        rows.append(_row(template, c, t, rng.uniform(100, 800), "temporal",
                         f"temporal_{e:05d}", f"INJ_temporal_{e:05d}_{c}",
                         merchant=rng.choice(merchants), merch_loc=merch_loc))
    return _finalize(base, rows)


def inject_category(base: pd.DataFrame, n_events: int, rarest_k: int = 3,
                    rng: np.random.Generator | None = None) -> pd.DataFrame:
    """A transaction in one of the card's `rarest_k` least-used categories, at a
    real merchant of that category. Knob: rarest_k."""
    rng = rng if rng is not None else np.random.default_rng()
    legit, template, merch_loc, cards, merchants, t_min, t_max = _pools(base)
    span_s = (t_max - t_min).total_seconds()
    all_cats = pd.Index(sorted(legit["category"].unique()))
    cat_counts = legit.groupby(["cc_num", "category"]).size()
    merch_by_cat = {cat: merch_loc.index[merch_loc["category"] == cat].to_numpy()
                    for cat in all_cats}

    rows = []
    for e in range(n_events):
        c = rng.choice(cards)
        cc = cat_counts.loc[c].reindex(all_cats, fill_value=0)
        target_cat = rng.choice(cc.nsmallest(rarest_k).index.to_numpy())
        m = rng.choice(merch_by_cat[target_cat])
        t = t_min + pd.Timedelta(seconds=float(rng.uniform(0, span_s)))
        rows.append(_row(template, c, t, rng.uniform(100, 800), "category",
                         f"category_{e:05d}", f"INJ_category_{e:05d}_{c}",
                         merchant=m, merch_loc=merch_loc))
    return _finalize(base, rows)


def inject_geo(base: pd.DataFrame, n_events: int, min_offset_deg: float = 8.0,
               rng: np.random.Generator | None = None) -> pd.DataFrame:
    """A merchant placed `min_offset_deg`+ degrees (~900+ km) from the card's
    home — the impossible-distance signal that does not exist in raw Sparkov.
    Knob: min_offset_deg."""
    rng = rng if rng is not None else np.random.default_rng()
    legit, template, merch_loc, cards, merchants, t_min, t_max = _pools(base)
    span_s = (t_max - t_min).total_seconds()

    rows = []
    for e in range(n_events):
        c = rng.choice(cards)
        home = template.loc[c]
        ang = rng.uniform(0, 2 * np.pi)
        d = rng.uniform(min_offset_deg, min_offset_deg + 5)
        m = rng.choice(merchants)
        t = t_min + pd.Timedelta(seconds=float(rng.uniform(0, span_s)))
        rows.append(_row(template, c, t, rng.uniform(100, 800), "geo",
                         f"geo_{e:05d}", f"INJ_geo_{e:05d}_{c}",
                         merchant=m, merch_loc=merch_loc,
                         merch_lat=float(home["lat"] + d * np.cos(ang)),
                         merch_long=float(home["long"] + d * np.sin(ang))))
    return _finalize(base, rows)


def build_controlled_dataset(df: pd.DataFrame, counts: dict | None = None,
                             seed: int = 0) -> pd.DataFrame:
    """Legit background + all five injected typologies (one typology per event).
    Returns the augmented frame with `typology` / `inj_event` answer-key columns."""
    counts = counts or DEFAULT_COUNTS
    rng = np.random.default_rng(seed)
    aug = legit_background(df)
    aug = inject_ring(aug, counts["ring"], rng=rng)
    aug = inject_velocity(aug, counts["velocity"], rng=rng)
    aug = inject_temporal(aug, counts["temporal"], rng=rng)
    aug = inject_category(aug, counts["category"], rng=rng)
    aug = inject_geo(aug, counts["geo"], rng=rng)
    return aug
