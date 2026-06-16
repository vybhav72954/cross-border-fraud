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

Controlled-benchmark invariant: injected rows match the legit distribution on
EVERY axis except the intended signature, so a detector can only succeed via the
real signal. Concretely: `amt` and the event timestamp are sampled from the
legit pools (no fraud-vs-legit or per-typology giveaway); merchant coords use
the legit home->merchant offset distribution (except `geo`, placed far); time is
a real legit timestamp (except `temporal`, whose hour is deliberately rare).

Overlap: `inject_overlap` stamps TWO compatible signatures on one event, so a
fraud carries >=2 typologies — the ground-truth source of `cross_border`. The
`typology` column holds a "+"-joined, sorted tag; `typology_dummies` /
`is_cross_border` read it back. Injectors sample from legit rows only, so they
compose safely when chained.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

TYPOLOGIES = ["ring", "velocity", "temporal", "category", "geo"]

TYPOLOGY_COL = "typology"   # "" for legit, else a "+"-joined sorted tag
EVENT_COL = "inj_event"     # groups rows belonging to the same injected event

DEFAULT_COUNTS = {"ring": 300, "velocity": 800, "temporal": 4000,
                  "category": 4000, "geo": 4000}

# Overlap events (counts are *events*; ring/velocity expand to multiple rows).
DEFAULT_OVERLAP = {
    ("geo", "temporal"): 800,
    ("category", "temporal"): 800,
    ("velocity", "geo"): 600,
    ("ring", "geo"): 200,
}


def legit_background(df: pd.DataFrame) -> pd.DataFrame:
    """Legit-only base. Injected fraud becomes the sole source of is_fraud=1,
    so the typology answer key is exact."""
    base = df.loc[df["is_fraud"] == 0].copy()
    base[TYPOLOGY_COL] = ""
    base[EVENT_COL] = pd.NA
    return base


# ── sampling pools & per-signature helpers ──────────────────────────────────

def _pools(base: pd.DataFrame):
    """Identity/merchant/amount/time pools drawn from legit rows only
    (safe under chaining)."""
    legit = base.loc[base[TYPOLOGY_COL] == ""]
    template = legit.drop_duplicates("cc_num", keep="first").set_index("cc_num")
    merch_loc = legit.drop_duplicates("merchant", keep="first").set_index("merchant")
    return (legit, template, merch_loc, template.index.to_numpy(),
            merch_loc.index.to_numpy(),
            legit["amt"].to_numpy(),
            legit["trans_date_trans_time"].to_numpy())


def _offset_pool(legit: pd.DataFrame):
    """Real legit home->merchant coordinate offsets, to reproduce Sparkov's
    near-home merchant placement on injected (non-geo) rows."""
    return ((legit["merch_lat"] - legit["lat"]).to_numpy(),
            (legit["merch_long"] - legit["long"]).to_numpy())


def _near_loc(home, off_lat, off_long, rng):
    k = int(rng.integers(0, off_lat.shape[0]))
    return float(home["lat"] + off_lat[k]), float(home["long"] + off_long[k])


def _far_loc(home, min_offset_deg, rng):
    ang = rng.uniform(0, 2 * np.pi)
    d = rng.uniform(min_offset_deg, min_offset_deg + 5)
    return float(home["lat"] + d * np.cos(ang)), float(home["long"] + d * np.sin(ang))


def _rand_amt(amt_pool, rng) -> float:
    return float(amt_pool[int(rng.integers(0, amt_pool.shape[0]))])


def _rand_time(time_pool, rng) -> pd.Timestamp:
    return pd.Timestamp(time_pool[int(rng.integers(0, time_pool.shape[0]))])


def _hour_counts(legit: pd.DataFrame) -> pd.Series:
    return (legit.assign(_h=legit["trans_date_trans_time"].dt.hour)
            .groupby(["cc_num", "_h"]).size())


def _rare_hour(hour_counts, c, all_hours, k, rng) -> int:
    hc = hour_counts.loc[c].reindex(all_hours, fill_value=0)
    return int(rng.choice(hc.nsmallest(k).index.to_numpy()))


def _rare_category(cat_counts, c, all_cats, k, rng):
    cc = cat_counts.loc[c].reindex(all_cats, fill_value=0)
    return rng.choice(cc.nsmallest(k).index.to_numpy())


def _rare_for_card_common_globally(cat_counts, global_freq, c, all_cats, k, top_g, rng):
    """A category rare FOR THIS CARD but among the ``top_g`` globally most common.

    The plain rare-category signature degenerates because rare-for-card is almost
    always rare-globally too, so a global one-hot dummy already flags it. Picking
    a category the card barely uses yet everyone else uses heavily makes the
    anomaly purely card-relative -- invisible to a global dummy, recoverable only
    by a card-conditioned model. Falls back to plain rarest if a card has no rare
    common category."""
    cc = cat_counts.loc[c].reindex(all_cats, fill_value=0)
    common = global_freq.nlargest(top_g).index
    cand = cc.loc[cc.index.isin(common)].nsmallest(k)
    pool = cand.index.to_numpy() if len(cand) else cc.nsmallest(k).index.to_numpy()
    return rng.choice(pool)


def _row(template, c, t, amt, typology, event, trans_num, merchant, merch_loc,
         merch_lat, merch_long, category=None) -> pd.Series:
    """Build one injected fraud row from card c's real identity template.
    Coordinates are supplied by the caller; category defaults to the merchant's."""
    row = template.loc[c].copy()
    row["cc_num"] = c
    row["merchant"] = merchant
    row["category"] = merch_loc.loc[merchant]["category"] if category is None else category
    row["merch_lat"] = merch_lat
    row["merch_long"] = merch_long
    row["trans_date_trans_time"] = t
    row["unix_time"] = int(t.timestamp())
    row["amt"] = round(float(amt), 2)
    row["is_fraud"] = 1
    row[TYPOLOGY_COL] = typology
    row[EVENT_COL] = event
    row["trans_num"] = trans_num
    return row


def _finalize(base: pd.DataFrame, rows: list[pd.Series]) -> pd.DataFrame:
    injected = pd.DataFrame(rows, columns=base.columns)
    return pd.concat([base, injected], ignore_index=True)


# ── single-typology injectors ───────────────────────────────────────────────

def inject_ring(base, n_rings, cards_per_ring=5, window_hours=2.0, rng=None):
    """`cards_per_ring` distinct cards at one shared merchant inside
    `window_hours` — the card->merchant fan-in a graph model should see."""
    rng = rng if rng is not None else np.random.default_rng()
    legit, template, merch_loc, cards, merchants, amt_pool, time_pool = _pools(base)
    off_lat, off_long = _offset_pool(legit)
    rows = []
    for r in range(n_rings):
        m = rng.choice(merchants)
        t0 = _rand_time(time_pool, rng)
        for c in rng.choice(cards, size=cards_per_ring, replace=False):
            home = template.loc[c]
            lat, lon = _near_loc(home, off_lat, off_long, rng)
            t = t0 + pd.Timedelta(hours=float(rng.uniform(0, window_hours)))
            rows.append(_row(template, c, t, _rand_amt(amt_pool, rng), "ring",
                             f"ring_{r:04d}", f"INJ_ring_{r:04d}_{c}", m, merch_loc, lat, lon))
    return _finalize(base, rows)


def inject_velocity(base, n_events, txn_per_burst=5, window_minutes=20.0, rng=None):
    """One card firing `txn_per_burst` transactions inside `window_minutes`."""
    rng = rng if rng is not None else np.random.default_rng()
    legit, template, merch_loc, cards, merchants, amt_pool, time_pool = _pools(base)
    off_lat, off_long = _offset_pool(legit)
    rows = []
    for e in range(n_events):
        c = rng.choice(cards)
        home = template.loc[c]
        t0 = _rand_time(time_pool, rng)
        for j in range(txn_per_burst):
            lat, lon = _near_loc(home, off_lat, off_long, rng)
            t = t0 + pd.Timedelta(minutes=float(rng.uniform(0, window_minutes)))
            rows.append(_row(template, c, t, _rand_amt(amt_pool, rng), "velocity",
                             f"velocity_{e:05d}", f"INJ_velocity_{e:05d}_{j}",
                             rng.choice(merchants), merch_loc, lat, lon))
    return _finalize(base, rows)


def inject_temporal(base, n_events, rarest_k=5, rng=None):
    """A transaction at one of the card's `rarest_k` least-used hours-of-day
    (date sampled from legit, only the hour is anomalous)."""
    rng = rng if rng is not None else np.random.default_rng()
    legit, template, merch_loc, cards, merchants, amt_pool, time_pool = _pools(base)
    off_lat, off_long = _offset_pool(legit)
    hour_counts = _hour_counts(legit)
    all_hours = pd.RangeIndex(24)
    rows = []
    for e in range(n_events):
        c = rng.choice(cards)
        home = template.loc[c]
        lat, lon = _near_loc(home, off_lat, off_long, rng)
        h = _rare_hour(hour_counts, c, all_hours, rarest_k, rng)
        t = _rand_time(time_pool, rng).normalize() + pd.Timedelta(hours=h, minutes=int(rng.integers(0, 60)))
        rows.append(_row(template, c, t, _rand_amt(amt_pool, rng), "temporal",
                         f"temporal_{e:05d}", f"INJ_temporal_{e:05d}_{c}",
                         rng.choice(merchants), merch_loc, lat, lon))
    return _finalize(base, rows)


def inject_category(base, n_events, rarest_k=3, globally_common_only=False,
                    global_common_k=5, rng=None):
    """A transaction in one of the card's `rarest_k` least-used categories.

    With ``globally_common_only`` the rare-for-card category is additionally
    constrained to the ``global_common_k`` most common categories overall, so the
    signal is purely card-relative (a global one-hot dummy can't catch it). See
    ``_rare_for_card_common_globally``."""
    rng = rng if rng is not None else np.random.default_rng()
    legit, template, merch_loc, cards, merchants, amt_pool, time_pool = _pools(base)
    off_lat, off_long = _offset_pool(legit)
    all_cats = pd.Index(sorted(legit["category"].unique()))
    cat_counts = legit.groupby(["cc_num", "category"]).size()
    global_freq = legit.groupby("category").size() if globally_common_only else None
    merch_by_cat = {cat: merch_loc.index[merch_loc["category"] == cat].to_numpy()
                    for cat in all_cats}
    rows = []
    for e in range(n_events):
        c = rng.choice(cards)
        home = template.loc[c]
        lat, lon = _near_loc(home, off_lat, off_long, rng)
        cat = (_rare_for_card_common_globally(cat_counts, global_freq, c, all_cats,
                                              rarest_k, global_common_k, rng)
               if globally_common_only
               else _rare_category(cat_counts, c, all_cats, rarest_k, rng))
        t = _rand_time(time_pool, rng)
        rows.append(_row(template, c, t, _rand_amt(amt_pool, rng), "category",
                         f"category_{e:05d}", f"INJ_category_{e:05d}_{c}",
                         rng.choice(merch_by_cat[cat]), merch_loc, lat, lon, category=cat))
    return _finalize(base, rows)


def inject_geo(base, n_events, min_offset_deg=8.0, rng=None):
    """A merchant placed `min_offset_deg`+ degrees (~900+ km) from the card's home."""
    rng = rng if rng is not None else np.random.default_rng()
    legit, template, merch_loc, cards, merchants, amt_pool, time_pool = _pools(base)
    rows = []
    for e in range(n_events):
        c = rng.choice(cards)
        home = template.loc[c]
        lat, lon = _far_loc(home, min_offset_deg, rng)
        t = _rand_time(time_pool, rng)
        rows.append(_row(template, c, t, _rand_amt(amt_pool, rng), "geo",
                         f"geo_{e:05d}", f"INJ_geo_{e:05d}_{c}",
                         rng.choice(merchants), merch_loc, lat, lon))
    return _finalize(base, rows)


# ── overlap injector (cross_border ground truth) ────────────────────────────

def inject_overlap(base, combo_counts=None, *, cards_per_ring=5, ring_window_hours=2.0,
                   txn_per_burst=5, velocity_window_minutes=20.0,
                   temporal_rarest_k=5, category_rarest_k=3, geo_min_offset_deg=8.0,
                   rng=None):
    """Inject events carrying TWO compatible typology signatures each, producing
    `cross_border` ground truth. `combo_counts` maps a typology pair to a number
    of events. A shape (ring/velocity, else single txn) + per-row modifiers
    (geo/temporal/category)."""
    rng = rng if rng is not None else np.random.default_rng()
    combo_counts = combo_counts if combo_counts is not None else DEFAULT_OVERLAP
    legit, template, merch_loc, cards, merchants, amt_pool, time_pool = _pools(base)
    off_lat, off_long = _offset_pool(legit)
    hour_counts = _hour_counts(legit)
    all_hours = pd.RangeIndex(24)
    all_cats = pd.Index(sorted(legit["category"].unique()))
    cat_counts = legit.groupby(["cc_num", "category"]).size()
    merch_by_cat = {cat: merch_loc.index[merch_loc["category"] == cat].to_numpy()
                    for cat in all_cats}

    def pick_time(c, mods):
        if "temporal" in mods:
            h = _rare_hour(hour_counts, c, all_hours, temporal_rarest_k, rng)
            return _rand_time(time_pool, rng).normalize() + pd.Timedelta(
                hours=h, minutes=int(rng.integers(0, 60)))
        return _rand_time(time_pool, rng)

    def pick_merchant(c, mods):
        if "category" in mods:
            cat = _rare_category(cat_counts, c, all_cats, category_rarest_k, rng)
            return rng.choice(merch_by_cat[cat]), cat
        return rng.choice(merchants), None

    def pick_loc(home, mods):
        if "geo" in mods:
            return _far_loc(home, geo_min_offset_deg, rng)
        return _near_loc(home, off_lat, off_long, rng)

    rows, eid = [], 0
    for combo, n in combo_counts.items():
        combo = tuple(sorted(combo))
        tag = "+".join(combo)
        mods = set(combo)
        for _ in range(n):
            event = f"{tag}_{eid:05d}"
            eid += 1
            if "ring" in mods:
                m = rng.choice(merchants)
                t0 = _rand_time(time_pool, rng)
                for c in rng.choice(cards, size=cards_per_ring, replace=False):
                    home = template.loc[c]
                    lat, lon = pick_loc(home, mods)
                    t = t0 + pd.Timedelta(hours=float(rng.uniform(0, ring_window_hours)))
                    rows.append(_row(template, c, t, _rand_amt(amt_pool, rng), tag,
                                     event, f"INJ_{event}_{c}", m, merch_loc, lat, lon))
            elif "velocity" in mods:
                c = rng.choice(cards)
                home = template.loc[c]
                t0 = pick_time(c, mods)
                for j in range(txn_per_burst):
                    m, cat = pick_merchant(c, mods)
                    lat, lon = pick_loc(home, mods)
                    t = t0 + pd.Timedelta(minutes=float(rng.uniform(0, velocity_window_minutes)))
                    rows.append(_row(template, c, t, _rand_amt(amt_pool, rng), tag,
                                     event, f"INJ_{event}_{j}", m, merch_loc, lat, lon, category=cat))
            else:
                c = rng.choice(cards)
                home = template.loc[c]
                t = pick_time(c, mods)
                m, cat = pick_merchant(c, mods)
                lat, lon = pick_loc(home, mods)
                rows.append(_row(template, c, t, _rand_amt(amt_pool, rng), tag,
                                 event, f"INJ_{event}_{c}", m, merch_loc, lat, lon, category=cat))
    return _finalize(base, rows)


# ── orchestrator & answer-key readers ───────────────────────────────────────

def build_controlled_dataset(df, counts=None, overlap=None, seed=0):
    """Legit background + single-typology injections + overlap events.
    Pass overlap={} to disable overlap."""
    counts = counts if counts is not None else DEFAULT_COUNTS
    overlap = overlap if overlap is not None else DEFAULT_OVERLAP
    rng = np.random.default_rng(seed)
    aug = legit_background(df)
    aug = inject_ring(aug, counts["ring"], rng=rng)
    aug = inject_velocity(aug, counts["velocity"], rng=rng)
    aug = inject_temporal(aug, counts["temporal"], rng=rng)
    aug = inject_category(aug, counts["category"], rng=rng)
    aug = inject_geo(aug, counts["geo"], rng=rng)
    if overlap:
        aug = inject_overlap(aug, overlap, rng=rng)
    return aug


def typology_dummies(df: pd.DataFrame) -> pd.DataFrame:
    """Expand the `typology` answer key into one 0/1 column per typology."""
    split = df[TYPOLOGY_COL].fillna("").str.split("+")
    return pd.DataFrame({t: split.apply(lambda lst: int(t in lst)) for t in TYPOLOGIES},
                        index=df.index)


def is_cross_border(df: pd.DataFrame) -> pd.Series:
    """True where a fraud carries >=2 typologies (an overlap event)."""
    return df[TYPOLOGY_COL].fillna("").str.contains("+", regex=False)
