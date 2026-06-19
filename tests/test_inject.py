"""Injection invariants — the controlled-benchmark answer key must be exact and
leak-free, and each signature must actually be present in its injected rows.
"""
import numpy as np
import pandas as pd
import pytest
from conftest import SMALL_COUNTS, SMALL_OVERLAP

from src import inject


def _injected(df):
    return df[df[inject.TYPOLOGY_COL].fillna("") != ""]


def test_determinism(legit_df):
    """Same seed -> byte-identical injection (deterministic answer key)."""
    a = inject.build_controlled_dataset(legit_df, SMALL_COUNTS, SMALL_OVERLAP, seed=0)
    b = inject.build_controlled_dataset(legit_df, SMALL_COUNTS, SMALL_OVERLAP, seed=0)
    cols = ["cc_num", "merchant", "category", "amt", "merch_lat", "merch_long",
            inject.TYPOLOGY_COL, inject.EVENT_COL, "trans_date_trans_time"]
    pd.testing.assert_frame_equal(a[cols], b[cols])


def test_different_seeds_differ(legit_df):
    a = inject.build_controlled_dataset(legit_df, SMALL_COUNTS, SMALL_OVERLAP, seed=0)
    b = inject.build_controlled_dataset(legit_df, SMALL_COUNTS, SMALL_OVERLAP, seed=1)
    assert not _injected(a)["merch_lat"].reset_index(drop=True).equals(
        _injected(b)["merch_lat"].reset_index(drop=True))


def test_legit_background_is_legit_only(legit_df):
    base = inject.legit_background(legit_df)
    assert (base["is_fraud"] == 0).all()
    assert (base[inject.TYPOLOGY_COL] == "").all()


def test_ring_is_a_fanin(legit_df):
    """Each ring event = cards_per_ring DISTINCT cards at ONE merchant in-window."""
    base = inject.legit_background(legit_df)
    out = inject.inject_ring(base, n_rings=3, cards_per_ring=5, window_hours=2.0,
                             rng=np.random.default_rng(0))
    rings = out[out[inject.TYPOLOGY_COL] == "ring"]
    for _, g in rings.groupby(inject.EVENT_COL):
        assert g["merchant"].nunique() == 1
        assert g["cc_num"].nunique() == 5
        span = g["trans_date_trans_time"].max() - g["trans_date_trans_time"].min()
        assert span <= pd.Timedelta(hours=2.0)


def test_velocity_is_a_burst(legit_df):
    base = inject.legit_background(legit_df)
    out = inject.inject_velocity(base, n_events=3, txn_per_burst=5,
                                 window_minutes=20.0, rng=np.random.default_rng(0))
    bursts = out[out[inject.TYPOLOGY_COL] == "velocity"]
    for _, g in bursts.groupby(inject.EVENT_COL):
        assert g["cc_num"].nunique() == 1
        assert len(g) == 5
        span = g["trans_date_trans_time"].max() - g["trans_date_trans_time"].min()
        assert span <= pd.Timedelta(minutes=20.0)


def test_temporal_hour_is_rare_for_card(legit_df):
    """Injected temporal hour must be among that card's rarest_k hours-of-day."""
    base = inject.legit_background(legit_df)
    k = 5
    out = inject.inject_temporal(base, n_events=20, rarest_k=k,
                                 rng=np.random.default_rng(0))
    legit = base
    hour_counts = (legit.assign(_h=legit["trans_date_trans_time"].dt.hour)
                   .groupby(["cc_num", "_h"]).size())
    inj = out[out[inject.TYPOLOGY_COL] == "temporal"]
    for _, r in inj.iterrows():
        c = r["cc_num"]
        rare = (hour_counts.loc[c].reindex(range(24), fill_value=0)
                .nsmallest(k).index.to_numpy())
        assert r["trans_date_trans_time"].hour in set(rare)


def test_geo_is_far_from_home(legit_df):
    base = inject.legit_background(legit_df)
    out = inject.inject_geo(base, n_events=20, min_offset_deg=8.0,
                            rng=np.random.default_rng(0))
    inj = out[out[inject.TYPOLOGY_COL] == "geo"]
    d = np.hypot(inj["merch_lat"] - inj["lat"], inj["merch_long"] - inj["long"])
    assert (d >= 8.0).all()


def test_leakfree_amt_drawn_from_legit_pool(legit_df):
    """amt on injected rows must come from the legit amt pool — no per-typology
    amount giveaway (the controlled-benchmark invariant)."""
    base = inject.legit_background(legit_df)
    pool = set(np.round(base["amt"].to_numpy(), 2))
    out = inject.build_controlled_dataset(legit_df, SMALL_COUNTS, {}, seed=0)
    inj = _injected(out)
    assert set(np.round(inj["amt"].to_numpy(), 2)).issubset(pool)


def test_answer_key_roundtrip(legit_df):
    """typology_dummies / is_cross_border must recover what was stamped."""
    out = inject.build_controlled_dataset(legit_df, SMALL_COUNTS, SMALL_OVERLAP, seed=0)
    dummies = inject.typology_dummies(out)
    # legit rows carry no typology; every injected row carries >=1
    inj_mask = out[inject.TYPOLOGY_COL].fillna("") != ""
    assert (dummies[inj_mask].sum(axis=1) >= 1).all()
    assert (dummies[~inj_mask].sum(axis=1) == 0).all()
    # cross_border <=> >=2 typologies on the row
    cb = inject.is_cross_border(out)
    assert (cb == (dummies.sum(axis=1) >= 2)).all()


def test_overlap_carries_two_signatures(legit_df):
    out = inject.build_controlled_dataset(legit_df, SMALL_COUNTS, SMALL_OVERLAP, seed=0)
    overlaps = out[inject.is_cross_border(out)]
    assert len(overlaps) > 0
    tags = overlaps[inject.TYPOLOGY_COL].unique()
    assert all("+" in t for t in tags)
    dummies = inject.typology_dummies(overlaps)
    assert (dummies.sum(axis=1) == 2).all()


@pytest.mark.parametrize("typ", inject.TYPOLOGIES)
def test_every_typology_present(legit_df, typ):
    out = inject.build_controlled_dataset(legit_df, SMALL_COUNTS, {}, seed=0)
    assert inject.typology_dummies(out)[typ].sum() > 0
