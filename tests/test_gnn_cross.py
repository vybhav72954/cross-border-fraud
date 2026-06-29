"""Cross-dataset GNN generalization (track G3): the ring extractors must thread
an arbitrary ``Schema`` end-to-end on a NON-Sparkov frame (custom column names,
no demographics, no coordinates), recovering the windowed merchant fan-in.

Guards the schema-threading of ``src.models.gnn`` and the demographic-feature
fallback in ``_card_node_table`` (``dob``/``city_pop`` absent on other bases).
Synthetic frame, so the suite needs no downloaded data.
"""
import numpy as np
import pandas as pd
import pytest

from src import inject
from src.features import cross_features
from src.models.gnn import RingSAGE, merchant_window_features, _card_node_table
from src.schema import Schema

# A non-Sparkov schema: different column names, no category, no location quad.
SCHEMA = Schema(entity="cust", target="shop", time="ts", amount="value", label="flag")


def synthetic_frame(seed: int = 0) -> pd.DataFrame:
    """Legit background + injected rings on a non-Sparkov-shaped frame."""
    rng = np.random.default_rng(seed)
    n = 700
    base = pd.DataFrame({
        "cust": [f"C{int(rng.integers(0, 25))}" for _ in range(n)],
        "shop": [f"S{int(rng.integers(0, 18))}" for _ in range(n)],
        "ts": pd.Timestamp("2020-01-01") + pd.to_timedelta(rng.uniform(0, 60 * 24, n), unit="h"),
        "value": np.round(rng.uniform(1, 500, n), 2),
        "flag": 0,
    })
    rng2 = np.random.default_rng(seed)
    bg = inject.legit_background(base, SCHEMA)
    return inject.inject_ring(bg, n_rings=12, cards_per_ring=5, window_hours=2.0,
                              schema=SCHEMA, rng=rng2).reset_index(drop=True)


def test_merchant_window_features_recovers_fanin_non_sparkov():
    df = synthetic_frame()
    typ = df[inject.TYPOLOGY_COL].fillna("").to_numpy()
    fan = merchant_window_features(df, window_hours=2.0, schema=SCHEMA,
                                   show_progress=False)["merch_win_cards"].to_numpy()
    ring_fan = fan[typ == "ring"]
    legit_fan = fan[typ == ""]
    # a 5-card ring lifts the windowed distinct-card count well above legit noise
    assert ring_fan.mean() > 4.0
    assert legit_fan.mean() < ring_fan.mean()
    assert np.median(ring_fan) >= 5


def test_card_node_table_fallback_when_no_demographics():
    df = synthetic_frame()
    feat, idx = _card_node_table(df, SCHEMA)            # no dob / city_pop -> fallback
    assert feat.shape == (len(idx), 2)
    assert np.isfinite(feat).all()
    assert set(idx) == set(df["cust"].unique())


def test_cross_features_schema_driven_non_sparkov():
    df = synthetic_frame()
    X = cross_features(df, SCHEMA)
    assert list(X.columns) == ["hour_sin", "hour_cos", "log_amt", "vel_1h", "is_weekend"]
    assert len(X) == len(df)
    assert np.isfinite(X.to_numpy()).all()
    assert (X["vel_1h"] >= 1).all()  # a row always counts itself


def test_ringsage_fit_score_non_sparkov():
    pytest.importorskip("torch_geometric")
    df = synthetic_frame()
    ring = inject.typology_dummies(df)["ring"].to_numpy()
    assert ring.sum() > 0
    sage = RingSAGE(window_hours=2.0, epochs=3, n_legit=5_000, seed=0,
                    schema=SCHEMA).fit(df, ring)
    score = sage.score(df)
    assert score.shape == (len(df),)
    assert np.isfinite(score).all()
    assert ((score >= 0) & (score <= 1)).all()  # sigmoid output
