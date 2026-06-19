"""Shared fixtures: a tiny legit-only Sparkov-schema frame for injection tests.

Small enough to run in milliseconds, structured enough that every injector has
something to bite on: 20 cards, 10 merchants over 6 categories, per-card hours
concentrated in the working day so the early-morning hours are genuinely rare.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

CATEGORIES = ["grocery", "gas", "shopping", "travel", "dining", "health"]


@pytest.fixture
def legit_df() -> pd.DataFrame:
    rng = np.random.default_rng(42)
    n_cards, n_merch = 20, 10
    cards = 1000 + np.arange(n_cards)
    merchants = [f"merch_{i}" for i in range(n_merch)]
    merch_cat = {m: CATEGORIES[i % len(CATEGORIES)] for i, m in enumerate(merchants)}
    merch_home = {m: (rng.uniform(30, 45), rng.uniform(-120, -75)) for m in merchants}
    card_home = {c: (rng.uniform(30, 45), rng.uniform(-120, -75)) for c in cards}

    rows = []
    t0 = pd.Timestamp("2019-01-01")
    for c in cards:
        for _ in range(60):
            m = merchants[int(rng.integers(0, n_merch))]
            # habitual hours: working day, so hours 0-5 stay rare/zero per card
            hour = int(rng.integers(9, 18))
            day = int(rng.integers(0, 300))
            ts = t0 + pd.Timedelta(days=day, hours=hour, minutes=int(rng.integers(0, 60)))
            lat, lon = card_home[c]
            mlat, mlon = merch_home[m]
            rows.append({
                "trans_date_trans_time": ts,
                "cc_num": int(c),
                "merchant": m,
                "category": merch_cat[m],
                "amt": round(float(rng.uniform(1, 400)), 2),
                "first": "A", "last": "B", "gender": "F",
                "street": "1 St", "city": "C", "state": "CA", "zip": "00000",
                "lat": lat, "long": lon, "city_pop": 10000,
                "job": "tester", "dob": pd.Timestamp("1990-01-01"),
                "trans_num": f"T{len(rows):07d}",
                "unix_time": int(ts.timestamp()),
                "merch_lat": mlat, "merch_long": mlon,
                "is_fraud": 0,
            })
    df = pd.DataFrame(rows)
    df["trans_date_trans_time"] = pd.to_datetime(df["trans_date_trans_time"])
    return df


SMALL_COUNTS = {"ring": 4, "velocity": 4, "temporal": 8, "category": 8, "geo": 8}
SMALL_OVERLAP = {("geo", "temporal"): 3, ("category", "temporal"): 3}
