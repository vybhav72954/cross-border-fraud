"""Base-dataset adapter contract (track G2) — on synthetic PaySim/BankSim-shaped
frames, so the suite needs no downloaded data.

Each adapter must (a) map roles to the right columns, (b) report the right
capability set, (c) synthesise a real timestamp from the integer ``step`` (PaySim
hourly, BankSim daily), (d) tidy dataset quirks (BankSim quotes its strings), and
(e) feed the SAME injection protocol so only the schema-supported typologies are
hosted.
"""
import numpy as np
import pandas as pd
import pytest

from src import inject
from src.adapters import adapt_paysim, adapt_banksim, ADAPTERS

SMALL = {"ring": 2, "velocity": 2, "temporal": 3, "category": 3, "geo": 3}


def paysim_raw() -> pd.DataFrame:
    """Minimal raw PaySim: hourly step, C-origins, M-destinations, no category/coords."""
    rng = np.random.default_rng(0)
    rows = []
    for _ in range(200):
        rows.append({
            "step": int(rng.integers(1, 96)),  # hours
            "type": rng.choice(["PAYMENT", "TRANSFER", "CASH_OUT"]),
            "amount": round(float(rng.uniform(1, 1000)), 2),
            "nameOrig": f"C{int(rng.integers(0, 15))}",
            "oldbalanceOrg": 0.0, "newbalanceOrig": 0.0,
            "nameDest": f"M{int(rng.integers(0, 8))}",
            "oldbalanceDest": 0.0, "newbalanceDest": 0.0,
            "isFraud": 0, "isFlaggedFraud": 0,
        })
    return pd.DataFrame(rows)


def banksim_raw() -> pd.DataFrame:
    """Minimal raw BankSim: daily step, every string field single-quoted, a
    category tied to each merchant."""
    rng = np.random.default_rng(1)
    cats = ["es_transportation", "es_food", "es_health", "es_tech"]
    rows = []
    for _ in range(200):
        m = int(rng.integers(0, 8))
        rows.append({
            "step": int(rng.integers(0, 30)),  # days
            "customer": f"'C{int(rng.integers(0, 15))}'",
            "age": "'4'", "gender": "'M'", "zipcodeOri": "'28007'",
            "merchant": f"'M{m}'", "zipMerchant": "'28007'",
            "category": f"'{cats[m % len(cats)]}'",
            "amount": round(float(rng.uniform(1, 500)), 2),
            "fraud": 0,
        })
    return pd.DataFrame(rows)


# ── PaySim ───────────────────────────────────────────────────────────────────

def test_paysim_schema_roles():
    _, s = adapt_paysim(paysim_raw())
    assert (s.entity, s.target, s.time, s.amount, s.label) == (
        "nameOrig", "nameDest", "ts", "amount", "isFraud")
    assert not s.has_category and not s.has_location
    assert s.supported_typologies() == ["ring", "velocity", "temporal"]


def test_paysim_timestamp_from_hourly_step():
    raw = paysim_raw()
    df, _ = adapt_paysim(raw)
    # midnight anchor -> hour-of-day is exactly step % 24 (temporal is meaningful)
    assert (df["ts"].dt.hour.to_numpy() == (df["step"].to_numpy() % 24)).all()
    assert "ts" not in raw.columns  # adapter must not mutate its input


# ── BankSim ──────────────────────────────────────────────────────────────────

def test_banksim_schema_roles():
    _, s = adapt_banksim(banksim_raw())
    assert (s.entity, s.target, s.category, s.label) == (
        "customer", "merchant", "category", "fraud")
    assert s.has_category and not s.has_location
    assert s.supported_typologies() == ["ring", "velocity", "temporal", "category"]


def test_banksim_strips_quotes():
    df, _ = adapt_banksim(banksim_raw())
    for col in ("customer", "merchant", "category"):
        assert not df[col].str.contains("'", regex=False).any()
    assert df["customer"].iloc[0].startswith("C")


def test_banksim_daily_step_is_hour_zero():
    """Daily step -> every row lands at hour 0 (the documented temporal-degeneracy
    that makes BankSim a ring/category base, not a timing base)."""
    df, _ = adapt_banksim(banksim_raw())
    assert (df["ts"].dt.hour == 0).all()


# ── adapter -> injection protocol ────────────────────────────────────────────

@pytest.mark.parametrize("adapt, raw, hosted, absent", [
    pytest.param(adapt_paysim, paysim_raw(), {"ring", "velocity", "temporal"},
                 {"category", "geo"}, id="paysim"),
    pytest.param(adapt_banksim, banksim_raw(),
                 {"ring", "velocity", "temporal", "category"}, {"geo"}, id="banksim"),
])
def test_adapter_build_hosts_only_supported(adapt, raw, hosted, absent):
    df, schema = adapt(raw)
    aug = inject.build_controlled_dataset(df, counts=SMALL, overlap={}, seed=0,
                                          schema=schema)
    dums = inject.typology_dummies(aug)
    assert {t for t in hosted if dums[t].sum() > 0} == hosted   # all hosted present
    assert all(dums[t].sum() == 0 for t in absent)              # none of the unsupported
    # answer-key roundtrip: injected rows carry >=1 typology, legit carry none
    inj_mask = aug[inject.TYPOLOGY_COL].fillna("") != ""
    assert (dums[inj_mask].sum(axis=1) >= 1).all()
    assert (dums[~inj_mask].sum(axis=1) == 0).all()


def test_adapter_registry():
    assert ADAPTERS == {"paysim": adapt_paysim, "banksim": adapt_banksim}
