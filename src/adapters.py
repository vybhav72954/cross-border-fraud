"""
Base-dataset adapters for the representation-recovery benchmark (track G2).

Each adapter is a thin ``adapt_x(df) -> (df, Schema)``: it normalises a raw public
transaction dataset just enough that the SAME injection protocol (``src.inject``)
and the SAME feature extractors (``src.models.ssm`` / ``src.models.sequence``) run
on it unchanged, and it returns the ``Schema`` describing which column plays which
role. Sparkov needs no adapter (it IS the default ``SPARKOV`` schema); these cover
the additional bases.

The only real work an adapter does is (a) synthesise a real timestamp from a
coarse integer ``step`` so ``.dt.hour`` / ``Timedelta`` logic is unchanged, and
(b) tidy up dataset quirks (BankSim quotes every string field). Capability is then
purely a function of which columns exist — see ``Schema.supported_typologies``:

    dataset   entity      target     time(from step)  amount  label    category  geo
    Sparkov   cc_num      merchant   trans_..._time   amt     is_fraud  yes       yes
    PaySim    nameOrig    nameDest   step (HOURLY)    amount  isFraud   no        no
    BankSim   customer    merchant   step (DAILY)     amount  fraud     yes       no

CAVEATS (documented, not bugs):
  * Neither PaySim nor BankSim has coordinates -> geo is un-hostable (Sparkov-only).
  * BankSim ``step`` is DAILY, so every legit row lands at hour 0: the temporal
    signature is degenerate there (injected rows become trivially separable by
    hour-of-day). ``supported_typologies`` still lists temporal because the columns
    exist; treat BankSim-temporal as out of scope in any cross-dataset comparison.
    PaySim ``step`` is HOURLY, so its hour-of-day is real and temporal is valid.
"""
from __future__ import annotations

import pandas as pd

from src.schema import Schema

# Arbitrary midnight anchors; only the *offset* from step matters. Anchoring at
# midnight makes hour-of-day == step % 24 for an hourly-step dataset.
PAYSIM_REF = pd.Timestamp("2019-01-01")
BANKSIM_REF = pd.Timestamp("2019-01-01")

PAYSIM_FILE = "PS_20174392719_1491204439457_log.csv"
BANKSIM_FILE = "bs140513_032310.csv"


def adapt_paysim(df: pd.DataFrame, time_col: str = "ts") -> tuple[pd.DataFrame, Schema]:
    """PaySim (Lopez-Rojas ``ealaxi/paysim1``). ``step`` is HOURLY.

    Roles: entity=nameOrig, target=nameDest, amount=amount, label=isFraud, time
    synthesised from the hourly step. No category, no coordinates -> hosts
    ring / velocity / temporal only."""
    df = df.copy()
    df[time_col] = PAYSIM_REF + pd.to_timedelta(df["step"], unit="h")
    schema = Schema(entity="nameOrig", target="nameDest", time=time_col,
                    amount="amount", label="isFraud")
    return df, schema


def adapt_banksim(df: pd.DataFrame, time_col: str = "ts") -> tuple[pd.DataFrame, Schema]:
    """BankSim (Lopez-Rojas ``ealaxi/banksim1``). ``step`` is DAILY.

    Every string field is single-quoted in the raw CSV (``'C123'``); we strip the
    quotes so keys are clean. Roles: entity=customer, target=merchant,
    category=category, amount=amount, label=fraud, time synthesised from the daily
    step. No coordinates -> hosts ring / velocity / temporal / category (temporal
    degenerate, see module caveat)."""
    df = df.copy()
    for col in df.columns:  # BankSim wraps every string field in '...'
        if pd.api.types.is_string_dtype(df[col]):
            df[col] = df[col].str.strip("'")
    df[time_col] = BANKSIM_REF + pd.to_timedelta(df["step"], unit="D")
    schema = Schema(entity="customer", target="merchant", time=time_col,
                    amount="amount", label="fraud", category="category")
    return df, schema


ADAPTERS = {"paysim": adapt_paysim, "banksim": adapt_banksim}
