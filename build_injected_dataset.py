"""
Build the controlled multi-typology fraud dataset.

Loads the raw Sparkov splits, strips them to legitimate transactions, injects
fraud with known typology signatures (ring / velocity / temporal / category /
geo), and writes the augmented frames + answer key to data/processed/.

Run from the project root:
    python build_injected_dataset.py
"""
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, ".")
from src.inject import (  # noqa: E402
    build_controlled_dataset, DEFAULT_COUNTS, TYPOLOGY_COL, TYPOLOGIES,
)
from src.labels import _haversine_series  # noqa: E402

RAW = Path("data/raw")
OUT = Path("data/processed")
OUT.mkdir(parents=True, exist_ok=True)

# Test split is ~43% of train; scale event counts so fraud rates are comparable.
SPLITS = [("train", "fraudTrain.csv", 1.0, 0),
          ("test", "fraudTest.csv", 0.43, 1)]


def summarize(aug: pd.DataFrame) -> None:
    n = len(aug)
    n_fraud = int((aug["is_fraud"] == 1).sum())
    print(f"  rows: {n:,}   fraud: {n_fraud:,} ({n_fraud / n * 100:.2f}%)")
    vc = aug[TYPOLOGY_COL].value_counts()
    for typ in TYPOLOGIES:
        cnt = int(vc.get(typ, 0))
        ev = aug.loc[aug[TYPOLOGY_COL] == typ, "inj_event"].nunique()
        print(f"    {typ:9s} rows={cnt:>6,}  events={ev:>5,}")
    # sanity: the geo injector should now produce far-from-home merchants
    geo = aug[aug[TYPOLOGY_COL] == "geo"]
    if len(geo):
        d = _haversine_series(geo["lat"], geo["long"], geo["merch_lat"], geo["merch_long"])
        print(f"    geo distance km: median={d.median():.0f}  min={d.min():.0f}  max={d.max():.0f}")


def main() -> None:
    for split, fname, scale, seed in SPLITS:
        path = RAW / fname
        if not path.exists():
            print(f"[{split}] SKIP — {path} not found")
            continue
        print(f"\n[{split}] loading {path} ...")
        df = pd.read_csv(path, index_col=0,
                         parse_dates=["trans_date_trans_time", "dob"])
        counts = {k: max(int(v * scale), 1) for k, v in DEFAULT_COUNTS.items()}
        aug = build_controlled_dataset(df, counts=counts, seed=seed)
        out = OUT / f"injected_{split}.parquet"
        aug.to_parquet(out, index=False)
        print(f"[{split}] wrote {out}")
        summarize(aug)


if __name__ == "__main__":
    main()
