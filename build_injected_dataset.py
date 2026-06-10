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
    build_controlled_dataset, DEFAULT_COUNTS, DEFAULT_OVERLAP,
    TYPOLOGY_COL, TYPOLOGIES, typology_dummies, is_cross_border,
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
    dums = typology_dummies(aug)
    for typ in TYPOLOGIES:
        print(f"    {typ:9s} rows (incl. overlap) = {int(dums[typ].sum()):>6,}")
    cb = is_cross_border(aug)
    print(f"    cross_border (>=2 typologies): {int(cb.sum()):,} rows")
    for combo, c in aug.loc[cb, TYPOLOGY_COL].value_counts().items():
        print(f"      {combo:24s} {int(c):>5,}")
    # sanity: every geo signature (single or overlap) should be far from home
    geo = aug.loc[dums["geo"] == 1]
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
        overlap = {k: max(int(v * scale), 1) for k, v in DEFAULT_OVERLAP.items()}
        aug = build_controlled_dataset(df, counts=counts, overlap=overlap, seed=seed)
        out = OUT / f"injected_{split}.parquet"
        aug.to_parquet(out, index=False)
        print(f"[{split}] wrote {out}")
        summarize(aug)


if __name__ == "__main__":
    main()
