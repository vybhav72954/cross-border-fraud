"""
Category headroom demonstration -- turning a tabular-solved slot into a genuine
card-relative one with an injection knob.

The default `category` signature (a txn in one of the card's rarest categories)
is ~0.87 for the tabular GLM, but that's a near-degenerate result: rare-FOR-CARD
is almost always rare-GLOBALLY too, so a global one-hot category dummy already
catches it. There is no real card-relative headroom for a sequence model to claim.

`inject_category(..., globally_common_only=True)` constrains the rare-for-card
category to the globally MOST common categories (grocery, gas, ...), the intent
being a purely card-relative anomaly a global dummy can't see.

The honest finding is a NEGATIVE one, and it is the point: on Sparkov the knob
does NOT create card-relative headroom, because the data lacks the structure.
Cards use every globally-common category at >=~7% (there are zero zero-usage
card x common-category pairs), so a common category is never genuinely card-rare.
Under the knob the global signal collapses AS INTENDED, but the card-relative
oracle collapses too -- there is no per-card sparsity to exploit. This is exactly
the structure hours-of-day HAVE (cards genuinely never transact at some hours ->
temporal is a real SSM slot) and categories LACK. Category therefore stays a
tabular slot; the knob documents a dataset limitation, not a neural opportunity.

  tabular GLM        : build_features one-hot logistic (the canonical baseline)
  global cat-rarity  : 1 - the category's GLOBAL share  (what a global dummy sees)
  card cat-rarity    : 1 - the card's share of this category (card-relative oracle)

Run from the project root:  python scripts/08_category_headroom.py
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, ".")
from src.features import build_features  # noqa: E402
from src.inject import legit_background, inject_category, TYPOLOGY_COL  # noqa: E402
from src.robustness import card_category_rarity  # noqa: E402

RAW = Path("data/raw")
TIME_COL = "trans_date_trans_time"
N_TRAIN, N_TEST = 4000, 1720


def load_legit(fname: str) -> pd.DataFrame:
    df = pd.read_csv(RAW / fname, index_col=0, parse_dates=[TIME_COL, "dob"])
    return legit_background(df)


def global_cat_rarity(df: pd.DataFrame, ref_freq: pd.Series) -> np.ndarray:
    """1 - the category's GLOBAL share (a category common across the population
    scores low here -- exactly what a global one-hot dummy can exploit)."""
    share = df["category"].map(ref_freq / ref_freq.sum())
    return (1.0 - share.to_numpy()).astype(float)


def isolated_auc(score: np.ndarray, typ: np.ndarray) -> float:
    solo, legit = typ == "category", typ == ""
    mask = solo | legit
    return roc_auc_score(solo[mask].astype(int), score[mask])


def tabular_auc(tr: pd.DataFrame, te: pd.DataFrame, ytr: np.ndarray,
                typ_te: np.ndarray) -> float:
    """Canonical tabular GLM: standardized L2-logistic over build_features."""
    Xtr = build_features(tr)
    Xte = build_features(te).reindex(columns=Xtr.columns, fill_value=0.0)
    sc = StandardScaler().fit(Xtr)
    clf = LogisticRegression(max_iter=1000, class_weight="balanced")
    clf.fit(sc.transform(Xtr), ytr)
    return isolated_auc(clf.predict_proba(sc.transform(Xte))[:, 1], typ_te)


def run(knob: bool, base_tr, base_te, ref_freq) -> dict:
    tr = inject_category(base_tr, N_TRAIN, globally_common_only=knob,
                         rng=np.random.default_rng(0))
    te = inject_category(base_te, N_TEST, globally_common_only=knob,
                         rng=np.random.default_rng(1))
    tr["trans_dt"] = pd.to_datetime(tr[TIME_COL])
    te["trans_dt"] = pd.to_datetime(te[TIME_COL])
    typ_te = te[TYPOLOGY_COL].fillna("").to_numpy()
    ytr = (tr[TYPOLOGY_COL].fillna("").to_numpy() == "category").astype(int)

    inj_cats = te.loc[typ_te == "category", "category"].value_counts().head(4)
    return {
        "tabular": tabular_auc(tr, te, ytr, typ_te),
        "global_rarity": isolated_auc(global_cat_rarity(te, ref_freq), typ_te),
        "card_rarity": isolated_auc(card_category_rarity(te).to_numpy(), typ_te),
        "top_injected": inj_cats.index.tolist(),
    }


def common_sparsity_diagnostic(base: pd.DataFrame, top_g: int = 5) -> dict:
    """Why the knob can't create headroom: do cards ever near-skip a globally
    common category? Reports zero-usage pairs + the smallest common-cat share."""
    gf = base["category"].value_counts()
    common = gf.nlargest(top_g).index
    mat = (base.groupby(["cc_num", "category"]).size().unstack(fill_value=0)
           .reindex(columns=gf.index, fill_value=0))
    card_tot = mat.sum(axis=1)
    least_common_share = (mat[common].min(axis=1) / card_tot)
    return {"top_g": top_g,
            "cards_with_zero_usage_common": int((mat[common] == 0).any(axis=1).sum()),
            "n_cards": int(len(mat)),
            "median_least_common_share": float(least_common_share.median()),
            "max_least_common_share": float(least_common_share.max())}


def main() -> None:
    print("== Category headroom: global-rare vs card-relative injection ==")
    base_tr, base_te = load_legit("fraudTrain.csv"), load_legit("fraudTest.csv")
    ref_freq = base_tr["category"].value_counts()

    off = run(False, base_tr, base_te, ref_freq)
    on = run(True, base_tr, base_te, ref_freq)

    print(f"\n{'injection knob':32s} {'tabular':>9s} {'global-rar':>11s} {'card-rar':>9s}")
    print(f"{'OFF  (rare-for-card, default)':32s} "
          f"{off['tabular']:9.3f} {off['global_rarity']:11.3f} {off['card_rarity']:9.3f}")
    print(f"{'ON   (rare-for-card, common-glob)':32s} "
          f"{on['tabular']:9.3f} {on['global_rarity']:11.3f} {on['card_rarity']:9.3f}")
    print(f"\n  injected categories (knob OFF): {off['top_injected']}")
    print(f"  injected categories (knob ON ): {on['top_injected']}")

    diag = common_sparsity_diagnostic(base_tr)
    print(f"\nper-card sparsity of the top-{diag['top_g']} common categories:")
    print(f"  cards that ever near-skip a common category: "
          f"{diag['cards_with_zero_usage_common']} / {diag['n_cards']}")
    print(f"  smallest common-category share per card: "
          f"median={diag['median_least_common_share']:.3f}  "
          f"max={diag['max_least_common_share']:.3f}")

    print("\nReading (a NEGATIVE result, and the point): knob ON collapses the "
          "global/tabular\nsignal as intended (global-rar -> ~0.26), but the "
          "card-relative oracle collapses\ntoo (card-rar 0.95 -> ~0.48). Sparkov "
          "cards use every common category (no zero-\nusage pairs; smallest share "
          "~7%), so a common category is never genuinely card-\nrare -- there is no "
          "card-relative headroom to expose. Unlike hours-of-day (real\nper-card "
          "sparsity -> temporal is an SSM slot), category usage is too uniform per "
          "card.\nThe modest gap that DOES exist is already visible knob-OFF "
          "(card-rar 0.95 > tabular\n0.87). (Residual tabular ~0.82 under the knob "
          "is partly an amt-pool artifact, not\ncategory signal.)")


if __name__ == "__main__":
    main()
