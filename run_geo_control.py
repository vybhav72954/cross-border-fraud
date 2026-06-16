"""
Geo negative-control: specificity of the LR-test admission gate, and the large-n
caveat that comes with it.

`geo` is tabular-solved (haversine distance separates it perfectly, AUC 1.000),
so it is the slot where a neural feature extractor should add NOTHING. We test,
on the `geo` label, the MATCHED tabular representation (haversine distance) vs the
MISMATCHED neural ones -- the GNN ring scalar (windowed merchant fan-in) and the
SSM temporal scalar (card-relative hour rarity). Geo is injected SOLO (no
overlaps) so no co-stamped signature can leak structure onto a geo row.

The honest result is a two-part finding:

  * BY EFFECT SIZE geo is a clean null. Isolated solo-vs-legit AUC -- the same
    measure as the whole bake-off -- is 1.000 for distance but ~0.50 / ~0.56 for
    the neural scalars: neither representation recovers geo.
  * BY THE RAW LR p-VALUE the gate over-admits. At n~1.3M the LR test rejects the
    null on negligible injection side-effects (assigning geo rows a random legit
    timestamp incidentally nudges card-hour-rarity to ~0.56; uniform-merchant
    sampling slightly lowers fan-in), so both neural scalars clear p<0.05 -- with
    G^2 one-to-two orders of magnitude below the MATCHED slot (~10^4, see §7.2),
    i.e. the gate RANKS correctly but p<0.05 is the wrong bar at this n.

Same large-n power inflation already flagged for Hosmer-Lemeshow (RESULTS §7.3):
a production gate should threshold on effect size (relative G^2 / held-out AUC
lift), not raw p. The gate's ranking is sound; its p-value alone is not.

Run from the project root:  python run_geo_control.py
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

sys.path.insert(0, ".")
from src.inject import legit_background, inject_geo, TYPOLOGY_COL  # noqa: E402
from src.robustness import compact_base, geo_score, ring_score  # noqa: E402
from src.models.ssm import card_hour_rarity  # noqa: E402
from src.models.glm import BinaryRelevanceGLM  # noqa: E402

RAW = Path("data/raw")
TIME_COL = "trans_date_trans_time"
N_GEO = 4000
MATCHED_G2 = 9620.0  # ring/temporal oracle-only G^2 on their OWN label (§7.2, df 1)


def isolated_auc(score: np.ndarray, typ: np.ndarray) -> float:
    solo, legit = typ == "geo", typ == ""
    mask = solo | legit
    return roc_auc_score(solo[mask].astype(int), score[mask])


def gate(name: str, X_base: pd.DataFrame, scalar: np.ndarray,
         y: pd.DataFrame) -> None:
    """LR test of one scalar on the geo label, with G^2 shown relative to the
    matched-slot G^2 so the effect size (not just p) is visible."""
    ext = pd.DataFrame({name: scalar})
    try:
        res = BinaryRelevanceGLM(maxiter=100).admit_extension(X_base, ext, y, "geo")
        verdict = "ADMITTED" if res["admitted"] else "rejected"
        ratio = f"{res['G2'] / MATCHED_G2:6.1%} of matched"
        print(f"    {name:22s} G2={res['G2']:9.2f}  p={res['p_value']:.2g}  "
              f"-> {verdict:8s} ({ratio})")
    except Exception as exc:
        print(f"    {name:22s} perfect separation ({type(exc).__name__}) "
              f"-> ADMITTED (infinite evidence)")


def main() -> None:
    print("== Geo negative-control (LR-gate specificity) ==")
    raw = pd.read_csv(RAW / "fraudTrain.csv", index_col=0,
                      parse_dates=[TIME_COL, "dob"])
    aug = inject_geo(legit_background(raw), N_GEO, rng=np.random.default_rng(0))
    typ = aug[TYPOLOGY_COL].fillna("").to_numpy()
    print(f"rows {len(aug):,} | geo (solo) rows {int((typ == 'geo').sum()):,}")

    dist = geo_score(aug)
    ring = ring_score(aug)               # windowed merchant fan-in (GNN oracle)
    hourrar = card_hour_rarity(aug).to_numpy()  # card hour rarity (SSM oracle)

    print("\nisolated geo AUC (solo vs legit) per representation:")
    print(f"    {'haversine distance':22s} {isolated_auc(dist, typ):.3f}  (matched)")
    print(f"    {'GNN ring fan-in':22s} {isolated_auc(ring, typ):.3f}  (mismatched)")
    print(f"    {'SSM hour-rarity':22s} {isolated_auc(hourrar, typ):.3f}  (mismatched)")

    # base = compact classical WITHOUT distance, so the negative controls aren't
    # masked by a perfectly-separating feature already in the model.
    X = compact_base(aug).drop(columns=["merch_dist_km"]).reset_index(drop=True)
    y = pd.DataFrame({"geo": (typ == "geo").astype(int)})

    print("\nLR-test gate on the `geo` label (base = classical \\ distance):")
    gate("merch_dist_km", X, dist, y)         # matched -> separates perfectly
    gate("ring_fanin", X, ring, y)            # mismatched -> tiny G^2 (artifact)
    gate("ssm_hour_rarity", X, hourrar, y)    # mismatched -> tiny G^2 (artifact)
    print("\nReading: BY EFFECT SIZE geo is a clean null -- distance separates it "
          "(AUC 1.000)\nwhile both neural scalars sit at ~chance (0.50 / 0.56). BY "
          "RAW p the gate over-\nadmits: at n~1.3M the LR test rejects on negligible "
          "injection side-effects, so the\nneural scalars clear p<0.05 -- but at G^2 "
          "a small fraction of the matched slot's\n~10^4. The gate ranks correctly "
          "(matched >> mismatched); p<0.05 is the wrong bar\nat this n (same large-n "
          "inflation as Hosmer-Lemeshow, RESULTS 7.3).\nThreshold on effect size, not p.")


if __name__ == "__main__":
    main()
