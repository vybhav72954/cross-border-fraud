"""
Robustness / statistical-rigor suite for the controlled multi-typology benchmark.

Four studies (driven by ``run_robustness.py``):

  D1  cross-border degradation -- per-typology matched-detector AUC vs the number
      of overlapping signatures stamped on an event (the project's namesake
      question: does a representation still recover ITS signature as typologies
      pile up on one fraud event?).
  D2  multi-seed variance -- rebuild the injection over several seeds and report
      each typology's isolated AUC as mean +/- std, plus LR-test stability.
  D3  calibration -- Hosmer-Lemeshow + reliability curves for the per-label logits.
  D4  threshold/overlap sensitivity -- move each injection knob by +/-20%, rebuild,
      and report how the isolated AUC grid shifts.

Design: everything here REUSES the production primitives -- the injectors in
``src.inject`` (``inject_overlap`` already stamps arbitrary-arity combos for the
single/velocity shapes), the oracle features in ``src.models`` and ``src.labels``,
and ``src.evaluation``. No detector is reimplemented; the studies only orchestrate.

Detector bank (oracle, deterministic, training-free -- the per-typology "ceiling"
recovery signal, so a degradation curve isolates overlap interference rather than
model-fit noise):
  ring      windowed merchant fan-in   (merchant_window_features)
  temporal  card-relative hour rarity  (card_hour_rarity)
  category  card-relative cat rarity   (card_category_rarity, the cat analogue)
  velocity  rolling 1h txn count       (mirrors features.build_features vel_1h)
  geo       haversine home->merchant   (labels._haversine_series)
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

from src.inject import (
    TYPOLOGIES, TYPOLOGY_COL, legit_background,
    inject_ring, inject_velocity, inject_temporal, inject_category, inject_geo,
    inject_overlap, DEFAULT_COUNTS, DEFAULT_OVERLAP,
)
from src.labels import _haversine_series
from src.models.gnn import merchant_window_features
from src.models.ssm import card_hour_rarity
from src.evaluation import hosmer_lemeshow, lr_test

RAW = Path("data/raw")
OUT = Path("data/processed")
FIG = Path("figures")
RING_WINDOW_H = 2.0  # match inject_ring's window_hours
TIME_COL = "trans_date_trans_time"


# ── oracle detector bank ────────────────────────────────────────────────────

def card_category_rarity(df: pd.DataFrame) -> pd.Series:
    """Per transaction: 1 - the cardholder's historical share of this category.

    The category analogue of ``card_hour_rarity`` -- high for a txn in a category
    the card rarely uses. The interpretable oracle for the `category` signature.
    """
    cc = df["cc_num"]
    cat = df["category"].astype(str)
    pair = cc.astype(str) + "_" + cat
    pair_n = pair.map(pair.value_counts())
    card_n = cc.map(cc.value_counts())
    return (1.0 - pair_n / card_n).astype(float)


def velocity_score(df: pd.DataFrame) -> np.ndarray:
    """Per-card rolling 1h transaction count -- mirrors features.build_features
    vel_1h exactly (the tabular velocity oracle)."""
    d = df.copy()
    d["trans_dt"] = pd.to_datetime(d[TIME_COL])
    order = d.sort_values(["cc_num", "trans_dt"]).index
    df_s = d.sort_values(["cc_num", "trans_dt"]).set_index("trans_dt")
    counts = (df_s.groupby("cc_num", group_keys=False)["trans_num"]
              .rolling("60min", closed="both").count())
    counts.index = counts.index.droplevel(0)
    df_s["_vel1h"] = counts
    df_s = df_s.reset_index()
    vel = df_s.set_index(order)["_vel1h"]
    return vel.reindex(d.index).fillna(1).astype(float).to_numpy()


def geo_score(df: pd.DataFrame) -> np.ndarray:
    return _haversine_series(df["lat"], df["long"],
                            df["merch_lat"], df["merch_long"]).to_numpy()


def ring_score(df: pd.DataFrame, window_hours: float = RING_WINDOW_H) -> np.ndarray:
    return merchant_window_features(
        df, window_hours=window_hours, show_progress=False
    )["merch_win_cards"].to_numpy()


def oracle_scores(df: pd.DataFrame, window_hours: float = RING_WINDOW_H) -> dict:
    """Compute every matched oracle detector once for a frame."""
    return {
        "ring": ring_score(df, window_hours),
        "velocity": velocity_score(df),
        "temporal": card_hour_rarity(df).to_numpy(),
        "category": card_category_rarity(df).to_numpy(),
        "geo": geo_score(df),
    }


def compact_base(df: pd.DataFrame) -> pd.DataFrame:
    """The few classical scalars the LR-test gate needs, computed directly.

    A focused subset of features.build_features (no one-hot dummies), so the
    multi-seed rebuild loop isn't paying for the full design matrix each seed.
    """
    dt = pd.to_datetime(df[TIME_COL])
    hour = dt.dt.hour
    age = (dt - pd.to_datetime(df["dob"])).dt.days / 365.25
    return pd.DataFrame({
        "log_amt": np.log1p(df["amt"].to_numpy()),
        "hour_sin": np.sin(2 * np.pi * hour / 24).to_numpy(),
        "hour_cos": np.cos(2 * np.pi * hour / 24).to_numpy(),
        "age": age.to_numpy(),
        "log_city_pop": np.log1p(df["city_pop"].to_numpy()),
        "merch_dist_km": geo_score(df),
        "vel_1h": velocity_score(df),
    })


# ── answer-key helpers ──────────────────────────────────────────────────────

def _typ(df: pd.DataFrame) -> np.ndarray:
    return df[TYPOLOGY_COL].fillna("").to_numpy()


def _depth(typ: np.ndarray) -> np.ndarray:
    """Number of stamped signatures per row (0 = legit)."""
    return np.array([0 if t == "" else t.count("+") + 1 for t in typ])


def isolated_auc(score: np.ndarray, typ: np.ndarray, typology: str) -> float:
    """AUC for one typology's SOLO rows vs legit (no overlap signal-borrowing)."""
    solo, legit = typ == typology, typ == ""
    mask = solo | legit
    return roc_auc_score(solo[mask].astype(int), score[mask])


def depth_auc(score: np.ndarray, typ: np.ndarray, typology: str,
              depth: int) -> tuple[float, int]:
    """AUC for `typology` rows AT a given overlap depth vs legit.

    A row contributes if it carries `typology` and exactly `depth` signatures.
    Scored against legit only, so this isolates whether the matched detector
    still recovers the signature when (depth-1) other signatures co-occur.
    """
    d = _depth(typ)
    has = np.array([typology in (t.split("+") if t else []) for t in typ])
    pos = has & (d == depth)
    legit = typ == ""
    mask = pos | legit
    if pos.sum() < 20:
        return float("nan"), int(pos.sum())
    return roc_auc_score(pos[mask].astype(int), score[mask]), int(pos.sum())


# ── dataset (re)builders from a cached legit background ──────────────────────

def load_legit(split: str = "train") -> pd.DataFrame:
    """Legit-only background from raw Sparkov (cache once, re-inject many times)."""
    fname = "fraudTrain.csv" if split == "train" else "fraudTest.csv"
    df = pd.read_csv(RAW / fname, index_col=0,
                     parse_dates=[TIME_COL, "dob"])
    return legit_background(df)


def build_standard(base: pd.DataFrame, counts: dict, overlap: dict,
                   seed: int, params: dict | None = None) -> pd.DataFrame:
    """Reproduce build_controlled_dataset from a pre-filtered legit base, with
    optional per-injector parameter overrides (for the sensitivity sweep)."""
    p = params or {}
    rng = np.random.default_rng(seed)
    aug = inject_ring(base, counts["ring"],
                      cards_per_ring=p.get("cards_per_ring", 5),
                      window_hours=p.get("window_hours", 2.0), rng=rng)
    aug = inject_velocity(aug, counts["velocity"],
                          txn_per_burst=p.get("txn_per_burst", 5),
                          window_minutes=p.get("velocity_window_minutes", 20.0), rng=rng)
    aug = inject_temporal(aug, counts["temporal"],
                          rarest_k=p.get("temporal_rarest_k", 5), rng=rng)
    aug = inject_category(aug, counts["category"],
                          rarest_k=p.get("category_rarest_k", 3), rng=rng)
    aug = inject_geo(aug, counts["geo"],
                     min_offset_deg=p.get("geo_min_offset_deg", 8.0), rng=rng)
    if overlap:
        aug = inject_overlap(aug, overlap, rng=rng)
    return aug


# ── D1: cross-border degradation ────────────────────────────────────────────

# combos that stamp each typology at depths 1..4. Single/velocity shapes compose
# cleanly with every modifier; ring composes cleanly only with geo (a shared
# merchant can't sit at each card's own rare hour/category), so ring is depth 1-2.
DEGRADE_OVERLAP = {
    # depth 2 (single-txn modifier pairs)
    ("geo", "temporal"): 2000, ("category", "geo"): 2000,
    ("category", "temporal"): 2000,
    # depth 3 (single-txn modifier triple)
    ("category", "geo", "temporal"): 2000,
    # velocity + modifiers (depth 2-4)
    ("geo", "velocity"): 1000, ("temporal", "velocity"): 1000,
    ("category", "velocity"): 1000,
    ("geo", "temporal", "velocity"): 1000, ("category", "geo", "velocity"): 1000,
    ("category", "temporal", "velocity"): 1000,
    ("category", "geo", "temporal", "velocity"): 1000,
    # ring + geo (depth 2)
    ("geo", "ring"): 400,
}
DEGRADE_SOLO = {"ring": 600, "velocity": 1500, "temporal": 4000,
                "category": 4000, "geo": 4000}


def study_degradation(base: pd.DataFrame, seed: int = 0) -> pd.DataFrame:
    """Per-typology matched-detector AUC at each overlap depth."""
    aug = build_standard(base, DEGRADE_SOLO, DEGRADE_OVERLAP, seed)
    typ = _typ(aug)
    sc = oracle_scores(aug)
    rows = []
    for t in TYPOLOGIES:
        for d in range(1, 5):
            auc, n = depth_auc(sc[t], typ, t, d)
            if not np.isnan(auc):
                rows.append({"typology": t, "depth": d, "auc": auc, "n_pos": n})
    return pd.DataFrame(rows)


# ── D2: multi-seed variance ─────────────────────────────────────────────────

def study_multiseed(base: pd.DataFrame, seeds: list[int],
                    counts: dict | None = None,
                    overlap: dict | None = None) -> pd.DataFrame:
    """Isolated AUC per typology across injection seeds + LR-test stability.

    LR test uses the matched oracle scalar as the extension (ring: windowed
    fan-in; temporal: hour-rarity) over a compact classical base, mirroring the
    production gate without per-seed neural training.
    """
    from src.models.glm import BinaryRelevanceGLM
    counts = counts or DEFAULT_COUNTS
    overlap = overlap or DEFAULT_OVERLAP
    rows = []
    for s in seeds:
        aug = build_standard(base, counts, overlap, seed=s)
        typ = _typ(aug)
        sc = oracle_scores(aug)

        rec = {"seed": s}
        for t in TYPOLOGIES:
            rec[f"auc_{t}"] = isolated_auc(sc[t], typ, t)

        # LR-test stability for the two neural slots (oracle scalar as extension)
        X = compact_base(aug).reset_index(drop=True)
        for t, scalar in [("ring", sc["ring"]), ("temporal", sc["temporal"])]:
            y = pd.DataFrame({t: (np.char.find(typ.astype(str), t) >= 0).astype(int)})
            ext = pd.DataFrame({f"{t}_oracle": scalar})
            try:
                res = BinaryRelevanceGLM(maxiter=80).admit_extension(X, ext, y, t)
                rec[f"lrG2_{t}"], rec[f"lradm_{t}"] = res["G2"], res["admitted"]
            except Exception:
                rec[f"lrG2_{t}"], rec[f"lradm_{t}"] = float("inf"), True
        rows.append(rec)
    return pd.DataFrame(rows)


# ── D3: calibration ─────────────────────────────────────────────────────────

def study_calibration(train: pd.DataFrame, test: pd.DataFrame,
                      g: int = 10) -> tuple[pd.DataFrame, dict]:
    """Per-label Hosmer-Lemeshow + reliability-curve points for the logits.

    Fits an UNWEIGHTED logistic per typology (class_weight=None -- balancing
    destroys probability calibration, which is exactly what we are measuring)
    over classical features + the matched oracle scalar for that typology.
    Returns (HL table, {typology: (bin_pred, bin_obs)}).
    """
    from src.features import build_features
    from src.inject import typology_dummies
    from sklearn.preprocessing import StandardScaler

    Xtr = build_features(train)
    Xte = build_features(test).reindex(columns=Xtr.columns, fill_value=0.0)
    Ytr = typology_dummies(train)[TYPOLOGIES]
    Yte = typology_dummies(test)[TYPOLOGIES]
    sc_tr, sc_te = oracle_scores(train), oracle_scores(test)

    rows, curves = [], {}
    for t in TYPOLOGIES:
        # standardize (even-handed reg across feature scales, as the baseline does)
        # + near-unregularized logit so calibration reflects the model, not shrinkage;
        # no class_weight, which would destroy probability calibration.
        scaler = StandardScaler().fit(Xtr.assign(_oracle=sc_tr[t]))
        Xtr_t = scaler.transform(Xtr.assign(_oracle=sc_tr[t]))
        Xte_t = scaler.transform(Xte.assign(_oracle=sc_te[t]))
        clf = LogisticRegression(max_iter=3000, C=1e4)
        clf.fit(Xtr_t, Ytr[t].to_numpy())
        p = clf.predict_proba(Xte_t)[:, 1]
        y = Yte[t].to_numpy()

        hl = hosmer_lemeshow(y, p, g=g)
        rows.append({"typology": t, "HL": hl["HL"], "df": hl["df"],
                     "p_value": hl["p_value"], "prevalence": float(y.mean()),
                     "mean_pred": float(p.mean())})
        d = pd.DataFrame({"y": y, "p": p})
        d["grp"] = pd.qcut(d["p"].rank(method="first"), q=g, labels=False)
        agg = d.groupby("grp").agg(pred=("p", "mean"), obs=("y", "mean"))
        curves[t] = (agg["pred"].to_numpy(), agg["obs"].to_numpy())
    return pd.DataFrame(rows), curves


# ── D4: threshold / overlap sensitivity ─────────────────────────────────────

# (param, affected typology, base value, the inject kwarg key)
SENS_PARAMS = [
    ("window_hours", "ring", 2.0),
    ("cards_per_ring", "ring", 5),
    ("velocity_window_minutes", "velocity", 20.0),
    ("txn_per_burst", "velocity", 5),
    ("temporal_rarest_k", "temporal", 5),
    ("category_rarest_k", "category", 3),
    ("geo_min_offset_deg", "geo", 8.0),
]


def _pm20(name: str, base):
    """-20% / base / +20%, rounded to int for count-like knobs."""
    lo, hi = base * 0.8, base * 1.2
    if isinstance(base, int):
        lo, hi = max(int(round(lo)), 1), int(round(hi))
    return [("-20%", lo), ("base", base), ("+20%", hi)]


def study_sensitivity(base: pd.DataFrame, seed: int = 0,
                      counts: dict | None = None,
                      overlap: dict | None = None) -> pd.DataFrame:
    """Vary each injection knob +/-20%, rebuild, recompute the affected
    typology's isolated oracle AUC. One knob moved at a time."""
    counts = counts or DEFAULT_COUNTS
    rows = []
    for name, typ, baseval in SENS_PARAMS:
        for tag, val in _pm20(name, baseval):
            params = {name: val}
            # ring fan-in detector must use the SAME window as injection
            win = val if name == "window_hours" else 2.0
            aug = build_standard(base, counts, overlap or {}, seed, params=params)
            t_arr = _typ(aug)
            if typ == "ring":
                sc = ring_score(aug, window_hours=win)
            elif typ == "velocity":
                sc = velocity_score(aug)
            elif typ == "temporal":
                sc = card_hour_rarity(aug).to_numpy()
            elif typ == "category":
                sc = card_category_rarity(aug).to_numpy()
            else:
                sc = geo_score(aug)
            rows.append({"param": name, "typology": typ, "setting": tag,
                         "value": val, "auc": isolated_auc(sc, t_arr, typ)})
    return pd.DataFrame(rows)
