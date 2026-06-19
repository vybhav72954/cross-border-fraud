"""The LR-test admission gate must admit a real signal and stay near-null on
noise — the mechanism that decides whether a neural extension enters the GLM.
"""
import numpy as np
import pandas as pd

from src.models.glm import BinaryRelevanceGLM


def _make_labelled(n=4000, seed=0):
    rng = np.random.default_rng(seed)
    signal = rng.normal(size=n)
    p = 1.0 / (1.0 + np.exp(-(-2.0 + 2.2 * signal)))
    y = pd.DataFrame({"target": (rng.uniform(size=n) < p).astype(int)})
    base = pd.DataFrame({"b": rng.normal(size=n)})
    return base, signal, y, rng


def test_gate_admits_signal_rejects_noise():
    base, signal, y, rng = _make_labelled()
    ext_signal = pd.DataFrame({"e": signal})
    ext_noise = pd.DataFrame({"e": rng.normal(size=len(y))})

    glm = BinaryRelevanceGLM()
    r_sig = glm.admit_extension(base, ext_signal, y, "target")
    r_noise = glm.admit_extension(base, ext_noise, y, "target")

    assert r_sig["p_value"] < 0.01          # real signal admitted
    assert r_sig["G2"] > r_noise["G2"]      # signal beats noise on the LR statistic
    assert r_sig["df"] == 1                 # one extension column -> one df


def test_gate_df_tracks_extension_width():
    base, signal, y, _ = _make_labelled()
    ext = pd.DataFrame({"e1": signal, "e2": signal ** 2})
    r = BinaryRelevanceGLM().admit_extension(base, ext, y, "target")
    assert r["df"] == 2


def test_binary_relevance_fits_and_predicts(legit_df):
    """End-to-end smoke: a binary-relevance GLM fits per-label and returns
    probabilities in [0, 1] with one column per provided label."""
    from src.labels import LABEL_COLS
    rng = np.random.default_rng(0)
    n = len(legit_df)
    X = pd.DataFrame({"f1": rng.normal(size=n), "f2": rng.normal(size=n)},
                     index=legit_df.index)
    y = pd.DataFrame({lab: rng.integers(0, 2, size=n) for lab in LABEL_COLS},
                     index=legit_df.index)
    glm = BinaryRelevanceGLM(maxiter=50).fit(X, y)
    proba = glm.predict_proba(X)
    assert list(proba.columns) == list(LABEL_COLS)
    assert ((proba.to_numpy() >= 0) & (proba.to_numpy() <= 1)).all()
