"""
GLM models for multi-label fraud detection.

Production model: BinaryRelevanceGLM — five independent binary logistic regressions.
Companion models: PowerSetMNL, ProportionalOdds, PoissonVelocity.
"""

from __future__ import annotations

import warnings
import numpy as np
import pandas as pd
import statsmodels.api as sm
from statsmodels.miscmodels.ordinal_model import OrderedModel

from src.labels import LABEL_COLS
from src.evaluation import lr_test


class BinaryRelevanceGLM:
    """Five independent binary logit models, one per fraud label.

    This is the production classifier. Each model is a full statsmodels
    Logit fit with coefficients, odds ratios, Wald tests, and calibration.
    Neural extension scalars are admitted per-label via LR test.
    """

    def __init__(self, maxiter: int = 300) -> None:
        self.maxiter = maxiter
        self.models: dict[str, sm.Logit] = {}
        self.results: dict[str, any] = {}

    def fit(self, X: pd.DataFrame, y: pd.DataFrame) -> "BinaryRelevanceGLM":
        """Fit one binary logit per column in y.

        Parameters
        ----------
        X: design matrix (no intercept — added internally)
        y: label matrix with columns matching LABEL_COLS
        """
        X_c = sm.add_constant(X, has_constant="add")
        for label in LABEL_COLS:
            if label not in y.columns:
                continue
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                res = sm.Logit(y[label], X_c).fit(
                    maxiter=self.maxiter, disp=False, warn_convergence=False
                )
            self.results[label] = res
        return self

    def predict_proba(self, X: pd.DataFrame) -> pd.DataFrame:
        """Return (n, K) probability matrix."""
        X_c = sm.add_constant(X, has_constant="add")
        probs = {label: res.predict(X_c) for label, res in self.results.items()}
        return pd.DataFrame(probs, index=X.index)

    def predict(self, X: pd.DataFrame, threshold: float = 0.5) -> pd.DataFrame:
        """Return (n, K) binary prediction matrix."""
        return (self.predict_proba(X) >= threshold).astype(int)

    def odds_ratios(self, label: str) -> pd.DataFrame:
        res = self.results[label]
        params = res.params
        ci = res.conf_int()
        return pd.DataFrame({
            "coef": params,
            "OR": np.exp(params),
            "OR_ci_lo": np.exp(ci[0]),
            "OR_ci_hi": np.exp(ci[1]),
            "pvalue": res.pvalues,
        })

    def log_likelihoods(self) -> dict[str, float]:
        return {label: res.llf for label, res in self.results.items()}

    def admit_extension(
        self,
        X_base: pd.DataFrame,
        X_ext: pd.DataFrame,
        y: pd.DataFrame,
        label: str,
    ) -> dict:
        """LR test: does X_ext add significant information over X_base for `label`?

        Returns the LR test result dict. Caller decides whether to admit.
        """
        X_b = sm.add_constant(X_base, has_constant="add")
        X_f = sm.add_constant(
            pd.concat([X_base, X_ext], axis=1), has_constant="add"
        )
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            m0 = sm.Logit(y[label], X_b).fit(maxiter=self.maxiter, disp=False)
            m1 = sm.Logit(y[label], X_f).fit(maxiter=self.maxiter, disp=False)
        result = lr_test(m1.llf, m0.llf, X_ext.shape[1])
        result["label"] = label
        return result

    def summary(self) -> pd.DataFrame:
        rows = []
        for label, res in self.results.items():
            rows.append({
                "label": label,
                "n_obs": int(res.nobs),
                "log_lik": res.llf,
                "pseudo_r2": res.prsquared,
                "converged": res.mle_retvals.get("converged", True),
            })
        return pd.DataFrame(rows).set_index("label")


class PowerSetMNL:
    """Multinomial logit on top-M most frequent label combinations.

    Companion analysis only — not the production model.
    """

    def __init__(self, top_m: int = 8, maxiter: int = 300) -> None:
        self.top_m = top_m
        self.maxiter = maxiter
        self.result = None
        self.top_combos: list[str] = []

    def fit(self, X: pd.DataFrame, df_labels: pd.DataFrame) -> "PowerSetMNL":
        combos = df_labels[LABEL_COLS].apply(
            lambda r: "".join(r.astype(str)), axis=1
        )
        self.top_combos = combos.value_counts().head(self.top_m).index.tolist()
        mask = combos.isin(self.top_combos)
        y_cat = pd.Categorical(combos[mask], categories=self.top_combos).codes
        X_c = sm.add_constant(X[mask], has_constant="add")
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            self.result = sm.MNLogit(y_cat, X_c).fit(
                maxiter=self.maxiter, disp=False
            )
        return self


class ProportionalOddsModel:
    """Cumulative logit (proportional-odds) model for severity = |L|."""

    def __init__(self) -> None:
        self.result = None

    def fit(self, X: pd.DataFrame, severity: pd.Series) -> "ProportionalOddsModel":
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            self.result = OrderedModel(
                severity, X, distr="logit"
            ).fit(method="bfgs", disp=False)
        return self


class PoissonVelocity:
    """Poisson (or NegBin) regression for transaction velocity counts."""

    def __init__(self, overdispersion_threshold: float = 1.5) -> None:
        self.threshold = overdispersion_threshold
        self.result = None
        self.model_type: str = "poisson"

    def fit(self, X: pd.DataFrame, V: pd.Series) -> "PoissonVelocity":
        X_c = sm.add_constant(X, has_constant="add")
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            res = sm.Poisson(V, X_c).fit(maxiter=200, disp=False)
            od = res.pearson_chi2 / res.df_resid
            if od > self.threshold:
                res = sm.NegativeBinomial(V, X_c).fit(disp=False)
                self.model_type = "negbin"
            self.result = res
        return self
