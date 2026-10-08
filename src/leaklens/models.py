"""The honest models: two simple rules, logistic regression, LightGBM."""

from __future__ import annotations

import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, brier_score_loss, log_loss, roc_auc_score
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder, StandardScaler

from .features import CATEGORICAL_FEATURES, FEATURES, NUMERIC_FEATURES


class NobodyChurns:
    """Baseline 1: predicts the average churn rate for everyone.

    It can't rank customers at all (AUC = 0.5), so any useful model has to
    beat it.
    """

    name = "Nobody churns (baseline)"

    def fit(self, X, y):
        self.rate_ = float(np.mean(y))
        return self

    def predict_proba(self, X):
        p = np.full(len(X), self.rate_)
        return np.column_stack([1 - p, p])


class AutoRenewRule:
    """Baseline 2: the rule a business would use without any model.

    "Customers who switched off auto-renew (or just cancelled) are at risk."
    Gives a churn chance per group, learned from the training months.
    """

    name = "Auto-renew rule"

    def _group(self, X):
        return (1 - X["auto_renew"].fillna(1)).clip(0, 1) + 2 * X["cancel_last"].fillna(0).clip(0, 1)

    def fit(self, X, y):
        g = self._group(X)
        self.rates_ = pd.Series(np.asarray(y), index=g.index).groupby(g.values).mean().to_dict()
        self.default_ = float(np.mean(y))
        return self

    def predict_proba(self, X):
        p = self._group(X).map(self.rates_).fillna(self.default_).to_numpy(dtype=float)
        return np.column_stack([1 - p, p])


def make_logistic():
    num = make_pipeline(
        SimpleImputer(strategy="median"),
        FunctionTransformer(lambda a: np.sign(a) * np.log1p(np.abs(a)), feature_names_out="one-to-one"),
        StandardScaler(),
    )
    cat = make_pipeline(
        SimpleImputer(strategy="most_frequent"),
        OneHotEncoder(handle_unknown="ignore", min_frequency=20),
    )
    pre = ColumnTransformer([("num", num, NUMERIC_FEATURES), ("cat", cat, CATEGORICAL_FEATURES)])
    model = Pipeline([("prep", pre), ("lr", LogisticRegression(max_iter=2000, C=0.5))])
    model.name = "Logistic regression"
    return model


def make_lgbm(seed: int = 0):
    m = LGBMClassifier(
        n_estimators=300, learning_rate=0.03, num_leaves=15, min_child_samples=80,
        subsample=0.8, subsample_freq=1, colsample_bytree=0.8, reg_lambda=1.0,
        random_state=seed, verbose=-1,
    )
    m.name = "LightGBM"
    return m


def prep_X(df: pd.DataFrame, categories: dict | None = None) -> tuple[pd.DataFrame, dict]:
    """Feature matrix with the same category lists as training (for LightGBM)."""
    X = df[FEATURES].copy()
    if categories is None:
        categories = {c: sorted(X[c].astype(str).unique().tolist()) for c in CATEGORICAL_FEATURES}
    for c in CATEGORICAL_FEATURES:
        vals = X[c].astype(str)
        # values never seen in training become missing (LightGBM handles that)
        X[c] = pd.Categorical(vals.where(vals.isin(categories[c])), categories=categories[c])
    return X, categories


def to_object_cats(X: pd.DataFrame) -> pd.DataFrame:
    """sklearn's one-hot encoder wants plain strings, not pandas categories."""
    X = X.copy()
    for c in CATEGORICAL_FEATURES:
        X[c] = X[c].astype(str)
    return X


def scores(y, p) -> dict:
    """The accuracy numbers we report. AUC is the headline."""
    y = np.asarray(y)
    p = np.clip(np.asarray(p, dtype=float), 1e-6, 1 - 1e-6)
    if len(np.unique(y)) < 2:
        return {"n": int(len(y)), "churn_rate": float(y.mean()) if len(y) else None}
    return {
        "n": int(len(y)),
        "churn_rate": round(float(y.mean()), 4),
        "auc": round(float(roc_auc_score(y, p)), 4),
        "pr_auc": round(float(average_precision_score(y, p)), 4),
        "log_loss": round(float(log_loss(y, p)), 4),
        "brier": round(float(brier_score_loss(y, p)), 4),
    }


def fit_predict_all(train: pd.DataFrame, test: pd.DataFrame, seed: int = 0) -> dict[str, np.ndarray]:
    """Train all four models on `train`, return churn chances for `test`."""
    y = train["is_churn"].to_numpy()
    Xtr, cats = prep_X(train)
    Xte, _ = prep_X(test, cats)
    out = {}
    for m in (NobodyChurns(), AutoRenewRule()):
        out[m.name] = m.fit(Xtr, y).predict_proba(Xte)[:, 1]
    lr = make_logistic().fit(to_object_cats(Xtr), y)
    out["Logistic regression"] = lr.predict_proba(to_object_cats(Xte))[:, 1]
    gbm = make_lgbm(seed).fit(Xtr, y)
    out["LightGBM"] = gbm.predict_proba(Xte)[:, 1]
    return out
