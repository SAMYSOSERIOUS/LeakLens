"""Three popular public KKBox churn notebooks, rebuilt so they can be re-tested.

Each recipe copies the published feature list, split and model as closely
as the public code and README allow. Where we had to substitute something,
it says so in `notes`. The point is not to copy their code line by line
but to keep the two things that decide whether a score can be trusted:
  1. WHICH data each feature is computed from, and
  2. HOW train and test customers were separated.

Each recipe builds features from a Dataset. "As published" hands it the
full data file (that is what the notebooks did). "Honest" hands it only
what was visible on the cut-off date.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from .data import LOG_NUM_COLS, Dataset


def _last_tx(tx: pd.DataFrame, users) -> pd.DataFrame:
    t = tx[tx["msno"].isin(users)]
    return (t.sort_values(["msno", "transaction_date", "membership_expire_date"])
            .groupby("msno").tail(1).set_index("msno").reindex(users))


def _yyyymmdd(s: pd.Series) -> pd.Series:
    return s.dt.year * 10000 + s.dt.month * 100 + s.dt.day


def _members(d: Dataset, users) -> pd.DataFrame:
    m = d.members.set_index("msno").reindex(users)
    return m


# --------------------------------------------------------------------------
# Recipe A - github.com/jsroa15/KKBOX
#   transaction aggregates + log averages + profile, random forest with
#   oversampling, random split, ROC AUC.
# --------------------------------------------------------------------------
def features_a(d: Dataset, users) -> pd.DataFrame:
    users = pd.Index(users, name="msno")
    tx = d.tx[d.tx["msno"].isin(users)]
    g = tx.groupby("msno")
    m = _members(d, users)
    f = pd.DataFrame(index=users)
    f["regist_trans"] = g.size()
    f["mode_plan_days"] = g["payment_plan_days"].agg(lambda s: s.mode().iat[0])
    f["mode_payment_method"] = g["payment_method_id"].agg(lambda s: s.mode().iat[0])
    f["revenue"] = g["actual_amount_paid"].sum()
    f["is_auto_renew"] = g["is_auto_renew"].mean()
    f["regist_cancels"] = g["is_cancel"].sum()
    f["mode_quarter"] = g["transaction_date"].agg(lambda s: s.dt.quarter.mode().iat[0])
    f["tenure"] = (g["transaction_date"].max() - m["registration_init_time"]).dt.days
    lg = d.logs[d.logs["msno"].isin(users)].groupby("msno")
    days = lg["days_active"].sum().replace(0, np.nan)
    for c in LOG_NUM_COLS:
        f[f"avg_{c}"] = np.log1p(lg[c].sum() / days)
    f["age"] = pd.to_numeric(m["bd"], errors="coerce").where(lambda a: (a > 0) & (a < 90))
    f["city"] = m["city"]
    f["gender_male"] = (m["gender"] == "male").astype(float)
    return f.astype(float)


def model_a(seed=0):
    return make_pipeline(SimpleImputer(strategy="median"),
                         RandomForestClassifier(n_estimators=200, min_samples_leaf=2,
                                                n_jobs=-1, random_state=seed))


# --------------------------------------------------------------------------
# Recipe B - github.com/apostaremczak/churn-prediction
#   last transaction from transactions_v2 + members_v3 columns as they come
#   (dates as numbers), "unbalanced" random forest, random split.
# --------------------------------------------------------------------------
def features_b(d: Dataset, users) -> pd.DataFrame:
    users = pd.Index(users, name="msno")
    last = _last_tx(d.tx, users)
    m = _members(d, users)
    f = pd.DataFrame(index=users)
    for c in ("payment_method_id", "payment_plan_days", "plan_list_price", "actual_amount_paid",
              "is_auto_renew", "is_cancel"):
        f[c] = last[c]
    f["transaction_date"] = _yyyymmdd(last["transaction_date"])
    f["membership_expire_date"] = _yyyymmdd(last["membership_expire_date"])
    f["city"] = m["city"]
    f["bd"] = m["bd"]
    f["gender_male"] = (m["gender"] == "male").astype(float)
    f["registered_via"] = m["registered_via"]
    f["registration_init_time"] = _yyyymmdd(m["registration_init_time"])
    return f.astype(float)


def model_b(seed=0):
    return make_pipeline(SimpleImputer(strategy="median"),
                         RandomForestClassifier(n_estimators=200, n_jobs=-1, random_state=seed))


# --------------------------------------------------------------------------
# Recipe C - github.com/naomifridman/Deep-VAE-prediction-of-churn-customer
#   transaction count + last transaction (with dates) + log counts and sums
#   + profile; their model compresses features with a VAE, then KNN (k=27).
# --------------------------------------------------------------------------
def features_c(d: Dataset, users) -> pd.DataFrame:
    users = pd.Index(users, name="msno")
    last = _last_tx(d.tx, users)
    m = _members(d, users)
    f = pd.DataFrame(index=users)
    f["trans_count"] = d.tx[d.tx["msno"].isin(users)].groupby("msno").size()
    for c in ("payment_method_id", "payment_plan_days", "plan_list_price", "actual_amount_paid",
              "is_auto_renew", "is_cancel"):
        f[c] = last[c]
    f["transaction_date"] = _yyyymmdd(last["transaction_date"])
    f["membership_expire_date"] = _yyyymmdd(last["membership_expire_date"])
    lg = d.logs[d.logs["msno"].isin(users)].groupby("msno")
    f["logs_count"] = lg["days_active"].sum()
    for c in LOG_NUM_COLS:
        f[c] = lg[c].sum()
    f["bd"] = m["bd"]
    f["registration_init_time"] = _yyyymmdd(m["registration_init_time"])
    for c, vals in (("city", [1, 4, 5, 13, 22]), ("registered_via", [3, 4, 7, 9])):
        for v in vals:
            f[f"{c}_{v}"] = (m[c] == v).astype(float)
    f["gender_male"] = (m["gender"] == "male").astype(float)
    f[["trans_count", "logs_count", *LOG_NUM_COLS]] = f[["trans_count", "logs_count", *LOG_NUM_COLS]].fillna(0)
    return f.astype(float)


def model_c(seed=0):
    return make_pipeline(SimpleImputer(strategy="median"), StandardScaler(),
                         KNeighborsClassifier(n_neighbors=27, n_jobs=-1))


@dataclass
class Recipe:
    key: str
    title: str
    url: str
    build: Callable[[Dataset, pd.Index], pd.DataFrame]
    model: Callable[[int], object]
    claimed: dict
    claimed_text: str
    oversample: bool = False
    notes: list[str] = field(default_factory=list)


RECIPES = [
    Recipe(
        key="A", title="jsroa15/KKBOX (random forest)",
        url="https://github.com/jsroa15/KKBOX",
        build=features_a, model=model_a, oversample=True,
        claimed={"auc": 0.9401},
        claimed_text="Random forest, ROC AUC 0.940 on its test set (README).",
        notes=["Cancellation count, auto-renew share, revenue and tenure are computed over the whole "
               "transactions file.", "Churners are oversampled in training, as in the original."],
    ),
    Recipe(
        key="B", title="apostaremczak/churn-prediction (random forest)",
        url="https://github.com/apostaremczak/churn-prediction",
        build=features_b, model=model_b,
        claimed={"accuracy": 0.927, "f1": 0.482, "precision": 0.825, "recall": 0.34},
        claimed_text="'Unbalanced' random forest: accuracy 92.7%, F1 0.48 (model_results.json).",
        notes=["Uses each user's last row of transactions_v2, which runs to the end of the month "
               "in which churn is decided."],
    ),
    Recipe(
        key="C", title="naomifridman/Deep-VAE-prediction-of-churn-customer (VAE + KNN)",
        url="https://github.com/naomifridman/Deep-VAE-prediction-of-churn-customer",
        build=features_c, model=model_c,
        claimed={"accuracy": 0.9503, "f1": 0.66},
        claimed_text="KNN on a VAE latent space: accuracy 95.0%, churn-class F1 0.66 (notebook output).",
        notes=["Substitution: we run the KNN (k=27) on scaled features directly instead of on a "
               "VAE's 2-D compression. The leakage sits in the inputs, which we keep exactly."],
    ),
]
