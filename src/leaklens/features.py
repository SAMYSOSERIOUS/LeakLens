"""Honest features: built only from what was known on the cut-off date.

There is exactly one door into the data here: `visible()`. It throws away
every transaction and every month of listening logs dated after the
cut-off. All feature code below only ever sees its output. The leakage
test in tests/test_leakage.py checks this from the outside by changing the
future and confirming that no feature moves.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .data import Dataset
from .dates import add_months, month_end
from .labels import known_status

NUMERIC_FEATURES = [
    # latest known transaction
    "plan_days", "list_price", "amount_paid", "discount", "auto_renew", "cancel_last",
    "days_to_expire",
    # history
    "n_tx", "n_tx_180d", "n_cancel", "n_cancel_365d", "tenure_days", "auto_renew_share",
    "n_plans", "avg_paid", "n_lapses", "days_since_last_tx",
    # profile
    "age", "reg_days",
    # listening (whole months up to the cut-off)
    "days_active_m1", "secs_m1", "days_active_m3", "secs_m3", "activity_trend",
    "completion_ratio", "unique_songs_m1", "months_since_active",
]
CATEGORICAL_FEATURES = ["payment_method", "city", "registered_via", "gender"]
FEATURES = NUMERIC_FEATURES + CATEGORICAL_FEATURES


def visible(ds: Dataset, cutoff) -> Dataset:
    """The only door: drop everything dated after the cut-off."""
    cutoff = month_end(cutoff)
    tx = ds.tx[ds.tx["transaction_date"] <= cutoff]
    logs = ds.logs[ds.logs["month_end"] <= cutoff]
    # Member profile has no change dates; we only keep people registered by then.
    members = ds.members[
        ds.members["registration_init_time"].isna()
        | (ds.members["registration_init_time"] <= cutoff)
    ]
    return Dataset(tx=tx, members=members, logs=logs)


def build_features(ds: Dataset, cutoff, users) -> pd.DataFrame:
    """Feature table for `users` as of `cutoff`. Index = msno, columns = FEATURES."""
    cutoff = month_end(cutoff)
    v = visible(ds, cutoff)
    users = pd.Index(users, name="msno")
    out = pd.DataFrame(index=users)

    # --- latest known transaction -------------------------------------
    last = known_status(v.tx, cutoff).reindex(users)
    out["plan_days"] = last["payment_plan_days"]
    out["list_price"] = last["plan_list_price"]
    out["amount_paid"] = last["actual_amount_paid"]
    out["discount"] = last["plan_list_price"] - last["actual_amount_paid"]
    out["auto_renew"] = last["is_auto_renew"]
    out["cancel_last"] = last["is_cancel"]
    out["days_to_expire"] = (last["membership_expire_date"] - cutoff).dt.days
    out["payment_method"] = last["payment_method_id"]
    out["days_since_last_tx"] = (cutoff - last["transaction_date"]).dt.days

    # --- transaction history ------------------------------------------
    tx = v.tx[v.tx["msno"].isin(users)]
    g = tx.groupby("msno")
    out["n_tx"] = g.size()
    out["n_tx_180d"] = tx[tx["transaction_date"] > cutoff - pd.Timedelta(days=180)].groupby("msno").size()
    out["n_cancel"] = g["is_cancel"].sum()
    out["n_cancel_365d"] = tx[tx["transaction_date"] > cutoff - pd.Timedelta(days=365)].groupby("msno")["is_cancel"].sum()
    out["tenure_days"] = (cutoff - g["transaction_date"].min()).dt.days
    out["auto_renew_share"] = g["is_auto_renew"].mean()
    out["n_plans"] = g["payment_plan_days"].nunique()
    out["avg_paid"] = g["actual_amount_paid"].mean()
    out["n_lapses"] = _count_lapses(tx).reindex(users)
    for c in ("n_tx", "n_tx_180d", "n_cancel", "n_cancel_365d", "n_lapses"):
        out[c] = out[c].fillna(0)

    # --- profile --------------------------------------------------------
    m = v.members.set_index("msno").reindex(users)
    age = pd.to_numeric(m.get("bd"), errors="coerce")
    out["age"] = age.where((age >= 10) & (age <= 80))
    out["reg_days"] = (cutoff - m["registration_init_time"]).dt.days
    out["city"] = m.get("city")
    out["registered_via"] = m.get("registered_via")
    out["gender"] = m.get("gender")

    # --- listening logs (whole months up to the cut-off) ---------------
    lg = v.logs[v.logs["msno"].isin(users)]
    m1 = lg[lg["month_end"] == cutoff].set_index("msno")
    m3 = lg[lg["month_end"] > add_months(cutoff, -3)].groupby("msno")
    out["days_active_m1"] = m1["days_active"].reindex(users).fillna(0)
    out["secs_m1"] = m1["total_secs"].reindex(users).fillna(0)
    out["unique_songs_m1"] = m1["num_unq"].reindex(users).fillna(0)
    out["days_active_m3"] = m3["days_active"].sum().reindex(users).fillna(0) / 3
    out["secs_m3"] = m3["total_secs"].sum().reindex(users).fillna(0) / 3
    out["activity_trend"] = (out["days_active_m1"] + 1) / (out["days_active_m3"] + 1)
    plays = m3[["num_25", "num_50", "num_75", "num_985", "num_100"]].sum()
    total = plays.sum(axis=1)
    out["completion_ratio"] = (plays["num_100"] / total.replace(0, np.nan)).reindex(users)
    last_active = lg.groupby("msno")["month_end"].max().reindex(users)
    months = (cutoff.year - last_active.dt.year) * 12 + (cutoff.month - last_active.dt.month)
    out["months_since_active"] = months.fillna(99)

    for c in CATEGORICAL_FEATURES:
        out[c] = out[c].astype("string").fillna("missing").astype("category")
    for c in NUMERIC_FEATURES:
        out[c] = pd.to_numeric(out[c], errors="coerce").astype("float64")
    return out[FEATURES]


def _count_lapses(tx: pd.DataFrame) -> pd.Series:
    """How often a customer let a membership run out for 30+ days, then came back."""
    t = tx[tx["is_cancel"] == 0].sort_values(["msno", "transaction_date"])
    prev_exp = t.groupby("msno")["membership_expire_date"].shift()
    gap = (t["transaction_date"] - prev_exp).dt.days
    return (gap > 30).groupby(t["msno"]).sum()


def build_training_table(ds: Dataset, cutoffs, label_tx: pd.DataFrame | None = None) -> pd.DataFrame:
    """Stack features + churn label for several cut-off dates.

    `label_tx` lets the caller restrict which transactions may be used for
    the answer key (the replay job passes only data up to "today").
    """
    from .labels import churn_labels, scoring_population

    label_tx = ds.tx if label_tx is None else label_tx
    frames = []
    for c in cutoffs:
        c = month_end(c)
        pop = scoring_population(ds.tx, c)
        if pop.empty:
            continue
        X = build_features(ds, c, pop.index)
        X["is_churn"] = churn_labels(label_tx, c, pop).reindex(X.index).values
        X["cutoff"] = c
        frames.append(X)
    if not frames:
        return pd.DataFrame(columns=[*FEATURES, "is_churn", "cutoff"])
    df = pd.concat(frames)
    for c in CATEGORICAL_FEATURES:
        df[c] = df[c].astype("string").astype("category")
    return df
