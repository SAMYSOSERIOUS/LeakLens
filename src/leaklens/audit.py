"""The leakage autopsy: steps 1-6 of the project in one run.

    1-3  each public recipe: as published -> clean features -> honest time split,
         plus a per-feature check of which inputs leaked the future
    4    our own honest models: nobody-churns, auto-renew rule, logistic, LightGBM
    5-6  euros: pick the threshold that saves the most, compare with 0.5

Time layout (all month-end cut-offs):
    data ends ............................................ D
    test cut-off T = latest month whose answers are known by D
    validation V = latest month whose answers were known on T
    training     = the N months before V
Nothing that is unknowable on T is used to train or tune - including the
answer key: training months are only those whose churn outcome was already
settled on T.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score
from sklearn.model_selection import train_test_split

from . import money
from .data import Dataset
from .dates import fmt, is_mature, month_end, month_ends_between
from .features import build_training_table, visible
from .labels import churn_labels, scoring_population
from .models import fit_predict_all, make_lgbm, prep_X, scores
from .recipes import RECIPES, Recipe


def pick_cutoffs(ds: Dataset, n_train: int = 4) -> dict:
    end = ds.data_end
    first = month_end(ds.tx["transaction_date"].min())
    months = month_ends_between(first, end)
    test = max(c for c in months if is_mature(c, end))
    known_at_test = [c for c in months if c < test and is_mature(c, test)]
    valid = known_at_test[-1]
    train = known_at_test[-1 - n_train:-1]
    return {"data_end": end, "test": test, "valid": valid, "train": train}


def _binary(y, p) -> dict:
    y = np.asarray(y)
    p = np.asarray(p, dtype=float)
    hard = (p >= 0.5).astype(int)
    return {
        "auc": round(float(roc_auc_score(y, p)), 4),
        "accuracy": round(float(accuracy_score(y, hard)), 4),
        "f1": round(float(f1_score(y, hard, zero_division=0)), 4),
        "precision": round(float(precision_score(y, hard, zero_division=0)), 4),
        "recall": round(float(recall_score(y, hard, zero_division=0)), 4),
    }


def _fit(recipe: Recipe, X, y, seed, max_rows, rng):
    X, y = X.reset_index(drop=True), pd.Series(np.asarray(y))
    if len(X) > max_rows:
        idx = rng.choice(len(X), max_rows, replace=False)
        X, y = X.iloc[idx], y.iloc[idx]
    if recipe.oversample:
        pos = np.flatnonzero(y.values == 1)
        extra = rng.choice(pos, max(0, int((y == 0).sum()) - len(pos)), replace=True)
        X = pd.concat([X, X.iloc[extra]])
        y = pd.concat([y, y.iloc[extra]])
    return recipe.model(seed).fit(X, y.values)


def _signal(y, x) -> float:
    """How well one feature on its own separates leavers (0.5 = not at all)."""
    x = pd.Series(x).astype(float)
    if x.nunique(dropna=True) < 2:
        return 0.5
    a = roc_auc_score(y, x.fillna(x.median()))
    return round(float(max(a, 1 - a)), 4)


def audit_recipe(recipe: Recipe, ds: Dataset, cut: dict, seed: int = 0, max_rows: int = 300_000) -> dict:
    rng = np.random.default_rng(seed)
    T = cut["test"]
    pop = scoring_population(ds.tx, T)
    y = churn_labels(ds.tx, T, pop).reindex(pop.index)

    published = recipe.build(ds, pop.index)                 # whole file, as the notebook did
    clean = recipe.build(visible(ds, T), pop.index)         # only what was known on T

    idx_tr, idx_te = train_test_split(np.arange(len(pop)), test_size=0.3, stratify=y, random_state=42)
    res = {}
    for name, F in (("as_published", published), ("clean_features_random_split", clean)):
        m = _fit(recipe, F.iloc[idx_tr], y.iloc[idx_tr], seed, max_rows, rng)
        res[name] = _binary(y.iloc[idx_te], m.predict_proba(F.iloc[idx_te])[:, 1])

    # honest: learn from earlier months (answers known on T), predict month T
    frames, ys = [], []
    for c in [*cut["train"], cut["valid"]]:
        p_c = scoring_population(ds.tx, c)
        frames.append(recipe.build(visible(ds, c), p_c.index))
        ys.append(churn_labels(ds.tx, c, p_c).reindex(p_c.index))
    m = _fit(recipe, pd.concat(frames), pd.concat(ys), seed, max_rows, rng)
    res["honest_time_split"] = _binary(y, m.predict_proba(clean)[:, 1])

    # which features leaked? compare each one built from the whole file vs. from the past only
    leaks = []
    for col in published.columns:
        a, b = published[col].to_numpy(dtype=float), clean[col].to_numpy(dtype=float)
        changed = float(np.mean(~np.isclose(a, b, equal_nan=True)))
        leaks.append({
            "feature": col,
            "changed_share": round(changed, 4),
            "signal_published": _signal(y.values, a),
            "signal_honest": _signal(y.values, b),
            "leaked": changed > 0.005,
        })
    leaks.sort(key=lambda r: (-r["signal_published"] + r["signal_honest"]))
    return {
        "key": recipe.key, "title": recipe.title, "url": recipe.url,
        "claimed": recipe.claimed, "claimed_text": recipe.claimed_text, "notes": recipe.notes,
        "results": res, "leaks": leaks,
        "auc_drop": round(res["as_published"]["auc"] - res["honest_time_split"]["auc"], 4),
    }


def audit_models(ds: Dataset, cut: dict, a: money.Assumptions, seed: int = 0) -> dict:
    train = build_training_table(ds, cut["train"])
    valid = build_training_table(ds, [cut["valid"]])
    test = build_training_table(ds, [cut["test"]])

    p_valid = fit_predict_all(train, valid, seed)
    p_test = fit_predict_all(train, test, seed)
    yv, yt = valid["is_churn"].to_numpy(), test["is_churn"].to_numpy()

    table = {}
    for name in p_test:
        t_star, _ = money.best_threshold(yv, p_valid[name], a)
        table[name] = {
            **scores(yt, p_test[name]),
            "money_threshold": t_star,
            "saved_eur_money_threshold": round(money.savings(yt, p_test[name], t_star, a), 2),
            "saved_eur_default_0_5": round(money.savings(yt, p_test[name], 0.5, a), 2),
        }

    best = "LightGBM"
    t_star, _ = money.best_threshold(yv, p_valid[best], a)
    t_hindsight, s_hindsight = money.best_threshold(yt, p_test[best], a)

    Xtr, cats = prep_X(train)
    gbm = make_lgbm(seed).fit(Xtr, train["is_churn"].to_numpy())
    imp = pd.Series(gbm.booster_.feature_importance("gain"), index=Xtr.columns)
    imp = (imp / imp.sum()).sort_values(ascending=False).head(10).round(4)

    return {
        "rows": {"train": int(len(train)), "valid": int(len(valid)), "test": int(len(test))},
        "table": table,
        "money": {
            "model": best,
            "threshold_chosen_on_validation": t_star,
            "break_even_chance": round(a.break_even, 4),
            "comparison": money.money_table(yt, p_test[best], a, t_star),
            "hindsight_best_threshold": t_hindsight,
            "hindsight_best_saved_eur": round(s_hindsight, 2),
        },
        "curves": {
            "valid": money.binned_counts(yv, p_valid[best]),
            "test": money.binned_counts(yt, p_test[best]),
        },
        "top_features": imp.to_dict(),
    }


def run_audit(ds: Dataset, a: money.Assumptions, is_sample: bool, n_train: int = 4,
              seed: int = 0, max_rows: int = 300_000) -> dict:
    cut = pick_cutoffs(ds, n_train)
    print(f"test month {fmt(cut['test'])}, validation {fmt(cut['valid'])}, "
          f"training {[fmt(c) for c in cut['train']]}")
    recipes = []
    for r in RECIPES:
        print(f"recipe {r.key}: {r.title}")
        recipes.append(audit_recipe(r, ds, cut, seed, max_rows))
    print("own models")
    models = audit_models(ds, cut, a, seed)
    return {
        "is_sample": is_sample,
        "data": {
            "first_date": fmt(ds.tx["transaction_date"].min()),
            "last_date": fmt(cut["data_end"]),
            "n_users": int(ds.tx["msno"].nunique()),
            "n_transactions": int(len(ds.tx)),
        },
        "cutoffs": {"test": fmt(cut["test"]), "valid": fmt(cut["valid"]),
                    "train": [fmt(c) for c in cut["train"]]},
        "assumptions": {"offer_cost": a.offer_cost, "customer_value": a.customer_value,
                        "save_rate": a.save_rate},
        "recipes": recipes,
        "models": models,
    }


def save(result: dict, *paths: str | Path) -> None:
    for p in paths:
        p = Path(p)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps(result, indent=2, default=str))
