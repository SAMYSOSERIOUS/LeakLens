"""Pretend the business is live: replay history one month per run.

Each run of `step()` is one "week" of the scheduled job and moves the
clock forward one month:

  1. today = next month end. The job may only see data up to today
     (everything goes through features.visible()).
  2. Check old predictions whose answers are now known (about two months
     later - customers get 30 days to renew) and record how accurate they
     were.
  3. Retrain rule: if the CURRENT model's latest checked accuracy (AUC) is
     below the floor in config.toml, retrain on the newest months whose
     answers are known, and log it.
  4. Score every customer whose membership runs out next month, and list
     who should get an offer (chance of leaving above the money threshold).
  5. Drift check: compare this month's customers with the training
     customers, feature by feature (PSI score).
  6. Write site/data/monitor.json for the web page.

When the clock passes the end of the data, the replay starts over.
State lives in state/ so a fresh machine (e.g. a GitHub Actions runner)
can pick up exactly where the last run stopped.
"""

from __future__ import annotations

import json
import shutil
from dataclasses import dataclass
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd

from . import money
from .data import Dataset
from .dates import add_months, fmt, is_mature, label_known_on, month_end, month_ends_between
from .features import NUMERIC_FEATURES, build_features, build_training_table, visible
from .labels import churn_labels, scoring_population
from .models import make_lgbm, prep_X, scores

# Not included on purpose: features that grow for everyone every month
# (time as a customer, number of payments). They "drift" by design and
# would raise a false alarm every single run.
DRIFT_FEATURES = [
    "auto_renew", "amount_paid", "discount", "plan_days", "n_lapses",
    "days_active_m1", "secs_m1", "completion_ratio", "age",
]
FEATURE_LABELS = {
    "auto_renew": "Auto-renew switched on",
    "amount_paid": "Price paid",
    "discount": "Discount received",
    "plan_days": "Plan length",
    "tenure_days": "Time as a customer",
    "n_lapses": "Past lapses",
    "days_active_m1": "Listening days last month",
    "secs_m1": "Listening time last month",
    "completion_ratio": "Songs played to the end",
    "age": "Age",
    "payment_method": "Payment method",
}


@dataclass
class ReplayConfig:
    start: str = "2016-01-31"
    train_months: int = 4
    retrain_below_auc: float = 0.80
    drift_psi_warn: float = 0.20
    offer_list_size: int = 200
    state_dir: str = "state"
    site_data_dir: str = "site/data"


# ---------------------------------------------------------------- drift
def _psi(ref: np.ndarray, cur: np.ndarray) -> float:
    eps = 1e-4
    ref, cur = np.clip(ref, eps, None), np.clip(cur, eps, None)
    return float(np.sum((cur - ref) * np.log(cur / ref)))


def drift_reference(X: pd.DataFrame) -> dict:
    ref = {}
    for f in DRIFT_FEATURES:
        x = X[f].astype(float).fillna(-1)
        edges = np.unique(np.quantile(x, np.linspace(0, 1, 11)))
        shares = np.histogram(x, bins=np.r_[-np.inf, edges[1:-1], np.inf])[0] / len(x)
        ref[f] = {"edges": edges[1:-1].tolist(), "shares": shares.tolist()}
    pm = X["payment_method"].astype(str).value_counts(normalize=True)
    ref["payment_method"] = {"shares": pm.to_dict()}
    return ref


def drift_scores(ref: dict, X: pd.DataFrame) -> dict[str, float]:
    out = {}
    for f in DRIFT_FEATURES:
        x = X[f].astype(float).fillna(-1)
        cur = np.histogram(x, bins=np.r_[-np.inf, ref[f]["edges"], np.inf])[0] / max(len(x), 1)
        out[f] = round(_psi(np.array(ref[f]["shares"]), cur), 4)
    rs = pd.Series(ref["payment_method"]["shares"])
    cs = X["payment_method"].astype(str).value_counts(normalize=True)
    idx = rs.index.union(cs.index)
    out["payment_method"] = round(_psi(rs.reindex(idx).fillna(0).values, cs.reindex(idx).fillna(0).values), 4)
    return out


# ---------------------------------------------------------------- job
class ReplayJob:
    def __init__(self, ds: Dataset, cfg: ReplayConfig, a: money.Assumptions, seed: int = 0):
        self.ds, self.cfg, self.a, self.seed = ds, cfg, a, seed
        self.state_dir = Path(cfg.state_dir)
        self.pred_dir = self.state_dir / "predictions"
        self.site_dir = Path(cfg.site_data_dir)
        self.state_path = self.state_dir / "replay_state.json"
        self.model_path = self.state_dir / "model.txt"
        self.end = max(c for c in month_ends_between(cfg.start, ds.data_end) if c <= ds.data_end)

    # ---- state ----
    def load_state(self) -> dict | None:
        return json.loads(self.state_path.read_text()) if self.state_path.exists() else None

    def save_state(self, s: dict) -> None:
        self.state_dir.mkdir(parents=True, exist_ok=True)
        self.state_path.write_text(json.dumps(s, indent=1, default=str))

    # ---- training ----
    def train(self, today: pd.Timestamp, s: dict, reason: str) -> None:
        """Fit LightGBM on months whose answers are known today."""
        seen = visible(self.ds, today)
        months = month_ends_between(add_months(today, -24), today)
        known = [c for c in months if c < today and is_mature(c, today)]
        valid, train = known[-1], known[-1 - self.cfg.train_months:-1]
        tr = build_training_table(seen, train)
        va = build_training_table(seen, [valid])
        Xtr, cats = prep_X(tr)
        Xva, _ = prep_X(va, cats)
        m = make_lgbm(self.seed).fit(Xtr, tr["is_churn"].to_numpy())
        p_va = m.predict_proba(Xva)[:, 1]
        threshold, _ = money.best_threshold(va["is_churn"].to_numpy(), p_va, self.a)
        valid_auc = scores(va["is_churn"], p_va).get("auc")
        # final model also learns from the validation month
        full = pd.concat([tr, va])
        Xfull, cats = prep_X(full)
        m = make_lgbm(self.seed).fit(Xfull, full["is_churn"].to_numpy())
        self.state_dir.mkdir(parents=True, exist_ok=True)
        m.booster_.save_model(str(self.model_path))

        s["model_version"] = s.get("model_version", 0) + 1
        s["model"] = {
            "version": s["model_version"], "trained_on": fmt(today),
            "months": [fmt(c) for c in [*train, valid]], "threshold": threshold,
            "valid_auc": valid_auc, "categories": cats,
        }
        s["reference"] = drift_reference(Xfull)
        s["events"].append({"date": fmt(today), "loop": s["loop"], "type": "retrain",
                            "model_version": s["model_version"], "reason": reason,
                            "valid_auc": valid_auc, "threshold": threshold})

    def predict(self, s: dict, X: pd.DataFrame) -> np.ndarray:
        booster = lgb.Booster(model_file=str(self.model_path))
        Xp, _ = prep_X(X, s["model"]["categories"])
        return booster.predict(Xp)

    # ---- one tick ----
    def step(self) -> dict:
        s = self.load_state()
        if s is None:
            today = month_end(self.cfg.start)
            s = {"loop": 1, "today": fmt(today), "events": [], "history": [], "drift": [], "pending": []}
            self.train(today, s, "first model")
        else:
            today = add_months(s["today"], 1)
            if today > self.end:
                today = month_end(self.cfg.start)
                s["loop"] += 1
                s["pending"], s["drift"] = [], []
                s["history"] = [h for h in s["history"] if h["loop"] >= s["loop"] - 1]
                shutil.rmtree(self.pred_dir, ignore_errors=True)
                s["events"].append({"date": fmt(today), "loop": s["loop"], "type": "restart",
                                    "reason": "reached the end of the data; replay starts over"})
                self.train(today, s, "replay restarted")
            s["today"] = fmt(today)

        seen = visible(self.ds, today)

        # 2. how did earlier predictions turn out?
        newly_checked = []
        for c in list(s["pending"]):
            if not is_mature(c, today):
                continue
            f = self.pred_dir / f"{c}.csv.gz"
            pred = pd.read_csv(f, index_col="msno")
            y = churn_labels(seen.tx, c, pred[["expire_date"]].assign(
                expire_date=pd.to_datetime(pred["expire_date"]))).reindex(pred.index)
            sc = scores(y, pred["p"])
            h = next(h for h in s["history"] if h["cutoff"] == c and h["loop"] == s["loop"])
            h.update({
                "checked_on": fmt(today), **{k: sc.get(k) for k in ("auc", "log_loss", "churn_rate")},
                "saved_eur": round(money.savings(y, pred["p"], h["threshold"], self.a), 2),
                "leavers": int(y.sum()),
                "leavers_reached": int(((pred["p"] >= h["threshold"]) & (y == 1)).sum()),
            })
            newly_checked.append(h)
            s["pending"].remove(c)
            f.unlink()

        # 3. retrain rule
        current = [h for h in newly_checked if h["model_version"] == s["model_version"]]
        if current and current[-1]["auc"] is not None and current[-1]["auc"] < self.cfg.retrain_below_auc:
            self.train(today, s, f"accuracy (AUC) {current[-1]['auc']:.3f} on {current[-1]['cutoff']} "
                                 f"fell below the floor of {self.cfg.retrain_below_auc:.2f}")

        # 4. score this month's customers
        pop = scoring_population(seen.tx, today)
        X = build_features(seen, today, pop.index)
        p = self.predict(s, X)
        th = s["model"]["threshold"]
        self.pred_dir.mkdir(parents=True, exist_ok=True)
        pd.DataFrame({"p": p, "expire_date": pop["expire_date"].dt.strftime("%Y-%m-%d")},
                     index=pop.index).to_csv(self.pred_dir / f"{fmt(today)}.csv.gz")
        s["pending"].append(fmt(today))
        s["history"].append({
            "loop": s["loop"], "cutoff": fmt(today), "answers_known_on": fmt(label_known_on(today)),
            "model_version": s["model_version"], "threshold": th, "n_scored": int(len(p)),
            "n_offers": int((p >= th).sum()), "expected_leavers": round(float(p.sum()), 1),
            "auc": None, "log_loss": None, "churn_rate": None, "saved_eur": None,
        })

        # 5. drift
        d = drift_scores(s["reference"], X)
        top = sorted(d.items(), key=lambda kv: -kv[1])
        s["drift"].append({"date": fmt(today), "max_psi": top[0][1], "top": top[:5],
                           "warning": top[0][1] >= self.cfg.drift_psi_warn})

        self.save_state(s)
        self.write_site(s, today, pop, X, p)
        return s

    # ---- web page data ----
    def write_site(self, s: dict, today, pop, X, p) -> None:
        a = self.a
        ev = p * a.save_rate * a.customer_value - a.offer_cost
        offers = pd.DataFrame({
            "customer": [m[:12] for m in X.index],
            "chance": np.round(p, 3),
            "expected_gain_eur": np.round(ev, 2),
            "expires": pop["expire_date"].dt.strftime("%Y-%m-%d").values,
            "auto_renew": X["auto_renew"].fillna(0).astype(int).values,
            "listening_days_last_month": X["days_active_m1"].astype(int).values,
        })
        offers = offers[p >= s["model"]["threshold"]].sort_values("chance", ascending=False)
        loop_hist = [h for h in s["history"] if h["loop"] == s["loop"]]
        monitor = {
            "today": fmt(today), "loop": s["loop"], "replay_start": self.cfg.start,
            "replay_end": fmt(self.end),
            "settings": {"retrain_below_auc": self.cfg.retrain_below_auc,
                         "drift_psi_warn": self.cfg.drift_psi_warn,
                         "offer_cost": a.offer_cost, "customer_value": a.customer_value,
                         "save_rate": a.save_rate},
            "model": {k: v for k, v in s["model"].items() if k != "categories"},
            "history": loop_hist,
            "events": s["events"][-30:],
            "drift_latest": s["drift"][-1],
            "drift_series": [{"date": d["date"], "max_psi": d["max_psi"]} for d in s["drift"]],
            "feature_labels": FEATURE_LABELS,
            "offers": {"count": int(len(offers)), "total_scored": int(len(p)),
                       "expected_gain_eur": round(float(offers["expected_gain_eur"].sum()), 2),
                       "top": offers.head(self.cfg.offer_list_size).to_dict(orient="records")},
        }
        self.site_dir.mkdir(parents=True, exist_ok=True)
        (self.site_dir / "monitor.json").write_text(json.dumps(monitor, indent=1, default=str))
