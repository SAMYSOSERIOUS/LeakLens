"""Make a small, made-up dataset in the exact KKBox file format.

Why: the real KKBox files are several GB and can't live in a GitHub repo,
but the tests, the CI pipeline and the weekly demo job all need data. This
generator writes transactions.csv, members_v3.csv and user_logs.csv with
the same columns and date formats as the real ones, so every line of the
pipeline runs exactly as it would on real data.

The numbers it produces are DEMO numbers. They are not KKBox results.

Built-in behaviour (so the audit and monitor have something to find):
  * Manual renewers churn more than auto-renewers; less active listeners
    churn more; people who cancelled or lapsed before churn more.
  * Customers who are about to leave listen less in their final month
    (an honest early-warning sign).
  * Auto-renew customers who leave usually file a cancel transaction a few
    days before their membership runs out, and renewals create a new
    transaction - the classic source of leakage when features are built
    from the whole file.
  * From September 2016 a new partner promotion brings in customers on a
    free first period (payment method 34). Many leave once they must pay.
    A model trained before that date has never seen them: this creates
    data drift and an accuracy drop the monitor should catch.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

START = pd.Timestamp("2015-01-01")
END = pd.Timestamp("2017-03-31")
PROMO_START = pd.Timestamp("2016-09-01")


def _sigmoid(z):
    return 1.0 / (1.0 + np.exp(-z))


def generate(n_users: int = 8000, seed: int = 7) -> dict[str, pd.DataFrame]:
    rng = np.random.default_rng(seed)
    months = pd.date_range(START, END, freq="ME")
    n_months = len(months)

    tx_rows, log_rows, members = [], [], []
    for u in range(n_users):
        msno = f"u{u:06d}" + "".join(rng.choice(list("abcdef0123456789"), 8))
        r = rng.random()
        promo = False
        if r < 0.35:      # customers who were already there
            start = START + pd.Timedelta(days=int(rng.integers(0, 31)))
            reg = start - pd.Timedelta(days=int(rng.integers(0, 1500)))
        elif r < 0.82:    # steady trickle of new customers
            start = START + pd.Timedelta(days=int(rng.integers(31, (pd.Timestamp("2017-02-28") - START).days)))
            reg = start - pd.Timedelta(days=int(rng.integers(0, 30)))
        else:             # the partner promotion wave, Sep 2016 - Jan 2017
            start = PROMO_START + pd.Timedelta(days=int(rng.integers(0, 150)))
            reg = start - pd.Timedelta(days=int(rng.integers(0, 10)))
            promo = True

        plan = int(rng.choice([30, 7, 90, 180, 410], p=[0.85, 0.03, 0.05, 0.03, 0.04]))
        if promo:
            plan = 30
        auto = bool(rng.random() < (0.78 if plan == 30 else 0.35)) or promo
        if promo:
            method = 34
        elif auto:
            method = int(rng.choice([41, 40, 36, 39], p=[0.6, 0.2, 0.1, 0.1]))
        else:
            method = int(rng.choice([38, 37, 32, 30], p=[0.4, 0.3, 0.2, 0.1]))
        list_price = {30: 149, 7: 0, 90: 447, 180: 894, 410: 1788}[plan]
        discounted = rng.random() < 0.1
        base_eng = rng.normal(0.0, 1.0)

        members.append({
            "msno": msno,
            "city": int(rng.choice([1, 4, 5, 13, 22, 15, 6], p=[0.45, 0.1, 0.1, 0.12, 0.08, 0.08, 0.07])),
            "bd": 0 if rng.random() < 0.55 else int(rng.integers(15, 60)),
            "gender": rng.choice(["", "male", "female"], p=[0.55, 0.23, 0.22]),
            "registered_via": int(rng.choice([7, 9, 3, 4, 13], p=[0.45, 0.25, 0.15, 0.1, 0.05])),
            "registration_init_time": int(reg.strftime("%Y%m%d")),
        })

        # engagement path, one value per calendar month
        eng = np.empty(n_months)
        e = base_eng
        for k in range(n_months):
            e = 0.8 * e + 0.2 * base_eng + rng.normal(0, 0.45)
            eng[k] = e
        active_months = np.zeros(n_months, dtype=bool)
        leaving_month = np.zeros(n_months, dtype=bool)

        def midx(d):
            return (d.year - START.year) * 12 + d.month - 1

        t, expire, period, trouble = start, start + pd.Timedelta(days=plan), 0, 0.0
        paid = 0 if promo else (int(list_price * 0.8) if discounted else list_price)
        tx_rows.append((msno, method, plan, list_price, paid, int(auto), t, expire, 0))
        while True:
            for k in range(max(midx(t), 0), min(midx(expire), n_months - 1) + 1):
                active_months[k] = True
            if expire > END:
                break
            ke = min(midx(expire), n_months - 1)
            z = (-2.9 + 2.3 * (not auto) - 0.85 * eng[ke] + 0.9 * (plan == 7)
                 - 0.4 * (plan >= 90) + trouble)
            if promo:
                # the first paid month is due at the end of period 2
                z += {0: -0.5, 1: 0.0, 2: 3.4}.get(period, 0.9)
            churn = rng.random() < _sigmoid(z)
            if churn:
                # promo leavers just let the free card payment fail: no cancel, no slowdown
                leaving_month[ke] = not promo
                if auto and not promo and rng.random() < 0.85:
                    cdate = expire - pd.Timedelta(days=int(rng.integers(1, 21)))
                    tx_rows.append((msno, method, plan, list_price, 0, 0, cdate, expire, 1))
                if rng.random() < 0.3:  # comes back later
                    t = expire + pd.Timedelta(days=int(rng.integers(45, 300)))
                    if t > END:
                        break
                    trouble += 0.6
                    auto = rng.random() < 0.5
                    method = int(rng.choice([41, 38, 40, 37]))
                    expire = t + pd.Timedelta(days=plan)
                    period += 1
                    paid = list_price
                    tx_rows.append((msno, method, plan, list_price, paid, int(auto), t, expire, 0))
                    continue
                break
            # renews
            if auto:
                t = expire - pd.Timedelta(days=int(rng.integers(0, 2)))
                new_exp = expire + pd.Timedelta(days=plan)
            else:
                t = expire + pd.Timedelta(days=int(rng.integers(-10, 21)))
                new_exp = max(expire, t) + pd.Timedelta(days=plan)
            if t > END:
                break
            period += 1
            if promo and period >= 3:
                paid = list_price
            trouble = max(0.0, trouble - 0.1)
            expire = new_exp
            tx_rows.append((msno, method, plan, list_price, paid, int(auto), t, expire, 0))

        # monthly listening -> daily rows
        for k in np.flatnonzero(active_months):
            p_act = _sigmoid(-0.3 + 1.1 * eng[k]) * (0.35 if leaving_month[k] else 1.0)
            days_in = months[k].day
            n_days = int(rng.binomial(days_in, p_act * 0.6))
            if n_days == 0:
                continue
            days = rng.choice(np.arange(1, days_in + 1), size=n_days, replace=False)
            songs = rng.poisson(np.exp(3.0 + 0.3 * eng[k]), size=n_days) + 1
            frac_full = np.clip(0.55 + 0.1 * eng[k] + rng.normal(0, 0.05, n_days), 0.05, 0.95)
            n100 = rng.binomial(songs, frac_full)
            rest = songs - n100
            n25 = rng.binomial(rest, 0.5)
            n50 = rng.binomial(rest - n25, 0.4)
            n75 = rest - n25 - n50
            for j, d in enumerate(days):
                log_rows.append((msno, int(months[k].strftime("%Y%m")) * 100 + int(d), int(n25[j]), int(n50[j]),
                                 int(n75[j]), 0, int(n100[j]), int(songs[j] * 0.9) + 1,
                                 float(songs[j] * rng.uniform(180, 240))))

    tx = pd.DataFrame(tx_rows, columns=[
        "msno", "payment_method_id", "payment_plan_days", "plan_list_price", "actual_amount_paid",
        "is_auto_renew", "transaction_date", "membership_expire_date", "is_cancel"])
    tx = tx[tx["transaction_date"] <= END]
    for c in ("transaction_date", "membership_expire_date"):
        tx[c] = tx[c].dt.strftime("%Y%m%d").astype(int)
    logs = pd.DataFrame(log_rows, columns=["msno", "date", "num_25", "num_50", "num_75", "num_985",
                                           "num_100", "num_unq", "total_secs"])
    return {"transactions.csv": tx, "members_v3.csv": pd.DataFrame(members), "user_logs.csv": logs}


def write_sample(raw_dir: str | Path, n_users: int = 8000, seed: int = 7) -> None:
    raw = Path(raw_dir)
    raw.mkdir(parents=True, exist_ok=True)
    for name, df in generate(n_users, seed).items():
        df.to_csv(raw / name, index=False)
        print(f"wrote {raw / name} ({len(df):,} rows)")
