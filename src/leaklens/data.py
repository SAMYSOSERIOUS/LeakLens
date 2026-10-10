"""Load the KKBox files and turn them into three compact tables.

Raw KKBox files (from the WSDM 2018 challenge, mirrored on Mendeley Data):
    transactions.csv, transactions_v2.csv   -> one row per payment / cancel
    members_v3.csv (or members.csv)          -> one row per user
    user_logs.csv, user_logs_v2.csv          -> one row per user per day (~30 GB)
    train.csv, train_v2.csv                  -> official labels (NOT used for
                                                training here: we rebuild labels
                                                ourselves for every month)

`prepare()` reads those once and writes small parquet files:
    transactions.parquet   dates parsed, duplicates removed
    members.parquet
    logs_monthly.parquet   daily listening logs summed per user per month

Monthly log totals are safe because every cut-off date is a month end, so
"everything up to the cut-off" is always a set of whole months.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

TX_COLS = [
    "msno", "payment_method_id", "payment_plan_days", "plan_list_price",
    "actual_amount_paid", "is_auto_renew", "transaction_date",
    "membership_expire_date", "is_cancel",
]
MEMBER_COLS = ["msno", "city", "bd", "gender", "registered_via", "registration_init_time"]
LOG_NUM_COLS = ["num_25", "num_50", "num_75", "num_985", "num_100", "num_unq", "total_secs"]


@dataclass
class Dataset:
    tx: pd.DataFrame        # transactions
    members: pd.DataFrame   # one row per user
    logs: pd.DataFrame      # user x month listening totals

    @property
    def data_end(self) -> pd.Timestamp:
        """The last date for which we have any transaction."""
        return self.tx["transaction_date"].max()


def _parse_kkbox_date(s: pd.Series) -> pd.Series:
    """KKBox stores dates as integers like 20170131."""
    return pd.to_datetime(s.astype("Int64").astype(str), format="%Y%m%d", errors="coerce")


def _clean_tx(tx: pd.DataFrame) -> pd.DataFrame:
    tx = tx[TX_COLS].copy()
    for c in ("transaction_date", "membership_expire_date"):
        if not np.issubdtype(tx[c].dtype, np.datetime64):
            tx[c] = _parse_kkbox_date(tx[c])
    tx = tx.dropna(subset=["transaction_date", "membership_expire_date"])
    for c in ("payment_method_id", "payment_plan_days", "plan_list_price",
              "actual_amount_paid", "is_auto_renew", "is_cancel"):
        tx[c] = pd.to_numeric(tx[c], errors="coerce").fillna(0).astype("int32")
    tx = tx.drop_duplicates()
    return tx.sort_values(["msno", "transaction_date", "membership_expire_date"]).reset_index(drop=True)


def _clean_members(m: pd.DataFrame) -> pd.DataFrame:
    m = m[[c for c in MEMBER_COLS if c in m.columns]].copy()
    if not np.issubdtype(m["registration_init_time"].dtype, np.datetime64):
        m["registration_init_time"] = _parse_kkbox_date(m["registration_init_time"])
    m["gender"] = m["gender"].fillna("unknown").astype(str)
    return m.drop_duplicates("msno").reset_index(drop=True)


def empty_members() -> pd.DataFrame:
    """A members table with the right columns and types but no rows."""
    return pd.DataFrame({
        "msno": pd.Series(dtype="object"), "city": pd.Series(dtype="float64"),
        "bd": pd.Series(dtype="float64"), "gender": pd.Series(dtype="object"),
        "registered_via": pd.Series(dtype="float64"),
        "registration_init_time": pd.Series(dtype="datetime64[ns]"),
    })


def aggregate_daily_logs(daily: pd.DataFrame) -> pd.DataFrame:
    """Sum daily listening rows into one row per user per month."""
    d = daily.copy()
    if not np.issubdtype(d["date"].dtype, np.datetime64):
        d["date"] = _parse_kkbox_date(d["date"])
    # total_secs in the raw data has absurd outliers; a day has 86,400 seconds.
    d["total_secs"] = d["total_secs"].clip(0, 86_400)
    d["month_end"] = d["date"] + pd.offsets.MonthEnd(0)
    g = d.groupby(["msno", "month_end"], sort=False)
    out = g[LOG_NUM_COLS].sum()
    out["days_active"] = g["date"].nunique()
    out["last_date"] = g["date"].max()
    return out.reset_index()


def _combine_monthly(parts: list[pd.DataFrame]) -> pd.DataFrame:
    if not parts:  # no listening logs: an empty table with the right types
        return pd.DataFrame({
            "msno": pd.Series(dtype="object"), "month_end": pd.Series(dtype="datetime64[ns]"),
            **{c: pd.Series(dtype="float64") for c in [*LOG_NUM_COLS, "days_active"]},
            "last_date": pd.Series(dtype="datetime64[ns]"),
        })
    df = pd.concat(parts, ignore_index=True)
    g = df.groupby(["msno", "month_end"], sort=False)
    out = g[LOG_NUM_COLS + ["days_active"]].sum()
    out["last_date"] = g["last_date"].max()
    return out.reset_index()


def prepare(raw_dir: str | Path, out_dir: str | Path, chunksize: int = 5_000_000) -> None:
    """Read raw KKBox CSVs once and write compact parquet tables."""
    raw, out = Path(raw_dir), Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    tx_files = [raw / f for f in ("transactions.csv", "transactions_v2.csv") if (raw / f).exists()]
    if not tx_files:
        raise FileNotFoundError(f"No transactions*.csv in {raw}")
    tx = _clean_tx(pd.concat([pd.read_csv(f) for f in tx_files], ignore_index=True))
    tx.to_parquet(out / "transactions.parquet", index=False)
    print(f"transactions: {len(tx):,} rows, {tx.msno.nunique():,} users")

    mfile = next((raw / f for f in ("members_v3.csv", "members.csv") if (raw / f).exists()), None)
    if mfile is None:
        # Optional: without it the profile features (age, city, sign-up date) stay empty.
        print(f"members: no members*.csv in {raw}, continuing without customer profiles")
        members = empty_members()
    else:
        members = _clean_members(pd.read_csv(mfile))
    members.to_parquet(out / "members.parquet", index=False)
    print(f"members: {len(members):,} rows")
    if len(tx) == 1_048_575:
        print("WARNING: exactly 1,048,575 transactions - the Excel row limit. "
              "This file was probably cut off by Excel; get a fresh copy and don't open it in Excel.")

    parts = []
    for f in ("user_logs.csv", "user_logs_v2.csv"):
        if not (raw / f).exists():
            continue
        for i, chunk in enumerate(pd.read_csv(raw / f, chunksize=chunksize)):
            parts.append(aggregate_daily_logs(chunk))
            if len(parts) >= 8:  # keep memory flat
                parts = [_combine_monthly(parts)]
            print(f"  {f}: chunk {i + 1} done")
    logs = _combine_monthly(parts)
    logs.to_parquet(out / "logs_monthly.parquet", index=False)
    print(f"logs_monthly: {len(logs):,} user-months")


def load_prepared(prepared_dir: str | Path) -> Dataset:
    p = Path(prepared_dir)
    tx = pd.read_parquet(p / "transactions.parquet")
    order = ["msno", "transaction_date", "membership_expire_date"]
    # prepare() writes the table sorted; check cheaply and remember it, so
    # every later "latest transaction per customer" can skip a full sort.
    if len(tx) > 1:
        m = tx["msno"].to_numpy()
        t = tx["transaction_date"].to_numpy()
        e = tx["membership_expire_date"].to_numpy()
        same_m, same_t = m[1:] == m[:-1], t[1:] == t[:-1]
        bad = (m[1:] < m[:-1]) | (same_m & (t[1:] < t[:-1])) | (same_m & same_t & (e[1:] < e[:-1]))
        if bad.any():
            tx = tx.sort_values(order, kind="mergesort").reset_index(drop=True)
    tx.attrs["sorted_by_customer_and_date"] = True
    members = pd.read_parquet(p / "members.parquet")
    logs_path = p / "logs_monthly.parquet"
    logs = pd.read_parquet(logs_path) if logs_path.exists() else _combine_monthly([])
    return Dataset(tx=tx, members=members, logs=logs)
