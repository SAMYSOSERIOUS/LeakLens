import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from leaklens.data import Dataset, load_prepared, prepare  # noqa: E402
from leaklens.sample import write_sample  # noqa: E402


@pytest.fixture(scope="session")
def ds(tmp_path_factory) -> Dataset:
    """A small made-up dataset, written as KKBox CSVs and loaded the real way."""
    root = tmp_path_factory.mktemp("kkbox")
    write_sample(root / "raw", n_users=4000, seed=11)
    prepare(root / "raw", root / "prepared")
    return load_prepared(root / "prepared")


def scramble_future(ds: Dataset, cutoff, seed: int = 0) -> Dataset:
    """Same past, nonsense future.

    Everything dated after the cut-off is altered and extra fake rows are
    added: a cancel and a renewal for every customer, and wild listening
    numbers. An honest feature must not change at all.
    """
    rng = np.random.default_rng(seed)
    cutoff = pd.Timestamp(cutoff)
    tx = ds.tx.copy()
    fut = tx["transaction_date"] > cutoff
    tx.loc[fut, "actual_amount_paid"] = rng.integers(0, 2000, fut.sum()).astype(tx["actual_amount_paid"].dtype)
    tx.loc[fut, "is_cancel"] = (1 - tx.loc[fut, "is_cancel"]).astype(tx["is_cancel"].dtype)
    tx.loc[fut, "membership_expire_date"] += pd.to_timedelta(rng.integers(1, 400, fut.sum()), unit="D")
    users = tx["msno"].unique()
    fake = pd.DataFrame({
        "msno": users, "payment_method_id": 99, "payment_plan_days": 999, "plan_list_price": 1,
        "actual_amount_paid": 1, "is_auto_renew": 1,
        "transaction_date": cutoff + pd.Timedelta(days=1),
        "membership_expire_date": cutoff + pd.Timedelta(days=900), "is_cancel": 1,
    })
    tx = pd.concat([tx, fake.astype({c: tx[c].dtype for c in fake.columns})], ignore_index=True)

    logs = ds.logs.copy()
    lf = logs["month_end"] > cutoff
    for c in ("days_active", "total_secs", "num_100", "num_unq"):
        logs[c] = logs[c].astype(float)
        logs.loc[lf, c] = logs.loc[lf, c] * 50 + 7
    return Dataset(tx=tx, members=ds.members.copy(), logs=logs)
