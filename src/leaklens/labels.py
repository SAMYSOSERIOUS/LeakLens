"""Who do we score on a given cut-off date, and did they churn?

The rule (same as the WSDM 2018 challenge):
  * On cut-off date C we look at customers whose membership, as far as we
    know on C, runs out during the next month.
  * Such a customer has churned if no new paid subscription starts within
    30 days after their membership runs out.

The list of customers uses only transactions dated on or before C.
The answer (churned or not) uses transactions after C - that is the whole
point: it's the future we are trying to predict.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .dates import CHURN_GRACE_DAYS, add_months, month_end


def known_status(tx: pd.DataFrame, cutoff) -> pd.DataFrame:
    """Each customer's latest transaction as known on the cut-off date."""
    cutoff = month_end(cutoff)
    seen = tx[tx["transaction_date"] <= cutoff]
    if not tx.attrs.get("sorted_by_customer_and_date"):
        # load_prepared() marks data that is already in this order, so the
        # 23-million-row sort is only needed for hand-made test data.
        seen = seen.sort_values(["msno", "transaction_date", "membership_expire_date"])
    last = (
        seen
        .groupby("msno", sort=False)
        .tail(1)
        .set_index("msno")
    )
    return last


def scoring_population(tx: pd.DataFrame, cutoff) -> pd.DataFrame:
    """Customers whose known membership runs out in the month after `cutoff`.

    Returns a frame indexed by msno with a column `expire_date`.
    """
    cutoff = month_end(cutoff)
    last = known_status(tx, cutoff)
    window_end = add_months(cutoff, 1)
    exp = last["membership_expire_date"]
    pop = last[(exp > cutoff) & (exp <= window_end)]
    return pd.DataFrame({"expire_date": pop["membership_expire_date"]}).sort_index()


def churn_labels(tx: pd.DataFrame, cutoff, population: pd.DataFrame | None = None) -> pd.Series:
    """1 = churned, 0 = renewed in time. Indexed by msno.

    Uses transactions after the cut-off on purpose: this is the answer key.
    """
    cutoff = month_end(cutoff)
    pop = scoring_population(tx, cutoff) if population is None else population
    if pop.empty:
        return pd.Series(dtype="int8", name="is_churn")

    later = tx[(tx["transaction_date"] > cutoff) & (tx["is_cancel"] == 0)]
    later = later[later["msno"].isin(pop.index)]
    later = later.join(pop["expire_date"], on="msno")
    renewed = later[
        (later["membership_expire_date"] > later["expire_date"])
        & (later["transaction_date"] <= later["expire_date"] + pd.Timedelta(days=CHURN_GRACE_DAYS))
    ]["msno"].unique()
    y = pd.Series(1, index=pop.index, dtype="int8", name="is_churn")
    y.loc[y.index.isin(renewed)] = 0
    return y


def churn_rate(y: pd.Series) -> float:
    return float(np.mean(y)) if len(y) else float("nan")
