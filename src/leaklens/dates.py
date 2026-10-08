"""Small date helpers. All cut-off dates are month ends."""

from __future__ import annotations

import pandas as pd

# A customer counts as churned if they do not renew within this many days
# after their membership runs out (the official WSDM 2018 definition).
CHURN_GRACE_DAYS = 30


def month_end(d) -> pd.Timestamp:
    """Return the last day of the month that `d` falls in."""
    ts = pd.Timestamp(d).normalize()
    return (ts + pd.offsets.MonthEnd(0)).normalize()


def add_months(cutoff, k: int) -> pd.Timestamp:
    """Move a month-end cut-off by `k` months, landing on a month end again."""
    ts = month_end(cutoff)
    return month_end(ts + pd.DateOffset(months=k))


def label_known_on(cutoff) -> pd.Timestamp:
    """The first day on which every churn label for this cut-off is settled.

    Customers scored on `cutoff` have a membership that expires within the
    next month. Each one then gets 30 days to renew. Only after that is the
    answer known.
    """
    return add_months(cutoff, 1) + pd.Timedelta(days=CHURN_GRACE_DAYS)


def is_mature(cutoff, now) -> bool:
    """True when the churn labels for `cutoff` can be known on date `now`."""
    return pd.Timestamp(now) >= label_known_on(cutoff)


def month_ends_between(start, end) -> list[pd.Timestamp]:
    """All month ends from `start` to `end`, both included."""
    out, cur = [], month_end(start)
    end = month_end(end)
    while cur <= end:
        out.append(cur)
        cur = add_months(cur, 1)
    return out


def fmt(d) -> str:
    return pd.Timestamp(d).strftime("%Y-%m-%d")
