"""Basic data rules: one row per customer per scoring date, clean labels."""

import pandas as pd
import pytest

from leaklens.dates import add_months
from leaklens.features import build_training_table
from leaklens.labels import churn_labels, scoring_population

CUTOFFS = ["2016-02-29", "2016-06-30", "2016-09-30", "2016-12-31"]


@pytest.fixture(scope="module")
def table(ds):
    return build_training_table(ds, CUTOFFS)


def test_every_customer_appears_once_per_scoring_date(table):
    dupes = table.reset_index().duplicated(["msno", "cutoff"])
    assert not dupes.any(), f"{dupes.sum()} duplicate customer rows"
    assert table["cutoff"].nunique() == len(CUTOFFS)


def test_labels_are_only_zero_or_one(table):
    assert set(table["is_churn"].unique()) <= {0, 1}
    assert table["is_churn"].notna().all()


def test_churn_rate_is_plausible(table):
    rate = table["is_churn"].mean()
    assert 0.02 < rate < 0.4, rate


@pytest.mark.parametrize("cutoff", CUTOFFS)
def test_scored_customers_expire_in_the_next_month(ds, cutoff):
    pop = scoring_population(ds.tx, cutoff)
    c = pd.Timestamp(cutoff)
    assert (pop["expire_date"] > c).all()
    assert (pop["expire_date"] <= add_months(c, 1)).all()


def test_label_rule_on_a_tiny_example():
    """Hand-made: A renews on time, B renews too late, C never, D cancels."""
    d = pd.Timestamp
    rows = [
        ("A", d("2016-01-15"), d("2016-02-15"), 0), ("A", d("2016-02-20"), d("2016-03-20"), 0),
        ("B", d("2016-01-10"), d("2016-02-10"), 0), ("B", d("2016-03-20"), d("2016-04-20"), 0),
        ("C", d("2016-01-05"), d("2016-02-05"), 0),
        ("D", d("2016-01-20"), d("2016-02-20"), 0), ("D", d("2016-02-10"), d("2016-02-20"), 1),
    ]
    tx = pd.DataFrame(rows, columns=["msno", "transaction_date", "membership_expire_date", "is_cancel"])
    y = churn_labels(tx, "2016-01-31")
    assert y.to_dict() == {"A": 0, "B": 1, "C": 1, "D": 1}


def test_csv_round_trip_parses_dates(ds):
    assert pd.api.types.is_datetime64_any_dtype(ds.tx["transaction_date"])
    assert ds.tx["transaction_date"].min() >= pd.Timestamp("2015-01-01")
    assert len(ds.logs) > 0 and {"days_active", "month_end"} <= set(ds.logs.columns)
