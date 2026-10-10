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


def test_listening_is_measured_per_day_so_february_is_not_a_drop():
    """Someone who listens every day must look the same in a 28-day and a 31-day month."""
    from leaklens.data import Dataset, empty_members
    from leaklens.features import build_features

    d = pd.Timestamp
    tx = pd.DataFrame({
        "msno": ["x", "x"], "payment_method_id": [41, 41], "payment_plan_days": [30, 30],
        "plan_list_price": [149, 149], "actual_amount_paid": [149, 149], "is_auto_renew": [1, 1],
        "transaction_date": [d("2017-01-10"), d("2017-02-10")],
        "membership_expire_date": [d("2017-02-10"), d("2017-03-10")], "is_cancel": [0, 0]})
    logs = pd.DataFrame({
        "msno": ["x", "x"], "month_end": [d("2017-01-31"), d("2017-02-28")],
        "num_25": [0, 0], "num_50": [0, 0], "num_75": [0, 0], "num_985": [0, 0], "num_100": [310, 280],
        "num_unq": [310, 280], "total_secs": [31 * 3600.0, 28 * 3600.0],
        "days_active": [31, 28], "last_date": [d("2017-01-31"), d("2017-02-28")]})
    ds = Dataset(tx=tx, members=empty_members(), logs=logs)
    jan = build_features(ds, "2017-01-31", ["x"]).iloc[0]
    feb = build_features(ds, "2017-02-28", ["x"]).iloc[0]
    assert jan["days_active_m1"] == feb["days_active_m1"] == 1.0
    assert jan["secs_m1"] == feb["secs_m1"] == 3600.0
    assert jan["unique_songs_m1"] == feb["unique_songs_m1"] == 10.0
