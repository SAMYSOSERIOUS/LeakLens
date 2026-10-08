"""The honest model must beat the two things you could do without it."""

import pytest

from leaklens.features import build_training_table
from leaklens.models import fit_predict_all, scores


@pytest.fixture(scope="module")
def results(ds):
    # Train on four months, test on a later month the model never saw. The
    # gap month (Jun) mirrors real life: its answers aren't known in time.
    # The test month is before the planted Sep-2016 promo shift; how the
    # model copes with that shift is the drift monitor's job, not this test's.
    train = build_training_table(ds, ["2016-01-31", "2016-02-29", "2016-03-31", "2016-04-30"])
    test = build_training_table(ds, ["2016-06-30"])
    preds = fit_predict_all(train, test)
    return {name: scores(test["is_churn"], p) for name, p in preds.items()}


def test_model_beats_nobody_churns(results):
    gbm, base = results["LightGBM"], results["Nobody churns (baseline)"]
    assert gbm["auc"] > base["auc"] + 0.15
    assert gbm["log_loss"] < base["log_loss"]


def test_model_beats_the_auto_renew_rule(results):
    gbm, rule = results["LightGBM"], results["Auto-renew rule"]
    assert gbm["auc"] > rule["auc"]
    assert gbm["log_loss"] < rule["log_loss"]


def test_logistic_regression_also_beats_baseline(results):
    assert results["Logistic regression"]["auc"] > results["Nobody churns (baseline)"]["auc"] + 0.15
