"""The most important tests in the project: no feature may use the future.

A churn model is used on a given day to predict what happens next. If any
input was computed from data dated after that day, the test score is a
lie. We check this from the outside: rewrite the future, and confirm that
not a single feature value moves.
"""

import numpy as np
import pandas as pd
import pytest

from conftest import scramble_future
from leaklens.audit import pick_cutoffs
from leaklens.dates import is_mature
from leaklens.features import FEATURES, build_features, visible
from leaklens.labels import scoring_population
from leaklens.recipes import RECIPES
from leaklens.replay import ReplayConfig, ReplayJob
from leaklens.money import Assumptions

CUTOFFS = ["2016-03-31", "2016-08-31", "2016-11-30", "2017-01-31"]


@pytest.mark.parametrize("cutoff", CUTOFFS)
def test_no_feature_uses_data_after_the_cutoff(ds, cutoff):
    users = scoring_population(ds.tx, cutoff).index
    assert len(users) > 100
    honest = build_features(ds, cutoff, users)
    after_scramble = build_features(scramble_future(ds, cutoff), cutoff, users)
    for col in FEATURES:
        pd.testing.assert_series_equal(honest[col], after_scramble[col], check_names=False,
                                       obj=f"feature '{col}' changed when the future changed")


@pytest.mark.parametrize("cutoff", CUTOFFS)
def test_visible_data_ends_at_the_cutoff(ds, cutoff):
    v = visible(ds, cutoff)
    c = pd.Timestamp(cutoff)
    assert v.tx["transaction_date"].max() <= c
    assert v.logs["month_end"].max() <= c
    assert (v.members["registration_init_time"].dropna() <= c).all()


def test_who_gets_scored_does_not_depend_on_the_future(ds):
    c = "2016-10-31"
    a = scoring_population(ds.tx, c)
    b = scoring_population(scramble_future(ds, c).tx, c)
    pd.testing.assert_frame_equal(a, b)


def test_the_leak_check_can_catch_a_leak(ds):
    """Sanity check on the check itself: the leaky public recipes MUST fail it."""
    c = "2016-10-31"
    users = scoring_population(ds.tx, c).index
    scrambled = scramble_future(ds, c)
    for r in RECIPES:
        published = r.build(ds, users)              # whole file, like the notebook
        published_scrambled = r.build(scrambled, users)
        moved = [col for col in published.columns
                 if not np.allclose(published[col], published_scrambled[col], equal_nan=True)]
        assert moved, f"recipe {r.key} should be caught leaking the future"
        honest = r.build(visible(ds, c), users)
        honest_scrambled = r.build(visible(scrambled, c), users)
        pd.testing.assert_frame_equal(honest, honest_scrambled)


def test_training_answers_were_already_known_on_the_test_date(ds):
    cut = pick_cutoffs(ds, 4)
    for c in [*cut["train"], cut["valid"]]:
        assert is_mature(c, cut["test"]), f"{c} answers were not known yet on {cut['test']}"
    assert is_mature(cut["test"], cut["data_end"])


def test_live_job_cannot_see_the_future(ds, tmp_path):
    """Run the weekly job twice: real future vs nonsense future. Same scores."""
    today = "2016-06-30"
    outs = []
    for name, data in (("real", ds), ("scrambled", scramble_future(ds, today))):
        cfg = ReplayConfig(start=today, state_dir=str(tmp_path / name / "state"),
                           site_data_dir=str(tmp_path / name / "site"))
        ReplayJob(data, cfg, Assumptions()).step()
        outs.append(pd.read_csv(tmp_path / name / "state" / "predictions" / f"{today}.csv.gz",
                                index_col="msno"))
    pd.testing.assert_frame_equal(outs[0], outs[1])
