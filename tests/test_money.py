"""The money maths, checked by hand on tiny examples."""

import numpy as np
import pytest

from leaklens.money import Assumptions, best_threshold, binned_counts, money_table, savings

# 4 customers: who actually left, and the model's chance of leaving
Y = [1, 0, 1, 0]
P = [0.9, 0.8, 0.3, 0.1]
A = Assumptions(offer_cost=5, customer_value=60)


def test_savings_by_hand():
    # t=0.5 -> offers to customers 1 and 2; one of them leaves: 1*60 - 2*5 = 50
    assert savings(Y, P, 0.5, A) == 50
    # t=0.2 -> offers to 1, 2, 3; two leave: 2*60 - 3*5 = 105
    assert savings(Y, P, 0.2, A) == 105
    # t=0.05 -> offers to everyone; two leave: 2*60 - 4*5 = 100
    assert savings(Y, P, 0.05, A) == 100
    # nobody gets an offer -> nothing saved, nothing spent
    assert savings(Y, P, 0.95, A) == 0


def test_save_rate_scales_the_benefit():
    # half of reached leavers stay: 1*0.5*60 - 2*5 = 20
    assert savings(Y, P, 0.5, Assumptions(5, 60, 0.5)) == 20


def test_best_threshold_by_hand():
    t, s = best_threshold(Y, P, A)
    assert s == 105
    assert 0.1 < t <= 0.3


def test_break_even():
    assert A.break_even == pytest.approx(5 / 60)


def test_money_table_counts():
    tab = money_table(Y, P, A, 0.2)
    assert tab["money-based threshold"]["offers_sent"] == 3
    assert tab["money-based threshold"]["leavers_missed"] == 0
    assert tab["default 0.5"]["saved_eur"] == 50
    assert tab["offer to everyone"]["saved_eur"] == 100


def test_binned_counts_reproduce_savings():
    """The web page rebuilds savings from these bins: it must match exactly."""
    rng = np.random.default_rng(0)
    p = rng.random(5000)
    y = (rng.random(5000) < p).astype(int)
    b = binned_counts(y, p, 100)
    for t_bin in (5, 20, 50):
        leavers = sum(b["leavers"][t_bin:])
        offers = leavers + sum(b["stayers"][t_bin:])
        assert leavers * 60 - offers * 5 == savings(y, p, t_bin / 100, A)
