"""Turn churn chances into euros.

Assumptions (change them in config.toml - they are guesses, not facts):
  * Sending a retention offer costs OFFER_COST euros per customer
    (default 5 EUR), whether or not they would have left.
  * A customer who leaves costs CUSTOMER_VALUE euros in lost future
    revenue (default 60 EUR).
  * SAVE_RATE: the share of would-be leavers who stay because of the
    offer. The brief assumes 1.0 (the offer always works). Real offers
    work far less often - try 0.3 and watch the best threshold rise.

Money saved, compared with sending no offers at all:
    saved = (leavers we reached) x SAVE_RATE x CUSTOMER_VALUE
            - (everyone we reached) x OFFER_COST

Sending an offer pays off when
    chance_of_leaving x SAVE_RATE x CUSTOMER_VALUE > OFFER_COST
so with 5 EUR / 60 EUR / 1.0 the break-even chance is 5/60 = 8.3%, far
below the usual default of 50%.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

THRESHOLDS = np.round(np.arange(0.01, 1.0, 0.01), 2)


@dataclass(frozen=True)
class Assumptions:
    offer_cost: float = 5.0
    customer_value: float = 60.0
    save_rate: float = 1.0

    @property
    def break_even(self) -> float:
        return self.offer_cost / (self.save_rate * self.customer_value)


def savings(y, p, threshold: float, a: Assumptions = Assumptions()) -> float:
    """Euros saved by offering to everyone with churn chance >= threshold."""
    y = np.asarray(y).astype(bool)
    target = np.asarray(p) >= threshold
    reached_leavers = np.sum(target & y)
    return float(reached_leavers * a.save_rate * a.customer_value - target.sum() * a.offer_cost)


def savings_curve(y, p, a: Assumptions = Assumptions(), thresholds=THRESHOLDS) -> np.ndarray:
    return np.array([savings(y, p, t, a) for t in thresholds])


def best_threshold(y, p, a: Assumptions = Assumptions(), thresholds=THRESHOLDS) -> tuple[float, float]:
    """The threshold that saves the most money on this data, and how much."""
    curve = savings_curve(y, p, a, thresholds)
    i = int(np.argmax(curve))
    return float(thresholds[i]), float(curve[i])


def money_table(y, p, a: Assumptions, chosen_threshold: float) -> dict:
    """Compare ways of choosing who gets an offer, in euros."""
    y = np.asarray(y)
    p = np.asarray(p)
    rows = {}
    for label, t in (("money-based threshold", chosen_threshold), ("default 0.5", 0.5)):
        target = p >= t
        rows[label] = {
            "threshold": round(float(t), 2),
            "offers_sent": int(target.sum()),
            "leavers_reached": int((target & (y == 1)).sum()),
            "leavers_missed": int((~target & (y == 1)).sum()),
            "saved_eur": round(savings(y, p, t, a), 2),
        }
    everyone = savings(y, np.ones_like(p), 0.5, a)
    rows["offer to everyone"] = {
        "threshold": 0.0, "offers_sent": int(len(y)), "leavers_reached": int(y.sum()),
        "leavers_missed": 0, "saved_eur": round(everyone, 2),
    }
    rows["no offers"] = {"threshold": 1.0, "offers_sent": 0, "leavers_reached": 0,
                         "leavers_missed": int(y.sum()), "saved_eur": 0.0}
    return rows


def binned_counts(y, p, n_bins: int = 100) -> dict:
    """Leavers and stayers per 1%-wide band of predicted churn chance.

    This is all a static web page needs to recompute savings for any
    offer cost and customer value, without a server.
    """
    y = np.asarray(y).astype(int)
    b = np.clip((np.asarray(p) * n_bins).astype(int), 0, n_bins - 1)
    churn = np.bincount(b, weights=y, minlength=n_bins).astype(int)
    total = np.bincount(b, minlength=n_bins).astype(int)
    return {"n_bins": n_bins, "leavers": churn.tolist(), "stayers": (total - churn).tolist()}
