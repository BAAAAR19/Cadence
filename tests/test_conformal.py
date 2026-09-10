"""Split-conformal calibration: the guarantee, and the places it stops holding.

The claim under test is not "the model is accurate". It is the conditional
statement that makes conformal prediction worth using at all:

    for exchangeable (X, Y), P(Y <= U(X)) >= 1 - alpha,
    for *any* underlying model, in finite samples.

So the tests below deliberately include a predictor that is wrong on purpose --
if coverage depended on the model being good, the method would be a heuristic
with a proof attached to it.
"""

from __future__ import annotations

import numpy as np
import pytest

from cadence.admission.conformal import (
    AdaptiveConformal,
    RollingConformalUpperBound,
    SplitConformalUpperBound,
    build_bound,
    conformal_level,
    level_is_attainable,
)
from cadence.admission.predictor import QuantileLatencyPredictor, pinball_loss


class _Oracle:
    """A predictor that knows the conditional quantile exactly."""

    def __init__(self, alpha: float) -> None:
        self.alpha = alpha

    def predict_quantiles(self, X):
        X = np.asarray(X, dtype=float)
        mu = 5.0 + X[:, 0]
        z = 1.2816 if self.alpha == 0.2 else 2.3263  # N quantiles at 0.9 / 0.99
        return mu - z, mu + z


class _Useless:
    """A predictor that says the same thing about every request.

    Not literally zero: the ratio score is multiplicative, and a bound of
    ``0 x anything`` would be zero for ever. A constant is the honest version
    of "this model carries no information", and it is what the constant-quantile
    baseline in the fitting script actually is.
    """

    def predict_quantiles(self, X):
        n = len(np.asarray(X, dtype=float))
        return np.full(n, 1.0), np.full(n, 1.0)


def _sample(rng, n, shift=0.0):
    """Strictly positive outcomes, because latency is.

    The offset matters for more than tidiness: the ratio score works in log
    space, so a synthetic "latency" that could come out negative would be
    testing the score on inputs the real thing cannot produce.
    """
    x = rng.uniform(0, 10, size=(n, 3))
    y = 5.0 + x[:, 0] + shift + rng.normal(0, 1.0, size=n)
    return x, y


# --- the guarantee -------------------------------------------------------
@pytest.mark.parametrize("alpha", [0.2, 0.05, 0.01])
@pytest.mark.parametrize("model", ["oracle", "useless"])
@pytest.mark.parametrize("score", ["absolute", "ratio", "scaled"])
def test_coverage_holds_for_any_model(alpha, model, score):
    """Validity is indifferent to both the model and the score function.

    That is the property the whole method rests on -- the rank argument never
    looks at what the score *means* -- and it is why the fitting script is free
    to choose a score for efficiency alone.
    """
    rng = np.random.default_rng(7)
    predictor = _Oracle(alpha) if model == "oracle" else _Useless()
    Xc, yc = _sample(rng, 4000)
    Xt, yt = _sample(rng, 8000)
    b = SplitConformalUpperBound(predictor, alpha=alpha, score=score).calibrate(Xc, yc)
    cov = b.empirical_coverage(Xt, yt)
    # Above nominal, up to sampling error on 8000 draws (3 sigma at alpha=0.01
    # is about 0.0033).
    assert cov >= 1 - alpha - 0.006, f"{model} at alpha={alpha}: coverage {cov}"


def test_the_bad_model_pays_in_width_not_in_coverage():
    """The whole reason the fitting script reports bound width.

    A useless predictor is *not* punished with lost coverage -- conformal
    calibration repairs that -- so the only place its uselessness shows up is
    in how wide the resulting bound is, and therefore in how many requests a
    controller using it would have to refuse.
    """
    rng = np.random.default_rng(11)
    Xc, yc = _sample(rng, 3000)
    Xt, yt = _sample(rng, 3000)
    good = SplitConformalUpperBound(_Oracle(0.05), alpha=0.05).calibrate(Xc, yc)
    bad = SplitConformalUpperBound(_Useless(), alpha=0.05).calibrate(Xc, yc)
    assert good.empirical_coverage(Xt, yt) >= 0.95 - 0.01
    assert bad.empirical_coverage(Xt, yt) >= 0.95 - 0.01
    assert bad.upper_bound(Xt).mean() > good.upper_bound(Xt).mean()

    # The operational version of the same statement, which is the one the
    # controller feels: against a fixed budget, the model that cannot tell one
    # request from another has to refuse most of them.
    budget = float(np.median(yt))
    admit_good = float(np.mean(good.upper_bound(Xt) <= budget))
    admit_bad = float(np.mean(bad.upper_bound(Xt) <= budget))
    assert admit_good > 0.15
    assert admit_bad < admit_good / 2


def test_marginal_guarantee_is_finite_sample():
    """The statement is about a *fresh* calibration set each time, not about one
    lucky draw, so the experiment is repeated end to end.

    Small calibration sets on purpose: the finite-sample correction is exactly
    what stops coverage falling below nominal when ``n`` is small, and an
    asymptotic version of the same procedure would fail this test.
    """
    rng = np.random.default_rng(3)
    alpha, n_calib, trials = 0.1, 120, 4000
    hits = 0
    for _ in range(trials):
        Xc, yc = _sample(rng, n_calib)
        b = SplitConformalUpperBound(_Oracle(alpha), alpha=alpha).calibrate(Xc, yc)
        Xn, yn = _sample(rng, 1)
        hits += int(yn[0] <= b.upper_bound(Xn)[0])
    cov = hits / trials
    assert cov >= 1 - alpha - 0.015, f"marginal coverage {cov} below nominal"


def test_the_correction_is_the_n_plus_one_one():
    # ceil((n+1)(1-alpha)) / n, and nothing else. The 100-row guard in
    # calibrate() is this arithmetic, not a superstition: below n=99 no order
    # statistic of the calibration set is high enough to be a 99% bound.
    assert conformal_level(200, 0.01) == np.ceil(201 * 0.99) / 200
    assert conformal_level(99, 0.01) == 1.0  # clipped; unattainable
    assert not level_is_attainable(98, 0.01)
    assert level_is_attainable(100, 0.01)
    # The uncorrected level is strictly smaller, which is the whole point: it
    # would give a bound one order statistic lower and no finite-sample claim.
    assert conformal_level(200, 0.1) > (1 - 0.1)


def test_calibration_set_too_small_is_refused():
    rng = np.random.default_rng(1)
    X, y = _sample(rng, 80)
    with pytest.raises(ValueError, match="too small"):
        SplitConformalUpperBound(_Oracle(0.01), alpha=0.01).calibrate(X, y)


def test_bound_before_calibration_is_an_error():
    with pytest.raises(RuntimeError, match="not calibrated"):
        SplitConformalUpperBound(_Useless(), alpha=0.1).upper_bound(np.zeros((1, 3)))


def test_tighter_level_gives_a_looser_bound():
    rng = np.random.default_rng(5)
    Xc, yc = _sample(rng, 2000)
    Xt, _ = _sample(rng, 100)
    prev = -np.inf
    for alpha in (0.2, 0.1, 0.05, 0.01):
        u = (
            SplitConformalUpperBound(_Oracle(alpha), alpha=alpha)
            .calibrate(Xc, yc)
            .upper_bound(Xt)
        )
        assert u.mean() > prev
        prev = u.mean()


# --- where it stops holding ---------------------------------------------
def test_distribution_shift_breaks_the_static_bound():
    """The honest negative result, asserted rather than admitted in prose.

    The controller's own shedding changes what runs, which is exactly this: a
    calibration set drawn from one distribution and a test set drawn from
    another.
    """
    rng = np.random.default_rng(13)
    Xc, yc = _sample(rng, 2000)
    Xt, yt = _sample(rng, 2000, shift=3.0)  # everything got slower
    b = SplitConformalUpperBound(_Oracle(0.05), alpha=0.05).calibrate(Xc, yc)
    assert b.empirical_coverage(Xt, yt) < 0.9, "a 3-sigma shift should have hurt"


def test_rolling_recalibration_recovers_from_the_shift():
    rng = np.random.default_rng(17)
    Xc, yc = _sample(rng, 2000)
    b = RollingConformalUpperBound(_Oracle(0.05), alpha=0.05, window=400, refresh=16)
    b.calibrate(Xc, yc)
    Xs, ys = _sample(rng, 1500, shift=3.0)
    for x, y in zip(Xs, ys, strict=True):
        b.observe(x, y)
    Xt, yt = _sample(rng, 3000, shift=3.0)
    assert b.empirical_coverage(Xt, yt) >= 0.94


def test_aci_drives_long_run_coverage_back_to_nominal():
    """The same stream, judged by a static bound and by an adaptive one.

    This is the trade the writeup describes, as a number: the static bound has
    the finite-sample marginal guarantee and loses it entirely under shift;
    ACI has no such guarantee for the next request and holds the long-run rate
    almost exactly.
    """
    rng = np.random.default_rng(19)
    alpha = 0.1
    Xc, yc = _sample(rng, 1000)
    Xs, ys = _sample(rng, 4000, shift=3.0)

    static = SplitConformalUpperBound(_Oracle(alpha), alpha=alpha).calibrate(Xc, yc)
    assert static.empirical_coverage(Xs, ys) < 0.5, "the shift should have been fatal"

    b = AdaptiveConformal(_Oracle(alpha), alpha=alpha, window=500, gamma=0.02)
    b.calibrate(Xc, yc)
    for x, y in zip(Xs, ys, strict=True):
        b.observe(x, y)
    assert 1 - alpha - 0.03 <= b.running_coverage <= 1 - alpha + 0.05, b.running_coverage
    # The working level is a random walk around the point where violations
    # arrive at rate alpha, so it moves; what it must not do is run away.
    assert 0.02 < b.alpha_t < 0.35


def test_build_bound_names_the_modes():
    for mode in ("static", "rolling", "aci"):
        b = build_bound(mode, _Useless(), 0.1, window=32, refresh=4, gamma=0.01)
        assert b.mode == mode
    with pytest.raises(ValueError, match="unknown conformal mode"):
        build_bound("magic", _Useless(), 0.1)


# --- the base model ------------------------------------------------------
def test_quantile_regressor_is_fitted_in_log_space_and_stays_positive():
    rng = np.random.default_rng(23)
    X = rng.uniform(0, 5, size=(600, 14))
    y = np.exp(0.4 * X[:, 0]) + rng.lognormal(0, 0.4, size=600)
    m = QuantileLatencyPredictor(alpha=0.1, n_estimators=60, max_depth=3).fit(X, y)
    lo, hi = m.predict_quantiles(X)
    assert (lo >= 0).all() and (hi >= lo).all()
    # A quantile model is scored with the pinball loss, not with MAE: the
    # 95th-percentile prediction *should* sit above most of the data.
    assert pinball_loss(y, hi, 0.95) < pinball_loss(y, np.full_like(y, y.mean()), 0.95)


def test_pinball_loss_is_minimised_at_the_true_quantile():
    rng = np.random.default_rng(29)
    y = rng.normal(0, 1, size=20000)
    best = pinball_loss(y, np.full_like(y, np.quantile(y, 0.9)), 0.9)
    for guess in (np.quantile(y, 0.7), np.quantile(y, 0.99)):
        assert pinball_loss(y, np.full_like(y, guess), 0.9) > best
