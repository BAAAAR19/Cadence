"""Split-conformal calibration, and the finite-sample upper bound.

The point of this module in one sentence: it converts a quantile regressor's
guess at the 99th percentile -- which is wrong by an unknown amount, because
the model is fitted, not true -- into a bound that is right at least 99% of the
time, with no assumption that the model is any good.

The procedure, one-sided because an SLO is one-sided (only being too *slow* is
a violation):

1. On a calibration set of size ``n`` that the model never saw, score each
   request by how badly the model under-predicted it: ``E_i = y_i - q_hi(x_i)``.
   Positive means the truth was above the predicted quantile.
2. Take ``Q``, the ``ceil((n + 1) * (1 - alpha)) / n``-th empirical quantile of
   those scores.
3. The bound for a new request is ``U(x) = q_hi(x) + Q``.

Then ``P(Y <= U(X)) >= 1 - alpha`` for a fresh request, in finite samples, for
*any* underlying model, assuming only that calibration and test requests are
exchangeable.

Why ``(n + 1)`` and not ``n``
-----------------------------
The argument is a rank argument, and it is worth being able to give. Under
exchangeability, the new request's own score ``E_{n+1}`` is equally likely to
occupy any rank among the ``n + 1`` scores ``{E_1..E_n, E_{n+1}}``. So
``P(E_{n+1} <= k-th smallest of the n + 1) = k / (n + 1)``. Choosing
``k = ceil((n + 1)(1 - alpha))`` gives probability at least ``1 - alpha``, and
that ``k``-th value of the ``n + 1`` is the ``k``-th of the ``n`` we can
actually see. Divide ``k`` by ``n`` and you get the level above. Using ``n``
instead of ``n + 1`` throws away exactly the correction that makes the
statement finite-sample rather than asymptotic.

It also explains the ``n >= 100`` guard below rather than leaving it as a
superstition: the level is only attainable when ``ceil((n+1)(1-alpha)) <= n``,
which for ``alpha = 0.01`` first happens at ``n = 99``. Below that, no order
statistic of the calibration set is high enough and the honest answer is that
a 99% bound cannot be formed from this much data.

Where the guarantee genuinely does not hold
-------------------------------------------
Exchangeability breaks the moment the controller starts shedding, because
shedding changes the distribution of what runs -- and the bound was calibrated
on traces collected with admission *off*. Two honest responses are implemented
here and both are measured in the writeup: :class:`RollingConformalUpperBound`
recalibrates on the most recent completions, and :class:`AdaptiveConformal`
(ACI) adjusts the working level online so that long-run coverage converges to
``1 - alpha`` even under shift, at the cost of the finite-sample marginal
guarantee. Neither is a fix for the theory; both are a way of noticing.
"""

from __future__ import annotations

import math
from collections import deque

import numpy as np


def conformal_level(n: int, alpha: float) -> float:
    """The finite-sample corrected quantile level, clipped at 1.

    Clipped because a level above 1 is unattainable: see the module docstring
    for what that means and when it happens.
    """
    return min(1.0, math.ceil((n + 1) * (1 - alpha)) / n)


def level_is_attainable(n: int, alpha: float) -> bool:
    return math.ceil((n + 1) * (1 - alpha)) <= n


# --- score functions -----------------------------------------------------
# Conformal validity does not depend on which of these is used: any function of
# (prediction, outcome) gives the same finite-sample guarantee, because the
# argument is about the *rank* of the new score among the calibration scores
# and never about what the score means. What the choice decides is efficiency,
# and on this workload that difference is the whole experiment.
#
# ``absolute`` is the build guide's, and it is the wrong one here. Latency
# spans two and a half orders of magnitude between an idle server and a
# saturated one, so a single additive correction is set by the worst regime and
# then applied to the best: measured Q is +22.8 s, which puts the bound for a
# request that will take 0.6 s at 23 s, and admits nothing.
#
# ``ratio`` is additive in log space -- the space the model is fitted in --
# so the correction is a multiplier: U = (1 + q_hi) * exp(Q) - 1. A bound that
# is 2.0x the model's estimate is meaningful at both ends of that range.
#
# ``scaled`` is the textbook locally-weighted alternative, dividing by the
# model's own interval width. It is included because it is the obvious
# competitor and because it fails here for a reason worth recording: the
# interval collapses towards zero width for the easiest requests, the score
# divides by it, and the calibration quantile comes out at Q = 1.3e3.


def _score_absolute(y, lo, hi):
    return y - hi


def _invert_absolute(lo, hi, q):
    return hi + q


def _score_ratio(y, lo, hi):
    # Both arguments are clamped at zero: the ratio score is only defined for a
    # positive quantity, which latency is, and a model free to predict a
    # negative one is not.
    return np.log1p(np.maximum(y, 0.0)) - np.log1p(np.maximum(hi, 0.0))


def _invert_ratio(lo, hi, q):
    return np.expm1(np.log1p(np.maximum(hi, 0.0)) + q)


def _width(lo, hi):
    return np.maximum(hi - lo, 1e-3)


def _score_scaled(y, lo, hi):
    return (y - hi) / _width(lo, hi)


def _invert_scaled(lo, hi, q):
    return hi + q * _width(lo, hi)


SCORES = {
    "absolute": (_score_absolute, _invert_absolute),
    "ratio": (_score_ratio, _invert_ratio),
    "scaled": (_score_scaled, _invert_scaled),
}


class SplitConformalUpperBound:
    """One-sided split-conformal calibration of a quantile regressor.

    Guarantee: for exchangeable ``(X, Y)``, ``P(Y <= U(X)) >= 1 - alpha``, for
    any underlying model, with no distributional assumptions, in finite
    samples.
    """

    mode = "static"

    def __init__(self, predictor, alpha: float = 0.01, score: str = "ratio") -> None:
        if score not in SCORES:
            raise ValueError(f"unknown nonconformity score {score!r}; have {sorted(SCORES)}")
        self.predictor = predictor
        self.alpha = alpha
        self.score = score
        self._score, self._invert = SCORES[score]
        self.q_: float | None = None
        self.n_calib_: int = 0
        self.scores_: np.ndarray = np.empty(0)
        """Calibration scores, sorted. The quantile is an order statistic, so
        this is the form every use of them wants."""
        self.scores_in_order_: np.ndarray = np.empty(0)
        """The same scores in the order they were handed over, which for a
        calibration set built from a time-sorted trace is chronological. Only
        the rolling window below cares, and it cares a lot: seeding a
        *most-recent-N* window from the sorted array would seed it with the N
        largest scores in the whole set."""

    # --- calibration ------------------------------------------------------
    def calibrate(self, X_calib, y_calib) -> SplitConformalUpperBound:
        lo, hi = self.predictor.predict_quantiles(X_calib)
        scores = self._score(np.asarray(y_calib, dtype=float), lo, hi)
        return self.calibrate_scores(scores)

    def calibrate_scores(self, scores) -> SplitConformalUpperBound:
        """Calibrate from precomputed nonconformity scores.

        Separated from :meth:`calibrate` because the online modes below have
        the scores already -- each completed request contributed one -- and
        re-running the model over a rolling window every few seconds to
        recompute numbers it already knows would be waste on the request path.
        """
        scores = np.asarray(scores, dtype=float)
        n = scores.size
        if n < 100:
            raise ValueError(
                f"calibration set too small for a {1 - self.alpha:.0%} bound: "
                f"n={n}, and the corrected level is unattainable below "
                f"n={math.ceil((1 - self.alpha) / self.alpha)}"
            )
        self.scores_in_order_ = scores
        self.scores_ = np.sort(scores)
        self.n_calib_ = n
        self.q_ = self._quantile(self.alpha)
        return self

    def _quantile(self, alpha: float) -> float:
        level = conformal_level(self.scores_.size, alpha)
        # ``method="higher"`` is the conservative choice and the correct one:
        # interpolating between order statistics would give a bound that is
        # not one of the observed scores and loses the rank argument the
        # guarantee rests on.
        return float(np.quantile(self.scores_, level, method="higher"))

    # --- use --------------------------------------------------------------
    def upper_bound(self, X) -> np.ndarray:
        if self.q_ is None:
            raise RuntimeError("not calibrated")
        lo, hi = self.predictor.predict_quantiles(X)
        return np.asarray(self._invert(lo, hi, self.q_), dtype=float)

    def score_of(self, y: float, lo: float, hi: float) -> float:
        """The nonconformity score of one completed request, for the online
        modes. Kept here so that the score function and its inverse can never
        be paired up wrongly by a caller."""
        return float(np.asarray(self._score(np.asarray([y]), np.asarray([lo]),
                                            np.asarray([hi])))[0])

    def empirical_coverage(self, X, y) -> float:
        return float(np.mean(np.asarray(y, dtype=float) <= self.upper_bound(X)))

    def observe(self, x, y_actual: float, hi: float | None = None) -> None:
        """A completed request. The static bound does not learn from it; the
        subclasses below do, and the controller calls this unconditionally so
        that switching modes is a configuration change and not a code path."""
        return None

    @property
    def alpha_working(self) -> float:
        """The level actually in force. Constant except under ACI."""
        return self.alpha


class RollingConformalUpperBound(SplitConformalUpperBound):
    """Recalibrate on a rolling window of the most recent completions.

    The cheap, obvious answer to distribution shift, and the one to reach for
    first: the offline calibration set was collected under a different policy
    (admission off) and on a machine in a different thermal state, and both of
    those are visible in the scores within a minute of a run starting.

    Seeded with the offline calibration scores so the bound is valid from the
    first request rather than after the window fills. Once ``window``
    completions have accumulated the seed has aged out entirely.
    """

    mode = "rolling"

    def __init__(self, predictor, alpha: float = 0.01, score: str = "ratio",
                 window: int = 512, refresh: int = 16) -> None:
        super().__init__(predictor, alpha, score)
        self.window = window
        self.refresh = refresh
        self._recent: deque[float] = deque(maxlen=window)
        self._since_refresh = 0

    def calibrate_scores(self, scores) -> RollingConformalUpperBound:
        super().calibrate_scores(scores)
        # The most recent calibration scores, not the largest ones.
        self._recent = deque(self.scores_in_order_[-self.window:], maxlen=self.window)
        return self

    def observe(self, x, y_actual: float, hi: float | None = None) -> None:
        lo, hi = self._quantiles_of(x, hi)
        self._recent.append(self.score_of(float(y_actual), lo, hi))
        self._since_refresh += 1
        if self._since_refresh >= self.refresh:
            self._since_refresh = 0
            self._requantile()

    def _quantiles_of(self, x, hi: float | None) -> tuple[float, float]:
        """The model's own quantile pair for a completed request.

        The controller already has ``hi`` -- it is what the request was
        admitted on -- so the common path costs nothing. The fallback runs the
        model, which is only for callers that did not keep it.
        """
        if hi is None:
            lo_a, hi_a = self.predictor.predict_quantiles(np.asarray(x, dtype=float)[None, :])
            return float(lo_a[0]), float(hi_a[0])
        if self.score == "scaled":
            lo_a, _ = self.predictor.predict_quantiles(np.asarray(x, dtype=float)[None, :])
            return float(lo_a[0]), float(hi)
        # Neither the absolute nor the ratio score looks at the lower quantile.
        return 0.0, float(hi)

    def _requantile(self) -> None:
        if len(self._recent) < 100:
            return  # too few to form the level; keep the offline bound
        self.scores_ = np.sort(np.fromiter(self._recent, dtype=float))
        self.n_calib_ = self.scores_.size
        self.q_ = self._quantile(self.alpha_working)


class AdaptiveConformal(RollingConformalUpperBound):
    """ACI: track coverage online so the bound survives distribution shift.

    ``alpha_{t+1} = alpha_t + gamma * (alpha - err_t)``, where ``err_t`` is 1
    if the last request violated its bound. The update is a gradient step on
    the pinball loss of the coverage target: a violation lowers the working
    alpha (a *higher* quantile, a looser bound), a covered request raises it
    slightly. Long-run coverage provably converges to ``1 - alpha`` for any
    sequence, adversarial included -- but the guarantee it gives is about the
    long-run average, not about the next request, which is the trade the
    docstring above describes.
    """

    mode = "aci"

    def __init__(self, predictor, alpha: float = 0.01, score: str = "ratio",
                 window: int = 512, refresh: int = 16, gamma: float = 0.005) -> None:
        super().__init__(predictor, alpha, score, window, refresh)
        self.gamma = gamma
        self.alpha_t = alpha
        self.n_observed = 0
        self.n_violations = 0

    @property
    def alpha_working(self) -> float:
        return self.alpha_t

    def observe(self, x, y_actual: float, hi: float | None = None) -> None:
        lo, hi = self._quantiles_of(x, hi)
        covered = float(y_actual) <= float(self._invert(lo, hi, self.q_ or 0.0))
        self.n_observed += 1
        self.n_violations += 0 if covered else 1
        self.alpha_t = float(
            np.clip(self.alpha_t + self.gamma * (self.alpha - (0.0 if covered else 1.0)),
                    1e-4, 0.5)
        )
        super().observe(x, y_actual, hi)
        # ACI must requantile on every step, not only on the refresh boundary:
        # the level moved, so the bound derived from it is stale even though
        # the score window has not changed much.
        self._requantile()

    @property
    def running_coverage(self) -> float:
        return 1.0 - self.n_violations / self.n_observed if self.n_observed else float("nan")


MODES = {
    SplitConformalUpperBound.mode: SplitConformalUpperBound,
    RollingConformalUpperBound.mode: RollingConformalUpperBound,
    AdaptiveConformal.mode: AdaptiveConformal,
}


def build_bound(mode: str, predictor, alpha: float, score: str = "ratio",
                **kw) -> SplitConformalUpperBound:
    if mode not in MODES:
        raise ValueError(f"unknown conformal mode {mode!r}; have {sorted(MODES)}")
    cls = MODES[mode]
    if cls is SplitConformalUpperBound:
        kw.pop("window", None)
        kw.pop("refresh", None)
        kw.pop("gamma", None)
    elif cls is RollingConformalUpperBound:
        kw.pop("gamma", None)
    return cls(predictor, alpha, score, **kw)
