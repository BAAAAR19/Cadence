"""The base predictor: conditional quantiles of end-to-end latency.

Quantile regression is the right base model because it already produces a
*heteroscedastic* interval -- wide when the system is loaded and the prompt is
long, narrow when it is idle -- and the conformal step in
:mod:`cadence.admission.conformal` then repairs its miscalibration without
destroying that adaptivity.

Two baselines live here as well, and they are not decoration. Conformal
calibration makes *any* predictor's upper bound cover at the nominal rate, so
coverage cannot distinguish a good model from a terrible one: what a good model
buys is a *tighter* bound at the same coverage, and therefore more admitted
requests at the same guarantee. Reporting the constant and the
throughput-arithmetic baselines calibrated the same way is what turns "my model
works" into a number: the width of the interval each of them needs.
"""

from __future__ import annotations

import numpy as np
from sklearn.ensemble import GradientBoostingRegressor

from cadence.admission.features import FEATURES


class QuantileLatencyPredictor:
    """Two gradient-boosted quantile regressors: a low and a high conditional
    quantile of end-to-end latency.

    Fitted in log space because latency is positive and right-skewed; the
    quantile loss is monotone-invariant, so a quantile fitted on ``log1p(y)``
    and mapped back with ``expm1`` is a quantile of ``y``. That is not true of
    the mean, which is why the same trick would be wrong for a least-squares
    model and is right here.

    ``base_alpha`` is the level the two regressors are actually fitted at, and
    it is separate from ``alpha``, the level the calibrated bound must hold at.
    Keeping them separate is not a refinement -- it is what makes the thing work
    at all here, and the measurement that says so is in ``results/w4_fit``.

    The build guide's recipe sets both to the same value. At alpha=0.01 that
    asks a gradient-boosted model to estimate the 99.5th conditional percentile
    from ~1 500 rows, of which about seven lie above it; the fitted "upper
    quantile" collapses to a near-constant 18 s for every input, which after
    calibration is perfectly *valid* (coverage 1.000) and completely useless
    (nothing is admitted, because the bound never drops below the SLO).

    Conformal validity does not depend on the base model at all -- that is the
    whole point of the method -- so the base level is free to be chosen for
    statistical stability rather than to match the guarantee. Fitted at 0.90 the
    quantile pair is estimated from hundreds of rows rather than a handful, it
    tracks the load and the requested output length instead of flattening, and
    the conformal step then inflates it to whatever the guarantee requires.
    """

    name = "cqr"

    def __init__(
        self,
        alpha: float = 0.01,
        base_alpha: float | None = None,
        n_estimators: int = 300,
        max_depth: int = 4,
        learning_rate: float = 0.05,
        subsample: float = 0.9,
        random_state: int = 0,
    ) -> None:
        self.alpha = alpha  # target miscoverage of the *calibrated* bound
        self.base_alpha = alpha if base_alpha is None else base_alpha
        common = dict(
            loss="quantile",
            n_estimators=n_estimators,
            max_depth=max_depth,
            learning_rate=learning_rate,
            subsample=subsample,
            random_state=random_state,
        )
        self.lo = GradientBoostingRegressor(alpha=self.base_alpha / 2, **common)
        self.hi = GradientBoostingRegressor(alpha=1 - self.base_alpha / 2, **common)
        self.feature_names: tuple[str, ...] = FEATURES

    def fit(self, X, y) -> QuantileLatencyPredictor:
        yl = np.log1p(np.asarray(y, dtype=float))
        X = np.asarray(X, dtype=float)
        self.lo.fit(X, yl)
        self.hi.fit(X, yl)
        return self

    def predict_quantiles(self, X) -> tuple[np.ndarray, np.ndarray]:
        X = np.asarray(X, dtype=float)
        return np.expm1(self.lo.predict(X)), np.expm1(self.hi.predict(X))

    def importances(self) -> dict[str, float]:
        """Feature importance of the *upper* quantile model, which is the one
        the admission bound is built on."""
        return dict(
            sorted(
                zip(self.feature_names, self.hi.feature_importances_, strict=True),
                key=lambda kv: -kv[1],
            )
        )


class ConstantQuantile:
    """Baseline 1: ignore the request entirely.

    The empirical quantiles of the training set, returned for every input.
    Conformalised, it yields a bound that is the same for a 16-token reply on
    an idle server as for a 512-token one on a saturated one -- which is
    exactly what a controller with no model would have to use, and exactly the
    thing the tail of this workload punishes.
    """

    name = "constant"

    def __init__(self, alpha: float = 0.01) -> None:
        self.alpha = alpha
        self.lo_: float = 0.0
        self.hi_: float = 0.0

    def fit(self, X, y) -> ConstantQuantile:
        y = np.asarray(y, dtype=float)
        self.lo_ = float(np.quantile(y, self.alpha / 2))
        self.hi_ = float(np.quantile(y, 1 - self.alpha / 2))
        return self

    def predict_quantiles(self, X) -> tuple[np.ndarray, np.ndarray]:
        n = len(np.asarray(X, dtype=float))
        return np.full(n, self.lo_), np.full(n, self.hi_)


class ThroughputArithmetic:
    """Baseline 2: the calculation anyone would do without a model.

    ``prefill_tokens / prefill_rate + max_tokens / decode_rate``, plus a queue
    term for the work already in front of this request, with the three rates
    fitted by least squares on the training split rather than guessed. This is
    the baseline that matters: if the learned model cannot beat *this*, the
    features are wrong, not the method.

    It is deliberately given the same information the learned model has, and
    the same conformal treatment, so the comparison is about functional form
    and nothing else.
    """

    name = "throughput"

    _COLS = ("n_new_prefill_tokens", "max_tokens", "sum_remaining_tokens")
    FLOOR_S = 0.05
    """No request can be served faster than a prefill and one decode step."""

    def __init__(self, alpha: float = 0.01) -> None:
        self.alpha = alpha
        self.coef_: np.ndarray = np.zeros(len(self._COLS) + 1)
        self.spread_lo_: float = 1.0
        self.spread_hi_: float = 1.0
        self._idx = [FEATURES.index(c) for c in self._COLS]

    def _design(self, X) -> np.ndarray:
        X = np.asarray(X, dtype=float)
        return np.column_stack([X[:, self._idx], np.ones(len(X))])

    def fit(self, X, y) -> ThroughputArithmetic:
        A = self._design(X)
        y = np.asarray(y, dtype=float)
        # Non-negative least squares would be more principled -- a negative
        # seconds-per-token is meaningless -- but plain least squares is what
        # someone reaching for this baseline would write, and clipping the
        # prediction below at zero is enough to keep it sane.
        self.coef_, *_ = np.linalg.lstsq(A, y, rcond=None)
        # Floored at a physical minimum rather than at zero. Unconstrained
        # least squares happily predicts a few milliseconds for a small
        # request, and the multiplicative spread below then divides by it: the
        # first version of this baseline reported a mean bound of 2.8e9
        # seconds, which is a division artifact and not a statement about the
        # baseline.
        pred = np.maximum(A @ self.coef_, self.FLOOR_S)
        # A point estimate has no quantiles, so give it multiplicative ones
        # from the training residual ratio: the same shape of interval a
        # practitioner would draw by eye around a linear fit.
        ratio = y / np.maximum(pred, 1e-6)
        self.spread_lo_ = float(np.quantile(ratio, self.alpha / 2))
        self.spread_hi_ = float(np.quantile(ratio, 1 - self.alpha / 2))
        return self

    def predict_quantiles(self, X) -> tuple[np.ndarray, np.ndarray]:
        pred = np.maximum(self._design(X) @ self.coef_, self.FLOOR_S)
        return pred * self.spread_lo_, pred * self.spread_hi_


def pinball_loss(y, q, level: float) -> float:
    """Quantile (pinball) loss at ``level``. The proper scoring rule for a
    quantile estimate: the mean absolute error would reward a median."""
    y = np.asarray(y, dtype=float)
    q = np.asarray(q, dtype=float)
    d = y - q
    return float(np.mean(np.maximum(level * d, (level - 1) * d)))


BASELINES = {
    ConstantQuantile.name: ConstantQuantile,
    ThroughputArithmetic.name: ThroughputArithmetic,
}
