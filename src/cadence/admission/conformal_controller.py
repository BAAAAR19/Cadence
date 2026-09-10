"""The Week 4 policy: admit if the conformal upper bound fits the SLO.

    U(x) = q_hi(x) + Q      (cadence.admission.conformal)
    admit  iff  U(x) * safety <= budget

with ``budget`` the SLO measured from arrival, tightened by hysteresis while
the controller is shedding.

Three details are the whole policy, and each of them is a decision that was
taken rather than a default that happened.

**The bound, not the point prediction.** A point prediction that is right on
average admits half the requests that will miss, because the thing being
bounded is a tail. The dominant source of uncertainty is the output length,
which is genuinely unknown at admission and lognormally distributed in this
workload, so the width of the interval is not a modelling failure to be tuned
away -- it is the actual uncertainty, and the controller has to act on it.

**Hysteresis.** Shedding is a positive feedback loop: shedding lowers load,
which lowers the predicted latency, which admits more, which raises load. The
standard damping is a Schmitt trigger, and that is what this is -- once
shedding, the controller requires the bound to fit inside ``hysteresis *
budget`` (0.9 by default) before it resumes admitting, so the two thresholds
differ and the system cannot chatter between them at the boundary. The writeup
plots admitted rate over time with it on.

**Retry-After that means something.** The queue's remaining decode work
divided by the measured token rate is an estimate of when capacity will exist,
which is what a client backing off wants to know. A constant would be a
placeholder wearing a header's clothes.
"""

from __future__ import annotations

import time

import numpy as np

from cadence.admission.artifact import PredictorArtifact
from cadence.admission.conformal import build_bound
from cadence.admission.controller import Decision
from cadence.admission.features import FEATURES, AdmitContext, extract
from cadence.obs.metrics import Metrics


def _check_domain(art: PredictorArtifact, cfg) -> str | None:
    """Say, loudly, when the bound is being asked about a distribution it was
    not calibrated on.

    A split-conformal bound guarantees coverage under exchangeability with the
    calibration set. Nothing about it degrades gracefully when that fails: it
    does not become a slightly worse bound, it becomes a number with no
    relationship to the quantity it claims to bound. The way that presents in
    this system is spectacular and, without this check, unexplained -- the
    Metal-fitted artifact loaded against the mock backend refuses every single
    request at zero offered load, because it is predicting four-second
    latencies for a backend whose requests take four hundred milliseconds, and
    every one of those refusals looks exactly like correct overload behaviour.

    Not fatal. A mismatch is legitimate while deliberately measuring one (the
    Week 5 writeup does), and refusing to start would be the wrong response to
    a warning that is right 100% of the time and important 1% of it. Silence
    would be worse than either.
    """
    fitted = (art.meta or {}).get("fitted_on") or {}
    backend = fitted.get("backend")
    if not backend:
        # Artifacts fitted before the fingerprint existed. Nothing can be
        # checked, and saying so once is better than implying it was checked.
        return (
            f"admission model {getattr(cfg, 'admission_model', '?')} carries no "
            f"calibration fingerprint; its domain cannot be verified"
        )
    if backend == cfg.backend:
        return None
    return (
        f"admission model was calibrated on backend {backend!r} and is being "
        f"served on {cfg.backend!r}. A conformal bound is only valid on data "
        f"exchangeable with its calibration set; on a different backend it is "
        f"not conservative, it is arbitrary, and the usual symptom is that "
        f"every request is shed at zero load. Refit with "
        f"`uv run bench/fit_predictor.py` on this backend, or run with "
        f"CADENCE_ADMISSION=none."
    )


class ConformalController:
    """Admit / shed on a conformal upper bound. Rung 5 of the ladder."""

    name = "conformal"

    def __init__(self, cfg, metrics: Metrics | None = None,
                 artifact: PredictorArtifact | None = None) -> None:
        self.cfg = cfg
        self.metrics = metrics if metrics is not None else Metrics(cfg.config_name)
        art = artifact if artifact is not None else PredictorArtifact.load(cfg.admission_model)
        if tuple(art.feature_names) != tuple(FEATURES):
            raise ValueError(
                "admission model was fitted on different features than this build "
                f"extracts:\n  model: {list(art.feature_names)}\n  build: {list(FEATURES)}"
            )
        self.artifact = art
        self.domain_warning = _check_domain(art, cfg)
        if self.domain_warning:
            print(f"warning: {self.domain_warning}", flush=True)
        self.alpha = cfg.admission_alpha if cfg.admission_alpha > 0 else art.alpha
        self.bound = build_bound(
            cfg.admission_mode,
            art.predictor,
            self.alpha,
            art.score,
            window=cfg.admission_window,
            refresh=cfg.admission_refresh,
            gamma=cfg.admission_aci_gamma,
        )
        self.bound.calibrate_scores(art.calib_scores)

        self.slo_s = cfg.slo_s
        self.safety = cfg.admission_safety
        self.hysteresis = cfg.admission_hysteresis
        self.max_waiting = cfg.max_waiting
        self._shedding = False

        self.n_admitted = 0
        self.n_shed = 0
        self.n_observed = 0
        self.n_violations = 0
        self.n_censored = 0
        self._publish_bound_state()

    # --- the decision -----------------------------------------------------
    def decide(self, ctx: AdmitContext, snap) -> Decision:
        t0 = time.perf_counter()
        x = extract(ctx, snap)
        u = float(self.bound.upper_bound(x[None, :])[0]) * self.safety
        budget = self.slo_s * (self.hysteresis if self._shedding else 1.0)

        if snap.queue_depth >= self.max_waiting:
            # The backstop from Week 1, kept: a model can be wrong, and running
            # out of memory is worse than shedding one request too many.
            d = self._shed(x, u, budget, snap, reason="queue_full")
        elif u <= budget:
            self._shedding = False
            self.n_admitted += 1
            d = Decision(
                action="admit", reason="fits", predicted_e2e_s=u, slack_s=budget, features=x
            )
        else:
            self._shedding = True
            d = self._shed(x, u, budget, snap, reason="predicted_slo_violation")

        self.metrics.admission_decision.observe(time.perf_counter() - t0)
        self.metrics.admission_bound.observe(u)
        return d

    def _shed(self, x, u: float, budget: float, snap, reason: str) -> Decision:
        self.n_shed += 1
        return Decision(
            action="shed",
            retry_after_s=self._retry_after(snap),
            reason=reason,
            predicted_e2e_s=u,
            slack_s=budget,
            features=x,
        )

    def _retry_after(self, snap) -> float:
        """When capacity is expected to exist, from the work already committed.

        ``sum_remaining_tokens`` is every running sequence's unwritten output;
        divided by the measured aggregate token rate it is the time to drain
        the batch. Bounded below by the configured floor so a client cannot be
        told to come back instantly, and above at 30 s so a transient spike in
        the estimate cannot park a caller for minutes.
        """
        rate = snap.ewma_tokens_per_s
        drain = snap.sum_remaining_tokens / rate if rate > 1e-6 else self.cfg.retry_after_s
        return float(np.clip(drain, self.cfg.retry_after_s, 30.0))

    # --- learning from what happened --------------------------------------
    def observe(self, rq) -> None:
        """One admitted request reached a terminal state.

        Feeds the online modes and, either way, counts coverage as it actually
        came out -- the number the writeup reports next to the offline one,
        because the controller's own shedding is what breaks the exchangeability
        the offline number assumes.
        """
        y = rq.e2e
        if y is None or rq.features is None:
            return
        if rq.cancelled or rq.finish_reason not in ("stop", "length"):
            # The client hung up, or the request failed. Its latency is a
            # censored observation of the thing being predicted -- the true
            # value is *at least* this -- and feeding it in as though it were
            # complete would drag the bound down exactly when the system is
            # slowest.
            self.n_censored += 1
            return
        self.n_observed += 1
        if rq.predicted_e2e_s is not None and y > rq.predicted_e2e_s:
            self.n_violations += 1
        self.bound.observe(rq.features, y, hi=self._hi_of(rq))
        self._publish_bound_state()

    def _hi_of(self, rq) -> float | None:
        """The model's own upper quantile for this request, recovered from the
        bound it was admitted on, so that the online recalibrator does not run
        the model a second time on the response path.

        Only the two scores that need no lower quantile can be inverted this
        cheaply; under the scaled score the recalibrator re-runs the model,
        which is why that score is not the one deployed.
        """
        if rq.predicted_e2e_s is None or self.bound.q_ is None:
            return None
        u = rq.predicted_e2e_s / self.safety
        if self.bound.score == "absolute":
            return u - self.bound.q_
        if self.bound.score == "ratio":
            return float(np.expm1(np.log1p(max(u, 0.0)) - self.bound.q_))
        return None

    def _publish_bound_state(self) -> None:
        self.metrics.admission_q.set(self.bound.q_ or 0.0)
        self.metrics.admission_alpha.set(self.bound.alpha_working)

    # --- introspection ----------------------------------------------------
    def stats(self) -> dict:
        n = self.n_admitted + self.n_shed
        return {
            "policy": self.name,
            "calibrated_on": (self.artifact.meta or {}).get("fitted_on"),
            # Surfaced and not only logged: a start-up warning scrolls past,
            # and the question "is this bound valid here" is one an operator
            # asks of a running process.
            "domain_warning": self.domain_warning,
            "mode": self.bound.mode,
            "score": self.bound.score,
            "alpha": self.alpha,
            "alpha_working": self.bound.alpha_working,
            "q_s": self.bound.q_,
            "n_calib": self.bound.n_calib_,
            "safety": self.safety,
            "admitted": self.n_admitted,
            "shed": self.n_shed,
            "shed_rate": self.n_shed / n if n else 0.0,
            "observed": self.n_observed,
            "censored": self.n_censored,
            # None, not NaN: this dict is serialised by /stats and /health,
            # and a NaN makes the whole response unencodable -- which shows up
            # as a gateway that never becomes healthy, with the real cause four
            # frames down a JSON encoder traceback.
            "online_coverage": (
                1.0 - self.n_violations / self.n_observed if self.n_observed else None
            ),
        }
