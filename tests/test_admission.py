"""The admission controller, from the feature vector to the 503.

Three groups of tests, and the first one is the one that matters most:

* **no lookahead.** A feature that is only knowable after the request ran
  produces a spectacular predictor and a worthless guarantee, and the defence
  against it here is structural rather than a convention -- so the structure is
  what is asserted.
* **the policy.** Admit, shed, hysteresis, the backstop, and what the
  ``Retry-After`` header is derived from.
* **the whole path.** A gateway configured with a real (tiny) fitted model,
  driven until it refuses, checked for the shape of what it refuses with.
"""

from __future__ import annotations

import dataclasses
import json
from pathlib import Path

import httpx
import numpy as np
import pytest
from httpx import ASGITransport

from cadence.admission.artifact import PredictorArtifact
from cadence.admission.conformal import SplitConformalUpperBound
from cadence.admission.conformal_controller import ConformalController
from cadence.admission.controller import PassThroughController, QueueCapController
from cadence.admission.features import (
    FEATURES,
    N_FEATURES,
    AdmitContext,
    extract,
    from_frame,
    to_dict,
)
from cadence.admission.predictor import QuantileLatencyPredictor
from cadence.api.app import create_app
from cadence.config import Settings
from cadence.engine.request import Request
from cadence.engine.scheduler.base import Snapshot

I_MAX_TOKENS = FEATURES.index("max_tokens")
I_QUEUE = FEATURES.index("queue_depth")


# --- a small, real, fitted model ----------------------------------------
def _synthetic(n: int, rng) -> tuple[np.ndarray, np.ndarray]:
    """Latency that depends on the two things the policy tests care about.

    A real ``GradientBoostingRegressor`` fitted on it, rather than a stub with
    a hand-written ``predict_quantiles``, so that the artifact, the pickle, the
    feature-name check and the sklearn call are all on the path under test.
    """
    X = np.zeros((n, N_FEATURES))
    X[:, FEATURES.index("n_prompt_tokens")] = rng.uniform(100, 700, n)
    X[:, I_MAX_TOKENS] = rng.uniform(16, 512, n)
    X[:, FEATURES.index("log_max_tokens")] = np.log1p(X[:, I_MAX_TOKENS])
    X[:, I_QUEUE] = rng.integers(0, 40, n)
    X[:, FEATURES.index("running_batch_size")] = rng.integers(0, 16, n)
    y = 0.3 + 0.01 * X[:, I_MAX_TOKENS] + 0.08 * X[:, I_QUEUE] + rng.normal(0, 0.08, n)
    return X, np.maximum(y, 0.05)


def _fit_toy(alpha: float = 0.1, seed: int = 0) -> PredictorArtifact:
    rng = np.random.default_rng(seed)
    Xtr, ytr = _synthetic(600, rng)
    Xca, yca = _synthetic(400, rng)
    # Enough boosting rounds and enough step size to actually leave the
    # initial constant: an under-fitted quantile model predicts the global
    # quantile for every input, which would make the policy tests below pass or
    # fail for reasons that have nothing to do with the policy.
    model = QuantileLatencyPredictor(
        alpha=alpha, n_estimators=300, max_depth=4, learning_rate=0.3,
        random_state=seed,
    ).fit(Xtr, ytr)
    bound = SplitConformalUpperBound(model, alpha=alpha).calibrate(Xca, yca)
    return PredictorArtifact(
        predictor=model, calib_scores=bound.scores_, alpha=alpha,
        feature_names=tuple(FEATURES), meta={"synthetic": True},
    )


@pytest.fixture(scope="module")
def toy() -> PredictorArtifact:
    return _fit_toy()


def _cfg(**kw) -> Settings:
    base = dict(backend="mock", config_name="test", scheduler="continuous",
                n_ctx=4096, max_batch=8, n_parallel=16, slo_s=4.0)
    base.update(kw)
    return Settings(**base)


def _ctx(max_tokens: int, n_prompt: int = 400, cached: int = 0) -> AdmitContext:
    return AdmitContext(prompt_ids=list(range(n_prompt)), max_tokens=max_tokens,
                        n_messages=2, cached_prefix_tokens=cached)


def _snap(**kw) -> Snapshot:
    return Snapshot(**kw)


# --- 1. no lookahead -----------------------------------------------------
def test_the_feature_extractor_cannot_reach_the_outcome():
    """The structural version of "do not leak the label".

    ``extract`` takes exactly two things, and neither of them can be walked
    back to the ``Request``: ``AdmitContext`` holds no reference to one, and
    ``Snapshot`` is a frozen record of scalars. A feature that depended on the
    realised output length could therefore not be written without changing one
    of these two signatures, which is a diff a reviewer would see.
    """
    ctx_fields = {f.name for f in dataclasses.fields(AdmitContext)}
    assert ctx_fields == {"prompt_ids", "max_tokens", "n_messages", "cached_prefix_tokens"}
    outcome_fields = {"output_ids", "finish_reason", "ts", "e2e", "ttft", "n_preemptions"}
    assert not (ctx_fields & outcome_fields)
    assert not (ctx_fields & {f.name for f in dataclasses.fields(Snapshot)})

    on_request = {f.name for f in dataclasses.fields(Request)} | {
        n for n in ("e2e", "ttft") if isinstance(getattr(Request, n, None), property)
    }
    assert outcome_fields <= on_request, "the leak-prone fields live on Request"
    for holder in (AdmitContext, Snapshot):
        for f in dataclasses.fields(holder):
            assert "Request" not in str(f.type), f"{holder.__name__}.{f.name} reaches a Request"


def test_extract_is_the_declared_row_and_nothing_else():
    x = extract(_ctx(128, n_prompt=500, cached=300), _snap(queue_depth=3, batch_size=2))
    assert x.shape == (N_FEATURES,)
    d = to_dict(x)
    assert list(d) == list(FEATURES)
    assert d["n_prompt_tokens"] == 500
    assert d["n_cached_prefix_tokens"] == 300
    assert d["n_new_prefill_tokens"] == 200
    assert d["has_shared_system_prompt"] == 1.0
    assert d["log_max_tokens"] == pytest.approx(np.log1p(128))


def test_a_cache_hit_longer_than_the_prompt_cannot_make_prefill_negative():
    # The probe is best-effort and unsynchronised, so it can report a hit from
    # a moment when the prompt was longer. Clamping is not defensive
    # programming here: a negative feature would be a value the model never saw
    # in training.
    x = to_dict(extract(_ctx(64, n_prompt=100, cached=500), _snap()))
    assert x["n_new_prefill_tokens"] == 0
    assert x["n_cached_prefix_tokens"] == 100


def test_trace_columns_round_trip_into_a_design_matrix():
    import pandas as pd

    rows = [to_dict(extract(_ctx(m), _snap(queue_depth=q))) for m, q in ((32, 0), (256, 9))]
    X = from_frame(pd.DataFrame(rows))
    assert X.shape == (2, N_FEATURES)
    assert X[1, I_QUEUE] == 9
    with pytest.raises(KeyError, match="missing feature columns"):
        from_frame(pd.DataFrame(rows).drop(columns=["queue_depth"]))


# --- 2. the policy -------------------------------------------------------
def test_admits_what_fits_and_sheds_what_does_not(toy):
    c = ConformalController(_cfg(admission="conformal"), artifact=toy)
    cheap = c.decide(_ctx(32), _snap())
    assert cheap.action == "admit" and cheap.reason == "fits"
    assert cheap.predicted_e2e_s is not None and cheap.predicted_e2e_s <= 4.0
    assert cheap.features is not None and cheap.features.shape == (N_FEATURES,)

    dear = c.decide(_ctx(480), _snap(queue_depth=25))
    assert dear.action == "shed" and dear.reason == "predicted_slo_violation"
    assert dear.predicted_e2e_s > 4.0


def test_the_bound_is_conditional_not_constant(toy):
    c = ConformalController(_cfg(admission="conformal"), artifact=toy)
    idle = c.decide(_ctx(128), _snap()).predicted_e2e_s
    busy = c.decide(_ctx(128), _snap(queue_depth=30)).predicted_e2e_s
    longer = c.decide(_ctx(400), _snap()).predicted_e2e_s
    assert busy > idle, "the same request on a busy server must not be cheaper"
    assert longer > idle, "a longer generation must not be cheaper"


def test_hysteresis_keeps_the_shed_admit_loop_from_chattering(toy):
    cfg = _cfg(admission="conformal", admission_hysteresis=0.9)
    c = ConformalController(cfg, artifact=toy)

    # A request whose bound sits between 0.9 * SLO and the SLO: admitted from
    # rest, refused while the controller is in its shedding state.
    marginal = None
    for m in range(16, 512, 4):
        u = c.decide(_ctx(m), _snap()).predicted_e2e_s
        if 0.9 * cfg.slo_s < u <= cfg.slo_s:
            marginal = m
            break
    assert marginal is not None, "no request lands in the hysteresis band"

    c._shedding = False
    assert c.decide(_ctx(marginal), _snap()).action == "admit"
    c.decide(_ctx(500), _snap(queue_depth=30))  # trip into shedding
    assert c._shedding
    assert c.decide(_ctx(marginal), _snap()).action == "shed", "no hysteresis"
    assert c.decide(_ctx(16), _snap()).action == "admit", "never resumed admitting"
    assert not c._shedding


def test_the_queue_cap_is_still_a_backstop(toy):
    c = ConformalController(_cfg(admission="conformal", max_waiting=4), artifact=toy)
    d = c.decide(_ctx(16), _snap(queue_depth=9))
    assert d.action == "shed" and d.reason == "queue_full", (
        "a model that says yes must not be able to override the memory backstop"
    )


def test_retry_after_is_derived_from_the_work_in_flight(toy):
    cfg = _cfg(admission="conformal", retry_after_s=0.5)
    c = ConformalController(cfg, artifact=toy)
    d = c.decide(
        _ctx(500),
        _snap(queue_depth=30, sum_remaining_tokens=2000, ewma_tokens_per_s=200.0),
    )
    assert d.action == "shed"
    assert d.retry_after_s == pytest.approx(10.0)  # 2000 tokens / 200 tok/s

    # Floor when the engine is idle enough that the estimate is meaningless,
    # ceiling so a spike cannot park a client for minutes.
    assert c._retry_after(_snap(sum_remaining_tokens=0, ewma_tokens_per_s=200.0)) == 0.5
    assert c._retry_after(_snap(sum_remaining_tokens=10**6, ewma_tokens_per_s=1.0)) == 30.0


def test_observe_counts_coverage_and_skips_censored_requests(toy):
    c = ConformalController(_cfg(admission="conformal", admission_mode="rolling"), artifact=toy)
    x = extract(_ctx(64), _snap())

    def _rq(e2e, reason="stop", cancelled=False, bound=3.0):
        rq = Request(rid="r", prompt_ids=[1], max_tokens=64, deadline=0.0)
        rq.features, rq.predicted_e2e_s = x, bound
        rq.finish_reason = reason
        rq._cancelled = cancelled
        rq.ts["finished"] = rq.arrival + e2e
        return rq

    c.observe(_rq(1.0))
    c.observe(_rq(9.0))  # over its bound: a violation
    assert c.n_observed == 2 and c.n_violations == 1
    assert c.stats()["online_coverage"] == pytest.approx(0.5)

    c.observe(_rq(300.0, reason=None, cancelled=True))
    assert c.n_censored == 1
    assert c.n_observed == 2, (
        "a client timeout is a lower bound on the latency, not an observation of it"
    )
    assert c.stats()["online_coverage"] == pytest.approx(0.5)


def test_pass_through_and_queue_cap_share_the_interface():
    snap = _snap(queue_depth=10)
    assert PassThroughController().decide(_ctx(64), snap).action == "admit"
    cap = QueueCapController(_cfg(max_waiting=4))
    assert cap.decide(_ctx(64), snap).action == "shed"
    assert cap.decide(_ctx(64), _snap(queue_depth=1)).action == "admit"


# --- the artifact --------------------------------------------------------
def test_artifact_round_trips_and_refuses_a_feature_mismatch(toy, tmp_path):
    path = tmp_path / "admission.pkl"
    toy.save(path)
    back = PredictorArtifact.load(path)
    assert back.alpha == toy.alpha
    assert tuple(back.feature_names) == tuple(FEATURES)
    np.testing.assert_allclose(back.calib_scores, toy.calib_scores)
    sidecar = json.loads((tmp_path / "admission.json").read_text())
    assert sidecar["n_calib"] == len(toy.calib_scores)

    stale = PredictorArtifact(
        predictor=toy.predictor, calib_scores=toy.calib_scores, alpha=toy.alpha,
        feature_names=tuple(FEATURES)[:-1],
    )
    with pytest.raises(ValueError, match="different features"):
        ConformalController(_cfg(admission="conformal"), artifact=stale)


def test_a_missing_model_says_what_to_do_about_it(tmp_path):
    with pytest.raises(FileNotFoundError, match="fit_predictor"):
        PredictorArtifact.load(tmp_path / "nope.pkl")


# --- the trap the warm-up exists to break --------------------------------
def test_the_engine_warms_itself_before_it_serves(toy, tmp_path):
    """A gateway that has never run a step reports zero for both EWMA
    features, and that combination is nowhere in the training set, because the
    collection drops each block's first twenty seconds.

    Left alone it is a closed loop: the bound comes out too large, the request
    is shed, the engine stays idle, the next request sees the same zeros.
    Measured live before the fix, a 1 rps load was shed 89 times out of 89 --
    including the load generator's own warm-up request, which arrives through
    the same door as everything else.

    So the warm-up happens inside the engine, where admission cannot refuse it.
    The assertion is on the mechanism rather than on the shed rate, because the
    shed rate depends on a fitted model and the invariant does not: after
    ``start()``, the state the controller reads describes an engine that has
    run.
    """
    from cadence.engine.engine import Engine

    model = tmp_path / "admission.pkl"
    toy.save(model)
    # An admission policy that refuses everything: if the warm-up went through
    # it, the engine would come up cold and this test would fail.
    cfg = _cfg(admission="conformal", admission_model=str(model), admission_safety=1e6)
    eng = Engine(cfg)
    try:
        before = eng.scheduler.snapshot()
        assert before.ewma_step_latency_s == 0.0
        eng.start()
        after = eng.scheduler.snapshot()
        assert after.ewma_step_latency_s > 0.0, "the engine served without ever stepping"
        assert after.ewma_tokens_per_s > 0.0
        # And it left nothing behind: the warm-up prompt is shorter than a KV
        # block, so it cannot be donated to the prefix cache, and the first
        # real arrival still finds a cold one.
        st = eng.stats()
        assert st["kv_blocks_free"] == st["kv_blocks_total"]
        assert st["prefix_hit_rate"] == 0.0
    finally:
        eng.stop()


# --- 3. the whole path ---------------------------------------------------
@pytest.mark.asyncio
async def test_the_gateway_sheds_with_a_503_and_a_retry_after(toy, tmp_path):
    model = tmp_path / "admission.pkl"
    toy.save(model)
    trace = tmp_path / "trace.jsonl"
    cfg = _cfg(
        admission="conformal", admission_model=str(model), slo_s=4.0,
        trace_log=str(trace), max_tokens_cap=512,
        # A slow mock, so that a burst really does build a queue the controller
        # can see rather than a queue that drains before it is measured.
        mock_step_overhead_s=0.02, mock_decode_s_per_seq=0.004,
    )
    app = create_app(cfg)
    async with app.router.lifespan_context(app), httpx.AsyncClient(
        transport=ASGITransport(app=app), base_url="http://test"
    ) as c:
        import asyncio

        async def one(max_tokens: int):
            return await c.post(
                "/v1/chat/completions",
                json={"model": "qwen", "stream": False, "max_tokens": max_tokens,
                      "messages": [{"role": "user", "content": "x" * 400}]},
            )

        first = await one(8)
        assert first.status_code == 200, "an idle server refused a trivial request"

        results = await asyncio.gather(*[one(512) for _ in range(24)])
        shed = [r for r in results if r.status_code == 503]
        assert shed, "nothing was shed under a burst of 512-token requests"
        body = shed[0].json()["error"]
        assert body["code"] == "slo_shed"
        assert float(shed[0].headers["Retry-After"]) >= 0.0
        assert all(r.status_code in (200, 503) for r in results)

        st = (await c.get("/stats")).json()
        assert st["admission"] == "conformal"
        assert st["controller"]["shed"] == len(shed)
        assert st["controller"]["mode"] == "static"
        assert st["controller"]["q_s"] is not None

    with Path(trace).open() as fh:
        rows = [json.loads(line) for line in fh]
    assert len(rows) >= 25, "the trace did not record every decision"
    assert {r["action"] for r in rows} == {"admit", "shed"}
    for r in rows:
        assert set(FEATURES) <= set(r), "a trace row is missing feature columns"
        assert r["u_bound_s"] is not None
        if r["action"] == "shed":
            assert "e2e_s" not in r, "a shed request cannot have an outcome"
        else:
            assert "e2e_s" in r and "censored" in r


@pytest.mark.asyncio
async def test_the_snapshot_the_controller_reads_actually_moves():
    """The features are only worth anything if they track the engine.

    A snapshot that stayed at its defaults would still fit a model, still pass
    every test above, and would be predicting from the request alone.
    """
    cfg = _cfg(trace_log=None, mock_step_overhead_s=0.01, mock_decode_s_per_seq=0.003)
    app = create_app(cfg)
    async with app.router.lifespan_context(app), httpx.AsyncClient(
        transport=ASGITransport(app=app), base_url="http://test"
    ) as c:
        import asyncio

        sched = app.state.engine.scheduler
        before = sched.snapshot()
        assert before.arrival_rate == 0.0

        async def one():
            return await c.post(
                "/v1/chat/completions",
                json={"model": "qwen", "stream": False, "max_tokens": 64,
                      "messages": [{"role": "user", "content": "y" * 300}]},
            )

        task = asyncio.gather(*[one() for _ in range(12)])
        seen = []
        for _ in range(200):
            seen.append(sched.snapshot())
            await asyncio.sleep(0.01)
            if task.done():
                break
        await task

        assert max(s.batch_size for s in seen) > 1
        assert max(s.sum_remaining_tokens for s in seen) > 0
        assert max(s.ewma_step_latency_s for s in seen) > 0
        assert max(s.ewma_tokens_per_s for s in seen) > 0
        assert sched.snapshot().arrival_rate > 0
        assert min(s.kv_free_frac for s in seen) < 1.0
