"""Readiness, draining and the concurrency cap.

Week 5's deployment checklist makes three operational claims -- separate
liveness and readiness, a drain that finishes what it started, and a hard cap
that keeps the process from being OOM-killed. Each of those is a claim about
behaviour under conditions that never occur in a normal test run, which is
exactly the kind of claim that is true in the README and false in the process.
So each one is asserted here.
"""

from __future__ import annotations

import asyncio
import json

import httpx
import pytest
from httpx import ASGITransport

from cadence.api.app import create_app
from cadence.api.lifecycle import Lifecycle
from cadence.config import Settings


def _cfg(**kw):
    base = dict(backend="mock", config_name="test", scheduler="continuous",
                n_ctx=4096, max_batch=8, n_parallel=16, slo_s=60.0)
    base.update(kw)
    return Settings(**base)


async def _client(app):
    return httpx.AsyncClient(transport=ASGITransport(app=app), base_url="http://test")


# --- the cap, as a unit ---------------------------------------------------


def test_cap_refuses_at_the_limit_and_recovers_on_release():
    life = Lifecycle(max_concurrent=2)
    assert life.acquire() and life.acquire()
    assert not life.acquire(), "the cap admitted one past its limit"
    assert life.n_rejected_capacity == 1
    life.release()
    assert life.acquire(), "a released slot was not reusable"
    assert life.peak_in_flight == 2


def test_release_cannot_drive_the_counter_negative():
    """A double release is a bug in the route. The right behaviour for it in
    production is a wrong gauge, not a 500 in the middle of a stream."""
    life = Lifecycle(max_concurrent=1)
    life.acquire()
    life.release()
    life.release()
    assert life.in_flight == 0


def test_drain_is_idempotent():
    life = Lifecycle(max_concurrent=1)
    life.begin_drain()
    t0 = life.drain_started_at
    life.begin_drain()
    assert life.drain_started_at == t0, "an impatient second SIGTERM reset the drain clock"


# --- readiness ------------------------------------------------------------


@pytest.mark.asyncio
async def test_ready_is_503_before_the_model_is_loaded():
    """Liveness and readiness are different questions. Before the lifespan has
    run, the process is alive and must not be sent traffic."""
    app = create_app(_cfg())
    async with await _client(app) as c:
        r = await c.get("/ready")
    assert r.status_code == 503
    assert r.json()["status"] == "loading"


@pytest.mark.asyncio
async def test_ready_then_draining():
    app = create_app(_cfg())
    async with app.router.lifespan_context(app), await _client(app) as c:
        r = await c.get("/ready")
        assert r.status_code == 200 and r.json()["status"] == "ready"

        app.state.lifecycle.begin_drain()

        r = await c.get("/ready")
        assert r.status_code == 503, "a draining process still asked for traffic"
        assert r.json()["status"] == "draining"

        # Liveness deliberately stays green: a supervisor that restarts a
        # process for being mid-drain is fighting the drain.
        assert (await c.get("/health")).status_code == 200


@pytest.mark.asyncio
async def test_draining_refuses_new_work_with_a_retry_after():
    app = create_app(_cfg())
    async with app.router.lifespan_context(app), await _client(app) as c:
        app.state.lifecycle.begin_drain()
        r = await c.post(
            "/v1/chat/completions",
            json={"model": "qwen", "stream": False, "max_tokens": 4,
                  "messages": [{"role": "user", "content": "hi"}]},
        )
    assert r.status_code == 503
    assert r.json()["error"]["code"] == "draining"
    assert float(r.headers["Retry-After"]) > 0, (
        "a refusal without a Retry-After is a failure; with one it is a "
        "scheduling instruction"
    )


# --- the cap, through the API --------------------------------------------


@pytest.mark.asyncio
async def test_cap_sheds_and_is_counted_apart_from_the_conformal_sheds():
    """A capacity refusal is not a statement about latency, so it must not be
    able to dilute a coverage number by hiding inside the same counter."""
    from cadence.obs.metrics import REGISTRY

    def counter(reason: str) -> float:
        # The registry is process-global, so read a delta rather than an
        # absolute: this test must not depend on what ran before it.
        v = REGISTRY.get_sample_value(
            "cadence_shed_total", {"config": "test", "reason": reason}
        )
        return 0.0 if v is None else float(v)

    before_cap = counter("capacity")
    before_slo = counter("predicted_slo_violation")

    app = create_app(_cfg(max_concurrent_requests=0))
    async with app.router.lifespan_context(app), await _client(app) as c:
        r = await c.post(
            "/v1/chat/completions",
            json={"model": "qwen", "stream": False, "max_tokens": 4,
                  "messages": [{"role": "user", "content": "hi"}]},
        )
        assert r.status_code == 503
        assert r.json()["error"]["code"] == "capacity"
        stats = (await c.get("/stats")).json()
    assert stats["rejected_capacity"] == 1

    assert counter("capacity") - before_cap == 1.0
    assert counter("predicted_slo_violation") == before_slo, (
        "an operational refusal was counted as a conformal shed"
    )


@pytest.mark.asyncio
async def test_in_flight_returns_to_zero_on_every_exit_path():
    """The slot is taken before the admission decision and given back on four
    different paths -- shed, error, non-streaming completion, and the end of a
    stream. A leak on any of them turns the cap into a slow strangulation of a
    healthy process, which is the failure mode it exists to prevent.
    """
    app = create_app(_cfg())
    payload = {"model": "qwen", "stream": False, "max_tokens": 4,
               "messages": [{"role": "user", "content": "hi"}]}
    async with app.router.lifespan_context(app), await _client(app) as c:
        await c.post("/v1/chat/completions", json=payload)
        async with c.stream(
            "POST", "/v1/chat/completions", json={**payload, "stream": True}
        ) as r:
            async for _ in r.aiter_lines():
                pass
        # A client that hangs up mid-stream: the slot has to come back with
        # the KV blocks, not after them.
        life = app.state.lifecycle
        assert life.in_flight == 0, f"{life.in_flight} slots leaked"
        assert life.peak_in_flight >= 1, "the cap never saw the requests at all"


@pytest.mark.asyncio
async def test_concurrent_streams_are_bounded_by_the_cap():
    """The interesting case is not one request at a time: it is many at once,
    which is the only condition under which the cap does anything."""
    app = create_app(_cfg(max_concurrent_requests=3))
    payload = {"model": "qwen", "stream": True, "max_tokens": 24,
               "messages": [{"role": "user", "content": "hello there"}]}

    async def one(c):
        async with c.stream("POST", "/v1/chat/completions", json=payload) as r:
            if r.status_code != 200:
                await r.aread()
                return r.status_code
            async for _ in r.aiter_lines():
                pass
            return 200

    async with app.router.lifespan_context(app), await _client(app) as c:
        codes = await asyncio.gather(*(one(c) for _ in range(8)))

    assert codes.count(200) >= 1
    assert 503 in codes, "eight concurrent streams against a cap of three shed nothing"
    life = app.state.lifecycle
    assert life.peak_in_flight <= 3, f"the cap was exceeded: peak {life.peak_in_flight}"
    assert life.in_flight == 0


@pytest.mark.asyncio
async def test_a_stream_started_before_the_drain_finishes_after_it():
    """The drain's whole point: stop taking work, finish what you took. A
    shutdown that truncates the running batch puts a burst of half-finished
    responses into exactly the tail this project claims to control.
    """
    app = create_app(_cfg())
    payload = {"model": "qwen", "stream": True, "max_tokens": 40,
               "messages": [{"role": "user", "content": "count for me"}]}
    async with app.router.lifespan_context(app), await _client(app) as c:
        drained = False
        frames = 0
        finish = None
        async with c.stream("POST", "/v1/chat/completions", json=payload) as r:
            assert r.status_code == 200
            async for line in r.aiter_lines():
                if not line.startswith("data: ") or line == "data: [DONE]":
                    continue
                chunk = json.loads(line[6:])
                frames += 1
                if frames == 2 and not drained:
                    app.state.lifecycle.begin_drain()
                    drained = True
                if chunk["choices"][0].get("finish_reason"):
                    finish = chunk["choices"][0]["finish_reason"]

        assert drained
        assert finish in {"stop", "length"}, (
            f"the drain truncated a running stream (finish_reason={finish!r})"
        )
        # And the door really was shut behind it.
        r = await c.post("/v1/chat/completions", json={**payload, "stream": False})
        assert r.status_code == 503
