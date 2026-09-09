"""Tests for the measurement harness itself.

The harness is the one component whose bugs are invisible in the results: a
closed loop hiding inside the generator produces plausible-looking numbers that
are simply wrong. So it is tested more carefully than anything it measures.
"""

from __future__ import annotations

import asyncio
import random

import numpy as np
import pytest

from loadgen import RunConfig, ks_exponential, run
from workloads import build_workload


class _FakeClient:
    """Records intended-vs-actual timing without any network."""

    def __init__(self, service_s: float, capacity: int) -> None:
        self.service_s = service_s
        self.sem = asyncio.Semaphore(capacity) if capacity else None
        self.sends: list[float] = []

    def stream(self, method, url, json=None, timeout=None):
        return _FakeStream(self, json)

    async def __aenter__(self):
        return self

    async def __aexit__(self, *a):
        return False


class _FakeStream:
    def __init__(self, client: _FakeClient, payload) -> None:
        self.client = client
        self.n = int(payload.get("max_tokens", 4))
        self.status_code = 200

    async def __aenter__(self):
        if self.client.sem is not None:
            await self.client.sem.acquire()
        await asyncio.sleep(self.client.service_s)
        return self

    async def __aexit__(self, *a):
        if self.client.sem is not None:
            self.client.sem.release()
        return False

    async def aiter_lines(self):
        for i in range(self.n):
            await asyncio.sleep(0)
            yield 'data: {"choices":[{"delta":{"content":"t' + str(i) + ' "}}]}'
        yield "data: [DONE]"

    async def aread(self):
        return b""


def _patched_run(monkeypatch, client):
    import httpx

    monkeypatch.setattr(httpx, "AsyncClient", lambda **kw: client)


def test_arrivals_are_exponential_not_fixed_interval():
    """A fixed-interval generator understates queueing; the KS test is what
    catches a well-meaning 'simplification' to sleep(1/rate)."""
    rng = random.Random(11)
    gaps = [rng.expovariate(8.0) for _ in range(4000)]
    d, p = ks_exponential(gaps, 8.0)
    assert p > 0.05, f"exponential gaps rejected: D={d}, p={p}"

    d_fixed, p_fixed = ks_exponential([1 / 8.0] * 4000, 8.0)
    assert p_fixed < 1e-6 and d_fixed > 0.5


@pytest.mark.parametrize(("rate", "duration"), [(5.0, 60.0), (1.4, 180.0)])
def test_ks_p_values_are_uniform_over_seeds(rate, duration):
    """The generator's own arrival process, tested the way it is actually
    driven -- gaps interleaved with workload sampling from the same RNG.

    The second case is the regime the published sweep runs in. It matters
    because the sweep fixes one seed in advance so every configuration replays
    the same arrivals, and that seed turns out to sit in the tail of this
    distribution: the process has to be established over seeds, not read off
    the one that was used.
    """
    wl = build_workload("mixed")
    ps, ns = [], []
    for seed in range(120):
        rng = random.Random(seed)
        t, gaps = 0.0, []
        while t < duration:
            g = rng.expovariate(rate)
            t += g
            gaps.append(g)
            wl.sample(rng)
        ns.append(len(gaps))
        ps.append(ks_exponential(gaps, rate)[1])
    ps, ns = np.array(ps), np.array(ns)
    assert np.mean(ps < 0.05) <= 0.15, f"reject rate {np.mean(ps < 0.05)}"
    assert 0.35 < np.median(ps) < 0.65, f"median p {np.median(ps)}"
    # Arrivals per run must look Poisson: mean ~= rate*duration, sd ~= sqrt(mean).
    expected = rate * duration
    assert abs(ns.mean() - expected) < 0.25 * np.sqrt(expected) * 3
    assert 0.6 < ns.std() / np.sqrt(expected) < 1.6


@pytest.mark.asyncio
async def test_outstanding_requests_do_not_delay_arrivals(monkeypatch):
    """The open-loop property, asserted directly: the service time is longer
    than the mean inter-arrival gap, and the realised arrival rate must be
    unaffected by that."""
    client = _FakeClient(service_s=0.5, capacity=0)
    _patched_run(monkeypatch, client)
    cfg = RunConfig(rate_rps=20.0, duration_s=4.0, warmup_s=0, cooldown_s=0, seed=5)
    df = await run(cfg, build_workload("uniform"))
    span = df.t_intended.max() - df.t_intended.min()
    realised = (len(df) - 1) / span
    assert 15.0 < realised < 26.0, f"offered load collapsed to {realised:.1f} rps"
    # The generator's own scheduling slip must stay far below the service time.
    assert df.sched_delay.max() < 0.1


@pytest.mark.asyncio
async def test_latency_is_measured_from_intended_arrival(monkeypatch):
    """Coordinated omission check: against a server that can only serve a
    fraction of the offered load, e2e must grow without bound rather than
    plateau at the service time."""
    client = _FakeClient(service_s=0.1, capacity=1)  # 10 rps of capacity
    _patched_run(monkeypatch, client)
    cfg = RunConfig(rate_rps=25.0, duration_s=6.0, warmup_s=0, cooldown_s=0, seed=1)
    df = (await run(cfg, build_workload("uniform"))).sort_values("t_rel")
    first = df[df.t_rel < 2.0].e2e.median()
    last = df[df.t_rel > 4.0].e2e.median()
    assert last > first * 2, f"latency plateaued ({first:.2f} -> {last:.2f}): closed loop?"
    assert last > 1.0, "queue never built; the generator is not open-loop"


@pytest.mark.asyncio
async def test_warmup_and_cooldown_are_flagged_not_dropped(monkeypatch):
    client = _FakeClient(service_s=0.01, capacity=0)
    _patched_run(monkeypatch, client)
    cfg = RunConfig(rate_rps=20.0, duration_s=6.0, warmup_s=2.0, cooldown_s=1.0, seed=2)
    df = await run(cfg, build_workload("uniform"))
    assert (~df.steady).sum() > 0
    assert df[df.steady].t_rel.min() > 2.0
    assert df[df.steady].t_rel.max() < 5.0
    assert len(df) > df.steady.sum()  # nothing was silently discarded
