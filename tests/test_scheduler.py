"""Scheduler behaviour, on the deterministic backend.

These are the properties the Week 2 measurements depend on being true. They run
without the model so that CI can enforce them.
"""

from __future__ import annotations

import asyncio
import contextlib

import httpx
import pytest
from httpx import ASGITransport

from cadence.api.app import create_app
from cadence.config import Settings


def _cfg(**kw):
    base = dict(
        backend="mock", config_name="test", n_ctx=8192, block_size=16,
        max_batch=8, n_parallel=24, scheduler="continuous", slo_s=60.0,
    )
    base.update(kw)
    return Settings(**base)


async def _fire(client, n: int, *, max_tokens=8, system="sys", user="u", stagger=0.0):
    async def one(i):
        if stagger:
            await asyncio.sleep(i * stagger)
        r = await client.post(
            "/v1/chat/completions",
            json={
                "model": "qwen", "stream": False, "max_tokens": max_tokens,
                "messages": [
                    {"role": "system", "content": system},
                    {"role": "user", "content": f"{user}-{i}"},
                ],
            },
            timeout=120.0,
        )
        return r

    return await asyncio.gather(*[one(i) for i in range(n)])


class _App:
    def __init__(self, cfg):
        self.cfg = cfg

    async def __aenter__(self):
        self.app = create_app(self.cfg)
        self._lifespan = self.app.router.lifespan_context(self.app)
        await self._lifespan.__aenter__()
        self.client = httpx.AsyncClient(
            transport=ASGITransport(app=self.app), base_url="http://test"
        )
        return self

    @property
    def engine(self):
        return self.app.state.engine

    async def __aexit__(self, *exc):
        await self.client.aclose()
        await self._lifespan.__aexit__(*exc)


@pytest.mark.asyncio
async def test_continuous_batching_runs_more_than_one_sequence_at_a_time():
    async with _App(_cfg()) as a:
        sizes: list[int] = []

        async def watch():
            while True:
                sizes.append(len(a.engine.scheduler.running))
                await asyncio.sleep(0.002)

        w = asyncio.create_task(watch())
        rs = await _fire(a.client, 8, max_tokens=24)
        w.cancel()
        assert all(r.status_code == 200 for r in rs)
        assert max(sizes) > 1, "the scheduler never batched anything"


@pytest.mark.asyncio
async def test_a_late_arrival_joins_the_running_batch():
    """The defining property of continuous batching: a request that arrives
    while others are decoding does not wait for them to finish."""
    async with _App(_cfg(max_batch=8)) as a:
        sched = a.engine.scheduler
        # ~1500 mock decode steps: long enough that the assertions below are
        # about scheduling rather than about how busy the machine happens to be.
        long_task = asyncio.create_task(_fire(a.client, 1, max_tokens=1500, user="long"))
        while not sched.running:
            await asyncio.sleep(0.005)

        joined = asyncio.get_running_loop().time()
        short = await _fire(a.client, 1, max_tokens=4, user="short")
        elapsed = asyncio.get_running_loop().time() - joined

        assert short[0].status_code == 200
        assert not long_task.done(), "the long request finished; nothing was tested"
        assert elapsed < 1.0, (
            f"a 4-token request took {elapsed:.2f}s to get through while a long "
            "one was decoding: it waited for the batch instead of joining it"
        )
        long_task.cancel()
        with contextlib.suppress(asyncio.CancelledError, httpx.ReadError):
            await long_task


@pytest.mark.asyncio
async def test_static_batching_holds_the_batch_open():
    """The contrast that makes rung 3's win legible: under static batching a
    short request submitted with a long one cannot finish before it."""
    async with _App(_cfg(scheduler="static", static_batch_size=4,
                         static_fill_timeout_s=0.2)) as a:
        async def one(max_tokens, tag):
            r = await a.client.post(
                "/v1/chat/completions",
                json={"model": "qwen", "stream": False, "max_tokens": max_tokens,
                      "messages": [{"role": "user", "content": tag}]},
                timeout=120.0,
            )
            return asyncio.get_running_loop().time(), r

        (t_short, r_short), (t_long, r_long) = await asyncio.gather(
            one(4, "short"), one(120, "long")
        )
        assert r_short.status_code == r_long.status_code == 200
        assert abs(t_short - t_long) < 0.3, (
            "the short request escaped the wave; this is not static batching"
        )


@pytest.mark.asyncio
async def test_kv_blocks_return_to_baseline_after_a_run():
    """Pitfall 9: leaked KV blocks make throughput decay silently over a long
    run, and every measurement after the leak drifts."""
    async with _App(_cfg(enable_prefix_cache=False)) as a:
        sched = a.engine.scheduler
        baseline = sched.blocks.free_blocks()
        await _fire(a.client, 12, max_tokens=16)
        for _ in range(200):
            if sched.blocks.free_blocks() == baseline and not sched.running:
                break
            await asyncio.sleep(0.01)
        assert sched.blocks.free_blocks() == baseline
        assert sched.seq_ids.available == sched.seq_ids.n


@pytest.mark.asyncio
async def test_client_disconnect_frees_blocks():
    async with _App(_cfg(enable_prefix_cache=False)) as a:
        sched = a.engine.scheduler
        baseline = sched.blocks.free_blocks()
        payload = {
            "model": "qwen", "stream": True, "max_tokens": 400,
            "messages": [{"role": "user", "content": "long one"}],
        }
        async with a.client.stream("POST", "/v1/chat/completions", json=payload) as r:
            n = 0
            async for _line in r.aiter_lines():
                n += 1
                if n > 5:
                    break  # walk away mid-stream
        for _ in range(300):
            if sched.blocks.free_blocks() == baseline and not sched.running:
                break
            await asyncio.sleep(0.01)
        assert sched.blocks.free_blocks() == baseline, "disconnect leaked KV blocks"


@pytest.mark.asyncio
async def test_preemption_requeues_rather_than_dropping():
    """Squeeze the block pool until sequences cannot grow, and assert that
    every request still completes -- preempted work is recomputed, never lost."""
    cfg = _cfg(kv_blocks=90, max_batch=8, enable_prefix_cache=False, max_tokens_cap=512)
    async with _App(cfg) as a:
        rs = await _fire(a.client, 10, max_tokens=64, system="s" * 400)
        assert all(r.status_code == 200 for r in rs)
        for r in rs:
            assert r.json()["choices"][0]["message"]["content"]
        assert a.engine.scheduler.blocks.free_blocks() >= 0


@pytest.mark.asyncio
async def test_prefix_cache_serves_a_shared_system_prompt():
    system = "shared system prompt. " * 200
    async with _App(_cfg()) as a:
        await _fire(a.client, 1, max_tokens=4, system=system, user="first")
        await _fire(a.client, 6, max_tokens=4, system=system, user="rest")
        rc = a.engine.scheduler.prefix
        assert rc.hit_tokens > 0, "nothing was reused"
        assert rc.hit_rate > 0.6, f"token-level hit rate only {rc.hit_rate:.2f}"


@pytest.mark.asyncio
async def test_prefix_cache_does_not_change_outputs():
    """The differential test: identical requests, cache on and cache off,
    identical text. Without it, a 'faster' configuration might simply be a
    wrong one."""
    system = "shared system prompt. " * 200
    outs = {}
    for enabled in (False, True):
        async with _App(_cfg(enable_prefix_cache=enabled)) as a:
            rs = await _fire(a.client, 5, max_tokens=12, system=system, user="q")
            outs[enabled] = [r.json()["choices"][0]["message"]["content"] for r in rs]
    assert outs[True] == outs[False]


@pytest.mark.asyncio
async def test_shed_returns_503_with_retry_after():
    async with _App(_cfg(max_waiting=0)) as a:
        r = await a.client.post(
            "/v1/chat/completions",
            json={"model": "qwen", "stream": False, "max_tokens": 4,
                  "messages": [{"role": "user", "content": "x"}]},
        )
        assert r.status_code == 503
        assert float(r.headers["Retry-After"]) > 0
        assert r.json()["error"]["code"] == "slo_shed"


@pytest.mark.asyncio
async def test_sequence_ids_are_conserved_with_the_prefix_cache_on():
    """Sequence ids are held by running requests *and* by cached prefixes.
    Every id must be in exactly one of those places or on the free list."""
    system = "shared. " * 300
    async with _App(_cfg(n_parallel=16, max_batch=4)) as a:
        sched = a.engine.scheduler
        total = sched.seq_ids.n
        for _ in range(4):
            await _fire(a.client, 6, max_tokens=6, system=system, user="q")
        for _ in range(300):
            if not sched.running:
                break
            await asyncio.sleep(0.01)
        owned = len(sched.prefix._seq_refs)
        assert sched.seq_ids.available + owned == total, (
            f"free={sched.seq_ids.available} cache-owned={owned} of {total}"
        )
        # And they all come back when the cache is emptied.
        sched.prefix.evict_all_unused()
        assert sched.seq_ids.available == total


@pytest.mark.asyncio
async def test_paged_allocation_sustains_a_larger_batch_than_contiguous():
    """Step 2.3's exit test.

    Same KV pool, same workload, same everything except the reservation policy:
    contiguous takes a slab sized for the server-wide output cap, paged takes
    the prompt and grows. The batch size that fits is the difference.
    """
    system = "s" * 1600  # ~400 mock tokens of prompt

    async def peak_batch(paged: bool) -> tuple[int, float]:
        cfg = _cfg(
            enable_paged_kv=paged, enable_prefix_cache=False,
            kv_blocks=400, max_batch=16, max_tokens_cap=512, n_parallel=32,
        )
        async with _App(cfg) as a:
            sched = a.engine.scheduler
            seen, frag = [], []

            async def watch():
                while True:
                    seen.append(len(sched.running))
                    live = [lv.rq.n_tokens for lv in sched.running]
                    if live:
                        frag.append(sched.blocks.fragmentation_ratio(live))
                    await asyncio.sleep(0.005)

            w = asyncio.create_task(watch())
            rs = await _fire(a.client, 24, max_tokens=24, system=system, user="q")
            w.cancel()
            assert all(r.status_code == 200 for r in rs)
            return max(seen), (sum(frag) / len(frag) if frag else 1.0)

    contig_batch, contig_frag = await peak_batch(paged=False)
    paged_batch, paged_frag = await peak_batch(paged=True)

    assert paged_batch > contig_batch, (
        f"paged sustained {paged_batch}, contiguous {contig_batch}: no memory win"
    )
    assert paged_frag > 0.9, f"paged fragmentation ratio {paged_frag:.2f}"
    assert paged_frag > contig_frag


@pytest.mark.asyncio
async def test_preemption_actually_fires_when_the_pool_is_outgrown():
    """Lazy allocation means several requests can each pass the admission gate
    and still collectively outgrow the pool. That is the condition preemption
    exists for, so it has to be reachable -- dead code would mean the recompute
    policy was never exercised."""
    from cadence.obs.metrics import preemptions

    # Sized so the arithmetic is visible: a ~141-token prompt is 9 blocks, the
    # admission gate asks for ceil((141+200)/16) = 22, and the pool holds 120.
    # Eleven requests each pass the gate on 9 blocks; grown to 22 apiece they
    # want 242. Something has to give.
    cfg = _cfg(kv_blocks=120, max_batch=16, enable_prefix_cache=False,
               max_tokens_cap=512, config_name="preempt-test")
    before = preemptions.labels(config="preempt-test", policy="recompute")._value.get()
    async with _App(cfg) as a:
        rs = await _fire(a.client, 12, max_tokens=200, system="s" * 140, user="q")
    after = preemptions.labels(config="preempt-test", policy="recompute")._value.get()

    assert all(r.status_code == 200 for r in rs), "preemption lost a request"
    for r in rs:
        assert r.json()["choices"][0]["message"]["content"]
    assert after > before, "the pool was never outgrown; the test proves nothing"
