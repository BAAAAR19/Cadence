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
async def test_sequence_ids_are_conserved_with_the_prefix_cache_on(kv_core):
    """Sequence ids are held by running requests *and* by cached prefixes.
    Every id must be in exactly one of those places or on the free list."""
    system = "shared. " * 300
    async with _App(_cfg(n_parallel=16, max_batch=4, kv_core=kv_core)) as a:
        sched = a.engine.scheduler
        total = sched.seq_ids.n
        for _ in range(4):
            await _fire(a.client, 6, max_tokens=6, system=system, user="q")
        for _ in range(300):
            if not sched.running:
                break
            await asyncio.sleep(0.01)
        owned = sched.prefix.n_owned_sequences()
        assert sched.seq_ids.available + owned == total, (
            f"free={sched.seq_ids.available} cache-owned={owned} of {total}"
        )
        # And they all come back when the cache is emptied.
        sched.prefix.evict_all_unused()
        assert sched.seq_ids.available == total


@pytest.mark.asyncio
async def test_paged_allocation_sustains_a_larger_batch_than_contiguous(kv_core):
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
            kv_core=kv_core,
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


# --- the pinned-match invariant ------------------------------------------
#
# Found by the Week 3 profiling run, which crashed the engine thread inside a
# minute at 4 rps on the mock backend: admission matched a prefix, then found
# itself short of blocks, then reclaimed by evicting the LRU leaf -- which was
# the node it had just matched. See ContinuousScheduler._pinned_match.


def _bare_scheduler(**kw):
    """A scheduler with no HTTP layer, so a single ``_schedule()`` call can be
    inspected."""
    from cadence.engine.backends.mock import MockRunner
    from cadence.engine.kv import build_kv
    from cadence.engine.scheduler.continuous import ContinuousScheduler
    from cadence.obs.metrics import Metrics

    cfg = _cfg(**kw)
    runner = MockRunner(cfg)
    blocks, prefix, _ = build_kv(cfg)
    sched = ContinuousScheduler(runner, cfg, Metrics("test", False), blocks=blocks,
                                prefix_cache=prefix)
    sched.prefix.on_seq_released = sched._release_owner_seq
    return sched


def _queue(sched, prompt_ids, max_tokens=8):
    import time as _time

    from cadence.engine.request import Request

    rq = Request(rid=f"r{len(sched.waiting)}", prompt_ids=list(prompt_ids),
                 max_tokens=max_tokens, deadline=_time.perf_counter() + 60.0)
    sched.waiting.append(rq)
    return rq


@pytest.mark.asyncio
async def test_reclaim_cannot_evict_the_prefix_admission_just_matched(kv_core):
    """The regression test for the crash: a request whose prefix is cached,
    admitted into a pool with too few free blocks, must not have that prefix
    evicted out from under it."""
    # 24 blocks of 16 tokens: room for one cached prefix and not much else, so
    # admission is forced down the reclaim path.
    sched = _bare_scheduler(kv_blocks=24, block_size=16, max_batch=4, n_parallel=8,
                            kv_core=kv_core)
    B = sched.blocks.block_size

    # A cached prefix of 8 blocks, donated by a finished request.
    shared_tokens = list(range(1, 8 * B + 1))
    blocks = sched.blocks.alloc(8)
    node = sched.prefix.insert(shared_tokens, blocks, owner_seq=0)
    sched.blocks.release(blocks)  # the cache holds the only reference now
    assert node is not None
    assert sched.prefix.match(shared_tokens + [999]).n_tokens == 8 * B

    # Occupy most of what is left, so admission must reclaim to fit.
    hog = sched.blocks.alloc(sched.blocks.free_blocks() - 2)

    rq = _queue(sched, shared_tokens + list(range(9000, 9000 + 2 * B)))
    prefill, _ = sched._schedule()  # used to raise "cannot share free block"

    if prefill:  # admitted: it must actually be holding the shared blocks
        assert rq.cached_prefix_len == 8 * B
        for b in rq.block_ids[:8]:
            assert sched.blocks.refcount[b] > 0
    else:  # or it declined to admit -- but it must not have spent the prefix
        assert sched.prefix.match(shared_tokens + [999]).n_tokens == 8 * B
    sched.blocks.release(hog)


@pytest.mark.asyncio
async def test_a_declined_admission_neither_spends_nor_leaks_the_prefix(kv_core):
    """The pin is a loan for the length of one decision, and both ways of
    getting it wrong are failures.

    Without it, a request that reclaims and then declines to admit has spent
    the cached prefix to buy a batch slot it did not take -- the node is gone
    when the next request asks for it. If the loan were never returned, the
    node could not be evicted again and the cache would fill with
    unreclaimable entries."""
    sched = _bare_scheduler(kv_blocks=24, block_size=16, max_batch=4, n_parallel=8,
                            kv_core=kv_core)
    B = sched.blocks.block_size
    shared_tokens = list(range(1, 4 * B + 1))
    blocks = sched.blocks.alloc(4)
    node = sched.prefix.insert(shared_tokens, blocks, owner_seq=0)
    sched.blocks.release(blocks)
    refs_before = node.refs

    hog = sched.blocks.alloc(sched.blocks.free_blocks())  # nothing left to admit into
    _queue(sched, shared_tokens + list(range(9000, 9000 + 8 * B)))
    sched._schedule()

    assert sched.prefix.match(shared_tokens + [999]).n_tokens == 4 * B, (
        "a declined admission spent the cached prefix"
    )
    assert node.refs == refs_before, "admission leaked a reference to the matched node"
    sched.blocks.release(hog)
    assert sched.prefix.evict(4) == 4, "the node stayed pinned after the decision"


# --- which KV core gets loaded -------------------------------------------


def test_kv_core_auto_falls_back_but_an_explicit_request_does_not(monkeypatch):
    """`auto` degrades quietly to the Python reference; `cpp` refuses to.

    The asymmetry is the point. A gateway on a machine without a compiler
    should still run; a *benchmark* that believes it measured the extension
    and quietly measured Python is worse than one that failed.
    """
    import cadence.engine.kv as kvmod
    from cadence.engine.kv import core_name

    monkeypatch.setattr(kvmod, "core_available", lambda: False)
    assert core_name("auto") == "python"
    assert core_name("python") == "python"
    with pytest.raises(RuntimeError, match="not built"):
        core_name("cpp")

    monkeypatch.setattr(kvmod, "core_available", lambda: True)
    assert core_name("auto") == "cpp"
    assert core_name("python") == "python", "an explicit choice is never overridden"


@pytest.mark.asyncio
async def test_stats_reports_which_kv_core_is_running(kv_core):
    """Every run records the implementation that produced it; the A/B in
    Week 3 is only interpretable because this is in the parquet's provenance."""
    async with _App(_cfg(kv_core=kv_core)) as a:
        r = await a.client.get("/stats")
        assert r.json()["kv_core"] == kv_core
