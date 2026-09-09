"""Cross-request KV isolation, against the real model.

This is the test that catches the failure the guide ranks third by cost: a
copy-on-write bug in the paged cache produces *cross-request text
contamination* -- one user's tokens appearing in another user's stream -- and it
is rare, non-deterministic, and terrifying.

The setup is the one that provokes it: two requests with an identical long
system prompt (so the prefix cache shares blocks between them) and different
user turns (so a sharing bug is visible in the output).
"""

from __future__ import annotations

import asyncio

import httpx
import pytest
from httpx import ASGITransport

from cadence.api.app import create_app
from cadence.config import Settings

SYSTEM = (
    "You are a precise assistant. Answer with a single short sentence and no "
    "preamble. Follow the user's instruction exactly. " * 40
)
USERS = [
    "Name the capital of France.",
    "What colour is a ripe banana?",
    "How many days are in a leap year?",
    "Name the largest ocean on Earth.",
]

pytestmark = pytest.mark.slow


def _cfg(model_path, **kw):
    base = dict(
        backend="llamacpp", model_path=model_path, config_name="test",
        scheduler="continuous", n_ctx=8192, n_batch=512, n_parallel=24,
        max_batch=8, temperature=0.0, slo_s=120.0,
    )
    base.update(kw)
    return Settings(**base)


async def _ask(client, user, max_tokens=20):
    r = await client.post(
        "/v1/chat/completions",
        json={
            "model": "qwen", "stream": False, "max_tokens": max_tokens,
            "messages": [
                {"role": "system", "content": SYSTEM},
                {"role": "user", "content": user},
            ],
        },
        timeout=300.0,
    )
    assert r.status_code == 200, r.text
    return r.json()["choices"][0]["message"]["content"]


class _App:
    def __init__(self, cfg):
        self.cfg = cfg

    async def __aenter__(self):
        self.app = create_app(self.cfg)
        self._ls = self.app.router.lifespan_context(self.app)
        await self._ls.__aenter__()
        self.client = httpx.AsyncClient(
            transport=ASGITransport(app=self.app), base_url="http://test"
        )
        return self

    @property
    def engine(self):
        return self.app.state.engine

    async def __aexit__(self, *e):
        await self.client.aclose()
        await self._ls.__aexit__(*e)


@pytest.mark.asyncio
async def test_shared_system_prompt_does_not_contaminate_outputs(model_path):
    # Ground truth: each request alone, no sharing possible.
    async with _App(_cfg(model_path, enable_prefix_cache=False, max_batch=1)) as a:
        alone = [await _ask(a.client, u) for u in USERS]

    # Now concurrently, with the prefix cache sharing the system prompt's KV.
    # One request first: entries are inserted when a request completes, so a
    # cold cache serves nothing to a simultaneous burst.
    async with _App(_cfg(model_path, enable_prefix_cache=True, max_batch=8)) as a:
        await _ask(a.client, "Warm the cache.", max_tokens=4)
        a.engine.scheduler.prefix.query_tokens = 0
        a.engine.scheduler.prefix.hit_tokens = 0
        together = await asyncio.gather(*[_ask(a.client, u) for u in USERS])
        hit_rate = a.engine.scheduler.prefix.hit_rate

    assert hit_rate > 0.5, f"the prompt was not actually shared (hit rate {hit_rate:.2f})"
    for user, want, got in zip(USERS, alone, together, strict=True):
        assert got == want, f"{user!r}\n  alone:    {want!r}\n  together: {got!r}"


@pytest.mark.asyncio
async def test_prefix_cache_on_and_off_agree(model_path):
    outs = {}
    for enabled in (False, True):
        async with _App(_cfg(model_path, enable_prefix_cache=enabled)) as a:
            outs[enabled] = await asyncio.gather(*[_ask(a.client, u) for u in USERS])
    assert outs[True] == outs[False]


@pytest.mark.asyncio
async def test_paged_and_contiguous_allocators_agree(model_path):
    outs = {}
    for paged in (False, True):
        async with _App(_cfg(model_path, enable_paged_kv=paged,
                             enable_prefix_cache=False)) as a:
            outs[paged] = await asyncio.gather(*[_ask(a.client, u) for u in USERS])
    assert outs[True] == outs[False]


@pytest.mark.asyncio
async def test_no_kv_or_sequence_leak_over_many_requests(model_path):
    async with _App(_cfg(model_path, enable_prefix_cache=False)) as a:
        sched = a.engine.scheduler
        baseline_blocks = sched.blocks.free_blocks()
        baseline_seqs = sched.seq_ids.available
        for _ in range(3):
            await asyncio.gather(*[_ask(a.client, u, max_tokens=8) for u in USERS])
        for _ in range(400):
            if sched.blocks.free_blocks() == baseline_blocks and not sched.running:
                break
            await asyncio.sleep(0.02)
        assert sched.blocks.free_blocks() == baseline_blocks
        assert sched.seq_ids.available == baseline_seqs
