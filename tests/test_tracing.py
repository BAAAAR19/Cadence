"""Step 2.1's exit test: a single request's trace shows queue -> prefill ->
decode with sensible durations, and every lifecycle transition appears as a
span event.

Spans are collected with an in-memory exporter rather than shipped to a
collector, so this runs in CI.
"""

from __future__ import annotations

import asyncio

import httpx
import pytest
from httpx import ASGITransport
from opentelemetry import trace as otel_trace
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter

from cadence.api.app import create_app
from cadence.config import Settings


@pytest.fixture
def exporter(monkeypatch):
    exp = InMemorySpanExporter()
    provider = TracerProvider()
    provider.add_span_processor(SimpleSpanProcessor(exp))
    # The global provider can only be set once per process, so the module's
    # tracer is repointed directly instead.
    import cadence.obs.tracing as tracing

    monkeypatch.setattr(tracing, "tracer", provider.get_tracer("cadence-test"))
    monkeypatch.setattr(tracing, "setup_tracing", lambda cfg: None)
    monkeypatch.setattr("cadence.api.app.setup_tracing", lambda cfg: None)
    yield exp
    otel_trace.NoOpTracerProvider  # noqa: B018  (keep the import meaningful)


def _cfg(**kw):
    base = dict(
        backend="mock", config_name="test", scheduler="continuous", n_ctx=4096,
        max_batch=8, n_parallel=16, slo_s=60.0, tracing_enabled=True,
    )
    base.update(kw)
    return Settings(**base)


async def _run_one(cfg, **body_kw):
    app = create_app(cfg)
    async with app.router.lifespan_context(app), httpx.AsyncClient(
        transport=ASGITransport(app=app), base_url="http://test"
    ) as c:
        body = {
            "model": "qwen", "stream": False, "max_tokens": 12,
            "messages": [
                {"role": "system", "content": "s" * 400},
                {"role": "user", "content": "hello"},
            ],
        }
        body.update(body_kw)
        r = await c.post("/v1/chat/completions", json=body, timeout=60.0)
        assert r.status_code == 200, r.text
        return app


@pytest.mark.asyncio
async def test_a_single_request_traces_queue_prefill_decode(exporter):
    await _run_one(_cfg())
    spans = exporter.get_finished_spans()
    names = [s.name for s in spans]
    assert "request" in names
    for phase in ("queue", "prefill", "decode"):
        assert phase in names, f"no {phase} span: {names}"

    by_name = {s.name: s for s in spans}
    root = by_name["request"]
    for phase in ("queue", "prefill", "decode"):
        child = by_name[phase]
        assert child.parent is not None
        assert child.parent.span_id == root.context.span_id, (
            f"{phase} is not parented under the request span"
        )
        dur = (child.end_time - child.start_time) / 1e9
        assert 0 < dur < 60, f"{phase} lasted {dur}s"

    # The phases tile the request without overlapping.
    order = sorted(("queue", "prefill", "decode"), key=lambda n: by_name[n].start_time)
    assert order == ["queue", "prefill", "decode"]
    assert by_name["queue"].end_time <= by_name["prefill"].start_time
    assert by_name["prefill"].end_time <= by_name["decode"].start_time

    events = {e.name for e in root.events}
    assert {"waiting", "prefill", "decoding", "done"} <= events, events
    assert root.attributes["cadence.output_tokens"] == 12
    assert root.attributes["cadence.prompt_tokens"] > 0


@pytest.mark.asyncio
async def test_concurrent_requests_do_not_parent_under_each_other(exporter):
    """The engine thread interleaves requests, so a phase span parented under
    whatever span happened to be 'current' would be confidently wrong."""
    cfg = _cfg()
    app = create_app(cfg)
    async with app.router.lifespan_context(app), httpx.AsyncClient(
        transport=ASGITransport(app=app), base_url="http://test"
    ) as c:
        async def one(i):
            return await c.post(
                "/v1/chat/completions",
                json={"model": "qwen", "stream": False, "max_tokens": 10,
                      "messages": [{"role": "user", "content": f"q{i}"}]},
                timeout=60.0,
            )

        rs = await asyncio.gather(*[one(i) for i in range(6)])
    assert all(r.status_code == 200 for r in rs)

    spans = exporter.get_finished_spans()
    roots = {s.context.span_id for s in spans if s.name == "request"}
    assert len(roots) == 6
    phases = [s for s in spans if s.name in {"queue", "prefill", "decode"}]
    assert len(phases) >= 18
    for s in phases:
        assert s.parent.span_id in roots, f"{s.name} parented under a non-request span"


@pytest.mark.asyncio
async def test_tracing_disabled_emits_nothing(exporter):
    await _run_one(_cfg(tracing_enabled=False))
    assert exporter.get_finished_spans() == ()
