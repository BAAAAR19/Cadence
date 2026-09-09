"""The OpenAI-compatible surface.

Step 1.2's exit test is that an unmodified OpenAI Python client streams against
this server, so that is asserted literally rather than approximated with a hand
written SSE parser.
"""

from __future__ import annotations

import asyncio
import json
import socket
import threading
import time

import httpx
import pytest
import uvicorn
from httpx import ASGITransport

from cadence.api.app import create_app
from cadence.config import Settings


def _cfg(**kw):
    base = dict(backend="mock", config_name="test", scheduler="continuous",
                n_ctx=4096, max_batch=8, n_parallel=16, slo_s=60.0)
    base.update(kw)
    return Settings(**base)


class _Server:
    """A real uvicorn on a real socket -- the only way to test a client that
    speaks HTTP rather than ASGI."""

    def __init__(self, cfg) -> None:
        s = socket.socket()
        s.bind(("127.0.0.1", 0))
        self.port = s.getsockname()[1]
        s.close()
        self.config = uvicorn.Config(
            create_app(cfg), host="127.0.0.1", port=self.port,
            log_level="error", access_log=False,
        )
        self.server = uvicorn.Server(self.config)

    def __enter__(self) -> _Server:
        self.thread = threading.Thread(target=self.server.run, daemon=True)
        self.thread.start()
        deadline = time.time() + 30
        while time.time() < deadline and not self.server.started:
            time.sleep(0.05)
        assert self.server.started, "server did not start"
        return self

    @property
    def base_url(self) -> str:
        return f"http://127.0.0.1:{self.port}/v1"

    def __exit__(self, *exc) -> None:
        self.server.should_exit = True
        self.thread.join(timeout=15)


@pytest.fixture
async def client():
    app = create_app(_cfg())
    async with app.router.lifespan_context(app), httpx.AsyncClient(
        transport=ASGITransport(app=app), base_url="http://test"
    ) as c:
        c._app = app
        yield c


@pytest.mark.asyncio
async def test_models_endpoint(client):
    r = await client.get("/v1/models")
    assert r.status_code == 200
    assert r.json()["data"][0]["id"] == "qwen"


@pytest.mark.asyncio
async def test_stream_frames_are_openai_shaped(client):
    payload = {
        "model": "qwen", "stream": True, "max_tokens": 6,
        "messages": [{"role": "user", "content": "hello"}],
    }
    frames, saw_done = [], False
    async with client.stream("POST", "/v1/chat/completions", json=payload) as r:
        assert r.status_code == 200
        assert r.headers["content-type"].startswith("text/event-stream")
        assert r.headers["x-accel-buffering"] == "no"
        async for line in r.aiter_lines():
            if not line.startswith("data: "):
                continue
            if line == "data: [DONE]":
                saw_done = True
                break
            frames.append(json.loads(line[6:]))

    assert saw_done, "stream did not terminate with [DONE]"
    assert frames[0]["choices"][0]["delta"]["role"] == "assistant", (
        "no opening role delta: StreamingResponse may not have flushed, which "
        "inflates TTFT by an amount that has nothing to do with the scheduler"
    )
    assert frames[0]["object"] == "chat.completion.chunk"
    assert frames[-1]["choices"][0]["finish_reason"] in {"stop", "length"}
    assert sum(1 for f in frames if f["choices"][0]["delta"].get("content")) >= 1


@pytest.mark.asyncio
async def test_non_streaming_response_shape(client):
    r = await client.post(
        "/v1/chat/completions",
        json={"model": "qwen", "stream": False, "max_tokens": 5,
              "messages": [{"role": "user", "content": "hi"}]},
    )
    assert r.status_code == 200
    d = r.json()
    assert d["object"] == "chat.completion"
    assert d["choices"][0]["message"]["role"] == "assistant"
    assert d["usage"]["completion_tokens"] == 5
    assert d["usage"]["total_tokens"] == d["usage"]["prompt_tokens"] + 5


@pytest.mark.asyncio
async def test_metrics_endpoint_exposes_the_collectors(client):
    await client.post(
        "/v1/chat/completions",
        json={"model": "qwen", "stream": False, "max_tokens": 4,
              "messages": [{"role": "user", "content": "hi"}]},
    )
    body = (await client.get("/metrics")).text
    for name in (
        "cadence_ttft_seconds", "cadence_e2e_seconds", "cadence_queue_depth",
        "cadence_batch_size", "cadence_kv_blocks_free",
        "cadence_prefix_cache_hit_tokens_total",
    ):
        assert name in body, f"{name} missing from /metrics"


def test_openai_python_client_streams_without_modification():
    """Step 1.2's exit test, taken literally."""
    from openai import OpenAI

    with _Server(_cfg()) as srv:
        oai = OpenAI(base_url=srv.base_url, api_key="not-needed")
        assert [m.id for m in oai.models.list().data] == ["qwen"]

        stream = oai.chat.completions.create(
            model="qwen",
            messages=[{"role": "user", "content": "count to three"}],
            max_tokens=8,
            stream=True,
        )
        deltas = [
            c.choices[0].delta.content
            for c in stream
            if c.choices and c.choices[0].delta.content
        ]
        assert len(deltas) >= 1
        assert "".join(deltas)

        done = oai.chat.completions.create(
            model="qwen",
            messages=[{"role": "user", "content": "and back down"}],
            max_tokens=6,
        )
        assert done.choices[0].message.content
        assert done.usage.completion_tokens == 6


@pytest.mark.asyncio
async def test_concurrent_streams_do_not_interleave_content(client):
    """Each stream must carry only its own tokens. On the mock backend the
    sampled ids are a pure function of the prompt, so cross-contamination is
    detectable exactly."""
    async def one(tag: str) -> str:
        payload = {
            "model": "qwen", "stream": True, "max_tokens": 10,
            "messages": [{"role": "user", "content": tag}],
        }
        out = []
        async with client.stream("POST", "/v1/chat/completions", json=payload) as r:
            async for line in r.aiter_lines():
                if line.startswith("data: ") and line != "data: [DONE]":
                    d = json.loads(line[6:])["choices"][0]["delta"].get("content")
                    if d:
                        out.append(d)
        return "".join(out)

    tags = [f"prompt-{i}" for i in range(6)]
    got = await asyncio.gather(*[one(t) for t in tags])

    expected = {}
    for t in tags:
        payload = {"model": "qwen", "stream": False, "max_tokens": 10,
                   "messages": [{"role": "user", "content": t}]}
        r = await client.post("/v1/chat/completions", json=payload)
        expected[t] = r.json()["choices"][0]["message"]["content"]

    for t, g in zip(tags, got, strict=True):
        assert g == expected[t], f"stream for {t} carried another request's tokens"


def test_dashboard_queries_reference_live_metric_names(tmp_path):
    """A renamed collector must break the build, not silently empty a panel."""
    import json
    import re
    import subprocess
    import sys
    from pathlib import Path

    root = Path(__file__).resolve().parents[1]
    dash = json.loads((root / "deploy/grafana/dashboards/cadence.json").read_text())
    referenced = set()
    for panel in dash["panels"]:
        for target in panel["targets"]:
            for m in re.findall(r"cadence_[a-z0-9_]+", target["expr"]):
                referenced.add(
                    m.removesuffix("_bucket").removesuffix("_sum").removesuffix("_count")
                )
    assert referenced, "the dashboard queries nothing"

    exported = subprocess.run(
        [sys.executable, "-c",
         "import sys; sys.path.insert(0, 'src');"
         "from prometheus_client import generate_latest;"
         "from cadence.obs.metrics import REGISTRY, Metrics;"
         "Metrics('test'); print(generate_latest(REGISTRY).decode())"],
        cwd=root, capture_output=True, text=True, check=True,
    ).stdout
    for name in sorted(referenced):
        assert name in exported, f"dashboard panel queries {name}, which /metrics never exports"
