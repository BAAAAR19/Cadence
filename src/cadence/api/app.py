"""FastAPI app factory."""

from __future__ import annotations

from contextlib import asynccontextmanager
from typing import Any

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, PlainTextResponse
from prometheus_client import CONTENT_TYPE_LATEST, generate_latest

from cadence.api.lifecycle import Lifecycle
from cadence.api.routes_openai import router as openai_router
from cadence.config import Settings, get_settings
from cadence.engine.engine import Engine
from cadence.obs.metrics import REGISTRY
from cadence.obs.tracing import setup_tracing


def create_app(cfg: Settings | None = None, engine: Engine | None = None) -> FastAPI:
    cfg = cfg or get_settings()
    life = Lifecycle(max_concurrent=cfg.max_concurrent_requests)

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        setup_tracing(cfg)
        eng = engine if engine is not None else Engine(cfg)
        eng.start()
        app.state.engine = eng
        app.state.cfg = cfg
        # Only now: ``Engine.start`` runs a generation through the scheduler
        # before returning, so this is the first moment at which a request
        # routed here would not be paying for the model load.
        life.mark_ready()
        prof = None
        if cfg.profile_out:
            from cadence.obs.profiler import SamplingProfiler

            # Started after the scheduler thread exists, stopped before it is
            # joined, so every sample it takes is of a thread that is running.
            prof = SamplingProfiler(cfg.profile_out, hz=cfg.profile_hz)
            prof.start()
        try:
            yield
        finally:
            # Reached only once uvicorn has finished waiting for the in-flight
            # streams, so the batch is already drained by the time the
            # scheduler is stopped. See ``DrainingServer``.
            life.begin_drain()
            if prof is not None:
                prof.stop()
            eng.stop()

    app = FastAPI(title="Cadence", version="0.1.0", lifespan=lifespan)
    app.state.cfg = cfg
    app.state.lifecycle = life
    app.include_router(openai_router)

    @app.get("/health")
    async def health(request: Request):
        """Liveness. Answers "is this process alive", and deliberately keeps
        answering 200 while draining -- a supervisor that restarts a process
        for being mid-drain is fighting the drain."""
        return {"status": "ok", **request.app.state.engine.stats()}

    @app.get("/ready")
    async def ready(request: Request):
        """Readiness. Answers "should traffic be sent here", which is false
        both before the model has loaded and after SIGTERM. This is the
        endpoint a load balancer polls; see ``cadence.api.lifecycle``."""
        life: Lifecycle = request.app.state.lifecycle
        body: dict[str, Any] = {
            "status": "ready" if life.serving else ("draining" if life.draining else "loading"),
            **life.status(),
        }
        if not life.serving:
            return JSONResponse(status_code=503, content=body)
        return body

    @app.get("/stats")
    async def stats(request: Request):
        return {
            **request.app.state.engine.stats(),
            **request.app.state.lifecycle.status(),
        }

    @app.get("/metrics")
    async def metrics():
        return PlainTextResponse(
            generate_latest(REGISTRY).decode(), media_type=CONTENT_TYPE_LATEST
        )

    if cfg.tracing_enabled:
        from opentelemetry.instrumentation.fastapi import FastAPIInstrumentor

        FastAPIInstrumentor.instrument_app(app)

    return app


def main() -> None:  # pragma: no cover - entry point
    import uvicorn

    cfg = get_settings()
    app = create_app(cfg)

    class DrainingServer(uvicorn.Server):
        """Graceful shutdown, in the order an operator needs it.

        uvicorn's own handling of SIGTERM already stops accepting connections
        and waits for the in-flight ones, which is most of a drain. What it
        does not do is tell anyone: ``/ready`` would keep answering 200 until
        the socket closed, so a load balancer polling on a 10 s interval
        keeps routing new requests into a process that is going away, and
        those are the requests that get refused or truncated.

        So the flag is flipped first, in the signal handler, and only then is
        the normal shutdown allowed to proceed:

            SIGTERM -> /ready 503 and new requests refused with Retry-After
                    -> the running batch finishes streaming (bounded by
                       ``drain_grace_s``)
                    -> lifespan shutdown stops the scheduler and the backend
        """

        def handle_exit(self, sig: int, frame: object) -> None:
            app.state.lifecycle.begin_drain()
            super().handle_exit(sig, frame)  # type: ignore[arg-type]

    server = DrainingServer(
        uvicorn.Config(
            app,
            host=cfg.host,
            port=cfg.port,
            log_level="warning",
            access_log=False,
            timeout_keep_alive=120,
            timeout_graceful_shutdown=int(cfg.drain_grace_s),
            # The API layer's own bound. The cap in ``Lifecycle`` refuses with
            # a 503 and a Retry-After, which a client can act on; this one is
            # the backstop underneath it, at the socket, and exists so that a
            # burst arriving faster than the event loop can refuse it still
            # cannot exhaust memory.
            limit_concurrency=max(cfg.max_concurrent_requests * 2, 64),
        )
    )
    server.run()


if __name__ == "__main__":  # pragma: no cover
    main()
