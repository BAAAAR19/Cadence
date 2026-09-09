"""FastAPI app factory."""

from __future__ import annotations

from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.responses import PlainTextResponse
from prometheus_client import CONTENT_TYPE_LATEST, generate_latest

from cadence.api.routes_openai import router as openai_router
from cadence.config import Settings, get_settings
from cadence.engine.engine import Engine
from cadence.obs.metrics import REGISTRY
from cadence.obs.tracing import setup_tracing


def create_app(cfg: Settings | None = None, engine: Engine | None = None) -> FastAPI:
    cfg = cfg or get_settings()

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        setup_tracing(cfg)
        eng = engine if engine is not None else Engine(cfg)
        eng.start()
        app.state.engine = eng
        app.state.cfg = cfg
        try:
            yield
        finally:
            eng.stop()

    app = FastAPI(title="Cadence", version="0.1.0", lifespan=lifespan)
    app.state.cfg = cfg
    app.include_router(openai_router)

    @app.get("/health")
    async def health(request: Request):
        return {"status": "ok", **request.app.state.engine.stats()}

    @app.get("/stats")
    async def stats(request: Request):
        return request.app.state.engine.stats()

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
    uvicorn.run(
        create_app(cfg),
        host=cfg.host,
        port=cfg.port,
        log_level="warning",
        access_log=False,
        timeout_keep_alive=120,
    )


if __name__ == "__main__":  # pragma: no cover
    main()
