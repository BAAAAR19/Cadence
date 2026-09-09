"""A model-free SSE server used to validate the load generator itself.

Two shapes, both needed by Step 1.4's exit test:

* ``--capacity 0`` -- unbounded: every request takes ``--service 0.1`` s. At
  lambda = 5 rps this should show ~lambda*duration arrivals, inter-arrival
  times that pass a KS test against Exp(lambda), and near-zero queueing.
* ``--capacity N`` -- a hard service limit of N concurrent requests, i.e. a
  server that can do ``N/service`` rps. Offer more than that and a correct
  open-loop generator must show p99 growing without bound. If it plateaus,
  there is still a closed loop somewhere.
"""

from __future__ import annotations

import argparse
import asyncio
import time

from fastapi import FastAPI
from fastapi.responses import StreamingResponse

from cadence.api.sse import DONE, SSE_HEADERS, chunk


def create_stub(service_s: float = 0.1, capacity: int = 0, n_tokens: int = 8) -> FastAPI:
    app = FastAPI()
    sem = asyncio.Semaphore(capacity) if capacity > 0 else None

    @app.post("/v1/chat/completions")
    async def completions(body: dict):
        cid = "stub-" + str(int(time.time() * 1e6))
        toks = int(body.get("max_tokens") or n_tokens)

        async def gen():
            yield chunk(cid, "stub", {"role": "assistant", "content": ""})
            if sem is not None:
                async with sem:
                    await asyncio.sleep(service_s)
            else:
                await asyncio.sleep(service_s)
            for i in range(toks):
                yield chunk(cid, "stub", {"content": f"t{i} "})
            yield chunk(cid, "stub", {}, finish="stop")
            yield DONE

        return StreamingResponse(gen(), media_type="text/event-stream", headers=SSE_HEADERS)

    @app.get("/health")
    async def health():
        return {"status": "ok", "capacity": capacity, "service_s": service_s}

    return app


def main() -> None:  # pragma: no cover - CLI
    import uvicorn

    p = argparse.ArgumentParser()
    p.add_argument("--service", type=float, default=0.1)
    p.add_argument("--capacity", type=int, default=0)
    p.add_argument("--tokens", type=int, default=8)
    p.add_argument("--port", type=int, default=8099)
    a = p.parse_args()
    uvicorn.run(
        create_stub(a.service, a.capacity, a.tokens),
        host="127.0.0.1",
        port=a.port,
        log_level="warning",
        access_log=False,
    )


if __name__ == "__main__":  # pragma: no cover
    main()
